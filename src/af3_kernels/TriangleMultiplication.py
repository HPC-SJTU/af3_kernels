###############################################################################
# Copyright (c) 2025 Xflops - All rights reserved.
#
# For information on the license, see the LICENSE file.
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Dragon Archer (Xflops)
###############################################################################

import torch
import torch.nn as nn
import torch.distributed as dist

import threading
import time
import queue
from math import ceil

import af3_kernels as kernels
from .tools import profile, pad_to_size, record_comm_time


class TriangleMultiplicationCpp(nn.Module):
    def __init__(self, c_pair: int = 128, _outgoing: bool = True) -> None:
        super(TriangleMultiplicationCpp, self).__init__()

        self.c_pair = c_pair
        self.left_norm_input = nn.LayerNorm(self.c_pair)
        self.projection = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.gate = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.center_norm = nn.LayerNorm(self.c_pair)
        self.output_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.gating_linear = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self._outgoing = _outgoing
        self.equation = "ckj,cki->cij"
        if _outgoing is True:
            self.equation = "cik,cjk->cij"

        self.first_run = True
        self.proj_gate_weight = None
        self.output_projection_weight = None
        self.gating_linear_weight = None

    @profile("C++ Triangle Multiplication")
    def forward(
        self,
        act: torch.Tensor,
        mask: torch.Tensor,
        BLK_SZ: int = 128,  # will be ignored if not compiled with DYNAMIC_TILING=1
    ) -> torch.Tensor:
        if self.first_run:  # Lazy init to allow dtype casting
            self.first_run = False
            self.proj_gate_weight = torch.cat([self.projection.weight, self.gate.weight], dim=0).contiguous()
            self.output_projection_weight = kernels.vnni_repack_tensor(
                self.output_projection.weight.T.contiguous(), 2, 0
            )
            self.gating_linear_weight = kernels.vnni_repack_tensor(self.gating_linear.weight.T.contiguous(), 2, 0)

        if act.size(0) * 2 < BLK_SZ:
            BLK_SZ = 16  # reduce overhead for small tensors

        mask = mask[..., None].to(act.dtype)
        out = kernels._C.af3_traingle_multiplication(
            act,
            mask,
            self._outgoing,
            self.left_norm_input.weight,
            self.left_norm_input.bias,
            self.proj_gate_weight,
            self.center_norm.weight,
            self.center_norm.bias,
            self.output_projection_weight,
            self.gating_linear_weight,
            BLK_SZ,
        )
        return out


class DistributedTriangleMultiplicationCpp(nn.Module):
    def __init__(self, c_pair: int = 128, _outgoing: bool = True) -> None:
        super(DistributedTriangleMultiplicationCpp, self).__init__()

        self.c_pair = c_pair
        self.left_norm_input = nn.LayerNorm(self.c_pair)
        self.projection = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.gate = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.center_norm = nn.LayerNorm(self.c_pair)
        self.output_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.gating_linear = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self._outgoing = _outgoing
        self.equation = "ckj,cki->cij"
        if _outgoing is True:
            self.equation = "cik,cjk->cij"

        self.first_run = True
        self.proj_gate_weight = None
        self.output_projection_weight = None
        self.gating_linear_weight = None

    @profile("C++ Distributed Triangle Multiplication")
    def forward(
        self,
        act: torch.Tensor,
        mask: torch.Tensor,
        COMM_SPLIT: int = 4,  # Communication split size
    ) -> torch.Tensor:
        rk, ws = dist.get_rank(), dist.get_world_size()
        split_size = ceil(self.c_pair / ws)
        split_slice = slice(2 * split_size * rk, 2 * split_size * (rk + 1))

        if self.first_run:  # Lazy init to allow dtype casting
            self.first_run = False
            self.proj_gate_weight = pad_to_size(
                torch.cat([self.projection.weight[split_slice], self.gate.weight[split_slice]], dim=0),
                4 * split_size,
                0,
            )
            self.gating_linear_weight = kernels.vnni_repack_tensor(self.gating_linear.weight.T.contiguous(), 2, 0)
            self.output_projection_weight = kernels.vnni_repack_tensor(
                self.output_projection.weight.T.contiguous(), 2, 0
            )

        mask = mask[..., None].to(act.dtype)
        BLK_SZ = 128 if act.shape[0] > 512 else 64

        input = act
        act, mid = kernels._C.traingle_multiplication_pre(
            act,
            mask,
            self.left_norm_input.weight,
            self.left_norm_input.bias,
            self.proj_gate_weight,
            self.gating_linear_weight,
            BLK_SZ,
        )

        gathered_raw = torch.empty((mid.shape[0] * ws // 2, *mid.shape[1:]), device=mid.device, dtype=mid.dtype)

        PIECE = ceil(split_size / COMM_SPLIT)

        comm_queue = queue.Queue(maxsize=-1)
        stop_flag = threading.Event()

        def communication_worker():
            current_work = None
            while not stop_flag.is_set() or not comm_queue.empty():
                i, out = comm_queue.get()

                begin = i * PIECE
                end = (i + 1) * PIECE if i < COMM_SPLIT - 1 else split_size
                gathered = [gathered_raw[j * split_size + begin : j * split_size + end] for j in range(ws)]

                if current_work is not None:
                    current_work.wait()

                current_work = dist.all_gather(gathered, out, async_op=False)
                comm_queue.task_done()

            if current_work is not None:
                current_work.wait()

        comm_thread = threading.Thread(target=communication_worker)
        comm_thread.start()

        for i in range(COMM_SPLIT):
            out = kernels._C.traingle_multiplication_einsum(
                mid,
                self._outgoing,
                i * PIECE,
                (i + 1) * PIECE if i < COMM_SPLIT - 1 else split_size,
                BLK_SZ,
            )
            comm_queue.put((i, out))

        start = time.time()
        stop_flag.set()
        comm_queue.join()
        comm_thread.join()
        record_comm_time("AG TM", time.time() - start, gathered_raw.numel() * gathered_raw.itemsize * (ws - 1) / ws)

        out = gathered_raw.narrow(0, 0, self.c_pair).contiguous()

        out = kernels._C.traingle_multiplication_post(
            act,
            out,
            input,
            self.center_norm.weight,
            self.center_norm.bias,
            self.output_projection_weight,
            BLK_SZ,
        )
        return out
