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


class GridSelfAttentionCpp(nn.Module):
    def __init__(self, c_pair: int = 128, num_head: int = 4, transpose: bool = False):
        super(GridSelfAttentionCpp, self).__init__()
        self.c_pair = c_pair
        self.num_head = num_head
        self.qkv_dim = self.c_pair // self.num_head
        self.transpose = transpose
        self.act_norm = nn.LayerNorm(self.c_pair)
        self.pair_bias_projection = nn.Linear(self.c_pair, self.num_head, bias=False)

        self.q_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.k_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.v_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.gating_query = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.output_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.first_run = True
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.gq = None
        self.out_proj = None

    @profile("C++ GridSelfAttention")
    def forward(self, pair: torch.Tensor, mask: torch.Tensor):
        BLK_SZ = 128 if pair.shape[1] > 256 else 64

        pair = self.act_norm(pair)
        bias = self.pair_bias_projection(pair).permute(2, 0, 1)

        if self.transpose:
            pair = pair.permute(1, 0, 2)
        if self.first_run:
            self.first_run = False
            self.q_proj = kernels.vnni_repack_tensor(self.q_projection.weight.T.contiguous(), 2, self.qkv_dim)
            self.k_proj = kernels.vnni_repack_tensor(self.k_projection.weight.T.contiguous(), 2, self.qkv_dim)
            self.v_proj = kernels.vnni_repack_tensor(self.v_projection.weight.T.contiguous(), 2, self.qkv_dim)
            self.gq = kernels.vnni_repack_tensor(self.gating_query.weight.T.contiguous(), 2, self.qkv_dim)
            self.out_proj = self.output_projection.weight.T.contiguous()

        pair = kernels._attention_cpp(
            pair,
            mask,
            bias,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.gq,
            self.out_proj,
            self.num_head,
            False,
            BLK_SZ,
            0,
        )

        if self.transpose:
            pair = pair.permute(1, 0, 2)

        return pair


class DistributedGridSelfAttentionCpp(nn.Module):
    def __init__(self, c_pair: int = 128, num_head: int = 4, transpose: bool = False):
        super(DistributedGridSelfAttentionCpp, self).__init__()
        self.c_pair = c_pair
        self.num_head = num_head
        self.qkv_dim = self.c_pair // self.num_head
        self.transpose = transpose

        self.act_norm = nn.LayerNorm(self.c_pair)
        self.pair_bias_projection = nn.Linear(self.c_pair, self.num_head, bias=False)

        self.q_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.k_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.v_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.gating_query = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.output_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.first_run = True
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.gq = None
        self.out_proj = None

    def _attention(self, pair: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        BLK_SZ = 128 if pair.shape[1] > 256 else 64
        if self.first_run:
            self.first_run = False
            self.q_proj = kernels.vnni_repack_tensor(self.q_projection.weight.T.contiguous(), 2, self.qkv_dim)
            self.k_proj = kernels.vnni_repack_tensor(self.k_projection.weight.T.contiguous(), 2, self.qkv_dim)
            self.v_proj = kernels.vnni_repack_tensor(self.v_projection.weight.T.contiguous(), 2, self.qkv_dim)
            self.gq = kernels.vnni_repack_tensor(self.gating_query.weight.T.contiguous(), 2, self.qkv_dim)
            self.out_proj = self.output_projection.weight.T.contiguous()

        out = kernels._attention_cpp(
            pair,
            mask,
            bias,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.gq,
            self.out_proj,
            self.num_head,
            False,
            BLK_SZ,
            0,
        )
        return out.contiguous()

    @profile("C++ Distributed GridSelfAttention")
    def forward(self, pair: torch.Tensor, mask: torch.Tensor, COMM_SPLIT: int = 4) -> torch.Tensor:
        """
        Args:
            pair (torch.Tensor): [N_token, N_token, c_pair]
            mask (torch.Tensor): [N_token, N_token]
        Returns:
            torch.Tensor: [N_token, N_token, c_pair]
        """
        rk, ws = dist.get_rank(), dist.get_world_size()
        seq_len = pair.shape[0]
        split_dim = 0
        split_size = ceil(seq_len / ws)
        split_slice = slice(split_size * rk, split_size * (rk + 1))
        local_mask = pad_to_size(mask[split_slice], split_size, split_dim)

        pair = self.act_norm(pair)
        nonbatched_bias = self.pair_bias_projection(pair).permute(2, 0, 1).contiguous()
        if self.transpose:
            pair = pair.permute(1, 0, 2)
        chunk = pad_to_size(pair[split_slice], split_size, split_dim)

        chunks_raw = torch.empty((split_size * ws, *pair.shape[1:]), device=pair.device, dtype=pair.dtype)

        PIECE = ceil(split_size / COMM_SPLIT)

        compute_queue = queue.Queue(maxsize=-1)
        stop_flag = threading.Event()

        def communication_worker():
            current_work = None
            while not stop_flag.is_set() or not compute_queue.empty():
                i, out = compute_queue.get()

                begin = i * PIECE
                end = (i + 1) * PIECE if i < COMM_SPLIT - 1 else split_size
                gathered = [chunks_raw[j * split_size + begin : j * split_size + end] for j in range(ws)]

                if current_work is not None:
                    current_work.wait()

                current_work = dist.all_gather(gathered, out, async_op=True)
                compute_queue.task_done()

            if current_work is not None:
                current_work.wait()

        comm_thread = threading.Thread(target=communication_worker)
        comm_thread.start()

        for i in range(COMM_SPLIT):
            begin = i * PIECE
            end = (i + 1) * PIECE if i < COMM_SPLIT - 1 else split_size
            out = self._attention(chunk[begin:end], local_mask[begin:end], nonbatched_bias)
            compute_queue.put((i, out))

        start = time.time()
        stop_flag.set()
        compute_queue.join()
        comm_thread.join()
        record_comm_time("AG Attn", time.time() - start, chunks_raw.numel() * chunks_raw.itemsize * (ws - 1) / ws)

        pair = chunks_raw.narrow(split_dim, 0, seq_len).contiguous()

        if self.transpose:
            pair = pair.permute(1, 0, 2)

        return pair
