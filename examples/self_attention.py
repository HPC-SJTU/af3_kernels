#!/usr/bin/env python3
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

import time
import einops
from typing import Optional

import af3_kernels
from af3_kernels import dot_product_attention_cpp, self_attention_cpp, vnni_repack_tensor

torch.manual_seed(1896)


class AttentionTorch(nn.Module):
    def __init__(
        self, c_x: int = 768, c_single_cond: int = 384, num_head: int = 16, use_single_cond: bool = False
    ) -> None:

        super(AttentionTorch, self).__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.num_head = num_head

        self.qkv_dim = self.c_x // self.num_head
        self.use_single_cond = use_single_cond

        self.q_projection = nn.Linear(self.c_x, self.c_x, bias=True)
        self.k_projection = nn.Linear(self.c_x, self.c_x, bias=False)
        self.v_projection = nn.Linear(self.c_x, self.c_x, bias=False)
        self.gating_query = nn.Linear(self.c_x, self.c_x, bias=False)

        self.first_run = True
        self.padded_pair_logits = None

    def dot_product_attention_torch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ):
        scaling = q.size(-1) ** -0.5
        q = q * scaling
        logits = torch.matmul(q, k.transpose(-1, -2))

        if bias is not None:
            logits += bias

        if mask is not None:
            if mask.dim() == 1:
                mask = mask[None, None, None, :].to(dtype=torch.bool)
            elif mask.dim() == 2:
                mask = mask[:, None, None, :].to(dtype=torch.bool)
            logits.masked_fill_(~mask, -1e9)

        weights = torch.nn.functional.softmax(logits, dim=-1)
        return torch.matmul(weights, v)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pair_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (num_tokens, ch)
            mask (torch.Tensor): (num_tokens,)
            pair_logits (torch.Tensor, optional): (num_heads, num_tokens, num_tokens)
        """
        with torch.profiler.record_function("Torch Attention"):
            q = self.q_projection(x)
            k = self.k_projection(x)
            v = self.v_projection(x)

            q, k, v = map(lambda t: einops.rearrange(t, "n (h c) -> h n c", h=self.num_head).unsqueeze(0), [q, k, v])

            weighted_avg = self.dot_product_attention_torch(q, k, v, mask=mask, bias=pair_logits)

            weighted_avg = weighted_avg.squeeze(0)
            weighted_avg = einops.rearrange(weighted_avg, "h q c -> q (h c)")

            gate_logits = self.gating_query(x)
            weighted_avg *= torch.sigmoid(gate_logits)
            return weighted_avg


class AttentionMix(nn.Module):
    def __init__(
        self, c_x: int = 768, c_single_cond: int = 384, num_head: int = 16, use_single_cond: bool = False
    ) -> None:

        super(AttentionMix, self).__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.num_head = num_head

        self.qkv_dim = self.c_x // self.num_head
        self.use_single_cond = use_single_cond

        self.q_projection = nn.Linear(self.c_x, self.c_x, bias=True)
        self.k_projection = nn.Linear(self.c_x, self.c_x, bias=False)
        self.v_projection = nn.Linear(self.c_x, self.c_x, bias=False)
        self.gating_query = nn.Linear(self.c_x, self.c_x, bias=False)

        self.first_run = True
        self.padded_pair_logits = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pair_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (num_tokens, ch)
            mask (torch.Tensor): (num_tokens,)
            pair_logits (torch.Tensor, optional): (num_heads, num_tokens, num_tokens)
        """
        with torch.profiler.record_function("Mixed Attention"):
            if self.first_run:
                self.first_run = False
                padded_size = ((x.shape[0] + BLK_SZ - 1) // BLK_SZ) * BLK_SZ
                self.padded_pair_logits = af3_kernels.pad_and_align_tensor(
                    pair_logits, (self.num_head, padded_size, padded_size)
                )

            q = self.q_projection(x)
            k = self.k_projection(x)
            v = self.v_projection(x)

            q, k, v = map(lambda t: einops.rearrange(t, "n (h c) -> h n c", h=self.num_head).unsqueeze(0), [q, k, v])

            weighted_avg = dot_product_attention_cpp(
                q, k, v, mask, self.padded_pair_logits, USE_FLASH, BLK_SZ, FLASH_SZ
            )

            weighted_avg = weighted_avg.squeeze(0)
            weighted_avg = einops.rearrange(weighted_avg, "h q c -> q (h c)")

            gate_logits = self.gating_query(x)
            weighted_avg *= torch.sigmoid(gate_logits)
            return weighted_avg


class AttentionCpp(nn.Module):
    def __init__(
        self, c_x: int = 768, c_single_cond: int = 384, num_head: int = 16, use_single_cond: bool = False
    ) -> None:

        super(AttentionCpp, self).__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.num_head = num_head

        self.qkv_dim = self.c_x // self.num_head
        self.use_single_cond = use_single_cond

        self.q_projection = nn.Linear(self.c_x, self.c_x, bias=True)
        self.k_projection = nn.Linear(self.c_x, self.c_x, bias=False)
        self.v_projection = nn.Linear(self.c_x, self.c_x, bias=False)
        self.gating_query = nn.Linear(self.c_x, self.c_x, bias=False)

        self.first_run = True
        self.padded_pair_logits = None
        self.q_proj = None
        self.q_bias = None
        self.q_bias = None
        self.k_proj = None
        self.v_proj = None
        self.gq = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pair_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.profiler.record_function("C++ Attention"):

            if self.first_run:
                self.first_run = False
                padded_size = ((x.shape[0] + BLK_SZ - 1) // BLK_SZ) * BLK_SZ
                self.padded_pair_logits = af3_kernels.pad_and_align_tensor(
                    pair_logits, (self.num_head, padded_size, padded_size)
                )

                self.q_proj = vnni_repack_tensor(self.q_projection.weight.T.contiguous(), 2, self.qkv_dim)
                self.q_bias = self.q_projection.bias.contiguous()
                self.k_proj = vnni_repack_tensor(self.k_projection.weight.T.contiguous(), 2, self.qkv_dim)
                self.v_proj = vnni_repack_tensor(self.v_projection.weight.T.contiguous(), 2, self.qkv_dim)
                self.gq = vnni_repack_tensor(self.gating_query.weight.T.contiguous(), 2, self.qkv_dim)

            out = self_attention_cpp(
                x,
                mask,
                self.padded_pair_logits,
                self.q_proj,
                self.q_bias,
                self.k_proj,
                self.v_proj,
                self.gq,
                self.num_head,
                USE_FLASH,
                BLK_SZ,
                FLASH_SZ,
            )
            return out


############## Config ##############
set_size = 3

if set_size == 1:
    length = 37
    qkv_shape = (length, 4, length, 32)
    bias_shape = (4, length, length)
    mask_shape = (length, length)
    COUNT = 1000
elif set_size == 2:
    length = 512
    c_pair = 128
    num_head = 4
    pair_shape = (length, length, c_pair)
    bias_shape = (num_head, c_pair)
    mask_shape = (length, length)
    proj_shape = (c_pair, c_pair)
    COUNT = 5
elif set_size == 3:
    length = 1536
    c_pair = 768
    num_head = 16
    pair_shape = (length, c_pair)
    bias_shape = (num_head, length, length)
    mask_shape = (length,)
    proj_shape = (c_pair, c_pair)
    proj_bias_shape = (c_pair,)
    COUNT = 1000
elif set_size == 4:
    length = 1491
    c_pair = 128
    num_head = 4
    pair_shape = (length // 2, length, c_pair)
    bias_shape = (4, length, length)
    mask_shape = (length // 2, length)
    proj_shape = (c_pair, c_pair)
    proj_bias_shape = (c_pair,)
    COUNT = 20

WARM = 50
DTYPE = torch.bfloat16
BLK_SZ = 32
FLASH_SZ = 768
USE_FLASH = False
TEST_ACCURACY = True
DO_PROFILE = False
SCALE = 2.0
BIAS = 3.0

######## Initialize ########

if TEST_ACCURACY:
    pair = torch.randn(pair_shape, dtype=DTYPE)
    mask = torch.ones(mask_shape, dtype=torch.bool)
    bias = torch.randn(bias_shape, dtype=DTYPE) * SCALE + BIAS

    q_proj = torch.randn(proj_shape, dtype=DTYPE) * SCALE + BIAS
    q_bias = torch.randn(proj_bias_shape, dtype=DTYPE) * SCALE + BIAS
    k_proj = torch.randn(proj_shape, dtype=DTYPE) * SCALE + BIAS
    v_proj = torch.randn(proj_shape, dtype=DTYPE) * SCALE + BIAS
    gq = torch.randn(proj_shape, dtype=DTYPE) * SCALE + BIAS
else:
    pair = torch.ones(pair_shape, dtype=DTYPE)
    mask = torch.ones(mask_shape, dtype=torch.bool)
    bias = torch.zeros(bias_shape, dtype=DTYPE)

    q_proj = torch.ones(proj_shape, dtype=DTYPE)
    q_bias = torch.zeros(proj_bias_shape, dtype=DTYPE)
    k_proj = torch.ones(proj_shape, dtype=DTYPE)
    v_proj = torch.ones(proj_shape, dtype=DTYPE)
    gq = torch.ones(proj_shape, dtype=DTYPE)


def init_weights(m):
    m.q_projection.weight = torch.nn.Parameter(q_proj)
    m.q_projection.bias = torch.nn.Parameter(q_bias)
    m.k_projection.weight = torch.nn.Parameter(k_proj)
    m.v_projection.weight = torch.nn.Parameter(v_proj)
    m.gating_query.weight = torch.nn.Parameter(gq)
    return m.eval()


net1 = AttentionTorch(c_pair, num_head)
net2 = AttentionMix(c_pair, num_head)
net3 = AttentionCpp(c_pair, num_head)

net1 = init_weights(net1)
net2 = init_weights(net2)
net3 = init_weights(net3)


def f1():
    if TEST_ACCURACY:
        return net1(pair, mask, bias)
    else:
        return None


def f2():
    if TEST_ACCURACY:
        return net2(pair, mask, bias)
    else:
        return None


def f3():
    return net3(pair, mask, bias)


def benchmark():
    t1, t2, t3 = 0.0, 0.0, 0.0
    diff1, diff2 = 0.0, 0.0
    for _ in range(WARM):
        z1 = f1()
        z2 = f2()
        z3 = f3()
        if TEST_ACCURACY:
            diff1 += torch.sum(torch.abs(z1 - z2)).item() / torch.sum(torch.abs(z1)) / WARM
            diff2 += torch.sum(torch.abs(z1 - z3)).item() / torch.sum(torch.abs(z1)) / WARM

    for _ in range(COUNT):
        start = time.time()
        z1 = f1()
        t1 += time.time() - start

    af3_kernels.reset_debug_timers()
    for _ in range(COUNT):
        start = time.time()
        z2 = f2()
        t2 += time.time() - start
    af3_kernels.print_debug_timers()

    af3_kernels.reset_debug_timers()
    for _ in range(COUNT):
        start = time.time()
        z3 = f3()
        t3 += time.time() - start
    af3_kernels.print_debug_timers()

    num_cores = torch.get_num_threads()
    print("Running benchmark for SelfAttention")
    print(f"DTYPE: {DTYPE}, SEQ_LEN: {length}, BLK_SZ: {BLK_SZ}, NCORES: {num_cores}")
    print("Model1: Torch,\tModel2: Mixed,\tModel3: C++")
    print(f"Diff:\t<1, 2>: {diff1*100:.6f}%,\t<1, 3>: {diff2*100:.6f}%")
    print(f"Time:\t<1>: {t1/COUNT*1000:.4f} ms,\t<2>: {t2/COUNT*1000:.4f} ms,\t<3>: {t3/COUNT*1000:.4f} ms")
    print(f"Boost:\t<1, 2>: {t1/t2:.2f}x,\t<1, 3>: {t1/t3:.2f}x,\t<2, 3>: {t2/t3:.2f}x")


if __name__ == "__main__":
    if DO_PROFILE:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as p:
            benchmark()
        p.export_chrome_trace("trace.json")
        print(p.key_averages().table(sort_by="cpu_time_total", row_limit=12))
    else:
        benchmark()
