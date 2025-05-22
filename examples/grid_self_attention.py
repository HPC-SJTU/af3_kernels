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
import einops
from typing import Optional
import af3_kernels
from af3_kernels import dot_product_attention_cpp, GridSelfAttentionCpp as AttentionCpp
import time

torch.manual_seed(1896)


class AttentionTorch(nn.Module):
    def __init__(self, c_pair: int = 128, num_head: int = 4, transpose: bool = False):
        super(AttentionTorch, self).__init__()
        self.c_pair = c_pair
        self.num_head = num_head
        self.transpose = transpose
        self.act_norm = nn.LayerNorm(self.c_pair)
        self.pair_bias_projection = nn.Linear(self.c_pair, self.num_head, bias=False)

        self.q_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.k_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.v_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.gating_query = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.output_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)

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

    def forward(self, pair: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            pair (torch.Tensor): [N_token, N_token, c_pair]
            mask (torch.Tensor): [N_token, N_token]
        Returns:
            torch.Tensor: [N_token, N_token, c_pair]
        """
        with torch.profiler.record_function("Torch Attention"):
            pair = self.act_norm(pair)
            bias = self.pair_bias_projection(pair).permute(2, 0, 1)

            if self.transpose:
                pair = pair.permute(1, 0, 2)
            q = self.q_projection(pair)
            k = self.k_projection(pair)
            v = self.v_projection(pair)

            q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_head), [q, k, v])

            weighted_avg = self.dot_product_attention_torch(q, k, v, mask, bias)

            weighted_avg = einops.rearrange(weighted_avg, "b h n d -> b n (h d)")

            gate_values = self.gating_query(pair)

            weighted_avg *= torch.sigmoid(gate_values)
            pair = self.output_projection(weighted_avg)

            if self.transpose:
                pair = pair.permute(1, 0, 2)

            return pair


class AttentionMix(nn.Module):
    def __init__(self, c_pair: int = 128, num_head: int = 4, transpose: bool = False):
        super(AttentionMix, self).__init__()
        self.c_pair = c_pair
        self.num_head = num_head
        self.transpose = transpose
        self.act_norm = nn.LayerNorm(self.c_pair)
        self.pair_bias_projection = nn.Linear(self.c_pair, self.num_head, bias=False)

        self.q_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.k_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.v_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.gating_query = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.output_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)

    def forward(self, pair: torch.Tensor, mask: torch.Tensor):
        with torch.profiler.record_function("Mixed Attention"):
            pair = self.act_norm(pair)
            bias = self.pair_bias_projection(pair).permute(2, 0, 1)

            if self.transpose:
                pair = pair.permute(1, 0, 2)
            q = self.q_projection(pair)
            k = self.k_projection(pair)
            v = self.v_projection(pair)

            q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_head), [q, k, v])

            weighted_avg = dot_product_attention_cpp(q, k, v, mask, bias, USE_FLASH, BLK_SZ, FLASH_SZ)

            weighted_avg = einops.rearrange(weighted_avg, "b h n d -> b n (h d)")

            gate_values = self.gating_query(pair)

            weighted_avg *= torch.sigmoid(gate_values)
            pair = self.output_projection(weighted_avg)

            if self.transpose:
                pair = pair.permute(1, 0, 2)

            return pair


############## Config ##############
set_size = 3

if set_size == 1:
    length = 37
    c_pair = 128
    num_head = 4
    pair_shape = (length, length, c_pair)
    bias_shape = (num_head, c_pair)
    mask_shape = (length, length)
    proj_shape = (c_pair, c_pair)
    COUNT = 1000
elif set_size == 2:
    length = 256
    c_pair = 128
    num_head = 4
    pair_shape = (length, length, c_pair)
    bias_shape = (num_head, c_pair)
    mask_shape = (length, length)
    proj_shape = (c_pair, c_pair)
    COUNT = 1000
elif set_size == 3:
    length = 1536
    c_pair = 128
    num_head = 4
    pair_shape = (length, length, c_pair)
    bias_shape = (num_head, c_pair)
    mask_shape = (length, length)
    proj_shape = (c_pair, c_pair)
    COUNT = 5
elif set_size == 4:
    length = 1491
    c_pair = 128
    num_head = 4
    pair_shape = (length // 2, length, c_pair)
    bias_shape = (4, length, length)
    mask_shape = (length // 2, length)
    proj_shape = (c_pair, c_pair)
    COUNT = 20

WARM = 2
DTYPE = torch.bfloat16
BLK_SZ = 128
FLASH_SZ = 768
USE_FLASH = False
TEST_ACCURACY = True
DO_PROFILE = False
SCALE = 1.0
BIAS = 0.0
TRANSPOSE = True

######## Initialize ########

if TEST_ACCURACY:
    pair = torch.randn(pair_shape, dtype=DTYPE)
    mask = torch.ones(mask_shape, dtype=torch.bool)

    q_proj = torch.randn(proj_shape, dtype=DTYPE) * SCALE + BIAS
    k_proj = torch.randn(proj_shape, dtype=DTYPE) * SCALE + BIAS
    v_proj = torch.randn(proj_shape, dtype=DTYPE) * SCALE + BIAS
    gq = torch.randn(proj_shape, dtype=DTYPE) * SCALE + BIAS
    out_proj = torch.randn(proj_shape, dtype=DTYPE) * SCALE + BIAS
    bias = torch.randn(bias_shape, dtype=DTYPE) * SCALE + BIAS
    norm_weight = torch.randn(c_pair, dtype=DTYPE) * SCALE + BIAS
    norm_bias = torch.randn(c_pair, dtype=DTYPE) * SCALE + BIAS
else:
    pair = torch.ones(pair_shape, dtype=DTYPE)
    bias = torch.zeros(bias_shape, dtype=DTYPE)
    mask = torch.ones(mask_shape, dtype=torch.bool)

    q_proj = torch.ones(proj_shape, dtype=DTYPE)
    k_proj = torch.ones(proj_shape, dtype=DTYPE)
    v_proj = torch.ones(proj_shape, dtype=DTYPE)
    gq = torch.ones(proj_shape, dtype=DTYPE)
    out_proj = torch.ones(proj_shape, dtype=DTYPE)
    norm_weight = torch.ones(c_pair, dtype=DTYPE)
    norm_bias = torch.ones(c_pair, dtype=DTYPE)


def init_weights(m):
    m.q_projection.weight = torch.nn.Parameter(q_proj)
    m.k_projection.weight = torch.nn.Parameter(k_proj)
    m.v_projection.weight = torch.nn.Parameter(v_proj)
    m.gating_query.weight = torch.nn.Parameter(gq)
    m.output_projection.weight = torch.nn.Parameter(out_proj)
    m.pair_bias_projection.weight = torch.nn.Parameter(bias)
    m.act_norm.weight = torch.nn.Parameter(norm_weight)
    m.act_norm.bias = torch.nn.Parameter(norm_bias)
    return m.eval()


net1 = AttentionTorch(c_pair, num_head, transpose=TRANSPOSE)
net2 = AttentionMix(c_pair, num_head, transpose=TRANSPOSE)
net3 = AttentionCpp(c_pair, num_head, transpose=TRANSPOSE)

net1 = init_weights(net1)
net2 = init_weights(net2)
net3 = init_weights(net3)


def f1():
    if TEST_ACCURACY:
        return net1(pair, mask)
    else:
        return None


def f2():
    if TEST_ACCURACY:
        return net2(pair, mask)
    else:
        return None


def f3():
    return net3(pair, mask)


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
    print("Running benchmark for GridSelfAttention")
    print(f"DTYPE: {DTYPE}, SEQ_LEN: {length}, TRANSPOSE: {TRANSPOSE}, BLK_SZ: {BLK_SZ}, NCORES: {num_cores}")
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
