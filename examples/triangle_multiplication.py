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
from torch import nn

import time

import af3_kernels
from af3_kernels import TriangleMultiplicationCpp

torch.manual_seed(1896)


class TriangleMultiplicationTorch(nn.Module):
    def __init__(self, c_pair: int = 128, _outgoing: bool = True) -> None:
        super(TriangleMultiplicationTorch, self).__init__()

        self.c_pair = c_pair
        self.left_norm_input = nn.LayerNorm(self.c_pair)
        self.projection = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.gate = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.center_norm = nn.LayerNorm(self.c_pair)
        self.output_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.gating_linear = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.equation = "ckj,cki->cij"
        if _outgoing is True:
            self.equation = "cik,cjk->cij"

    def forward(self, pair: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair (torch.Tensor): [N_token, N_token, c_pair]
            mask (torch.Tensor): [N_token]
        Returns:
            torch.Tensor: [N_token, N_token, c_pair]
        """
        input = pair
        pair = self.left_norm_input(pair)
        input_pair = pair

        projection = self.projection(pair)
        projection = projection.permute(2, 0, 1)
        if mask is not None:
            projection *= mask[None, ...]

        gate = self.gate(pair)
        gate = gate.permute(2, 0, 1)
        projection *= torch.sigmoid(gate)

        projection = projection.reshape(self.c_pair, 2, *projection.shape[1:])

        a, b = torch.chunk(projection, 2, dim=1)
        a, b = torch.squeeze(a, dim=1), torch.squeeze(b, dim=1)
        pair = torch.einsum(self.equation, a, b)

        pair = pair.permute(1, 2, 0)
        pair = self.center_norm(pair)
        pair = self.output_projection(pair)

        gate_out = self.gating_linear(input_pair)
        pair *= torch.sigmoid(gate_out)
        input += pair
        return input


set = 3
if set == 1:
    n_token = 37
    c_pair = 64
    COUNT = 1000
elif set == 2:
    n_token = 4095
    c_pair = 128
    COUNT = 10
elif set == 3:
    n_token = 1536
    c_pair = 128
    COUNT = 10
elif set == 4:
    n_token = 4096
    c_pair = 128
    COUNT = 5

WARM = 3
DTYPE = torch.bfloat16
is_outgoing = False
BLK_SZ = 128
TEST_ACCURACY = True
DO_PROFILE = False
SCALE = 2.0
BIAS = 3.0

net1 = TriangleMultiplicationTorch(c_pair, _outgoing=is_outgoing).eval()
net2 = TriangleMultiplicationCpp(c_pair, _outgoing=is_outgoing).eval()


def f1():
    if TEST_ACCURACY:
        return net1(act.clone(), mask)
    else:
        return None


def f2():
    return net2(act.clone(), mask, BLK_SZ)


if TEST_ACCURACY:
    act = torch.randn(n_token, n_token, c_pair, dtype=DTYPE) * SCALE + BIAS
    mask = torch.ones(n_token, n_token, dtype=torch.bool)

    left_norm_input_weight = torch.randn(c_pair, dtype=DTYPE) * SCALE + BIAS
    left_norm_input_bias = torch.randn(c_pair, dtype=DTYPE) * SCALE + BIAS
    projection_weight = torch.randn(2 * c_pair, c_pair, dtype=DTYPE) * SCALE + BIAS
    gate_weight = torch.randn(2 * c_pair, c_pair, dtype=DTYPE) * SCALE + BIAS

    output_projection_weight = torch.randn(c_pair, c_pair, dtype=DTYPE) * SCALE + BIAS
    center_norm_weight = torch.randn(c_pair, dtype=DTYPE) * SCALE + BIAS
    center_norm_bias = torch.randn(c_pair, dtype=DTYPE) * SCALE + BIAS
    gating_linear_weight = torch.randn(c_pair, c_pair, dtype=DTYPE) * SCALE + BIAS
else:
    act = torch.ones(n_token, n_token, c_pair, dtype=DTYPE)
    mask = torch.ones(n_token, n_token, dtype=torch.bool)

    left_norm_input_weight = torch.ones(c_pair, dtype=DTYPE)
    left_norm_input_bias = torch.ones(c_pair, dtype=DTYPE)
    projection_weight = torch.ones(2 * c_pair, c_pair, dtype=DTYPE)
    gate_weight = torch.ones(2 * c_pair, c_pair, dtype=DTYPE)

    output_projection_weight = torch.ones(c_pair, c_pair, dtype=DTYPE)
    center_norm_weight = torch.ones(c_pair, dtype=DTYPE)
    center_norm_bias = torch.ones(c_pair, dtype=DTYPE)
    gating_linear_weight = torch.ones(c_pair, c_pair, dtype=DTYPE)


def init_weights(m):
    m.left_norm_input.weight = torch.nn.Parameter(left_norm_input_weight)
    m.left_norm_input.bias = torch.nn.Parameter(left_norm_input_bias)
    m.projection.weight = torch.nn.Parameter(projection_weight)
    m.gate.weight = torch.nn.Parameter(gate_weight)
    m.center_norm.weight = torch.nn.Parameter(center_norm_weight)
    m.center_norm.bias = torch.nn.Parameter(center_norm_bias)
    m.output_projection.weight = torch.nn.Parameter(output_projection_weight)
    m.gating_linear.weight = torch.nn.Parameter(gating_linear_weight)
    return m.eval()


net1 = init_weights(net1)
net2 = init_weights(net2)

diff = 0
for _ in range(WARM):
    z1 = f1()
    z2 = f2()
    if TEST_ACCURACY:
        diff += torch.sum(torch.abs(z1 - z2)).item() / torch.sum(torch.abs(z1)) / WARM


def benchmark():
    t1 = 0
    t2 = 0
    for _ in range(COUNT):
        start = time.time()
        z1 = f1()
        t1 += time.time() - start

    for _ in range(COUNT):
        start = time.time()
        z2 = f2()
        t2 += time.time() - start

    num_cores = torch.get_num_threads()
    print("Running benchmark for TriangleMultiplication")
    print(f"DTYPE: {DTYPE}, SEQ_LEN: {n_token}, BLK_SZ: {BLK_SZ}, NCORES: {num_cores}")
    print(f"Diff : {diff*100:.6f}%")
    print(f"Torch: {t1/COUNT*1000:.4f} ms")
    print(f"C++  : {t2/COUNT*1000:.4f} ms")
    print(f"Boost: {t1/t2:.2f}x")


if __name__ == "__main__":
    af3_kernels.reset_debug_timers()
    if DO_PROFILE:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as p:
            benchmark()
        p.export_chrome_trace("trace.json")
    else:
        benchmark()
    af3_kernels.print_debug_timers()
