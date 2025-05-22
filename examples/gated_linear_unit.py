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
import af3_kernels
from af3_kernels import gated_linear_unit_cpp
import time

torch.manual_seed(1896)


def gated_linear_unit_torch(x, weight):
    y = torch.matmul(x, weight)
    a, b = torch.chunk(y, 2, dim=-1)
    out = torch.nn.functional.silu(a) * b
    return out


set_size = 3
if set_size == 1:
    length = 128
    x_shape = (37, 37, length)
    weight_shape = (length, 512)
    COUNT = 1000
elif set_size == 2:
    length = 256
    x_shape = (1024, 1024, length)
    weight_shape = (length, 1024)
    COUNT = 2
elif set_size == 3:
    length = 128
    x_shape = (1536, 1536, length)
    weight_shape = (length, 1024)
    COUNT = 10

WARM = 5
DTYPE = torch.bfloat16
M_BLK = 128
N_BLK = 128
TEST_ACCURACY = True
DO_PROFILE = False
SCALE = 2.0
BIAS = 3.0

if TEST_ACCURACY:
    input = torch.randn(x_shape, dtype=DTYPE) * SCALE + BIAS
    weight = torch.randn(weight_shape, dtype=DTYPE) * SCALE + BIAS
else:
    input = torch.ones(x_shape, dtype=DTYPE)
    weight = torch.ones(weight_shape, dtype=DTYPE)


def f1():
    if TEST_ACCURACY:
        return gated_linear_unit_torch(input, weight)
    else:
        return None


def f2():
    return gated_linear_unit_cpp(input, weight, False, M_BLK, N_BLK)


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
    print("Running benchmark for Gated Linear Unit")
    print(
        f"DTYPE: {DTYPE}, x_shape: {x_shape}, weight_shape: {weight_shape}, BLK_SZ: {M_BLK}x{N_BLK}, NCORES: {num_cores}"
    )
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
