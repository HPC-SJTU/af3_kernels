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
from af3_kernels import Linear as LinearCpp
from torch.nn import Linear as LinearTorch
import time

torch.manual_seed(1896)

set_size = 4
if set_size == 1:
    length = 128
    x_shape = (37, 37, length)
    weight_shape = (length, 512)
    COUNT = 1000
elif set_size == 2:
    in_features = 256
    out_features = 1024
    x_shape = (163840, in_features)
    COUNT = 10
elif set_size == 3:
    in_features = 128
    out_features = 1024
    x_shape = (1491, 1491, in_features)
    COUNT = 10
elif set_size == 4:  # _attention
    in_features = 128
    out_features = 128
    x_shape = (1491, 1491, in_features)
    COUNT = 10
elif set_size == 5:  # SelfAttention
    in_features = 768
    out_features = 768
    x_shape = (1491, in_features)
    COUNT = 1000
elif set_size == 6:  # DiffusitionTransition
    in_features = 384
    out_features = 1536
    x_shape = (1536, in_features)
    COUNT = 100000

WARM = 50
DTYPE = torch.bfloat16
M_BLK = 128
N_BLK = 128
TEST_ACCURACY = True
DO_PROFILE = False
SCALE = 2.0
BIAS = 3.0

net1 = LinearTorch(in_features, out_features, False, dtype=DTYPE).eval()
net2 = LinearCpp(in_features, out_features, False, dtype=DTYPE, M_BLK=M_BLK, N_BLK=N_BLK).eval()


def f1():
    if TEST_ACCURACY:
        return net1(input)
    else:
        return None


def f2():
    return net2(input)


if TEST_ACCURACY:
    input = torch.randn(x_shape, dtype=DTYPE) * SCALE + BIAS
    weight = torch.randn(out_features, in_features, dtype=DTYPE) * SCALE + BIAS
else:
    input = torch.ones(x_shape, dtype=DTYPE)
    weight = torch.ones(out_features, in_features, dtype=DTYPE)

net1.weight = torch.nn.Parameter(weight)
net2.weight = torch.nn.Parameter(weight)

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

    total_flops = input.numel() * out_features * 2 * COUNT
    num_cores = torch.get_num_threads()
    print("Running benchmark for Linear")
    print(
        f"DTYPE: {DTYPE}, x_shape: {x_shape}, in_features: {in_features}, out_features: {out_features}, BLK_SZ: {M_BLK}x{N_BLK}, NCORES: {num_cores}"
    )
    print(f"Diff : {diff*100:.6f}%")
    print(f"Torch: {t1/COUNT*1000:.4f} ms, {total_flops / t1 / 1e12:.2f} TFLOPS")
    print(f"C++  : {t2/COUNT*1000:.4f} ms, {total_flops / t2 / 1e12:.2f} TFLOPS")
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
