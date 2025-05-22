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
from typing import Optional
import af3_kernels
from af3_kernels import dot_product_attention_cpp
import time

torch.manual_seed(1896)


def dot_product_attention_torch(
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


############## Config ##############
set_size = 3

if set_size == 1:
    length = 37
    qkv_shape = (length, 4, length, 32)
    bias_shape = (4, length, length)
    mask_shape = (length, length)
    COUNT = 1000
elif set_size == 2:
    length = 2048
    qkv_shape = (length, 4, length, 32)
    bias_shape = (4, length, length)
    mask_shape = (length, length)
    COUNT = 10
elif set_size == 3:
    length = 1491
    qkv_shape = (length, 4, length, 32)
    bias_shape = (4, length, length)
    mask_shape = (length, length)
    COUNT = 5
# Below is for Diffusion
elif set_size == 4:
    length = 37
    qkv_shape = (1, 16, length, 48)
    bias_shape = (16, length, length)
    mask_shape = (length,)
    COUNT = 1000
elif set_size == 5:
    length = 128
    qkv_shape = (1, 16, length, 48)
    bias_shape = (16, length, length)
    mask_shape = (length,)
    COUNT = 100
elif set_size == 6:
    length = 1491
    qkv_shape = (1, 16, length, 24)
    bias_shape = (16, length, length)
    mask_shape = (length,)
    COUNT = 5

WARM = 2
DTYPE = torch.bfloat16
BLK_SZ = 128
FLASH_SZ = 1024
USE_FLASH = False
TEST_ACCURACY = True
USE_BIAS = True
USE_MASK = False
DO_PROFILE = False
SCALE = 2.0
BIAS = 3.0

######## Initialize ########

if TEST_ACCURACY:
    q = torch.randn(qkv_shape, dtype=DTYPE) * SCALE + BIAS
    k = torch.randn(qkv_shape, dtype=DTYPE) * SCALE + BIAS
    v = torch.randn(qkv_shape, dtype=DTYPE) * SCALE + BIAS
    if USE_BIAS:
        bias = torch.randn(bias_shape, dtype=DTYPE) * SCALE + BIAS
    else:
        bias = torch.zeros(bias_shape, dtype=DTYPE)
    mask = torch.ones(mask_shape, dtype=torch.bool)
else:
    q = torch.ones(qkv_shape, dtype=DTYPE)
    k = torch.ones(qkv_shape, dtype=DTYPE)
    v = torch.ones(qkv_shape, dtype=DTYPE)
    bias = torch.zeros(bias_shape, dtype=DTYPE)
    mask = torch.ones(mask_shape, dtype=torch.bool)


def f1():
    if TEST_ACCURACY:
        return dot_product_attention_torch(q, k, v, mask, bias)
    else:
        return None


def f2():
    return dot_product_attention_cpp(q, k, v, mask, bias, USE_FLASH, BLK_SZ, FLASH_SZ)


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
    print(f"DTYPE: {DTYPE}, SEQ_LEN: {length}, BLK_SZ: {BLK_SZ}, NCORES: {num_cores}")
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
