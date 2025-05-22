###############################################################################
# Copyright (c) 2025 Xflops - All rights reserved.
#
# For information on the license, see the LICENSE file.
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Dragon Archer (Xflops)
###############################################################################

from functools import wraps
from os import environ
import torch
from torch.distributed import get_rank

DO_PROFILE = environ.get("DO_PROFILE", "0") == "1"

USE_DIST = environ.get("USE_DIST", "0") == "1"


def profile(name: str = None):
    def decorator(func):
        if DO_PROFILE:

            @wraps(func)
            def wrapper(*args, **kwargs):
                with torch.profiler.record_function(func.__name__ if name is None else name):
                    return func(*args, **kwargs)

            return wrapper
        else:
            return func

    return decorator


comm_time_counters = {}


def record_comm_time(name: str, time: float, size: int = 0):
    global comm_time_counters
    if USE_DIST and get_rank() == 0:
        if name not in comm_time_counters:
            comm_time_counters[name] = [0.0, 0.0]
        comm_time_counters[name][0] += time
        comm_time_counters[name][1] += size / 1e9


def print_comm_time():
    global comm_time_counters
    if USE_DIST and get_rank() == 0:
        for name, times in comm_time_counters.items():
            print(f"{name}\t:\t{times[0]:.4f} s\t{times[1]/times[0]:.4f} GB/s\t{times[1]:.4f} GB")


def pad_to_size(tensor: torch.Tensor, size: int, dim: int = 0) -> torch.Tensor:
    if tensor.shape[dim] == size:
        return tensor.contiguous()
    shape = list(tensor.shape)
    shape[dim] = size - tensor.shape[dim]
    pad = torch.zeros(shape, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=dim).contiguous()
