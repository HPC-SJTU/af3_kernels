###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.
# Copyright (c) 2025 Xflops - All rights reserved.
#
# For information on the license, see the LICENSE file.
# SPDX-License-Identifier: BSD-3-Clause
#
# Authors: Dhiraj Kalamkar (Intel Corp.)
#          Dragon Archer (Xflops)
###############################################################################

import torch
import torch.nn as nn

from . import _C

from .TriangleMultiplication import TriangleMultiplicationCpp, DistributedTriangleMultiplicationCpp
from .Attention import GridSelfAttentionCpp, DistributedGridSelfAttentionCpp


def gated_linear_unit_cpp(
    x: torch.Tensor,
    weight: torch.Tensor,
    vnni_repacked: bool = False,
    M_BLK: int = 128,
    N_BLK: int = 128,
) -> torch.Tensor:
    return _C.gated_linear_unit(x, weight, vnni_repacked, M_BLK, N_BLK)


def dot_product_attention_cpp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    bias: torch.Tensor,
    use_flash_attention: bool = False,
    BLK_SZ: int = 128,
    FLASH_SZ: int = 1024,
) -> torch.Tensor:
    if mask.dim() == 1:
        mask = mask[None, None, None, :].to(dtype=torch.bool)
    elif mask.dim() == 2:
        mask = mask[:, None, None, :].to(dtype=torch.bool)
    mask = ~mask
    dtype = q.dtype
    mask = mask.to(dtype)
    bias = bias.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)
    out = _C.dot_product_attention(q, k, v, mask, bias, use_flash_attention, BLK_SZ, FLASH_SZ)
    return out


def _attention_cpp(
    pair: torch.Tensor,
    mask: torch.Tensor,
    bias: torch.Tensor,
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    gating_query: torch.Tensor,
    out_proj: torch.Tensor,
    head: int,
    use_flash_attention: bool = False,  # unused
    BLK_SZ: int = 128,
    FLASH_SZ: int = 1024,  # unused
) -> torch.Tensor:
    if mask.dim() == 1:
        mask = mask[None, None, None, :].to(dtype=torch.bool)
    elif mask.dim() == 2:
        mask = mask[:, None, None, :].to(dtype=torch.bool)
    mask = ~mask
    dtype = pair.dtype
    mask = mask.to(dtype)
    bias = bias.to(dtype)
    out = _C._attention(
        pair, mask, bias, q_proj, k_proj, v_proj, gating_query, out_proj, head, use_flash_attention, BLK_SZ, FLASH_SZ
    )
    return out


def self_attention_cpp(
    pair: torch.Tensor,
    mask: torch.Tensor,
    bias: torch.Tensor,
    q_proj: torch.Tensor,
    q_bias: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    gating_query: torch.Tensor,
    head: int,
    use_flash_attention: bool = False,  # unused
    BLK_SZ: int = 128,
    FLASH_SZ: int = 1024,  # unused
) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool)
    mask = ~mask
    dtype = pair.dtype
    mask = mask.to(dtype)
    bias = bias.to(dtype)
    out = _C.self_attention(
        pair, mask, bias, q_proj, q_bias, k_proj, v_proj, gating_query, head, use_flash_attention, BLK_SZ, FLASH_SZ
    )
    return out


def pad_and_align_tensor(
    x: torch.Tensor,
    shape: torch.Size,
    need_init: bool = True,
    init: float = 0.0,
    alignment: int = 64,
) -> torch.Tensor:
    return _C.pad_and_align_tensor(x, shape, need_init, init, alignment)


def vnni_repack_tensor(
    x: torch.Tensor,
    type: int = 2,
    BLK_SZ: int = 0,
) -> torch.Tensor:
    return _C.vnni_repack_tensor(x, type, BLK_SZ)


def reset_debug_timers():
    _C.reset_debug_timers()


def print_debug_timers(tid=0, detailed=True):
    _C.print_debug_timers(tid, detailed)


def print_debug_thread_imbalance():
    _C.print_debug_thread_imbalance()


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        M_BLK: int = 128,
        N_BLK: int = 128,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        assert bias is False
        self.register_parameter("bias", None)
        self.M_BLK = M_BLK
        self.N_BLK = N_BLK
        self.first_run = True
        self.weight_v = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.first_run:
            self.first_run = False
            self.weight_v = vnni_repack_tensor(self.weight.T.contiguous(), 2, self.N_BLK)
        return _C.linear(input, self.weight_v, self.M_BLK, self.N_BLK)


reset_debug_timers()
