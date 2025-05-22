/******************************************************************************
 * Copyright (c) 2025 Xflops - All rights reserved.
 *
 * For information on the license, see the LICENSE file.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Author: Dragon Archer (Xflops)
 ******************************************************************************/

#ifndef _TPP_DOT_PRODUCT_ATTENTION_H_
#define _TPP_DOT_PRODUCT_ATTENTION_H_

#include <torch/extension.h>

#ifdef DYNAMIC_TILING // defined in setup.py
	#define DPA_OPT_ARGS     int64_t BLK_SZ, int64_t FLASH_SZ
	#define DPA_OPT_ARGS_FWD BLK_SZ, FLASH_SZ
#else
	#define DPA_OPT_ARGS     int64_t dummy_BLK_SZ, int64_t dummy_FLASH_SZ
	#define DPA_OPT_ARGS_FWD dummy_BLK_SZ, dummy_FLASH_SZ
constexpr int64_t BLK_SZ   = 128;
constexpr int64_t FLASH_SZ = 1024;
#endif

at::Tensor dot_product_attention(
	at::Tensor& q,
	at::Tensor& k,
	at::Tensor& v,
	at::Tensor& mask,
	at::Tensor& bias,
	bool        use_flash_attention,
	DPA_OPT_ARGS);

at::Tensor _attention(
	at::Tensor& pair,
	at::Tensor& mask,
	at::Tensor& bias,
	at::Tensor& q_proj,
	at::Tensor& k_proj,
	at::Tensor& v_proj,
	at::Tensor& gating_query,
	at::Tensor& out_proj,
	int64_t     head,
	bool        use_flash_attention,
	DPA_OPT_ARGS);

at::Tensor self_attention(
	at::Tensor& pair,
	at::Tensor& mask,
	at::Tensor& bias,
	at::Tensor& q_proj,
	at::Tensor& q_weight,
	at::Tensor& k_proj,
	at::Tensor& v_proj,
	at::Tensor& gating_query,
	int64_t     head,
	bool        use_flash_attention,
	DPA_OPT_ARGS);

#endif //_TPP_DOT_PRODUCT_ATTENTION_H_
