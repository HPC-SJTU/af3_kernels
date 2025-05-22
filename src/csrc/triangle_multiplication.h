/******************************************************************************
 * Copyright (c) 2025 Xflops - All rights reserved.
 *
 * For information on the license, see the LICENSE file.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Author: Dragon Archer (Xflops)
 ******************************************************************************/

#ifndef _TPP_TRIANGLE_MULTIPLICATION_H_
#define _TPP_TRIANGLE_MULTIPLICATION_H_

#include <torch/extension.h>

#ifdef DYNAMIC_TILING // defined in setup.py
	#define TM_OPT_ARGS     int64_t BLK_SZ
	#define TM_OPT_ARGS_FWD BLK_SZ
#else
	#define TM_OPT_ARGS     int64_t dummy_BLK_SZ
	#define TM_OPT_ARGS_FWD dummy_BLK_SZ
constexpr int64_t BLK_SZ = 128;
#endif

std::tuple<at::Tensor, at::Tensor> traingle_multiplication_pre(
	at::Tensor& act,
	at::Tensor& mask,
	at::Tensor& left_norm_input_weight,
	at::Tensor& left_norm_input_bias,
	at::Tensor& proj_gate_weight,
	at::Tensor& gating_linear_weight,
	TM_OPT_ARGS);

at::Tensor traingle_multiplication_einsum(
	at::Tensor& mid,
	bool        is_outgoing,
	int64_t     from,
	int64_t     to,
	TM_OPT_ARGS);

at::Tensor traingle_multiplication_post(
	at::Tensor& act,
	at::Tensor& out,
	at::Tensor& input,
	at::Tensor& center_norm_weight,
	at::Tensor& center_norm_bias,
	at::Tensor& output_projection_weight,
	TM_OPT_ARGS);

at::Tensor af3_traingle_multiplication(
	at::Tensor& act,
	at::Tensor& mask,
	bool        is_outgoing,
	at::Tensor& left_norm_input_weight,
	at::Tensor& left_norm_input_bias,
	at::Tensor& proj_gate_weight,
	at::Tensor& center_norm_weight,
	at::Tensor& center_norm_bias,
	at::Tensor& output_projection_weight,
	at::Tensor& gating_linear_weight,
	TM_OPT_ARGS);

#endif //_TPP_TRIANGLE_MULTIPLICATION_H_
