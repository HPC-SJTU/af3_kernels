/******************************************************************************
 * Copyright (c) 2025 Xflops - All rights reserved.
 *
 * For information on the license, see the LICENSE file.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Author: Dragon Archer (Xflops)
 ******************************************************************************/

#ifndef _TPP_GATED_LINEAR_UNIT_H_
#define _TPP_GATED_LINEAR_UNIT_H_

#include <torch/extension.h>

#ifdef DYNAMIC_TILING // defined in setup.py
	#define GLU_OPT_ARGS     int64_t M_BLK, int64_t N_BLK
	#define GLU_OPT_ARGS_FWD M_BLK, N_BLK
#else
	#define GLU_OPT_ARGS     int64_t dummy_M_BLK, int64_t dummy_N_BLK
	#define GLU_OPT_ARGS_FWD dummy_M_BLK, dummy_N_BLK
constexpr int64_t M_BLK = 128;
constexpr int64_t N_BLK = 128;
#endif

at::Tensor gated_linear_unit(at::Tensor& input, at::Tensor& weight, bool vnni_repacked, GLU_OPT_ARGS);

at::Tensor linear(at::Tensor& input, at::Tensor& weight, GLU_OPT_ARGS);

#endif //_TPP_GATED_LINEAR_UNIT_H_
