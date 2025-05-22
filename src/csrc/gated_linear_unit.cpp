/******************************************************************************
 * Copyright (c) 2025 Xflops - All rights reserved.
 *
 * For information on the license, see the LICENSE file.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Author: Dragon Archer (Xflops)
 ******************************************************************************/

#include "gated_linear_unit.h"

#include <ATen/record_function.h>
#include "timing.h"
#include "tools.h"
#include "xsmm_functors.h"

REGISTER_LOCAL_SCOPE(glu_gemm, "glu_gemm");
REGISTER_LOCAL_SCOPE(linear, "linear");

template<typename T>
at::Tensor gated_linear_unit_impl(at::Tensor& input, at::Tensor& weight, bool vnni_repacked, GLU_OPT_ARGS) {
	RECORD_FUNCTION("C++ Gated Linear Unit", {input, weight});

	int64_t raw_dim  = input.dim();
	int64_t raw_dim1 = input.sizes()[0];
	int64_t raw_dim2 = input.sizes()[1];

	int64_t Mp       = input.numel() / input.sizes()[input.dim() - 1];
	int64_t K        = weight.sizes()[0];     // Always power of 2, mostly 64 / 128
	int64_t N        = weight.sizes()[1] / 2; // Always power of 2, mostly >= 512
	int64_t M        = Mp;
	int64_t leftover = 0;
	if(M % M_BLK != 0) {
		M        = (M / M_BLK) * M_BLK;
		leftover = Mp - M;
	}
	if(N_BLK > N) {
		N_BLK = N;
	}
	input      = input.contiguous();
	weight     = weight.contiguous();
	int64_t Mb = M / M_BLK;
	int64_t Nb = N / N_BLK;
	// input: [M, K], weight: [K, 2N]
	auto input_a  = GetVLAPtr<T>(input, {M_BLK, K});       // [Mb, M_BLK, K]
	auto weight_a = GetVLAPtr<T>(weight, {2 * Nb, N_BLK}); // [K, 2 * Nb, N_BLK]
	auto out      = new_aligned_tensor(input, {Mp, N});
	auto out_a    = GetVLAPtr<T>(out, {M_BLK, Nb, N_BLK}); // [Mb, M_BLK, Nb, N_BLK]

	{
		RECORD_SCOPE(glu_gemm, {input, weight, out});
		auto gemm      = SCOPEIT((tpp::BrgemmTPP<T, T>(M_BLK, N_BLK, K, 0, 0, K, 2 * N, N_BLK, 0.0f, 0, 1, vnni_repacked, 0, 0)), BRGEMM);
		auto silu      = SCOPEIT(tpp::SiLUFwdTPP<T>(M_BLK, N_BLK), SILU);
		auto mul       = SCOPEIT(tpp::MulTPP<T>(M_BLK, N_BLK, N_BLK, N), EW_MUL);
		auto gemm_left = SCOPEIT((tpp::BrgemmTPP<T, T>(leftover, N_BLK, K, 0, 0, K, 2 * N, N_BLK, 0.0f, 0, 1, vnni_repacked, 0, 0)), MARGIN);
		auto silu_left = SCOPEIT(tpp::SiLUFwdTPP<T>(leftover, N_BLK), MARGIN);
		auto mul_left  = SCOPEIT(tpp::MulTPP<T>(leftover, N_BLK, N_BLK, N), MARGIN);
#pragma omp parallel for collapse(2)
		for(int i = 0; i <= Mb; i++) {
			for(int j = 0; j < Nb; j++) {
				LIBXSMM_ALIGNED(T tmp[M_BLK][N_BLK], 64);
				LIBXSMM_ALIGNED(T tmp2[M_BLK][N_BLK], 64);
				if(i != Mb) {
					gemm.config();
					gemm(&input_a[i][0][0], &weight_a[0][j][0], &tmp[0][0], 1, true);
					silu(&tmp[0][0], &tmp[0][0], &tmp2[0][0]);
					gemm(&input_a[i][0][0], &weight_a[0][j + Nb][0], &tmp2[0][0], 1, true);
					mul(&tmp[0][0], &tmp2[0][0], &out_a[i][0][j][0]);
					gemm.release();
				} else if(leftover) {
					gemm_left.config();
					gemm_left(&input_a[Mb][0][0], &weight_a[0][j][0], &tmp[0][0], 1, true);
					silu_left(&tmp[0][0], &tmp[0][0], &tmp2[0][0]);
					gemm_left(&input_a[Mb][0][0], &weight_a[0][j + Nb][0], &tmp2[0][0], 1, true);
					mul_left(&tmp[0][0], &tmp2[0][0], &out_a[Mb][0][j][0]);
					gemm_left.release();
				}
			}
		}
	}
	if(raw_dim == 3) {
		out = out.view({raw_dim1, raw_dim2, N});
	}
	return out;
}

template<typename T>
at::Tensor linear_impl(at::Tensor& input, at::Tensor& weight, GLU_OPT_ARGS) {
	RECORD_FUNCTION("C++ Linear", {input, weight});

	int64_t Mp       = input.numel() / input.sizes()[input.dim() - 1];
	int64_t K        = weight.sizes()[0];
	int64_t N        = weight.sizes()[1];
	int64_t M        = Mp;
	int64_t leftover = 0;
	if(M % M_BLK != 0) {
		M        = (M / M_BLK) * M_BLK;
		leftover = Mp - M;
	}
	if(N_BLK > N) {
		N_BLK = N;
	}

	TPP_ASSERT(N % N_BLK == 0, "N must be divisible by N_BLK");

	auto vec            = input.sizes().vec();
	vec[vec.size() - 1] = N;

	input            = input.contiguous();
	weight           = weight.contiguous();
	int64_t Mb       = M / M_BLK;
	int64_t Nb       = N / N_BLK;
	auto    input_a  = GetVLAPtr<T>(input, {M_BLK, K});  // [Mb, M_BLK, K]
	auto    weight_a = GetVLAPtr<T>(weight, {K, N_BLK}); // [K, Nb, N_BLK]
	auto    out      = new_aligned_tensor(input, vec);
	auto    out_a    = GetVLAPtr<T>(out, {M_BLK, Nb, N_BLK}); // [Mb, M_BLK, Nb, N_BLK]

	{
		RECORD_SCOPE(linear, {input, weight, out});
		auto gemm      = SCOPEIT((tpp::BrgemmTPP<T, T>(M_BLK, N_BLK, K, 0, 0, K, N_BLK, N, 0.0f, 0, 1, 1, 0)), BRGEMM);
		auto gemm_left = SCOPEIT((tpp::BrgemmTPP<T, T>(leftover, N_BLK, K, 0, 0, K, N_BLK, N, 0.0f, 0, 1, 1, 0)), MARGIN);
#pragma omp parallel for collapse(2)
		for(int i = 0; i <= Mb; i++) {
			for(int j = 0; j < Nb; j++) {
				if(i != Mb) {
					gemm(&input_a[i][0][0], &weight_a[j][0][0], &out_a[i][0][j][0], 1);
				} else if(leftover) {
					gemm_left(&input_a[Mb][0][0], &weight_a[j][0][0], &out_a[Mb][0][j][0], 1);
				}
			}
		}
	}
	return out;
}

at::Tensor gated_linear_unit(at::Tensor& input, at::Tensor& weight, bool vnni_repacked, GLU_OPT_ARGS) {
	GlobalPass _gp(FWD);
	if(input.dtype() == at::kFloat) {
		return gated_linear_unit_impl<float>(input, weight, vnni_repacked, GLU_OPT_ARGS_FWD);
	} else {
		return gated_linear_unit_impl<tpp::bfloat16>(input, weight, vnni_repacked, GLU_OPT_ARGS_FWD);
	}
}

at::Tensor linear(at::Tensor& input, at::Tensor& weight, GLU_OPT_ARGS) {
	GlobalPass _gp(FWD);
	if(input.dtype() == at::kFloat) {
		return linear_impl<float>(input, weight, GLU_OPT_ARGS_FWD);
	} else {
		return linear_impl<tpp::bfloat16>(input, weight, GLU_OPT_ARGS_FWD);
	}
}
