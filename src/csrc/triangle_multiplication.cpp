/******************************************************************************
 * Copyright (c) 2025 Xflops - All rights reserved.
 *
 * For information on the license, see the LICENSE file.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Author: Dragon Archer (Xflops)
 ******************************************************************************/

#include "triangle_multiplication.h"

#include <ATen/record_function.h>
#include "timing.h"
#include "tools.h"
#include "xsmm_functors.h"

REGISTER_LOCAL_SCOPE(tm_prep, "tm_prep");
REGISTER_LOCAL_SCOPE(pre_einsum, "pre_einsum");
REGISTER_LOCAL_SCOPE(einsum, "einsum");
REGISTER_LOCAL_SCOPE(post_einsum, "post_einsum");

template<typename T>
std::tuple<at::Tensor, at::Tensor> traingle_multiplication_pre_impl(
	at::Tensor& act,
	at::Tensor& mask,
	at::Tensor& left_norm_input_weight,
	at::Tensor& left_norm_input_bias,
	at::Tensor& proj_gate_weight,
	at::Tensor& gating_linear_weight,
	TM_OPT_ARGS) {
	RECORD_FUNCTION("C++ TM Pre", {act, mask});

	const int64_t Mp = act.sizes()[0];
	const int64_t Np = act.sizes()[1];
	const int64_t K  = act.sizes()[2];
	const int64_t C  = proj_gate_weight.sizes()[0] / 4;
	const int64_t M  = ((Mp + BLK_SZ - 1) / BLK_SZ) * BLK_SZ;
	const int64_t N  = ((Np + BLK_SZ - 1) / BLK_SZ) * BLK_SZ;
	const int64_t Mb = M / BLK_SZ;
	const int64_t Nb = N / BLK_SZ;

	{
		RECORD_SCOPE(tm_prep, {act, mask});
		act  = pad_and_align_tensor(act, {M, N, K}); // TODO: try to find if we can eliminate the zero-init while keep out NaNs
		mask = pad_and_align_tensor(mask, {M, N, 1L});
	}

	auto act_out   = new_aligned_tensor(act, {M, N, K});
	auto mid       = new_aligned_tensor(act, {2 * C, M, N});
	auto act_out_a = GetVLAPtr<T>(act_out, {Nb, BLK_SZ, K});      // [M, Nb, BLK_SZ, K]
	auto act_a     = GetVLAPtr<T>(act, {Nb, BLK_SZ, K});          // [M, Nb, BLK_SZ, K]
	auto mask_a    = GetVLAPtr<T>(mask, {Nb, BLK_SZ, 1L});        // [M, Nb, BLK_SZ, 1]
	auto mid_a     = GetVLAPtr<T>(mid, {Mb, BLK_SZ, Nb, BLK_SZ}); // [2K, Mb, BLK_SZ, Nb, BLK_SZ]

	{
		RECORD_SCOPE(pre_einsum, {act, mask});

		auto left_norm_gamma_a  = GetVLAPtr<T>(left_norm_input_weight, {1L}); // [K, 1L]
		auto left_norm_beta_a   = GetVLAPtr<T>(left_norm_input_bias, {1L});   // [K, 1L]
		auto proj_gate_weight_a = GetVLAPtr<T>(proj_gate_weight, {K});        // [2 * 2C, K]
		auto out_gate_weight_a  = GetVLAPtr<T>(gating_linear_weight, {K});    // [K, K]

		auto layernorm      = SCOPEIT(tpp::LayerNormFwdTPP<T>(1, BLK_SZ, K, 1e-5f), LAYER_NORM);
		auto proj_gate_gemm = SCOPEIT((tpp::BrgemmTPP<T, T>(4 * C, BLK_SZ, K, 0, 0, K, K, BLK_SZ, 0.0f, 0, 1, 0, 1, 0)), BRGEMM);
		auto mask_mul       = SCOPEIT((tpp::BCastMul2TPP<T, T>(2 * C, BLK_SZ)), MARGIN); // intended type
		auto gate_sigmoid   = SCOPEIT(tpp::SigmoidFwdTPP<T>(2 * C, BLK_SZ), SILU);
		auto proj_gate_mul  = SCOPEIT(tpp::MulTPP<T>(2 * C, BLK_SZ), EW_MUL);
		auto cpy            = SCOPEIT(tpp::CpyTPP<T>(2 * C, BLK_SZ, BLK_SZ, M * N), EW_COPY);
		auto out_gemm       = SCOPEIT((tpp::BrgemmTPP<T, T>(BLK_SZ, K, K, 0, 0, K, K, K, 0.0f, 0, 1, 1, 0, 0)), BRGEMM);
		auto out_sigmoid    = SCOPEIT(tpp::SigmoidFwdTPP<T>(BLK_SZ, K), SILU);

#pragma omp parallel for
		for(int i = 0; i < M; i++) {
			for(int j = 0; j < Nb; j++) {
				LIBXSMM_ALIGNED(T tmp[std::max(4 * C, K)][BLK_SZ], 64);
				layernorm( // pair = self.left_norm_input(pair)
					&act_a[i][j][0][0],
					&left_norm_gamma_a[0][0],
					&left_norm_beta_a[0][0],
					nullptr,
					nullptr,
					&act_out_a[i][j][0][0]);
				// ORIGIN: projection = self.projection(pair), gate = self.gate(pair)
				// <=>     C_1  = A    @ B_1, C_2  = A    @ B_2
				// <=>     C_1' = B_1' @ A',  C_2' = B_2' @ A'
				proj_gate_gemm(
					&proj_gate_weight_a[0][0],
					&act_out_a[i][j][0][0],
					&tmp[0][0],
					1);
				mask_mul( // projection *= mask
					&mask_a[i][j][0][0],
					&tmp[0][0],
					&tmp[0][0]);
				gate_sigmoid( // gate = torch.sigmoid(gate)
					&tmp[2 * C][0],
					&tmp[2 * C][0]);
				proj_gate_mul( // projection *= gate
					&tmp[0][0],
					&tmp[2 * C][0],
					&tmp[0][0]);
				cpy( // mid = projection.permute(2, 0, 1)
					&tmp[0][0],
					&mid_a[0][0][i][j][0]);
				out_gemm( // gate_out = self.gating_linear(input_pair)
					&act_out_a[i][j][0][0],
					&out_gate_weight_a[0][0],
					&tmp[0][0],
					1);
				out_sigmoid( // gate_out = torch.sigmoid(gate_out)
					&tmp[0][0],
					&act_out_a[i][j][0][0]);
			}
		}
	}

	act_out = act_out.narrow(0, 0, Mp);
	return std::make_tuple(act_out, mid);
}

template<typename T>
at::Tensor traingle_multiplication_einsum_impl(
	at::Tensor& mid,
	bool        is_outgoing,
	int64_t     from,
	int64_t     to,
	TM_OPT_ARGS) {
	RECORD_FUNCTION("C++ TM Einsum", {mid, is_outgoing});

	const int64_t Mp = mid.sizes()[1];
	const int64_t Np = mid.sizes()[2];
	const int64_t K  = std::min(to, mid.sizes()[0] / 2) - from;
	const int64_t M  = ((Mp + BLK_SZ - 1) / BLK_SZ) * BLK_SZ;
	const int64_t N  = ((Np + BLK_SZ - 1) / BLK_SZ) * BLK_SZ;
	const int64_t Mb = M / BLK_SZ;
	const int64_t Nb = N / BLK_SZ;

	const int64_t MN  = is_outgoing ? M : N;
	const int64_t MNb = is_outgoing ? Mb : Nb;

	if(MN != std::min(M, N)) {
		std::cerr << "Warning: MN != min(M, N) " << MN << " != " << std::min(M, N) << std::endl;
		std::cerr << "The program may fail silently." << std::endl;
	}

	auto out   = new_aligned_tensor(mid, {K, MN, MN});
	auto out_a = GetVLAPtr<T>(out, {MNb, BLK_SZ, MNb, BLK_SZ}); // [K, MNb, BLK_SZ, MNb, BLK_SZ]
	auto mid_a = GetVLAPtr<T>(mid, {Mb, BLK_SZ, Nb, BLK_SZ});   // [2K, Mb, BLK_SZ, Nb, BLK_SZ]

	{
		RECORD_SCOPE(einsum, {mid, out});

		if(is_outgoing) {
			auto gemm_out = SCOPEIT((tpp::BrgemmTPP<T, T>(M, BLK_SZ, N, 0, 0, N, N, M, 0.0f, 0, 1, 0, 1, 0)), BRGEMM);

#pragma omp parallel for collapse(2)
			for(int k = from; k < to; k++) {
				for(int j = 0; j < Mb; j++) {
					// No need to tiling deeper
					gemm_out( // out = a @ b.T
						&mid_a[2 * k][0][0][0][0],
						&mid_a[2 * k + 1][j][0][0][0],
						&out_a[k - from][0][0][j][0],
						1);
				}
			}
		} else {
			auto gemm_in    = SCOPEIT((tpp::BrgemmTPP<T, T>(BLK_SZ, BLK_SZ, M, 0, 0, M, N, N, 0.0f, 0, 1, 0, 0, 0)), BRGEMM);
			auto trans_b_in = SCOPEIT(tpp::XformExtTPP<T>(M, BLK_SZ, BLK_SZ, M, N, M, tpp::XformTPP::XFORM_XPOSE_TPP), XPOSE);

#pragma omp parallel for collapse(2)
			for(int k = from; k < to; k++) {
				for(int j = 0; j < Nb; j++) {
					auto tmp = reinterpret_cast<T*>(std::aligned_alloc(64, BLK_SZ * M * sizeof(T)));

					trans_b_in( // tmp = b.T
						&mid_a[2 * k + 1][0][0][j][0],
						tmp);

					gemm_in.config();
					for(int i = 0; i < Nb; i++) {
						gemm_in( // out = tmp @ a
							tmp,
							&mid_a[2 * k][0][0][i][0],
							&out_a[k - from][j][0][i][0],
							1,
							true);
					}
					gemm_in.release();
					std::free(tmp);
				}
			}
		}
	}
	return out;
}

template<typename T>
at::Tensor traingle_multiplication_post_impl(
	at::Tensor& act,
	at::Tensor& out,
	at::Tensor& input,
	at::Tensor& center_norm_weight,
	at::Tensor& center_norm_bias,
	at::Tensor& output_projection_weight,
	TM_OPT_ARGS) {
	RECORD_FUNCTION("C++ TM Post", {act, out});

	const int64_t MNp = act.sizes()[0]; // NOTE: Actually Mp
	const int64_t K   = act.sizes()[2];
	const int64_t MN  = ((MNp + BLK_SZ - 1) / BLK_SZ) * BLK_SZ;
	const int64_t MNb = MN / BLK_SZ;

	auto out_tmp   = new_aligned_tensor(out, {MN, MN, K});
	auto out_a     = GetVLAPtr<T>(out, {MNb, BLK_SZ, MNb, BLK_SZ});   // [K, MNb, BLK_SZ, MNb, BLK_SZ]
	auto out_tmp_a = GetVLAPtr<T>(out_tmp, {BLK_SZ, MNb, BLK_SZ, K}); // [MNb, BLK_SZ, MNb, BLK_SZ, K]
	{
		RECORD_SCOPE(post_einsum, {act, out});

		if(MN <= 2816) { // TODO: Find out why LIBXSMM will crash with larger size
			auto trans = SCOPEIT(tpp::XformExtTPP<T>(K, MN, MN, K, MN * MN, K, tpp::XformTPP::XFORM_XPOSE_TPP), XPOSE);
#pragma omp parallel for
			for(int i = 0; i < MN; i++) { // TODO: find out why this can't be MNp
				trans(                    // out = out.permute(1, 2, 0)
					&out_a[0][0][i][0][0],
					&out_tmp_a[0][i][0][0][0]);
			}
		} else {
			ScopedTimer _t(XPOSE);
			out_tmp   = out.permute({1, 2, 0}).contiguous();
			out_tmp_a = GetVLAPtr<T>(out_tmp, {BLK_SZ, MNb, BLK_SZ, K}); // [MNb, BLK_SZ, MNb, BLK_SZ, K]
		}
		input                    = pad_and_align_tensor(input, {MN, MN, K});
		auto act_a               = GetVLAPtr<T>(act, {MNb, BLK_SZ, K});         // [M, Nb, BLK_SZ, C]
		auto input_a             = GetVLAPtr<T>(input, {MNb, BLK_SZ, K});       // [M, Nb, BLK_SZ, C]
		auto center_norm_gamma_a = GetVLAPtr<T>(center_norm_weight, {1L});      // [K, 1L]
		auto center_norm_beta_a  = GetVLAPtr<T>(center_norm_bias, {1L});        // [K, 1L]
		auto out_proj_weight_a   = GetVLAPtr<T>(output_projection_weight, {K}); // [K, K]

		auto layernorm       = SCOPEIT(tpp::LayerNormFwdTPP<T>(1, BLK_SZ, K, 1e-5f), LAYER_NORM);
		auto projection_gemm = SCOPEIT((tpp::BrgemmTPP<T, T>(BLK_SZ, K, K, 0, 0, K, K, K, 0.0f, 0, 1, 1, 0, 0)), BRGEMM);
		auto proj_gate_mul   = SCOPEIT(tpp::MulTPP<T>(BLK_SZ, K), EW_MUL);
		auto add             = SCOPEIT(tpp::AddTPP<T>(BLK_SZ, K), BIAS);

#pragma omp parallel for
		for(int i = 0; i < MN; i++) {
			for(int j = 0; j < MNb; j++) {
				LIBXSMM_ALIGNED(T tmp[BLK_SZ][K], 64);
				layernorm( // pair = self.center_norm(pair)
					&out_tmp_a[0][i][j][0][0],
					&center_norm_gamma_a[0][0],
					&center_norm_beta_a[0][0],
					nullptr,
					nullptr,
					&tmp[0][0]);
				projection_gemm( // pair = self.output_projection(pair)
					&tmp[0][0],
					&out_proj_weight_a[0][0],
					&out_tmp_a[0][i][j][0][0],
					1);
				proj_gate_mul( // projection *= gate
					&out_tmp_a[0][i][j][0][0],
					&act_a[i][j][0][0],
					&out_tmp_a[0][i][j][0][0]);
				add( // projection += input
					&out_tmp_a[0][i][j][0][0],
					&input_a[i][j][0][0],
					&out_tmp_a[0][i][j][0][0]);
			}
		}
	}
	if(MN != MNp) {
		out_tmp = out_tmp.narrow(1, 0, MNp);
		out_tmp = out_tmp.narrow(0, 0, MNp);
	}
	return out_tmp;
}

template<typename T>
at::Tensor af3_traingle_multiplication_impl(
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
	TM_OPT_ARGS) {
	RECORD_FUNCTION("C++ Triangle Multiplication", {act, mask, is_outgoing});

	const int64_t Mp = act.sizes()[0];
	const int64_t K  = act.sizes()[2];
	const int64_t M  = ((Mp + BLK_SZ - 1) / BLK_SZ) * BLK_SZ;
	const int64_t Mb = M / BLK_SZ;

	{
		RECORD_SCOPE(tm_prep, {act, mask});
		act  = pad_and_align_tensor(act, {M, M, K}); // TODO: try to find if we can eliminate the zero-init while keep out NaNs
		mask = pad_and_align_tensor(mask, {M, M, 1L});
	}

	auto act_a     = GetVLAPtr<T>(act, {Mb, BLK_SZ, K});          // [M, Mb, BLK_SZ, K]
	auto mask_a    = GetVLAPtr<T>(mask, {Mb, BLK_SZ, 1L});        // [M, Mb, BLK_SZ, 1]
	auto act_mid   = new_aligned_tensor(act, {M, M, K});          // for holding the act
	auto act_mid_a = GetVLAPtr<T>(act_mid, {Mb, BLK_SZ, K});      // [M, Mb, BLK_SZ, K]
	auto out       = new_aligned_tensor(act, {2 * M, M, K});      // for holding the mid
	auto out_a     = GetVLAPtr<T>(out, {Mb, BLK_SZ, Mb, BLK_SZ}); // [2K, Mb, BLK_SZ, Mb, BLK_SZ]

	{
		RECORD_SCOPE(pre_einsum, {act, mask});

		auto left_norm_gamma_a  = GetVLAPtr<T>(left_norm_input_weight, {1L});  // [K, 1L]
		auto left_norm_beta_a   = GetVLAPtr<T>(left_norm_input_bias, {1L});    // [K, 1L]
		auto proj_gate_weight_a = GetVLAPtr<T>(proj_gate_weight, {2 * 2 * K}); // [K, 2N]

		auto layernorm      = SCOPEIT(tpp::LayerNormFwdTPP<T>(1, BLK_SZ, K, 1e-5f), LAYER_NORM);
		auto proj_gate_gemm = SCOPEIT((tpp::BrgemmTPP<T, T>(4 * K, BLK_SZ, K, 0, 0, K, K, BLK_SZ, 0.0f, 0, 1, 0, 1, 0)), BRGEMM);
		auto mask_mul       = SCOPEIT((tpp::MulTPP<T, T>(BLK_SZ)), MARGIN); // intended type
		auto gate_sigmoid   = SCOPEIT(tpp::SigmoidFwdTPP<T>(2 * K, BLK_SZ), SILU);
		auto proj_gate_mul  = SCOPEIT(tpp::MulTPP<T>(2 * K, BLK_SZ), EW_MUL);
		auto cpy            = SCOPEIT(tpp::CpyTPP<T>(2 * K, BLK_SZ, BLK_SZ, M * M), EW_COPY);

#pragma omp parallel for
		for(int i = 0; i < M; i++) {
			for(int j = 0; j < Mb; j++) {
				LIBXSMM_ALIGNED(T tmp[4 * K][BLK_SZ], 64);
				layernorm( // pair = self.left_norm_input(pair)
					&act_a[i][j][0][0],
					&left_norm_gamma_a[0][0],
					&left_norm_beta_a[0][0],
					nullptr,
					nullptr,
					&act_mid_a[i][j][0][0]);
				// ORIGIN: projection = self.projection(pair), gate = self.gate(pair)
				// <=>     C_1  = A    @ B_1, C_2  = A    @ B_2
				// <=>     C_1' = B_1' @ A',  C_2' = B_2' @ A'
				proj_gate_gemm(
					&proj_gate_weight_a[0][0],
					&act_mid_a[i][j][0][0],
					&tmp[0][0],
					1);
				for(int k = 0; k < 2 * K; k++) {
					mask_mul( // projection *= mask
						&mask_a[i][j][0][0],
						&tmp[k][0],
						&tmp[k][0]);
				}
				gate_sigmoid( // gate = torch.sigmoid(gate)
					&tmp[2 * K][0],
					&tmp[2 * K][0]);
				proj_gate_mul( // projection *= gate
					&tmp[0][0],
					&tmp[2 * K][0],
					&tmp[0][0]);
				cpy( // mid = projection.permute(2, 0, 1)
					&tmp[0][0],
					&out_a[0][0][i][j][0]);
			}
		}
	}
	auto out_tmp   = new_aligned_tensor(act, {K, M, M});
	auto out_tmp_a = GetVLAPtr<T>(out_tmp, {Mb, BLK_SZ, Mb, BLK_SZ}); // [K, Mb, BLK_SZ, Mb, BLK_SZ]

	{
		RECORD_SCOPE(einsum, {out});

		if(is_outgoing) {
			auto gemm_out = SCOPEIT((tpp::BrgemmTPP<T, T>(M, BLK_SZ, M, 0, 0, M, M, M, 0.0f, 0, 1, 0, 1, 0)), BRGEMM);

#pragma omp parallel for collapse(2)
			for(int k = 0; k < K; k++) {
				for(int j = 0; j < Mb; j++) {
					// No need to tiling deeper
					gemm_out( // out = a @ b.T
						&out_a[2 * k][0][0][0][0],
						&out_a[2 * k + 1][j][0][0][0],
						&out_tmp_a[k][0][0][j][0],
						1);
				}
			}
		} else {
			auto gemm_in    = SCOPEIT((tpp::BrgemmTPP<T, T>(BLK_SZ, BLK_SZ, M, 0, 0, M, M, M, 0.0f, 0, 1, 0, 0, 0)), BRGEMM);
			auto trans_b_in = SCOPEIT(tpp::XformExtTPP<T>(M, BLK_SZ, BLK_SZ, M, M, M, tpp::XformTPP::XFORM_XPOSE_TPP), XPOSE);

#pragma omp parallel for collapse(2)
			for(int k = 0; k < K; k++) {
				for(int j = 0; j < Mb; j++) {
					auto tmp = reinterpret_cast<T*>(std::aligned_alloc(64, BLK_SZ * M * sizeof(T)));

					trans_b_in( // tmp = b.T
						&out_a[2 * k + 1][0][0][j][0],
						tmp);

					gemm_in.config();
					for(int i = 0; i < Mb; i++) {
						gemm_in( // out = tmp @ a
							tmp,
							&out_a[2 * k][0][0][i][0],
							&out_tmp_a[k][j][0][i][0],
							1,
							true);
					}
					gemm_in.release();
					std::free(tmp);
				}
			}
		}
		if(M <= 2816) { // TODO: Find out why LIBXSMM will crash with larger size
			auto trans = SCOPEIT(tpp::XformExtTPP<T>(K, M, M, K, M * M, K, tpp::XformTPP::XFORM_XPOSE_TPP), XPOSE);
			out_a      = GetVLAPtr<T>(out, {BLK_SZ, Mb, BLK_SZ, K}); // [Mb, BLK_SZ, Mb, BLK_SZ, K]

#pragma omp parallel for
			for(int i = 0; i < M; i++) {
				trans( // out = out.permute(1, 2, 0)
					&out_tmp_a[0][0][i][0][0],
					&out_a[0][i][0][0][0]);
			}
		} else {
			ScopedTimer _t(XPOSE);
			out   = out_tmp.permute({1, 2, 0}).contiguous();
			out_a = GetVLAPtr<T>(out, {BLK_SZ, Mb, BLK_SZ, K}); // [Mb, BLK_SZ, Mb, BLK_SZ, K]
		}
	}

	{
		RECORD_SCOPE(post_einsum, {act, out});

		auto center_norm_gamma_a = GetVLAPtr<T>(center_norm_weight, {1L});      // [K, 1L]
		auto center_norm_beta_a  = GetVLAPtr<T>(center_norm_bias, {1L});        // [K, 1L]
		auto out_proj_weight_a   = GetVLAPtr<T>(output_projection_weight, {K}); // [K, K]
		auto out_gate_weight_a   = GetVLAPtr<T>(gating_linear_weight, {K});     // [K, K]

		auto layernorm       = SCOPEIT(tpp::LayerNormFwdTPP<T>(1, BLK_SZ, K, 1e-5f), LAYER_NORM);
		auto projection_gemm = SCOPEIT((tpp::BrgemmTPP<T, T>(BLK_SZ, K, K, 0, 0, K, K, K, 0.0f, 0, 1, 1, 0, 0)), BRGEMM);
		auto gate_sigmoid    = SCOPEIT(tpp::SigmoidFwdTPP<T>(BLK_SZ, K), SILU);
		auto proj_gate_mul   = SCOPEIT(tpp::MulTPP<T>(BLK_SZ * K), EW_MUL);
		auto fma             = SCOPEIT(tpp::AddTPP<T>(BLK_SZ * K), BIAS); // TODO: Fuse with proj_gate_mul

#pragma omp parallel for
		for(int i = 0; i < M; i++) {
			for(int j = 0; j < Mb; j++) {
				LIBXSMM_ALIGNED(T tmp[BLK_SZ][K], 64);
				layernorm( // pair = self.center_norm(pair)
					&out_a[0][i][j][0][0],
					&center_norm_gamma_a[0][0],
					&center_norm_beta_a[0][0],
					nullptr,
					nullptr,
					&tmp[0][0]);
				projection_gemm.config();
				projection_gemm( // pair = self.output_projection(pair)
					&tmp[0][0],
					&out_proj_weight_a[0][0],
					&out_a[0][i][j][0][0],
					1,
					true);
				projection_gemm( // gate_out = self.gating_linear(input_pair)
					&act_mid_a[i][j][0][0],
					&out_gate_weight_a[0][0],
					&tmp[0][0],
					1,
					true);
				projection_gemm.release();
				gate_sigmoid( // gate_out = torch.sigmoid(gate_out)
					&tmp[0][0],
					&tmp[0][0]);
				proj_gate_mul( // projection *= gate
					&tmp[0][0],
					&out_a[0][i][j][0][0],
					&out_a[0][i][j][0][0]);
				fma( // pair += Triangle(pair)
					&act_a[i][j][0][0],
					&out_a[0][i][j][0][0],
					&out_a[0][i][j][0][0]);
			}
		}
	}
	if(M != Mp) {
		out = out.narrow(1, 0, Mp);
	}
	out = out.narrow(0, 0, Mp);
	return out;
}

std::tuple<at::Tensor, at::Tensor> traingle_multiplication_pre(
	at::Tensor& act,
	at::Tensor& mask,
	at::Tensor& left_norm_input_weight,
	at::Tensor& left_norm_input_bias,
	at::Tensor& proj_gate_weight,
	at::Tensor& gating_linear_weight,
	TM_OPT_ARGS) {
	GlobalPass _gp(FWD);
	if(act.dtype() == at::kFloat) {
		return traingle_multiplication_pre_impl<float>(
			act,
			mask,
			left_norm_input_weight,
			left_norm_input_bias,
			proj_gate_weight,
			gating_linear_weight,
			TM_OPT_ARGS_FWD);
	} else {
		return traingle_multiplication_pre_impl<tpp::bfloat16>(
			act,
			mask,
			left_norm_input_weight,
			left_norm_input_bias,
			proj_gate_weight,
			gating_linear_weight,
			TM_OPT_ARGS_FWD);
	}
}

at::Tensor traingle_multiplication_einsum(
	at::Tensor& mid,
	bool        is_outgoing,
	int64_t     from,
	int64_t     to,
	TM_OPT_ARGS) {
	GlobalPass _gp(FWD);
	if(mid.dtype() == at::kFloat) {
		return traingle_multiplication_einsum_impl<float>(
			mid,
			is_outgoing,
			from,
			to,
			TM_OPT_ARGS_FWD);
	} else {
		return traingle_multiplication_einsum_impl<tpp::bfloat16>(
			mid,
			is_outgoing,
			from,
			to,
			TM_OPT_ARGS_FWD);
	}
}

at::Tensor traingle_multiplication_post(
	at::Tensor& act,
	at::Tensor& out,
	at::Tensor& input,
	at::Tensor& center_norm_weight,
	at::Tensor& center_norm_bias,
	at::Tensor& output_projection_weight,
	TM_OPT_ARGS) {
	GlobalPass _gp(FWD);
	if(act.dtype() == at::kFloat) {
		return traingle_multiplication_post_impl<float>(
			act,
			out,
			input,
			center_norm_weight,
			center_norm_bias,
			output_projection_weight,
			TM_OPT_ARGS_FWD);
	} else {
		return traingle_multiplication_post_impl<tpp::bfloat16>(
			act,
			out,
			input,
			center_norm_weight,
			center_norm_bias,
			output_projection_weight,
			TM_OPT_ARGS_FWD);
	}
}

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
	TM_OPT_ARGS) {
	GlobalPass _gp(FWD);
	if(act.dtype() == at::kFloat) {
		return af3_traingle_multiplication_impl<float>(
			act,
			mask,
			is_outgoing,
			left_norm_input_weight,
			left_norm_input_bias,
			proj_gate_weight,
			center_norm_weight,
			center_norm_bias,
			output_projection_weight,
			gating_linear_weight,
			TM_OPT_ARGS_FWD);
	} else {
		return af3_traingle_multiplication_impl<tpp::bfloat16>(
			act,
			mask,
			is_outgoing,
			left_norm_input_weight,
			left_norm_input_bias,
			proj_gate_weight,
			center_norm_weight,
			center_norm_bias,
			output_projection_weight,
			gating_linear_weight,
			TM_OPT_ARGS_FWD);
	}
}
