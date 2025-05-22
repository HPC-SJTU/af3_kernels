/******************************************************************************
 * Copyright (c) 2025 Xflops - All rights reserved.
 *
 * For information on the license, see the LICENSE file.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Author: Dragon Archer (Xflops)
 ******************************************************************************/

#include "dot_product_attention.h"

#include <ATen/record_function.h>
#include "timing.h"
#include "tools.h"
#include "xsmm_functors.h"

REGISTER_SCOPE(dpa_prep, "dpa_prep");
REGISTER_SCOPE(dpa_main, "dpa_main");
REGISTER_SCOPE(attn_prep, "attn_prep");
REGISTER_SCOPE(attn_main, "attn_main");
REGISTER_SCOPE(attn_out, "attn_out");
REGISTER_SCOPE(self_prep, "self_prep");
REGISTER_SCOPE(self_attn, "self_attn");

template<typename T>
at::Tensor dot_product_attention_impl(
	at::Tensor& q,
	at::Tensor& k,
	at::Tensor& v,
	at::Tensor& mask,
	at::Tensor& bias,
	bool        use_flash_attention,
	DPA_OPT_ARGS) {
	RECORD_FUNCTION("C++ Dot Product Attention", {q, k, v, mask, bias});

	int64_t raw_dim1 = q.sizes()[0];
	int64_t raw_dim2 = q.sizes()[1];
	int64_t raw_dim3 = q.sizes()[2];
	int64_t raw_dim4 = q.sizes()[3];

	int64_t B1 = raw_dim1; // M or 1
	int64_t B2 = raw_dim2; // 4 or 16
	int64_t Mp = raw_dim3; // 37 to 1491
	int64_t K  = raw_dim4; // 24 or 32 or 48
	int64_t M  = ((Mp + BLK_SZ - 1) / BLK_SZ) * BLK_SZ;
	int64_t Mb = M / BLK_SZ;

	{
		RECORD_SCOPE(dpa_prep, {q, k, v, mask, bias});
		q    = pad_and_align_tensor(q, {B1, B2, M, K});
		k    = pad_and_align_tensor(k, {B1, B2, M, K});
		v    = pad_and_align_tensor(v, {B1, B2, M, K});
		bias = pad_and_align_tensor(bias, {B2, M, M});
		mask = pad_and_align_tensor(mask, {B1, 1L, 1L, M}, true, 1.0f);
	}

	auto q_a    = GetVLAPtr<T>(q, {B2, Mb, BLK_SZ, K});         // [B1, B2, Mb, BLK_SZ, K]
	auto k_a    = GetVLAPtr<T>(k, {B2, Mb, BLK_SZ, K});         // [B1, B2, Mb, BLK_SZ, K]
	auto v_a    = GetVLAPtr<T>(v, {B2, Mb, BLK_SZ, K});         // [B1, B2, Mb, BLK_SZ, K]
	auto bias_a = GetVLAPtr<T>(bias, {Mb, BLK_SZ, Mb, BLK_SZ}); // [B2, Mb, BLK_SZ, Mb, BLK_SZ]
	auto mask_a = GetVLAPtr<T>(mask, {Mb, BLK_SZ});             // [B1, Mb, BLK_SZ]

	{
		RECORD_SCOPE(dpa_prep, {mask});

		float scaling = -1e9f;

		auto scale = SCOPEIT((tpp::ScaleTPP<T, T>(M)), EW_SCL);

#pragma omp parallel for
		for(int i = 0; i < B1; i++) {
			scale(&mask_a[i][0][0], &mask_a[i][0][0], scaling);
		}
	}

	if(!use_flash_attention || M <= FLASH_SZ) {
		RECORD_SCOPE(dpa_main, {q, k, v, bias});

		float scaling = 1.0f / sqrtf(float(K));

		auto scale       = SCOPEIT((tpp::ScaleTPP<T, T>(BLK_SZ * K)), EW_SCL);
		auto logits_gemm = SCOPEIT((tpp::BrgemmTPP<T, T>(BLK_SZ, M, K, 0, 0, K, K, M, 0.0f, 0, 1, 0, 1, 0)), BRGEMM);
		auto add_bias    = SCOPEIT((tpp::AddTPP<T, T, T>(BLK_SZ * M)), BIAS);
		auto add_mask    = SCOPEIT((tpp::AddBiasTPP<T, T>(BLK_SZ, M)), BIAS);
		auto softmax     = SCOPEIT((tpp::SoftMaxFwdTPP<T, T>(1, BLK_SZ, M)), SOFTMAX);
		auto out_gemm    = SCOPEIT((tpp::BrgemmTPP<T, T>(BLK_SZ, K, M, 0, 0, M, K, K, 0.0f, 0, 1, 0, 0, 0)), BRGEMM);

#pragma omp parallel for collapse(3)
		for(int i = 0; i < B1; i++) {
			for(int j = 0; j < B2; j++) {
				for(int k = 0; k < Mb; k++) {
					auto mode = _MM_GET_FLUSH_ZERO_MODE();
					_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
					LIBXSMM_ALIGNED(T tmp[BLK_SZ * M], 64);
					scale( // q = q * scaling
						&q_a[i][j][k][0][0],
						&q_a[i][j][k][0][0],
						scaling);
					logits_gemm( // logits = torch.matmul(q, k.transpose(-1, -2))
						&q_a[i][j][k][0][0],
						&k_a[i][j][0][0][0],
						tmp,
						1);
					add_bias( // logits += bias
						&bias_a[j][k][0][0][0],
						tmp,
						tmp);
					add_mask( // logits.masked_fill_(mask, -1e9)
						&mask_a[i][0][0],
						tmp);
					softmax( // weights = torch.softmax(logits, dim=-1)
						tmp,
						tmp);
					out_gemm( // out = torch.matmul(weights, v)
						tmp,
						&v_a[i][j][0][0][0],
						&q_a[i][j][k][0][0],
						1);
					_MM_SET_FLUSH_ZERO_MODE(mode);
				}
			}
		}
	} else { // Enable Flash Attention
		RECORD_SCOPE(dpa_main, {q, k, v, bias});

		float scaling = 1.0f / sqrtf(float(K));

		auto scale         = SCOPEIT((tpp::ScaleTPP<T, T>(BLK_SZ * K)), EW_SCL);
		auto logits_gemm   = SCOPEIT((tpp::BrgemmTPP<T, T>(BLK_SZ, FLASH_SZ, K, 0, 0, K, K, FLASH_SZ, 0.0f, 0, 1, 0, 1, 0)), BRGEMM);
		auto add_bias      = SCOPEIT(tpp::AddTPP<T>(BLK_SZ, FLASH_SZ, M, FLASH_SZ, FLASH_SZ), BIAS);
		auto add_mask      = SCOPEIT((tpp::AddBiasTPP<T, T>(BLK_SZ, FLASH_SZ)), MARGIN); // just to distinguish
		auto softmax       = SCOPEIT((tpp::VarSoftMaxFwdTPP<T, T>(BLK_SZ, FLASH_SZ, true)), SOFTMAX);
		auto softmax_fix   = SCOPEIT(tpp::SoftMaxFixUpTPP<T>(BLK_SZ, K, true), VNNI);        // just to distinguish
		auto softmax_scale = SCOPEIT(tpp::SoftMaxFlashScaleTPP<T>(BLK_SZ, K, true), EW_MUL); // just to distinguish
		auto out_gemm      = SCOPEIT((tpp::BrgemmTPP<T, T>(BLK_SZ, K, FLASH_SZ, 0, 0, FLASH_SZ, K, K, 0.0f, 0, 1, 0, 0, 0)), BRGEMM);
		auto cpy           = SCOPEIT(tpp::CpyTPP<T>(BLK_SZ * K), EW_COPY);

		int64_t margin_sz = M % FLASH_SZ;
		if(margin_sz == 0) {
			margin_sz = FLASH_SZ;
		}
		auto logits_gemm_edge = SCOPEIT((tpp::BrgemmTPP<T, T>(BLK_SZ, margin_sz, K, 0, 0, K, K, margin_sz, 0.0f, 0, 1, 0, 1, 0)), BRGEMM);
		auto add_bias_edge    = SCOPEIT(tpp::AddTPP<T>(BLK_SZ, margin_sz, M, margin_sz, margin_sz), BIAS);
		auto add_mask_edge    = SCOPEIT((tpp::AddBiasTPP<T, T>(BLK_SZ, margin_sz)), MARGIN); // just to distinguish
		auto softmax_edge     = SCOPEIT((tpp::VarSoftMaxFwdTPP<T, T>(BLK_SZ, margin_sz, true)), SOFTMAX);
		auto out_gemm_edge    = SCOPEIT((tpp::BrgemmTPP<T, T>(BLK_SZ, K, margin_sz, 0, 0, margin_sz, K, K, 0.0f, 0, 1, 0, 0, 0)), BRGEMM);

#pragma omp parallel for collapse(3)
		for(int i = 0; i < B1; i++) {
			for(int j = 0; j < B2; j++) {
				for(int k = 0; k < Mb; k++) {
					LIBXSMM_ALIGNED(T tmp[BLK_SZ * K], 64);
					LIBXSMM_ALIGNED(T tmp_2[BLK_SZ * K], 64);
					LIBXSMM_ALIGNED(T tmp_3[BLK_SZ * FLASH_SZ], 64);
					LIBXSMM_ALIGNED(float omax[BLK_SZ], 64);
					LIBXSMM_ALIGNED(float osum[BLK_SZ], 64);
					LIBXSMM_ALIGNED(float cmax[BLK_SZ], 64);
					LIBXSMM_ALIGNED(float csum[BLK_SZ], 64);

					scale(&q_a[i][j][k][0][0], &q_a[i][j][k][0][0], scaling);

					for(int l = 0; l < (M / FLASH_SZ) * FLASH_SZ; l += FLASH_SZ) {
						logits_gemm(&q_a[i][j][k][0][0], &k_a[i][j][0][l][0], tmp_3, 1);
						add_bias(&bias_a[j][k][0][0][l], tmp_3, tmp_3);
						add_mask(&mask_a[i][0][l], tmp_3);
						if(l == 0) {
							softmax(1, tmp_3, tmp_3, omax, osum, nullptr);
						} else {
							softmax(1, tmp_3, tmp_3, cmax, csum, omax);
						}
						auto p = l == 0 ? tmp_2 : tmp;
						out_gemm(tmp_3, &v_a[i][j][0][l][0], p, 1);
						if(l != 0) {
							softmax_fix(tmp, tmp_2, cmax, csum, omax, osum);
						}
					}
					if(M % FLASH_SZ != 0) {
						LIBXSMM_ALIGNED(T tmp_4[BLK_SZ * margin_sz], 64);
						int64_t l = (M / FLASH_SZ) * FLASH_SZ;
						logits_gemm_edge(&q_a[i][j][k][0][0], &k_a[i][j][0][l][0], tmp_4, 1);
						add_bias_edge(&bias_a[j][k][0][0][l], tmp_4, tmp_4);
						add_mask_edge(&mask_a[i][0][l], tmp_4);
						softmax_edge(1, tmp_4, tmp_4, cmax, csum, omax);
						out_gemm_edge(tmp_4, &v_a[i][j][0][l][0], tmp, 1);
						softmax_fix(tmp, tmp_2, cmax, csum, omax, osum);
					}
					softmax_scale(tmp_2, osum);
					cpy(tmp_2, &q_a[i][j][k][0][0]);
				}
			}
		}
	}

	if(M != Mp) {
		q = q.narrow(2, 0, Mp);
	}
	return q;
}

template<typename T>
at::Tensor _attention_impl(
	at::Tensor& pair,
	at::Tensor& mask,
	at::Tensor& bias,
	at::Tensor& q_proj,
	at::Tensor& k_proj,
	at::Tensor& v_proj,
	at::Tensor& gating_query,
	at::Tensor& out_proj,
	int64_t     head,
	bool        use_flash_attention, // unused for now
	DPA_OPT_ARGS) {
	RECORD_FUNCTION("C++ Attention", {pair, mask, bias});

	int64_t raw_dim1 = pair.sizes()[0];
	int64_t raw_dim2 = pair.sizes()[1];
	int64_t raw_dim3 = pair.sizes()[2];

	int64_t B  = raw_dim1;        // 37 to 1491
	int64_t H  = head;            // 4
	int64_t Mp = raw_dim2;        // B
	int64_t K  = raw_dim3 / head; // mostly 32, seldom 16
	int64_t M  = ((Mp + BLK_SZ - 1) / BLK_SZ) * BLK_SZ;
	int64_t Mb = M / BLK_SZ;

	{
		RECORD_SCOPE(attn_prep, {pair, mask, bias});
		pair = pad_and_align_tensor(pair, {B, M, H * K});
		bias = pad_and_align_tensor(bias, {H, M, M});
		mask = pad_and_align_tensor(mask, {B, 1L, 1L, M}, true, 1.0f);
	}

	auto out    = new_aligned_tensor(pair, {B, M, H * K});
	auto out_a  = GetVLAPtr<T>(out, {Mb, BLK_SZ, H, K});        // [B, Mb, BLK_SZ, H, K]
	auto pair_a = GetVLAPtr<T>(pair, {Mb, BLK_SZ, H * K});      // [B, Mb, BLK_SZ, H * K]
	auto bias_a = GetVLAPtr<T>(bias, {Mb, BLK_SZ, Mb, BLK_SZ}); // [H, Mb, BLK_SZ, Mb, BLK_SZ]
	auto mask_a = GetVLAPtr<T>(mask, {Mb, BLK_SZ});             // [B, Mb, BLK_SZ]
	auto q_a    = GetVLAPtr<T>(q_proj, {H * K * K});            // [H, H * K * K]
	auto k_a    = GetVLAPtr<T>(k_proj, {H * K * K});            // [H, H * K * K]
	auto v_a    = GetVLAPtr<T>(v_proj, {H * K * K});            // [H, H * K * K]
	auto g_a    = GetVLAPtr<T>(gating_query, {H * K * K});      // [H, H * K * K]
	auto o_a    = GetVLAPtr<T>(out_proj, {H * K});              // [H * K, H * K]

	{
		RECORD_SCOPE(attn_prep, {mask});

		float scaling = -1e9f;

		auto scale = SCOPEIT((tpp::ScaleTPP<T, T>(M)), EW_SCL);

#pragma omp parallel for
		for(int i = 0; i < B; i++) {
			scale(&mask_a[i][0][0], &mask_a[i][0][0], scaling);
		}
	}

	if(true) { // TODO: Flash Attention
		RECORD_SCOPE(attn_main, {pair, bias});

		float scaling = 1.0f / sqrtf(float(K));

		auto gemm_q      = SCOPEIT((tpp::BrgemmTPP<T, T>(BLK_SZ, K, H * K, 0, 0, H * K, K, K, 0.0f, 0, 1, 1, 0, 0)), BRGEMM);
		auto gemm_kv     = SCOPEIT((tpp::BrgemmTPP<T, T>(M, K, H * K, 0, 0, H * K, K, K, 0.0f, 0, 1, 1, 0, 0)), BRGEMM);
		auto scale       = SCOPEIT((tpp::ScaleTPP<T, T>(BLK_SZ * K)), EW_SCL);
		auto logits_gemm = SCOPEIT((tpp::BrgemmTPP<T, T>(BLK_SZ, M, K, 0, 0, K, K, M, 0.0f, 0, 1, 0, 1, 0)), BRGEMM);
		auto add_bias    = SCOPEIT((tpp::AddTPP<T, T, T>(BLK_SZ * M)), BIAS);
		auto add_mask    = SCOPEIT((tpp::AddBiasTPP<T, T>(BLK_SZ, M)), BIAS);
		// TODO: Find out why this has accuracy error when inputs are small (e.g. generated by unscaled and unbiased torch.randn)
		// auto add_bias_mask = SCOPEIT_REF((tpp::Add2TPP<T, T, T>(BLK_SZ, M)), BIAS);
		auto softmax     = SCOPEIT_REF((tpp::SoftMaxFwdTPP<T, T>(1, BLK_SZ, M)), SOFTMAX); // faster than original
		auto weight_gemm = SCOPEIT((tpp::BrgemmTPP<T, T>(BLK_SZ, K, M, 0, 0, M, K, K, 0.0f, 0, 1, 0, 0, 0)), BRGEMM);
		auto sig         = SCOPEIT((tpp::SigmoidFwdTPP<T, T>(BLK_SZ, K)), SILU);
		auto mul         = SCOPEIT((tpp::MulTPP<T, T>(BLK_SZ, K, K, H * K)), EW_MUL);

#pragma omp parallel for collapse(2)
		for(int i = 0; i < B; i++) {
			for(int j = 0; j < H; j++) {
				auto mode = _MM_GET_FLUSH_ZERO_MODE();
				_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
				auto tmp_q = reinterpret_cast<T*>(std::aligned_alloc(64, BLK_SZ * K * sizeof(T)));
				auto tmp_k = reinterpret_cast<T*>(std::aligned_alloc(64, M * K * sizeof(T)));
				auto tmp_v = reinterpret_cast<T*>(std::aligned_alloc(64, M * K * sizeof(T)));
				auto tmp_o = reinterpret_cast<T*>(std::aligned_alloc(64, BLK_SZ * std::max(M, K) * sizeof(T)));

				gemm_kv.config();
				gemm_kv( // k = self.k_projection(pair)
					&pair_a[i][0][0][0],
					&k_a[j][0],
					tmp_k,
					1,
					true);
				gemm_kv( // v = self.v_projection(pair)
					&pair_a[i][0][0][0],
					&v_a[j][0],
					tmp_v,
					1,
					true);
				gemm_kv.release();

				for(int k = 0; k < Mb; k++) {
					gemm_q( // q = self.q_projection(pair)
						&pair_a[i][k][0][0],
						&q_a[j][0],
						tmp_q,
						1);
					scale( // q = q * scaling
						tmp_q,
						tmp_q,
						scaling);
					logits_gemm( // logits = torch.matmul(q, k.transpose(-1, -2))
						tmp_q,
						tmp_k,
						tmp_o,
						1);
					add_bias( // logits += bias
						&bias_a[j][k][0][0][0],
						tmp_o,
						tmp_o);
					add_mask( // logits.masked_fill_(mask, -1e9)
						&mask_a[i][0][0],
						tmp_o);
					// add_bias_mask( // logits += bias; logits.masked_fill_(mask, -1e9)
					// 	&bias_a[j][k][0][0][0],
					// 	&mask_a[i][0][0],
					// 	tmp_o);
					softmax( // weights = torch.softmax(logits, dim=-1)
						tmp_o,
						tmp_o);
					weight_gemm( // weighted_avg = torch.matmul(weights, v)
						tmp_o,
						tmp_v,
						tmp_q,
						1);
					gemm_q( // gate_values = self.gating_query(pair)
						&pair_a[i][k][0][0],
						&g_a[j][0],
						tmp_o,
						1);
					sig( // gate_values = torch.sigmoid(gate_values)
						tmp_o,
						tmp_o);
					mul( // weighted_avg = weighted_avg * gate_values
						tmp_o,
						tmp_q,
						&out_a[i][k][0][j][0]);
				}

				std::free(tmp_q);
				std::free(tmp_k);
				std::free(tmp_v);
				std::free(tmp_o);
				_MM_SET_FLUSH_ZERO_MODE(mode);
			}
		}
	}
	{
		RECORD_SCOPE(attn_out, {out, pair});
		auto out_gemm = SCOPEIT((tpp::BrgemmTPP<T, T>(BLK_SZ, H * K, H * K, 0, 0, H * K, H * K, H * K, 0.0f, 0, 1, 0, 0, 0)), BRGEMM);
#pragma omp parallel for collapse(2)
		for(int i = 0; i < B; i++) {
			for(int j = 0; j < Mb; j++) {
				auto mode = _MM_GET_FLUSH_ZERO_MODE();
				_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
				out_gemm( // out = self.out_projection(weighted_avg)
					&out_a[i][j][0][0][0],
					&o_a[0][0],
					&pair_a[i][j][0][0],
					1);
				_MM_SET_FLUSH_ZERO_MODE(mode);
			}
		}
	}

	if(M != Mp) {
		pair = pair.narrow(1, 0, Mp);
	}
	return pair;
}

template<typename T>
at::Tensor self_attention_impl(
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
	DPA_OPT_ARGS) {
	RECORD_FUNCTION("C++ Attention", {pair, mask, bias});

	int64_t raw_dim1 = pair.sizes()[0];
	int64_t raw_dim2 = pair.sizes()[1];

	int64_t H  = head;            // 16
	int64_t Mp = raw_dim1;        // 64 to 1536
	int64_t K  = raw_dim2 / head; // 48
	int64_t M  = ((Mp + BLK_SZ - 1) / BLK_SZ) * BLK_SZ;
	int64_t Mb = M / BLK_SZ;

	{
		RECORD_SCOPE(self_prep, {pair, mask, bias});
		pair = pad_and_align_tensor(pair, {M, H * K});
		bias = pad_and_align_tensor(bias, {H, M, M});
		mask = pad_and_align_tensor(mask, {M}, true, 1.0f);
	}

	auto out    = new_aligned_tensor(pair, {M, H * K});
	auto out_a  = GetVLAPtr<T>(out, {BLK_SZ, H, K});            // [Mb, BLK_SZ, H, K]
	auto pair_a = GetVLAPtr<T>(pair, {BLK_SZ, H * K});          // [Mb, BLK_SZ, H * K]
	auto bias_a = GetVLAPtr<T>(bias, {Mb, BLK_SZ, Mb, BLK_SZ}); // [H, Mb, BLK_SZ, Mb, BLK_SZ]
	auto mask_a = GetVLAPtr<T>(mask, {BLK_SZ});                 // [Mb, BLK_SZ]
	auto q_a    = GetVLAPtr<T>(q_proj, {H * K * K});            // [H, H * K * K]
	auto q_w_a  = GetVLAPtr<T>(q_weight, {K});                  // [H, K]
	auto k_a    = GetVLAPtr<T>(k_proj, {H * K * K});            // [H, H * K * K]
	auto v_a    = GetVLAPtr<T>(v_proj, {H * K * K});            // [H, H * K * K]
	auto g_a    = GetVLAPtr<T>(gating_query, {H * K * K});      // [H, H * K * K]

	{
		RECORD_SCOPE(self_prep, {mask});

		float scaling = -1e9f;

		auto scale = SCOPEIT((tpp::ScaleTPP<T, T>(M)), EW_SCL);

		scale(&mask_a[0][0], &mask_a[0][0], scaling);
	}

	if(true || !use_flash_attention || M <= FLASH_SZ) {

		RECORD_SCOPE(self_attn, {pair, bias});
		float scaling = 1.0f / sqrtf(float(K));

		auto gemm_qkvg   = SCOPEIT((tpp::BrgemmTPP<T, T>(BLK_SZ, K, H * K, 0, 0, H * K, K, K, 0.0f, 0, 1, 1, 0, 0)), BRGEMM);
		auto scale       = SCOPEIT((tpp::ScaleTPP<T, T>(BLK_SZ * K)), EW_SCL);
		auto logits_gemm = SCOPEIT((tpp::BrgemmTPP<T, T>(BLK_SZ, M, K, 0, 0, K, K, M, 0.0f, 0, 1, 0, 1, 0)), BRGEMM);
		auto add_q       = SCOPEIT((tpp::AddBiasTPP<T, T>(BLK_SZ, K)), BIAS);
		auto add_bias    = SCOPEIT((tpp::AddTPP<T, T, T>(BLK_SZ * M)), BIAS);
		auto add_mask    = SCOPEIT((tpp::AddBiasTPP<T, T>(BLK_SZ, M)), BIAS);
		auto softmax     = SCOPEIT_REF((tpp::SoftMaxFwdTPP<T, T>(1, BLK_SZ, M)), SOFTMAX);
		auto weight_gemm = SCOPEIT((tpp::BrgemmTPP<T, T>(BLK_SZ, K, M, 0, 0, M, K, K, 0.0f, 0, 1, 0, 0, 0)), BRGEMM);
		auto sig         = SCOPEIT((tpp::SigmoidFwdTPP<T, T>(BLK_SZ, K)), SILU);
		auto mul         = SCOPEIT((tpp::MulTPP<T, T>(BLK_SZ, K, K, H * K)), EW_MUL);

		auto tmp_q_ = reinterpret_cast<T*>(std::aligned_alloc(64, H * M * K * sizeof(T)));
		auto tmp_k  = reinterpret_cast<T*>(std::aligned_alloc(64, H * M * K * sizeof(T)));
		auto tmp_v  = reinterpret_cast<T*>(std::aligned_alloc(64, H * M * K * sizeof(T)));

		// NOTE: The following order is tested to act the best, however I don't know why.
#pragma omp parallel for collapse(2)
		for(int j = 0; j < H; j++) {
			for(int k = 0; k < Mb; k++) {
				auto mode = _MM_GET_FLUSH_ZERO_MODE();
				_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
				gemm_qkvg.config();
				gemm_qkvg( // k = self.k_projection(pair)
					&pair_a[k][0][0],
					&k_a[j][0],
					&tmp_k[j * M * K + k * BLK_SZ * K],
					1,
					true);
				gemm_qkvg( // v = self.v_projection(pair)
					&pair_a[k][0][0],
					&v_a[j][0],
					&tmp_v[j * M * K + k * BLK_SZ * K],
					1,
					true);
				gemm_qkvg( // q = self.q_projection(pair)
					&pair_a[k][0][0],
					&q_a[j][0],
					&tmp_q_[j * M * K + k * BLK_SZ * K],
					1,
					true);
				gemm_qkvg.release();
				_MM_SET_FLUSH_ZERO_MODE(mode);
			}
		}
#pragma omp parallel for collapse(2)
		for(int j = 0; j < H; j++) {
			for(int k = 0; k < Mb; k++) {
				auto mode = _MM_GET_FLUSH_ZERO_MODE();
				_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

				auto tmp_q = &tmp_q_[j * M * K + k * BLK_SZ * K];
				auto tmp_o = reinterpret_cast<T*>(std::aligned_alloc(64, BLK_SZ * std::max(M, K) * sizeof(T)));

				add_q( // q = q + q_weight
					&q_w_a[j][0],
					tmp_q);
				scale( // q = q * scaling
					tmp_q,
					tmp_q,
					scaling);
				logits_gemm( // logits = torch.matmul(q, k.transpose(-1, -2))
					tmp_q,
					&tmp_k[j * M * K],
					tmp_o,
					1);
				add_bias( // logits += bias
					&bias_a[j][k][0][0][0],
					tmp_o,
					tmp_o);
				add_mask( // logits.masked_fill_(mask, -1e9)
					&mask_a[0][0],
					tmp_o);
				softmax( // weights = torch.softmax(logits, dim=-1)
					tmp_o,
					tmp_o);
				weight_gemm( // weighted_avg = torch.matmul(weights, v)
					tmp_o,
					&tmp_v[j * M * K],
					tmp_q,
					1);
				gemm_qkvg( // gate_values = self.gating_query(pair)
					&pair_a[k][0][0],
					&g_a[j][0],
					tmp_o,
					1);
				sig( // gate_values = torch.sigmoid(gate_values)
					tmp_o,
					tmp_o);
				mul( // weighted_avg = weighted_avg * gate_values
					tmp_o,
					tmp_q,
					&out_a[k][0][j][0]);

				std::free(tmp_o);
				_MM_SET_FLUSH_ZERO_MODE(mode);
			}
		}
		std::free(tmp_q_);
		std::free(tmp_k);
		std::free(tmp_v);
	}

	if(M != Mp) {
		out = out.narrow(0, 0, Mp);
	}
	return out;
}

at::Tensor dot_product_attention(
	at::Tensor& q,
	at::Tensor& k,
	at::Tensor& v,
	at::Tensor& mask,
	at::Tensor& bias,
	bool        use_flash_attention,
	DPA_OPT_ARGS) {
	GlobalPass _gp(FWD);
	if(q.dtype() == at::kFloat) {
		return dot_product_attention_impl<float>(q, k, v, mask, bias, use_flash_attention, DPA_OPT_ARGS_FWD);
	} else {
		return dot_product_attention_impl<at::BFloat16>(q, k, v, mask, bias, use_flash_attention, DPA_OPT_ARGS_FWD);
	}
}

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
	DPA_OPT_ARGS) {
	GlobalPass _gp(FWD);
	if(pair.dtype() == at::kFloat) {
		return _attention_impl<float>(pair, mask, bias, q_proj, k_proj, v_proj, gating_query, out_proj, head, use_flash_attention, DPA_OPT_ARGS_FWD);
	} else {
		return _attention_impl<at::BFloat16>(pair, mask, bias, q_proj, k_proj, v_proj, gating_query, out_proj, head, use_flash_attention, DPA_OPT_ARGS_FWD);
	}
}

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
	DPA_OPT_ARGS) {
	GlobalPass _gp(FWD);
	if(pair.dtype() == at::kFloat) {
		return self_attention_impl<float>(pair, mask, bias, q_proj, q_weight, k_proj, v_proj, gating_query, head, use_flash_attention, DPA_OPT_ARGS_FWD);
	} else {
		return self_attention_impl<at::BFloat16>(pair, mask, bias, q_proj, q_weight, k_proj, v_proj, gating_query, head, use_flash_attention, DPA_OPT_ARGS_FWD);
	}
}
