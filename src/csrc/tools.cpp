/******************************************************************************
 * Copyright (c) 2025 Xflops - All rights reserved.
 *
 * For information on the license, see the LICENSE file.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Author: Dragon Archer (Xflops)
 ******************************************************************************/

#include "tools.h"
#include "timing.h"
#include "utils.h"
#include "xsmm_functors.h"

// TODO: Try to use Intel DSA
void fast_memcpy(void* dest, void* src, int64_t size, int64_t in_rows, int64_t in_cols, int64_t out_rows, int64_t out_cols, bool zero = true) {
	RECORD_FUNCTION("C++ Memcpy", {size, in_rows, in_cols, out_rows, out_cols, zero});
	ScopedTimer _t(EW_COPY);

	in_cols *= size;
	out_cols *= size;
	if(zero) {
#pragma omp parallel for
		for(int64_t i = 0; i < in_rows; ++i) {
			memcpy((char*)dest + i * out_cols, (char*)src + i * in_cols, in_cols);
			memset((char*)dest + i * out_cols + in_cols, 0, out_cols - in_cols);
		}
		if(out_rows > in_rows) {
#pragma omp parallel for
			for(int64_t i = in_rows; i < out_rows; ++i) {
				memset((char*)dest + i * out_cols, 0, out_cols);
			}
		}
	} else {
#pragma omp parallel for
		for(int64_t i = 0; i < in_rows; ++i) {
			memcpy((char*)dest + i * out_cols, (char*)src + i * in_cols, in_cols);
		}
	}
}

at::Tensor pad_and_align_tensor(
	const at::Tensor& tensor,
	at::IntArrayRef   padded_shape,
	bool              need_init,
	float             init,
	int64_t           alignment) {
	RECORD_FUNCTION("C++ Pad & Align Tensor", {tensor});

	auto    raw_shape    = tensor.sizes();
	int64_t padded_numel = 1, dims = padded_shape.size();
	int64_t in_rows = 1, in_cols = 1, out_rows = 0, out_cols = 0;
	for(int64_t dim = dims - 1; dim >= 0; --dim) {
		padded_numel *= padded_shape[dim];
		if(out_cols == 0) {
			if(padded_shape[dim] == raw_shape[dim]) {
				in_cols *= padded_shape[dim];
			} else {
				out_cols = in_cols * padded_shape[dim];
				in_cols *= raw_shape[dim];
			}
		} else if(out_rows == 0) {
			if(padded_shape[dim] == raw_shape[dim]) {
				in_rows *= padded_shape[dim];
			} else {
				out_rows = in_rows * padded_shape[dim];
				in_rows *= raw_shape[dim];
			}
		} else {
			out_rows = -1;
		}
	}
	if(out_rows == 0) {
		out_rows = in_rows;
	}
	size_t data_size = padded_numel * tensor.itemsize();

	if(padded_numel == tensor.numel()
	   && tensor.is_contiguous()
	   && (reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0)) {
		return tensor;
	}

	data_size = (data_size + alignment - 1) / alignment * alignment;
	void* ptr = std::aligned_alloc(alignment, data_size);
	TPP_ASSERT(ptr != nullptr, "Alloc failed");

	auto deleter = [ptr](void*) {
		std::free(ptr);
	};

	at::Tensor aligned_tensor = at::from_blob(
		ptr,
		padded_shape,
		deleter,
		tensor.options());

	if(tensor.is_contiguous() && (!need_init || init == 0.0f) && out_rows != -1) {
		fast_memcpy(
			ptr,
			tensor.data_ptr(),
			tensor.itemsize(),
			in_rows,
			in_cols,
			out_rows,
			out_cols,
			need_init);
	} else {
		if(need_init) {
			ScopedTimer _t(EW_ZERO);
			aligned_tensor.fill_(init);
		}
		at::Tensor view = aligned_tensor;
		for(int64_t dim = 0; dim < dims; ++dim) {
			view = view.narrow(dim, 0, raw_shape[dim]);
		}
		{
			ScopedTimer _t(EW_COPY);
			view.copy_(tensor);
		}
	}

	return aligned_tensor;
}

at::Tensor new_aligned_tensor(const at::Tensor& tensor, at::IntArrayRef shape, int64_t alignment) {
	RECORD_FUNCTION("C++ New Aligned Tensor", {shape});

	int64_t numel = 1;
	for(auto dim : shape) {
		numel *= dim;
	}
	size_t data_size = numel * tensor.itemsize();

	data_size = (data_size + alignment - 1) / alignment * alignment;
	void* ptr = std::aligned_alloc(alignment, data_size);
	TPP_ASSERT(ptr != nullptr, "Alloc failed");

	auto deleter = [ptr](void*) {
		std::free(ptr);
	};

	at::Tensor aligned_tensor = at::from_blob(
		ptr,
		shape,
		deleter,
		tensor.options());
	return aligned_tensor;
}

at::Tensor vnni_repack_tensor(const at::Tensor& tensor, int type, int BLK_SZ) {
	RECORD_FUNCTION("C++ VNNI Repack Tensor", {tensor});

	TPP_ASSERT(tensor.dim() == 2, "VNNI Repack only support 2D tensor");
	TPP_ASSERT(tensor.dtype() == at::kBFloat16, "VNNI Repack only support bf16 tensor");
	TPP_ASSERT(tensor.is_contiguous(), "VNNI Repack only support contiguous tensor");

	auto M     = tensor.sizes()[0];
	auto N     = tensor.sizes()[1];
	auto ret   = new_aligned_tensor(tensor, {M, N});
	auto ret_a = GetVLAPtr<tpp::bfloat16>(ret, {M});
	auto t_a   = GetVLAPtr<tpp::bfloat16>(tensor, {N});
	if(BLK_SZ == 0) {
		BLK_SZ = N;
	}

	TPP_ASSERT(N % BLK_SZ == 0, "N must be divisible by BLK_SZ");

	auto tpp = tpp::XformExtTPP<tpp::bfloat16>(M, BLK_SZ, M, BLK_SZ, N, BLK_SZ, static_cast<tpp::XformTPP::XFORM_TYPE>(type));

	for(int i = 0; i < N; i += BLK_SZ) {
		tpp(&t_a[0][i], &ret_a[i][0]);
	}

	return ret;
}
