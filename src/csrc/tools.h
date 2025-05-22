/******************************************************************************
 * Copyright (c) 2025 Xflops - All rights reserved.
 *
 * For information on the license, see the LICENSE file.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Author: Dragon Archer (Xflops)
 ******************************************************************************/

#ifndef _TPP_TOOLS_H_
#define _TPP_TOOLS_H_

#include <torch/extension.h>

at::Tensor pad_and_align_tensor(
	const at::Tensor& tensor,
	at::IntArrayRef   padded_shape,
	bool              need_init = true,
	float             init      = 0.0f,
	int64_t           alignment = 64);

at::Tensor new_aligned_tensor(const at::Tensor& tensor, at::IntArrayRef shape, int64_t alignment = 64);

at::Tensor vnni_repack_tensor(const at::Tensor& tensor, int type, int BLK_SZ = 0);

#endif //_TPP_TOOLS_H_
