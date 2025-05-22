/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.
 * Copyright (c) 2025 Xflops - All rights reserved.
 *
 * For information on the license, see the LICENSE file.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Authors: Dhiraj Kalamkar (Intel Corp.)
 *          Dragon Archer (Xflops)
 ******************************************************************************/

#include "dot_product_attention.h"
#include "gated_linear_unit.h"
#include "timing.h"
#include "tools.h"
#include "triangle_multiplication.h"
#include "utils.h"

PassType globalPass = OTH;

int globalScope = 0;

long long hsh_key, hsh_ret;

// TODO: Find a better way to get the CPU frequency
inline double getFreq() {
	long long int s = __rdtsc();
	sleep(1);
	long long int e = __rdtsc();
	return (e - s) * 1.0;
}

inline int env2int(const char* env_name, int def_val = 0) {
	int  val = def_val;
	auto env = getenv(env_name);
	if(env) {
		val = atoi(env);
	}
	// printf("Using %s = %d\n", env_name, val);
	return val;
}

int guess_mpi_rank() {
	const char* env_names[] = {
		"RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"};
	static int guessed_rank = -1;
	if(guessed_rank >= 0) {
		return guessed_rank;
	}
	for(int i = 0; i < 4; i++) {
		if(getenv(env_names[i]) != NULL) {
			int r = atoi(getenv(env_names[i]));
			if(r >= 0) {
				// printf("My guessed rank = %d\n", r);
				guessed_rank = r;
				return guessed_rank;
			}
		}
	}
	guessed_rank = 0;
	return guessed_rank;
}

double     ifreq                     = 1.0 / getFreq();
int        TPP_VERBOSE               = env2int("TPP_VERBOSE", 0);
static int TPP_DEBUG_TIMER_TIDS_UPTO = env2int("TPP_DEBUG_TIMER_TIDS_UPTO", 0);
static int TPP_DEBUG_TIMER_RANK      = env2int("TPP_DEBUG_TIMER_RANK", 0);
static int TPP_DEBUG_TIMER_DETAILED  = env2int("TPP_DEBUG_TIMER_DETAILED", 0);
#ifdef DEBUG_TRACE_TPP
int tpp_debug_trace = env2int("TPP_DEBUG_TRACE", 0);
#endif

template<int maxlen>
class SafePrint {
	public:
	SafePrint() { }
	template<typename... Types>
	int operator()(Types... vars) {
		int l = snprintf(&buf[len], 2 * maxlen - len, vars...);
		if(len + l >= 2 * maxlen) {
			buf[len] = 0;
			print();
			l = snprintf(&buf[len], 2 * maxlen - len, vars...);
		}
		len += l;
		if(len >= maxlen) {
			print();
		}
		return l;
	}
	void print() {
		printf("%s", buf);
		len    = 0;
		buf[0] = 0;
	}

	private:
	char buf[2 * maxlen];
	int  len = 0;
};

void reset_debug_timers() {
	hsh_key = 0;
	hsh_ret = 0;
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		for(auto& scope : get_pass_list()) {
			if(scope.master_timer == 0.0) {
				continue;
			}
			for(int t = 0; t < NUM_TIMERS; t++) {
				scope.detailed_timers[tid][t] = 0.0;
			}
			scope.flops[tid][0] = 0;
		}
		for(auto& scope : get_scope_list()) {
			if(scope.master_timer == 0.0) {
				continue;
			}
			for(int t = 0; t < NUM_TIMERS; t++) {
				scope.detailed_timers[tid][t] = 0.0;
			}
			scope.flops[tid][0] = 0;
		}
	}
	for(auto& scope : get_pass_list()) {
		if(scope.master_timer == 0.0) {
			continue;
		}
		scope.master_timer = 0.0;
		scope.omp_timer    = 0.0;
		scope.count        = 0;
	}
	for(auto& scope : get_scope_list()) {
		if(scope.master_timer == 0.0) {
			continue;
		}
		scope.master_timer = 0.0;
		scope.omp_timer    = 0.0;
		scope.count        = 0;
	}
}

void print_debug_timers(int tid, bool detailed) {
	detailed    = detailed || (TPP_DEBUG_TIMER_DETAILED != 0);
	int my_rank = guess_mpi_rank();
	if(my_rank != TPP_DEBUG_TIMER_RANK && TPP_DEBUG_TIMER_RANK != -1) {
		return;
	}
	int               max_threads = omp_get_max_threads();
	constexpr int     maxlen      = 10000;
	SafePrint<maxlen> printf;
	printf("%-11s: ", "#KEY#");
	for(int t = 0; t < LAST_TIMER; t++) {
		if(detailed || t == 0) {
			printf(" %7s", DebugTimerName(t));
		}
	}
	printf(
		" %8s  %8s  %5s %8s (%4s) %6s\n",
		"Total",
		"MTotal",
		"Count",
		"TotalGFS",
		"IMBL",
		"TF/s");
	for(int i = 0; i < max_threads; i++) {
		if(tid == -1 || tid == i || TPP_DEBUG_TIMER_TIDS_UPTO > i) {
			auto print_scope = [&](const Scope& scope) {
				if(scope.master_timer == 0.0) {
					return;
				}
				double total = 0.0;
				printf("%-11s: ", scope.name.c_str());
				for(int t = 0; t < LAST_TIMER; t++) {
					if(detailed || t == 0) {
						printf(" %7.1f", scope.detailed_timers[i][t] * 1e3);
					}
					total += scope.detailed_timers[i][t];
				}
				long t_flops = 0;
				for(int f = 0; f < max_threads; f++) {
					t_flops += scope.flops[f][0];
				}
				if(t_flops > 0.0) {
					printf(
						" %8.1f  %8.1f  %5ld %8.3f (%4.2f) %6.3f\n",
						total * 1e3,
						scope.master_timer * 1e3,
						scope.count,
						t_flops * 1e-9,
						t_flops * 100.0 / (scope.flops[i][0] * max_threads),
						t_flops * 1e-12 / scope.detailed_timers[i][BRGEMM]);
				} else {
					printf(
						" %8.1f  %8.1f  %5ld\n",
						total * 1e3,
						scope.master_timer * 1e3,
						scope.count);
				}
			};
			for(auto& scope : get_pass_list()) {
				print_scope(scope);
			}
			for(auto& scope : get_scope_list()) {
				print_scope(scope);
			}
		}
	}
	printf(
		"Hash create: %.3f ms   Hash search: %.3f ms\n",
		hsh_key * ifreq * 1e3,
		hsh_ret * ifreq * 1e3);
	printf.print();
}

void print_debug_thread_imbalance() {
	int my_rank = guess_mpi_rank();
	if(my_rank != 0) {
		return;
	}
	int               max_threads = omp_get_max_threads();
	constexpr int     maxlen      = 10000;
	SafePrint<maxlen> printf;
	printf("%-11s: ", "#KEY#");
	printf("TID %7s %7s %7s  ", DebugTimerName(0), "ELTW", "Total");
	printf("MIN %7s %7s %7s  ", DebugTimerName(0), "ELTW", "Total");
	printf("MAX %7s %7s %7s  ", DebugTimerName(0), "ELTW", "Total");
	printf(
		" %8s  %9s (%5s) %6s   %9s %9s %9s\n",
		"MTotal",
		"GF_Total",
		"IMBL",
		"TF/s",
		"GF_T0",
		"GF_Tmin",
		"GF_Tmax");
	auto print_scope = [&](const Scope& scope) {
		if(scope.master_timer == 0.0) {
			return;
		}
		double total_0 = 0.0;
		for(int t = 0; t < LAST_TIMER; t++) {
			total_0 += scope.detailed_timers[0][t];
		}
		double total_min  = total_0;
		double total_max  = total_0;
		int    total_imin = 0;
		int    total_imax = 0;
		for(int i = 1; i < max_threads; i++) {
			double total = 0.0;
			for(int t = 0; t < LAST_TIMER; t++) {
				total += scope.detailed_timers[i][t];
			}
			if(total < total_min) {
				total_imin = i;
				total_min  = total;
			}
			if(total > total_max) {
				total_imax = i;
				total_max  = total;
			}
		}
		printf("%-11s: ", scope.name.c_str());
		printf(
			"T%02d %7.1f %7.1f %7.1f  ",
			0,
			scope.detailed_timers[0][0] * 1e3,
			(total_0 - scope.detailed_timers[0][0]) * 1e3,
			total_0 * 1e3);
		printf(
			"T%02d %7.1f %7.1f %7.1f  ",
			total_imin,
			scope.detailed_timers[total_imin][0] * 1e3,
			(total_min - scope.detailed_timers[total_imin][0]) * 1e3,
			total_min * 1e3);
		printf(
			"T%02d %7.1f %7.1f %7.1f  ",
			total_imax,
			scope.detailed_timers[total_imax][0] * 1e3,
			(total_max - scope.detailed_timers[total_imax][0]) * 1e3,
			total_max * 1e3);
		long t_flops = 0;
		for(int f = 0; f < max_threads; f++) {
			t_flops += scope.flops[f][0];
		}
		if(t_flops > 0.0) {
			printf(
				" %8.1f  %9.3f (%5.2f) %6.3f   %9.3f %9.3f %9.3f\n",
				scope.master_timer * 1e3,
				t_flops * 1e-9,
				t_flops * 100.0 / (scope.flops[0][0] * max_threads),
				t_flops * 1e-12 / scope.detailed_timers[0][BRGEMM],
				scope.flops[0][0] * 1e-9,
				scope.flops[total_imin][0] * 1e-9,
				scope.flops[total_imax][0] * 1e-9);
		} else {
			printf(" %8.1f\n", scope.master_timer * 1e3);
		}
	};
	for(auto& scope : get_pass_list()) {
		print_scope(scope);
	}
	for(auto& scope : get_scope_list()) {
		print_scope(scope);
	}
	printf.print();
}

PYBIND11_MODULE(_C, m) {
	m.def("print_debug_timers",
		  &print_debug_timers,
		  "print_debug_timers");
	m.def("print_debug_thread_imbalance",
		  &print_debug_thread_imbalance,
		  "print_debug_thread_imbalance");
	m.def("reset_debug_timers",
		  &reset_debug_timers,
		  "reset_debug_timers");

	m.def("pad_and_align_tensor",
		  &pad_and_align_tensor,
		  "Pad and Align Tensor");
	m.def("vnni_repack_tensor",
		  &vnni_repack_tensor,
		  "VNNI Repack Tensor");

	m.def("gated_linear_unit",
		  &gated_linear_unit,
		  "Gated Linear Unit");
	m.def("linear",
		  &linear,
		  "Linear");

	m.def("traingle_multiplication_pre",
		  &traingle_multiplication_pre,
		  "Pre Triangle Multiplication");
	m.def("traingle_multiplication_einsum",
		  &traingle_multiplication_einsum,
		  "Einsum Triangle Multiplication");
	m.def("traingle_multiplication_post",
		  &traingle_multiplication_post,
		  "Post Triangle Multiplication");
	m.def("af3_traingle_multiplication",
		  &af3_traingle_multiplication,
		  "Triangle Multiplication");

	m.def("dot_product_attention",
		  &dot_product_attention,
		  "Dot Product Attention");
	m.def("_attention",
		  &_attention,
		  "Attention");
	m.def("self_attention",
		  &self_attention,
		  "Self Attention");
};
