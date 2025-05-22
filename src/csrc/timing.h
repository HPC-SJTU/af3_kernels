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

#ifndef _TPP_TIMING_H_
#define _TPP_TIMING_H_

#include <ATen/record_function.h>
#include "utils.h"
#include "xsmm_functors.h"
#include <cxxabi.h>
#include <omp.h>
#include <string>
#include <vector>
#include <x86gprintrin.h> // for __rdtsc

enum DebugTimer {
	BRGEMM,
	XPOSE,
	SILU,
	LAYER_NORM,
	SOFTMAX,
	EW_RCP = SOFTMAX,
	BIAS,
	VNNI,
	EW_COPY,
	EW_SCL,
	EW_MUL,
	EW_ZERO,
	MARGIN,
	LAST_TIMER
};

extern double ifreq; // defined in init.cpp

inline double getTime() {
	return __rdtsc() * ifreq;
}

inline const char* DebugTimerName(int t) {
	const char* names[] = {
		"BRGEMM", "XPOSE", "SILU", "LYR_NRM", "SOFTMAX", "BIAS",
		"VNNI", "COPY", "SCALE", "MUL",
		"ZERO", "MARGIN", "LAST_TIMER"};
	return names[t];
}

enum PassType {
	OTH,
	FWD,
	BWD,
	UPD
};

#define MAX_THREADS 640

#ifdef DEBUG_TRACE_TPP
extern int tpp_debug_trace;
#endif
extern PassType globalPass;
extern int      globalScope;
constexpr int   NUM_TIMERS = ((LAST_TIMER + 8) / 8) * 8;
extern double   pass_timers[MAX_THREADS][3][NUM_TIMERS];
extern double   master_pass_timers[3];

struct Scope {
	Scope(std::string const& name)
		: name(name)
		, master_timer(0.0)
		, omp_timer(0.0)
		, detailed_timers{0.0}
		, flops{0.0}
		, count(0) { }
	const std::string name;
	double            master_timer;
	double            omp_timer;
	double            detailed_timers[MAX_THREADS][NUM_TIMERS];
	double            flops[MAX_THREADS][8];
	long              count;
};

inline std::vector<Scope>& get_scope_list() {
	static std::vector<Scope> _scope_list{Scope("Reserved")};
	return _scope_list;
}

inline std::vector<Scope>& get_pass_list() {
	static std::vector<Scope> _pass_list{
		Scope("OTH"), Scope("FWD"), Scope("BWD"), Scope("UPD")};
	return _pass_list;
}

inline int register_scope(std::string name) {
	auto& _scope_list = get_scope_list();
	_scope_list.emplace_back(name);
	int idx = _scope_list.size() - 1;
	// printf("Registering %s scope @%d\n", name.c_str(), idx);
	return idx;
}

#define REGISTER_LOCAL_SCOPE(id, name) static int sc_##id = register_scope(name)
#define REGISTER_SCOPE(id, name)       int sc_##id = register_scope(name)
#define USING_SCOPE(id)                extern int sc_##id

class ScopedTimer {
	public:
	ScopedTimer(DebugTimer t, long f = 0)
		: type(t)
		, flops(f)
		, start(getTime()) { }
	~ScopedTimer() {
		auto  time = getTime() - start;
		int   tid  = omp_get_thread_num();
		auto& pass = get_pass_list()[globalPass];
		pass.detailed_timers[tid][type] += time;
		if(type == BRGEMM) {
			pass.flops[tid][0] += flops;
		}
		if(globalPass == 0 && tid == 0) {
			pass.master_timer += time;
		}

		auto& scope = get_scope_list()[globalScope];
		scope.detailed_timers[tid][type] += time;
		if(type == BRGEMM) {
			scope.flops[tid][0] += flops;
		}
		if(globalScope == 0 && tid == 0) {
			scope.master_timer += time;
		}
	}
	DebugTimer type;
	long       flops;
	double     start;
};

class GlobalScope {
	public:
	GlobalScope(int t)
		: oldScope(globalScope)
		, start(getTime()) {
		TPP_ASSERT(t < (int)get_scope_list().size(), "Invalid scope initialized");
		globalScope = t;
#ifdef DEBUG_TRACE_TPP
		if(tpp_debug_trace >= 2) {
			printf(
				"Scope Enter: %d %s\n",
				t,
				get_scope_list()[globalScope].name.c_str());
		}
#endif
	}
	~GlobalScope() {
		auto  time  = getTime() - start;
		auto& scope = get_scope_list()[globalScope];
		scope.master_timer += time;
		scope.count++;
		if(oldScope != 0) {
			// Remove time from outer scope
			auto& outer_scope = get_scope_list()[oldScope];
			outer_scope.master_timer -= time;
		}
#ifdef DEBUG_TRACE_TPP
		if(tpp_debug_trace >= 2) {
			printf(
				"Scope Exit:  %d %s  Time: %.3f ms\n",
				globalScope,
				scope.name.c_str(),
				time * 1e3);
		}
#endif
		globalScope = oldScope;
	}
	int    oldScope;
	double start;
};

class OMPScope {
	public:
	OMPScope()
		: start(getTime()) { }
	~OMPScope() {
		auto  time  = getTime() - start;
		auto& scope = get_scope_list()[globalScope];
		scope.omp_timer += time;
		auto& pass = get_pass_list()[globalPass];
		pass.omp_timer += time;
	}
	double start;
};

class GlobalPass {
	public:
	GlobalPass(PassType p)
		: oldPass(globalPass)
		, start(getTime()) {
		globalPass = p;
	}
	~GlobalPass() {
		auto  time = getTime() - start;
		auto& pass = get_pass_list()[globalPass];
		pass.master_timer += time;
		pass.count++;
		if(oldPass != 0) {
			auto& outer_pass = get_pass_list()[oldPass];
			outer_pass.master_timer -= time;
		}
		globalPass = oldPass;
	}
	PassType oldPass;
	double   start;
};

template<typename T>
inline std::string get_class_name() {
	auto        cname = abi::__cxa_demangle(typeid(T).name(), 0, 0, NULL);
	std::string name(cname);
	free(cname);
	return name;
}

#ifdef DEBUG_TRACE_TPP
static thread_local std::string prev_class_name = "";
#endif
template<typename T, int impl = 0>
class ScopedTPP {
	public:
	ScopedTPP() { }
	ScopedTPP(T&& func, DebugTimer t)
		: func(std::move(func))
		, t(t) { }
	template<typename... Types>
	void operator()(Types... vars) {
		ScopedTimer _t(t);
#ifdef DEBUG_TRACE_TPP
		if(tpp_debug_trace >= 3 && omp_get_thread_num() == 0) {
			auto cur_class_name = get_class_name<T>();
			if(cur_class_name != prev_class_name) {
				std::cout << "Calling impl " << impl << " for " << cur_class_name
						  << std::endl;
				prev_class_name = cur_class_name;
			}
		}
#endif
		if constexpr(impl == 0) {
			func(vars...);
		} else if constexpr(impl == 1) {
			func.ref(vars...);
		} else {
			printf("invalid impl requested\n");
			exit(1);
		}
	}

	private:
	T          func;
	DebugTimer t;
};

template<typename Tin, typename Tout, int impl>
class ScopedTPP<tpp::BrgemmTPP<Tin, Tout>, impl> {
	public:
	ScopedTPP() { }
	ScopedTPP(tpp::BrgemmTPP<Tin, Tout> func, DebugTimer t = BRGEMM)
		: func(std::move(func))
		, t(t) { }
	void operator()(
		Tin*  A,
		Tin*  B,
		Tout* C,
		long  count,
		bool  no_tile_cfg = false) {
		ScopedTimer _t(t, func.flops() * count);
		if constexpr(impl == 0) {
			func(A, B, C, count, no_tile_cfg);
		} else if constexpr(impl == 1) {
			func.ref(A, B, C, count, no_tile_cfg);
		} else {
			printf("invalid impl requested\n");
			exit(1);
		}
	}

	void operator()(
		Tin*  A,
		Tin*  B,
		Tin*  B_scales,
		Tout* C,
		long  count,
		bool  no_tile_cfg = false) {
		ScopedTimer _t(t, func.flops() * count);
		if constexpr(impl == 0) {
			func(A, B, B_scales, C, count, no_tile_cfg);
		} else if constexpr(impl == 1) {
			func.ref(A, B, B_scales, C, count, no_tile_cfg);
		} else {
			printf("invalid impl requested\n");
			exit(1);
		}
	}

	void operator()(
		Tin*   A,
		float* A_scales,
		Tin*   B,
		Tin*   B_scales,
		Tout*  C,
		long   count,
		bool   no_tile_cfg = false) {
		ScopedTimer _t(t, func.flops() * count);
		if constexpr(impl == 0) {
			func(A, A_scales, B, B_scales, C, count, no_tile_cfg);
		} else if constexpr(impl == 1) {
			func.ref(A, A_scales, B, B_scales, C, count, no_tile_cfg);
		} else {
			printf("invalid impl requested\n");
			exit(1);
		}
	}

	void config(void* ptr = nullptr) {
		func.config(ptr);
	}

	void release(void* ptr = nullptr) {
		func.release(ptr);
	}

	private:
	tpp::BrgemmTPP<Tin, Tout> func;
	DebugTimer                t;
};

template<typename Tin, typename Tout, typename Tw, int impl>
class ScopedTPP<tpp::BrgemmTPP<Tin, Tout, Tw>, impl> {
	public:
	ScopedTPP() { }
	ScopedTPP(tpp::BrgemmTPP<Tin, Tout, Tw> func, DebugTimer t = BRGEMM)
		: func(std::move(func))
		, t(t) { }
	void operator()(
		Tin*  A,
		Tw*   B,
		Tout* C,
		long  count,
		bool  no_tile_cfg = false) {
		ScopedTimer _t(t, func.flops() * count);
		if constexpr(impl == 0) {
			func(A, B, C, count, no_tile_cfg);
		} else if constexpr(impl == 1) {
			func.ref(A, B, C, count, no_tile_cfg);
		} else {
			printf("invalid impl requested\n");
			exit(1);
		}
	}

	void operator()(
		Tin*  A,
		Tw*   B,
		Tw*   B_scales,
		Tout* C,
		long  count,
		bool  no_tile_cfg = false) {
		ScopedTimer _t(t, func.flops() * count);
		if constexpr(impl == 0) {
			func(A, B, B_scales, C, count, no_tile_cfg);
		} else if constexpr(impl == 1) {
			func.ref(A, B, B_scales, C, count, no_tile_cfg);
		} else {
			printf("invalid impl requested\n");
			exit(1);
		}
	}

	void operator()(
		Tin*   A,
		float* A_scales,
		Tw*    B,
		Tw*    B_scales,
		Tout*  C,
		long   count,
		bool   no_tile_cfg = false) {
		ScopedTimer _t(t, func.flops() * count);
		if constexpr(impl == 0) {
			func(A, A_scales, B, B_scales, C, count, no_tile_cfg);
		} else if constexpr(impl == 1) {
			func.ref(A, A_scales, B, B_scales, C, count, no_tile_cfg);
		} else {
			printf("invalid impl requested\n");
			exit(1);
		}
	}

	void config(void* ptr = nullptr) {
		func.config(ptr);
	}

	void release(void* ptr = nullptr) {
		func.release(ptr);
	}

	private:
	tpp::BrgemmTPP<Tin, Tout, Tw> func;
	DebugTimer                    t;
};

#define SCOPEIT(f, ...)     ScopedTPP<std::remove_reference_t<decltype(f)>, 0>(f, ##__VA_ARGS__)
#define SCOPEIT_REF(f, ...) ScopedTPP<std::remove_reference_t<decltype(f)>, 1>(f, ##__VA_ARGS__)

#define RECORD_SCOPE(scope, ...) \
	GlobalScope gs_(sc_##scope); \
	RECORD_FUNCTION(#scope, std::vector<c10::IValue>(__VA_ARGS__))

#define SCOPE_ARG(scope)  sc_##scope, #scope
#define RECORD_OMP_TIME() OMPScope os_

#endif //_TPP_TIMING_H_
