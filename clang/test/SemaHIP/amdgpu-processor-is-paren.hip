// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -fsyntax-only -verify -triple amdgcn -target-cpu gfx900 \
// RUN:   -std=c++17 -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64 -aux-triple amdgcn \
// RUN:   -target-cpu x86-64 -aux-target-cpu gfx900 -std=c++17 \
// RUN:   -Wno-unused-value %s

// expected-no-diagnostics

// Test that __builtin_amdgcn_processor_is works when wrapped in parentheses
// (e.g. from macro expansion) and used in logical expressions.

#define __device__ __attribute__((device))

#define IS_GFX900() (__builtin_amdgcn_processor_is("gfx900"))
#define IS_GFX1010() (__builtin_amdgcn_processor_is("gfx1010"))
#define IS_GFX1100() (__builtin_amdgcn_processor_is("gfx1100"))

__device__ void test_or() {
    if (IS_GFX900() || IS_GFX1010())
        __builtin_trap();
}

__device__ void test_not() {
    if (!IS_GFX900())
        __builtin_trap();
}

__device__ void test_constexpr_or() {
    if constexpr (IS_GFX900() || IS_GFX1010())
        __builtin_trap();
}

__device__ void test_triple_or() {
    if (IS_GFX900() || IS_GFX1010() || IS_GFX1100())
        __builtin_trap();
}

__device__ void test_and_or() {
    if ((IS_GFX900() || IS_GFX1010()) && IS_GFX1100())
        __builtin_trap();
}

__device__ void test_not_or() {
    if (!(IS_GFX900() || IS_GFX1010()))
        __builtin_trap();
}

__device__ void test_nested_parens() {
    if (((IS_GFX900())) || ((IS_GFX1010())))
        __builtin_trap();
}

__device__ void test_constexpr_complex() {
    if constexpr ((IS_GFX900() || IS_GFX1010()) || IS_GFX1100())
        __builtin_trap();
}

__device__ void test_mixed_bare_paren() {
    if (__builtin_amdgcn_processor_is("gfx900") || IS_GFX1010())
        __builtin_trap();
}

__device__ void test_constexpr_not() {
    if constexpr (!IS_GFX900())
        __builtin_trap();
}
