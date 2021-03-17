// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -std=c++11 -O3 -mllvm -amdgpu-internalize-symbols -emit-llvm -o - \
// RUN:   | FileCheck %s

#include "Inputs/cuda.h"

// AMDGPU internalize unused global variables for whole-program compilation
// (-fno-gpu-rdc for each TU, or -fgpu-rdc for LTO), which are then
// eliminated by global DCE. If there are invisible unused address space casts
// for global variables, the internalization and elimination of unused global
// variales will be hindered. This test makes sure no such address space
// casts.

// Check unused device/constant variables are eliminated.

// CHECK-NOT: @v1
__device__ int v1;

// CHECK-NOT: @v2
__constant__ int v2;

// CHECK-NOT: @_ZL2v3
constexpr int v3 = 1;

// Check managed variables are always kept.

// CHECK: @v4
__managed__ int v4;

// Check used device/constant variables are not eliminated.
// CHECK: @u1
__device__ int u1;

// CHECK: @u2
__constant__ int u2;

// Check u3 is kept because its address is taken.
// CHECK: @_ZL2u3
constexpr int u3 = 2;

// Check u4 is not kept because it is not ODR-use.
// CHECK-NOT: @_ZL2u4
constexpr int u4 = 3;

__device__ int fun1(const int& x);

__global__ void kern1(int *x) {
  *x = u1 + u2 + fun1(u3) + u4;
}
