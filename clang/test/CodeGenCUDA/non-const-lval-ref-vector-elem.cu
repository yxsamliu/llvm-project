// RUN: %clang_cc1 -emit-llvm %s -o - -fcuda-is-device -triple nvptx-unknown-unknown | FileCheck %s
// RUN: %clang_cc1 -emit-llvm %s -o - -fcuda-is-device -triple amdgcn | FileCheck %s

#include "Inputs/cuda.h"
typedef double __attribute__((vector_size(32))) native_double4;

struct alignas(32) double4_struct {
    double x,y,z,w;
    __device__ operator native_double4& () { return (native_double4&)(*this); }
};

__device__ void test_write(double4_struct& x, int i) {
  x[i] = 1;
}

__device__ void test_read(double& y, double4_struct& x, int i) {
  y = x[i];
}
