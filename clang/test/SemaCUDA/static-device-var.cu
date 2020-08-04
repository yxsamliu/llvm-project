// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple nvptx -fcuda-is-device \
// RUN:    -emit-llvm -o - %s -fsyntax-only -verify

// RUN: %clang_cc1 -triple x86_64-gnu-linux \
// RUN:    -emit-llvm -o - %s -fsyntax-only -verify

#include "Inputs/cuda.h"

__device__ void f1() {
  const static int b = 123;
  static int a;
  // expected-error@-1 {{within a __device__ function, only __shared__ variables or const variables without device memory qualifier may be marked 'static'}}
}

__global__ void k1() {
  const static int b = 123;
  static int a;
  // expected-error@-1 {{within a __global__ function, only __shared__ variables or const variables without device memory qualifier may be marked 'static'}}
}

static __device__ int x;
static __constant__ int y;

__global__ void kernel(int *a) {
  a[0] = x;
  a[1] = y;
}

int* getDeviceSymbol(int *x);

void foo() {
  getDeviceSymbol(&x);
  getDeviceSymbol(&y);
}
