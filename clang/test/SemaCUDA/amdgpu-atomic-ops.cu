// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 %s -verify -emit-llvm - -triple=amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device -target-cpu gfx906 -fnative-half-type \
// RUN:   -fnative-half-arguments-and-returns

#include "Inputs/cuda.h"
#include <stdatomic.h>

__device__ _Float16 test_Flot16(_Float16 *p) {
  return __atomic_fetch_sub(p, 1.0f16, memory_order_relaxed);
}

__device__ __fp16 test_fp16(__fp16 *p) {
  return __atomic_fetch_sub(p, 1.0f16, memory_order_relaxed);
}

struct BigStruct {
  int data[128];
};

__device__ void test_big(BigStruct *p1, BigStruct *p2) {
  __atomic_load(p1, p2, memory_order_relaxed);
  // expected-error@-1 {{large atomic operation not supported; the access size (512 bytes) exceeds the max lock-free size (8  bytes)}}
}
