// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 %s -verify -fsyntax-only -triple=amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device -target-cpu gfx906 -fnative-half-type \
// RUN:   -fnative-half-arguments-and-returns

#include "Inputs/cuda.h"
#include <stdatomic.h>

__device__ _Float16 test_Flot16(_Float16 *p) {
  return __atomic_fetch_sub(p, 1.0f16, memory_order_relaxed);
  // expected-error@-1 {{atomic add/sub of '_Float16' type requires runtime support that is not available for this target}}
}

__device__ __fp16 test_fp16(__fp16 *p) {
  return __atomic_fetch_sub(p, 1.0f16, memory_order_relaxed);
  // expected-error@-1 {{atomic add/sub of '__fp16' type requires runtime support that is not available for this target}}
}

struct BigStruct {
  int data[128];
};

void test_big(BigStruct *p1, BigStruct *p2) {
  __atomic_load(p1, p2, memory_order_relaxed);
  // expected-error@-1 {{atomic load/store of 'BigStruct' type requires runtime support that is not available for this target}}
}
