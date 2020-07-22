// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -fsyntax-only -verify %s

#include "Inputs/cuda.h"

inline __device__ void device_fn(int n);
inline __device__ void device_fn2() { device_fn(42); }

__global__ void kernel() { device_fn2(); }

inline __device__ void device_fn(int n) {
  int vla[n]; // expected-error {{variable-length array}}
}
