// RUN: %clang_cc1 -std=c++20 -x hip -fcuda-is-device \
// RUN:   -fsyntax-only %s
// RUN: %clang_cc1 -std=c++20 -x hip -fcuda-is-device \
// RUN:   -fsyntax-only -verify -DDEVICE_USE %s

#include "Inputs/cuda.h"

__host__ constexpr int direct_host_only() {
  return 1;
}
// expected-note@-3 {{'direct_host_only' declared here}}

template <class T> constexpr T direct_reference() {
  return static_cast<T>(direct_host_only());
  // expected-error@-1 {{reference to __host__ function 'direct_host_only'}}
}

// A host-side explicit instantiation should not make this implicit
// host/device template a device diagnostic root.
template int direct_reference<int>();

#ifdef DEVICE_USE
__global__ void direct_kernel(int *out) {
  *out = direct_reference<int>();
  // expected-note@-1 {{called by 'direct_kernel'}}
}
#endif

__host__ constexpr int existing_host_only() {
  return 3;
}
// expected-note@-3 2{{'existing_host_only' declared here}}

constexpr int existing_constexpr() {
  return existing_host_only();
  // expected-error@-1 {{reference to __host__ function 'existing_host_only'}}
}

void host_constexpr_use() {
  (void)existing_constexpr();
}

void host_lambda_use() {
  auto lambda = [] { return existing_host_only(); };
  (void)lambda();
}

#ifdef DEVICE_USE
__global__ void existing_implicit_hd_kernel(int *out) {
  auto lambda = [] { return existing_host_only(); };
  // expected-error@-1 {{reference to __host__ function 'existing_host_only'}}
  *out = existing_constexpr();
  // expected-note@-1 {{called by 'existing_implicit_hd_kernel'}}
  *out += lambda();
  // expected-note@-1 {{called by 'existing_implicit_hd_kernel'}}
}
#endif
