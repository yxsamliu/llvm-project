// RUN: %clang_cc1 -std=c++17 -fsyntax-only -x hip -verify %s -fhip-lambda-host-device

#include "Inputs/cuda.h"

template<class F>
__global__ void kernel(F f) { f(); }
// expected-error@-1 3{{no matching function for call to object of type}}

constexpr __host__ __device__ void hd();

int main(void) {
  auto lambda_kernel = [&]__global__(){};
  // expected-error@-1 {{kernel function 'operator()' must be a free function or static member function}}

  int b;

  kernel<<<1,1>>>([](){ hd(); });

  kernel<<<1,1>>>([=](){ hd(); });

  kernel<<<1,1>>>([b](){ hd(); });

  kernel<<<1,1>>>([&]()constexpr{ hd(); });

  kernel<<<1,1>>>([&](){ hd(); });
  // expected-note@-1 {{in instantiation of function template specialization 'kernel<(lambda at}}
  // expected-note@-2 {{candidate function not viable: call to __host__ function from __global__ function}}

  kernel<<<1,1>>>([=, &b](){ hd(); });
  // expected-note@-1 {{in instantiation of function template specialization 'kernel<(lambda at}}
  // expected-note@-2 {{candidate function not viable: call to __host__ function from __global__ function}}

  kernel<<<1,1>>>([&, b](){ hd(); });
  // expected-note@-1 {{in instantiation of function template specialization 'kernel<(lambda at}}
  // expected-note@-2 {{candidate function not viable: call to __host__ function from __global__ function}}

  kernel<<<1,1>>>([=](){
      auto f = [&]{ hd(); };
      f();
  });

  return 0;
}
