// RUN: %clang_cc1 -fcuda-is-device -fsyntax-only -verify=dev,com %s \
// RUN:   -std=c++11 -fgpu-defer-diag
// RUN: %clang_cc1 -fsyntax-only -verify=host,com %s \
// RUN:   -std=c++11 -fgpu-defer-diag

#include "Inputs/cuda.h"

__device__ void callee(int);
__host__ void callee(float); // host-note 2{{candidate function}}
__host__ void callee(double); // host-note 2{{candidate function}}

// Check no semantic diagnostics for this function since it is never
// called. Other kinds of diagnostics are still emitted.

inline __host__ __device__ void hdf_not_called() {
  callee(1);
  typedef 123; // com-error {{expected unqualified-id}}
  bad_line // this is a semantic error
}

// When emitted on device, there is syntax error.
// When emitted on host, there is ambiguity and syntax error.
  
inline __host__ __device__ void hdf_called() {
  callee(1); // host-error {{call to 'callee' is ambiguous}}
  bad_line // com-error {{use of undeclared identifier 'bad_line'}}
}

// This is similar to the above but is always emitted on
// both sides.

__host__ __device__ void hdf_always_emitted() {
  callee(1); // host-error {{call to 'callee' is ambiguous}}
  bad_line // com-error {{use of undeclared identifier 'bad_line'}}
}

void hf() {
 hdf_called(); // host-note {{called by 'hf'}}
}
 
__device__ void df() {
 hdf_called(); // dev-note {{called by 'df'}}
}

struct A { int x; typedef int type; };
struct B { int x; };

// This function is invalid for A and B by SFINAE.
// This fails to substitue for A but no diagnostic
// should be emitted.
template<typename T, typename T::foo* = nullptr>
__host__ __device__ void sfinae(T t) { // com-note {{candidate template ignored: substitution failure [with T = B]}}
  t.x = 1;
}

// This function is defined for A only by SFINAE.
// Calling it with A should succeed, with B should fail.
// The error should not be deferred since it happens in
// file scope.

template<typename T, typename T::type* = nullptr>
__host__ __device__ void sfinae(T t) { // com-note {{candidate template ignored: substitution failure [with T = B]}}
  t.x = 1;
}

void test_sfinae() {
  sfinae(A());
  sfinae(B()); // com-error{{no matching function for call to 'sfinae'}}
}

// If a syntax error causes a function not declared, it cannot
// be deferred.

inline __host__ __device__ void bad_func() { // com-note {{to match this '{'}}
// com-error {{expected '}'}}
