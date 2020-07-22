// RUN: %clang_cc1 %s --std=c++11 -triple x86_64-linux-unknown \
// RUN:   -fsyntax-only -o - -verify=com,host
// RUN: %clang_cc1 %s --std=c++11 -triple nvptx -fcuda-is-device \
// RUN:   -fsyntax-only -o - -verify=com,dev

#include "Inputs/cuda.h"

struct A {
  int a;
  __device__ A() { a = 1; }
  __device__ ~A() { a = 2; }
};

// This can be a global var since ctor/dtors of data members are not called.
union B {
  A a;
  __device__ B() {}
  __device__ ~B() {}
};

// This cannot be a global var since it has a dynamic ctor.
union C {
  A a;
  __device__ C() { a.a = 3; }
  __device__ ~C() {}
};

// This cannot be a global var since it has a dynamic dtor.
union D {
  A a;
  __device__ D() { }
  __device__ ~D() { a.a = 4; }
};

__device__ B b;
__device__ C c;
// com-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__device__ D d;
// com-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ void foo() {
  __shared__ B b;
  __shared__ C c;
  // com-error@-1 {{initialization is not supported for __shared__ variables.}}
  __shared__ D d;
  // com-error@-1 {{initialization is not supported for __shared__ variables.}}
}
