// RUN: %clang_cc1 -std=c++14 -fsyntax-only -verify=host,com -x hip %s
// RUN: %clang_cc1 -std=c++14 -fsyntax-only -fcuda-is-device -verify=dev,com -x hip %s
// RUN: %clang_cc1 -std=c++14 -fsyntax-only -fgpu-rdc -verify=host,com -x hip %s
// RUN: %clang_cc1 -std=c++14 -fsyntax-only -fgpu-rdc -fcuda-is-device -verify=dev,com -x hip %s

#include "Inputs/cuda.h"

struct A {
  static int a;
  static __device__ int fun(); 
};

int A::a;
__device__ int A::fun() {
  return a;
  // dev-error@-1 {{reference to __host__ variable 'a' in __device__ function}}
}

// Assuming this function accepts a pointer to a device variable and calculate some result.
__device__ __host__ int work(const int *x);

int fun1(int x) {
  static __device__ int a = sizeof(a);
  static __device__ int b = x;
  // com-error@-1 {{dynamic initialization is not supported for __device__, __constant__, __shared__, and __managed__ variables}}
  static const __device__ int c = sizeof(a);
  static constexpr __device__ int d = sizeof(a);
  static __constant__ __device__ int e = sizeof(a);
  static __managed__ __device__ int f = sizeof(a);
  static int a2 = sizeof(a);
  static int b2 = x;
  static const int c2 = sizeof(a);
  static constexpr int d2 = sizeof(a);
  static __constant__ int e2 = sizeof(a);
  static __managed__ int f2 = sizeof(a);
  return work(&a) + work(&b) + work(&c) + work(&d) + work(&e) + f + a2 + b2 + c2 + d2 + work(&e2) + f2;
}

__device__ int fun2(int x) {
  static __device__ int a = sizeof(a);
  static __device__ int b = x;
  // com-error@-1 {{dynamic initialization is not supported for __device__, __constant__, __shared__, and __managed__ variables}}
  static const __device__ int c = sizeof(a);
  static constexpr __device__ int d = sizeof(a);
  static __constant__ __device__ int e = sizeof(a);
  static __managed__ __device__ int f = sizeof(a);
  static int a2 = sizeof(a);
  static int b2 = x;
  // com-error@-1 {{dynamic initialization is not supported for __device__, __constant__, __shared__, and __managed__ variables}}
  static const int c2 = sizeof(a);
  static constexpr int d2 = sizeof(a);
  static __constant__ int e2 = sizeof(a);
  static __managed__ int f2 = sizeof(a);
  return a + b + c + d + e + f + a2 + b2 + c2 + d2 + e2 + f2;
}

__device__ __host__ int fun3(int x) {
  static __device__ int a = sizeof(a);
  static __device__ int b = x;
  // com-error@-1 {{dynamic initialization is not supported for __device__, __constant__, __shared__, and __managed__ variables}}
  static const __device__ int c = sizeof(a);
  static constexpr __device__ int d = sizeof(a);
  static __constant__ __device__ int e = sizeof(a);
  static __managed__ __device__ int f = sizeof(a);
  static int a2 = sizeof(a);
  static int b2 = x;
  // dev-error@-1 {{dynamic initialization is not supported for __device__, __constant__, __shared__, and __managed__ variables}}
  static const int c2 = sizeof(a);
  static constexpr int d2 = sizeof(a);
  static __constant__ int e2 = sizeof(a);
  static __managed__ int f2 = sizeof(a);
  return work(&a) + work(&b) + work(&c) + work(&d) + work(&e) + f + a2 + b2 + c2 + d2 + work(&e2) + f2;
}

template<typename T>
__device__ __host__ int fun4(T x) {
  static __device__ int a = sizeof(x);
  static __device__ int b = x;
  // com-error@-1 {{dynamic initialization is not supported for __device__, __constant__, __shared__, and __managed__ variables}}
  static const __device__ int c = sizeof(x);
  static constexpr __device__ int d = sizeof(x);
  static __constant__ __device__ int e = sizeof(a);
  static __managed__ __device__ int f = sizeof(a);
  static int a2 = sizeof(x);
  static int b2 = x;
  // dev-error@-1 {{dynamic initialization is not supported for __device__, __constant__, __shared__, and __managed__ variables}}
  static const int c2 = sizeof(x);
  static constexpr int d2 = sizeof(x);
  static __constant__ int e2 = sizeof(a);
  static __managed__ int f2 = sizeof(a);
  return work(&a) + work(&b) + work(&c) + work(&d) + work(&e) + f + a2 + b2 + c2 + d2 + work(&e2) + f2;
}

__device__ __host__ int fun4_caller() {
  return fun4(1);
  // com-note@-1 {{in instantiation of function template specialization 'fun4<int>' requested here}}
}

__global__ void fun5(int x, int *y) {
  static __device__ int a = sizeof(a);
  static __device__ int b = x;
  // com-error@-1 {{dynamic initialization is not supported for __device__, __constant__, __shared__, and __managed__ variables}}
  static const __device__ int c = sizeof(a);
  static constexpr __device__ int d = sizeof(a);
  static __constant__ __device__ int e = sizeof(a);
  static __managed__ __device__ int f = sizeof(a);
  static int a2 = sizeof(a);
  static int b2 = x;
  // com-error@-1 {{dynamic initialization is not supported for __device__, __constant__, __shared__, and __managed__ variables}}
  static const int c2 = sizeof(a);
  static constexpr int d2 = sizeof(a);
  static __constant__ int e2 = sizeof(a);
  static __managed__ int f2 = sizeof(a);
  *y = a + b + c + d + e + f + a2 + b2 + c2 + d2 + e2 + f2;
}
