// REQUIRES: x86-registered-target, amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -std=c++11 \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=DEV,NORDC %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -std=c++11 \
// RUN:   -emit-llvm -fgpu-rdc -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=DEV,RDC %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux -std=c++11 \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=HOST %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux -std=c++11 \
// RUN:   -emit-llvm -fgpu-rdc -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=HOST %s

#include "Inputs/cuda.h"

// In device functions, static device variables are not externalized nor shadowed.
// Static managed variable behaves like a normal static device variable.

// DEV: @_ZZ4fun1vE1a = internal addrspace(1) global i32 1
// HOST-NOT: @_ZZ4fun1vE1a
// DEV: @_ZZ4fun1vE1b = internal addrspace(1) global i32 2
// HOST-NOT: @_ZZ4fun1vE1b
// DEV: @_ZZ4fun1vE1c = internal addrspace(4) constant i32 3
// HOST-NOT: @_ZZ4fun1vE1c
// DEV: @_ZZ4fun1vE1d = internal addrspace(4) constant i32 4
// HOST-NOT: @_ZZ4fun1vE1d
// DEV: @_ZZ4fun1vE1e = internal addrspace(4) global i32 5
// HOST-NOT: @_ZZ4fun1vE1e
// DEV: @_ZZ4fun1vE1f = internal addrspace(1) global i32 6
// HOST-NOT: @_ZZ4fun1vE1f
__device__ int fun1() {
  static int a = 1;
  static __device__ int b = 2;
  static const int c = 3;
  static constexpr int d = 4;
  static __constant__ int e = 5;
  static __managed__ int f = 6;
  return a + b + c + d + e + f;
}

// Assuming this function accepts a device pointer and does some work. 
__host__ __device__ int work(int *x);

// In host function, static device variables are externalized if used and shadowed.

// DEV-NOT: @_ZZ4fun2vE1a
// HOST: @_ZZ4fun2vE1a = internal global i32 1
// NORDC: @_ZZ4fun2vE1b = dso_local addrspace(1) global i32 2
// RDC: @_ZZ4fun2vE1b = internal addrspace(1) global i32 2
// HOST: @_ZZ4fun2vE1b = internal global i32 2
// DEV-NOT: @_ZZ4fun2vE1c
// HOST: @_ZZ4fun2vE1c = internal constant i32 3
// DEV-NOT: @_ZZ4fun2vE1d
// HOST: @_ZZ4fun2vE1d = internal constant i32 4
// NORDC: @_ZZ4fun2vE1e = dso_local addrspace(4) global i32 5
// RDC: @_ZZ4fun2vE1e = internal addrspace(4) global i32 5
// HOST: @_ZZ4fun2vE1e = internal global i32 5
// DEV: @_ZZ4fun2vE1f = internal addrspace(1) global i32* addrspacecast (i32 addrspace(1)* @_ZZ4fun2vE1b to i32*)
// HOST: @_ZZ4fun2vE1f = internal global i32* @_ZZ4fun2vE1b
// NORDC: @_ZZ4fun2vE1b_0 = dso_local addrspace(1) global i32 6
// RDC: @_ZZ4fun2vE1b_0 = internal addrspace(1) global i32 6
// HOST: @_ZZ4fun2vE1b_0 = internal global i32 6
// NORDC: @_ZZ4fun2vE1g = dso_local addrspace(1) externally_initialized global i32 undef
// RDC: @_ZZ4fun2vE1g = external dso_local addrspace(1) global i32
// HOST: @_ZZ4fun2vE1g = internal global i32 7
int fun2() {
  static int a = 1;
  static __device__ int b = 2;
  static const int c = 3;
  static constexpr int d = 4;
  static __constant__ int e = 5;
  static __device__ int *f = &b;
  for (int i = 0; i < 10; i++) {
    static __device__ int b = 6;
    work(&b);
  }
  static __managed__ int g = 7;
  return a + c + d + work(&e) + g;
}

// In host device function, explicit static device variables are externalized
// if used and registered. Static variables w/o attributes are implicit device
// variables in device compilation and host variables in host compilation.
// The variable emitted in host compilation is not the shadow variable of the
// variable emitted in device compilation.

// DEV: @_ZZ4fun3vE1a = internal addrspace(1) global i32 1
// HOST: @_ZZ4fun3vE1a = internal global i32 1
// NORDC: @_ZZ4fun3vE1b = dso_local addrspace(1) global i32 2
// RDC: @_ZZ4fun3vE1b = internal addrspace(1) global i32 2
// HOST: @_ZZ4fun3vE1b = internal global i32 2
// DEV: @_ZZ4fun3vE1c = internal addrspace(4) constant i32 3
// HOST: @_ZZ4fun3vE1c = internal constant i32 3
// DEV: @_ZZ4fun3vE1d = internal addrspace(4) constant i32 4
// HOST: @_ZZ4fun3vE1d = internal constant i32 4
// NORDC: @_ZZ4fun3vE1e = dso_local addrspace(4) global i32 5
// RDC: @_ZZ4fun3vE1e = internal addrspace(4) global i32 5
// HOST: @_ZZ4fun3vE1e = internal global i32 5
// DEV: @_ZZ4fun3vE1f = internal addrspace(1) global i32* addrspacecast (i32 addrspace(1)* @_ZZ4fun3vE1b to i32*)
// HOST: @_ZZ4fun3vE1f = internal global i32* @_ZZ4fun3vE1b
// NORDC: @_ZZ4fun3vE1b_0 = dso_local addrspace(1) global i32 6
// RDC: @_ZZ4fun3vE1b_0 = internal addrspace(1) global i32 6
// HOST: @_ZZ4fun3vE1b_0 = internal global i32 6
// NORDC: @_ZZ4fun3vE1g = dso_local addrspace(1) externally_initialized global i32 undef
// RDC: @_ZZ4fun3vE1g = external dso_local addrspace(1) global i32
// HOST: @_ZZ4fun3vE1g = internal global i32 7
__host__ __device__ int fun3() {
  static int a = 1;
  static __device__ int b = 2;
  static const int c = 3;
  static constexpr int d = 4;
  static __constant__ int e = 5;
  static __device__ int *f = &b;
  for (int i = 0; i < 10; i++) {
    static __device__ int b = 6;
    work(&b);
  }
  static __managed__ int g = 7;
  return a + c + d + work(&e) + g;
}

// In kernels, static device variables are not externalized nor shadowed
// since they cannot be accessed by host code. Static managed variable behaves
// like a normal static device variable.

// DEV: @_ZZ4fun4vE1a = internal addrspace(1) global i32 1
// HOST-NOT: @_ZZ4fun4vE1a
// DEV: @_ZZ4fun4vE1b = internal addrspace(1) global i32 2
// HOST-NOT: @_ZZ4fun4vE1b
// DEV: @_ZZ4fun4vE1c = internal addrspace(4) constant i32 3
// HOST-NOT: @_ZZ4fun4vE1c
// DEV: @_ZZ4fun4vE1d = internal addrspace(4) constant i32 4
// HOST-NOT: @_ZZ4fun4vE1d
// DEV: @_ZZ4fun4vE1e = internal addrspace(4) global i32 5
// HOST-NOT: @_ZZ4fun4vE1e
// DEV: @_ZZ4fun4vE1f = internal addrspace(1) global i32 6
// HOST-NOT: @_ZZ4fun4vE1f
__global__ void fun4() {
  static int a = 1;
  static __device__ int b = 2;
  static const int c = 3;
  static constexpr int d = 4;
  static __constant__ int e = 5;
  static __managed__ int f = 6;
}

// HOST-NOT: call void @__hipRegisterVar({{.*}}@_ZZ4fun1vE1f
// HOST-NOT: call void @__hipRegisterManagedVar({{.*}}@_ZZ4fun1vE1f
// HOST: call void @__hipRegisterVar({{.*}}@_ZZ4fun2vE1b
// HOST: call void @__hipRegisterVar({{.*}}@_ZZ4fun2vE1e
// HOST: call void @__hipRegisterVar({{.*}}@_ZZ4fun2vE1f
// HOST: call void @__hipRegisterVar({{.*}}@_ZZ4fun2vE1b_0
// HOST: call void @__hipRegisterManagedVar({{.*}}@_ZZ4fun2vE1g
// HOST-NOT: call void @__hipRegisterVar({{.*}}@_ZZ4fun3vE1a
// HOST: call void @__hipRegisterVar({{.*}}@_ZZ4fun3vE1b
// HOST-NOT: call void @__hipRegisterVar({{.*}}@_ZZ4fun3vE1c
// HOST-NOT: call void @__hipRegisterVar({{.*}}@_ZZ4fun3vE1d
// HOST: call void @__hipRegisterVar({{.*}}@_ZZ4fun3vE1e
// HOST: call void @__hipRegisterVar({{.*}}@_ZZ4fun3vE1f
// HOST: call void @__hipRegisterVar({{.*}}@_ZZ4fun3vE1b_0
// HOST: call void @__hipRegisterManagedVar({{.*}}@_ZZ4fun3vE1g
// HOST-NOT: call void @__hipRegisterVar({{.*}}@_ZZ4fun4vE1f
// HOST-NOT: call void @__hipRegisterManagedVar({{.*}}@_ZZ4fun4vE1f
