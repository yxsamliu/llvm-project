// RUN: %clang_cc1 -std=c++11 -ast-dump -x hip %s | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -ast-dump -fcuda-is-device -x hip %s | FileCheck %s

#include "Inputs/cuda.h"

// CHECK-LABEL: FunctionDecl {{.*}} fun1
// CHECK: VarDecl {{.*}} a 'int' static
// CHECK-NOT: CUDADeviceAttr
// CHECK: VarDecl {{.*}} b 'int' static
// CHECK-NEXT: CUDADeviceAttr {{.*}}cuda.h
// CHECK: VarDecl {{.*}} c 'const int' static cinit
// CHECK-NOT: CUDADeviceAttr
// CHECK: VarDecl {{.*}} d 'const int' static constexpr cinit
// CHECK-NOT: CUDADeviceAttr
// CHECK: VarDecl {{.*}} e 'int' static cinit
// CHECK: CUDAConstantAttr {{.*}}cuda.h
// CHECK: VarDecl {{.*}} f 'int' static cinit
// CHECK: HIPManagedAttr {{.*}}cuda.h
// CHECK: CUDADeviceAttr {{.*}}Implicit
// CHECK-NOT: CUDADeviceAttr
void fun1() {
  static int a;
  static __device__ int b;
  static const int c = 1;
  static constexpr int d = 1;
  static __constant__ int e = 1;
  static __managed__ int f = 1;
}

// CHECK-LABEL: FunctionDecl {{.*}} fun2
// CHECK: VarDecl {{.*}} a 'int' static
// CHECK-NEXT: CUDADeviceAttr {{.*}}Implicit
// CHECK: VarDecl {{.*}} b 'int' static
// CHECK-NEXT: CUDADeviceAttr {{.*}}cuda.h
// CHECK: VarDecl {{.*}} c 'const int' static cinit
// CHECK: CUDAConstantAttr {{.*}}Implicit
// CHECK-NOT: CUDADeviceAttr
// CHECK: VarDecl {{.*}} d 'const int' static constexpr cinit
// CHECK: CUDAConstantAttr {{.*}}Implicit
// CHECK-NOT: CUDADeviceAttr
// CHECK: VarDecl {{.*}} e 'int' static cinit
// CHECK: CUDAConstantAttr {{.*}}cuda.h
// CHECK: VarDecl {{.*}} f 'int' static cinit
// CHECK: HIPManagedAttr {{.*}}cuda.h
// CHECK: CUDADeviceAttr {{.*}}Implicit
__device__ void fun2() {
  static int a;
  static __device__ int b;
  static const int c = 1;
  static constexpr int d = 1;
  static __constant__ int e = 1;
  static __managed__ int f = 1;
}
