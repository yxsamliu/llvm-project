// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -cuid=abc \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -std=c++11 -fgpu-rdc \
// RUN:   -emit-llvm -o - -x hip %s > %t.dev

// RUN: %clang_cc1 -triple x86_64-gnu-linux -cuid=abc \
// RUN:   -aux-triple amdgcn-amd-amdhsa -std=c++11 -fgpu-rdc \
// RUN:   -emit-llvm -o - -x hip %s > %t.host

// RUN: cat %t.dev %t.host | FileCheck %s

// RUN: %clang_cc1 -triple nvptx -fcuda-is-device -cuid=abc \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -std=c++11 -fgpu-rdc \
// RUN:   -emit-llvm -o - %s | FileCheck -check-prefix=CUDA %s

#include "Inputs/cuda.h"

// CHECK-DAG: define weak_odr {{.*}}void @[[KERN1:_ZN12_GLOBAL__N_16kernelEv\.intern\.b04fd23c98500190]](
// CHECK-DAG: define weak_odr {{.*}}void @[[KERN2:_Z8tempKernIN12_GLOBAL__N_11XEEvT_\.intern\.b04fd23c98500190]](
// CHECK-DAG: define weak_odr {{.*}}void @[[KERN3:_Z8tempKernIN12_GLOBAL__N_1UlvE_EEvT_\.intern\.b04fd23c98500190]](
// CHECK-DAG: @[[STR1:.*]] = {{.*}} c"[[KERN1]]\00"
// CHECK-DAG: @[[STR2:.*]] = {{.*}} c"[[KERN2]]\00"
// CHECK-DAG: @[[STR3:.*]] = {{.*}} c"[[KERN3]]\00"
// CHECK-DAG: call i32 @__hipRegisterFunction({{.*}}@[[STR1]]
// CHECK-DAG: call i32 @__hipRegisterFunction({{.*}}@[[STR2]]
// CHECK-DAG: call i32 @__hipRegisterFunction({{.*}}@[[STR3]]

// CUDA: define weak_odr {{.*}}void @[[KERN1:_ZN12_GLOBAL__N_16kernelEv__intern__b04fd23c98500190]](

template <typename T>
__global__ void tempKern(T x) {}

namespace {
  __global__ void kernel() {}
  struct X {};
  X x;
  auto lambda = [](){};
}

void test() {
  kernel<<<1, 1>>>();

  tempKern<<<1, 1>>>(x);

  tempKern<<<1, 1>>>(lambda);
}
