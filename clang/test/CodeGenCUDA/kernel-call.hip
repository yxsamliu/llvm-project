// RUN: %clang_cc1 -fhip-new-launch-api -triple x86_64-unknown-linux-gnu \
// RUN:   -std=c++11 -emit-llvm %s -o - | FileCheck %s

#include "Inputs/cuda.h"

struct A { int a[10]; };

__global__ void g1(int x) {}
__global__ void g2(A x) {}
__global__ void g3(A &x) {}
template<typename F> __global__ void g4(F f, int *x) { *x = f(); }
void (*pg1)(int x) = g1;

// CHECK-LABEL: define{{.*}}test1
void test1() {
  // CHECK: call void @_ZN4dim3C1Ejjj(%struct.dim3* {{.*}}, i32 2, i32 1, i32 1)
  // CHECK: call void @_ZN4dim3C1Ejjj(%struct.dim3* {{.*}}, i32 3, i32 1, i32 1)
  // CHECK: call i32 @hipLaunchKernel({{.*}}@_Z17__device_stub__g1i{{.*}}, i64 0, %struct.hipStream* null)
  g1<<<2, 3>>>(0);

  // CHECK: call void @_ZN4dim3C1Ejjj(%struct.dim3* {{.*}}, i32 4, i32 5, i32 6)
  // CHECK: call void @_ZN4dim3C1Ejjj(%struct.dim3* {{.*}}, i32 7, i32 8, i32 9)
  // CHECK: call i32 @hipLaunchKernel({{.*}}@_Z17__device_stub__g1i{{.*}}, i64 10, {{.*}}inttoptr (i64 11
  g1<<<dim3(4,5,6), dim3(7,8,9), 10, (hipStream_t)11>>>(0);

  // CHECK: %[[LD:.*]] = load void (i32)*, void (i32)** @pg1
  // CHECK: %[[PTR:.*]] = bitcast void (i32)* %[[LD]] to i8*
  // CHECK: call i32 @hipLaunchKernel({{.*}}%[[PTR]]{{.*}}, i64 0, %struct.hipStream* null)
  pg1<<<1, 1>>>(0);
}

// CHECK-LABEL: define{{.*}}test2
void test2() {
  A a;
  // CHECK: %agg.tmp = alloca %struct.A, align 4
  // CHECK: %kernel_args = alloca i8*, i64 1, align 16
  // CHECK: %[[CAST:.*]] = bitcast %struct.A* %agg.tmp to i8*
  // CHECK: %[[GEP:.*]] = getelementptr i8*, i8** %kernel_args, i32 0
  // CHECK: store i8* %[[CAST]], i8** %[[GEP]], align 8
  // CHECK: call i32 @hipLaunchKernel({{.*}}@_Z17__device_stub__g21A{{.*}}, i64 0, %struct.hipStream* null)
  g2<<<1, 1>>>(a);
}

// CHECK-LABEL: define{{.*}}test3
void test3() {
  A a;
  // CHECK: %a = alloca %struct.A, align 4
  // CHECK: %kernel_arg = alloca %struct.A*, align 8
  // CHECK: %kernel_args = alloca i8*, i64 1, align 16
  // CHECK: store %struct.A* %a, %struct.A** %kernel_arg, align 8
  // CHECK: %[[CAST:.*]] = bitcast %struct.A** %kernel_arg to i8*
  // CHECK: %[[GEP:.*]] = getelementptr i8*, i8** %kernel_args, i32 0
  // CHECK: store i8* %[[CAST]], i8** %[[GEP]], align 8
  // CHECK: call i32 @hipLaunchKernel({{.*}}@_Z17__device_stub__g3R1A{{.*}}, i64 0, %struct.hipStream* null)
  g3<<<1, 1>>>(a);
}

// CHECK-LABEL: define{{.*}}test4
void test4() {
  int x = 123;
  int y;
  // CHECK: %agg.tmp = alloca %class.anon, align 4
  // CHECK: %kernel_args = alloca i8*, i64 2, align 16
  // CHECK: %[[CAST:.*]] = bitcast %class.anon* %agg.tmp to i8*
  // CHECK: %[[GEP:.*]] = getelementptr i8*, i8** %kernel_args, i32 0
  // CHECK: store i8* %[[CAST]], i8** %[[GEP]], align 8
  // CHECK: call i32 @hipLaunchKernel({{.*}}@_Z17__device_stub__g4IZ5test4vEUlvE_EvT_Pi{{.*}}, i64 0, %struct.hipStream* null)
  g4<<<1, 1>>>([=]() { return x; }, &y);
}
