// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device \
// RUN:   -emit-llvm -o - -x hip %s -debug-info-kind=limited \
// RUN:   | FileCheck %s

// Check no assertion with debug info.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device \
// RUN:   -S -o %t.s -x hip %s \
// RUN:   -debug-info-kind=limited

#include "Inputs/cuda.h"
 
struct A {
  int x[100];
  __device__ A();
};

struct B {
  int x[100];
};

__device__ B b;

__device__ void callee(A *a);

// CHECK-LABEL: @_Z5func1v(
// CHECK-SAME: %struct.A addrspace(5)* noalias sret(%struct.A) align 4 %[[RET:.*]])
// CHECK: %x = alloca [100 x i32], align 16, addrspace(5)
// CHECK: %x.ascast = addrspacecast [100 x i32] addrspace(5)* %x to [100 x i32]*
// CHECK: %p = alloca %struct.A*, align 8, addrspace(5)
// CHECK: %p.ascast = addrspacecast %struct.A* addrspace(5)* %p to %struct.A**
// CHECK: %[[RET_CAST:.*]] = addrspacecast %struct.A addrspace(5)* %[[RET]] to %struct.A*
// CHECK: call void @llvm.dbg.declare(metadata %struct.A addrspace(5)* %[[RET]]
// CHECK: call void @_ZN1AC1Ev(%struct.A* nonnull dereferenceable(400) %[[RET_CAST]])
// CHECK: call void @llvm.dbg.declare(metadata [100 x i32] addrspace(5)* %x
// CHECK: call void @_Z6calleeP1A(%struct.A* %[[RET_CAST]])
// CHECK: %[[RET_CAST2:.*]] = bitcast %struct.A* %[[RET_CAST]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %[[RET_CAST2]], i8* align 16 %{{.*}}, i64 400, i1 false)
// CHECK: call void @llvm.dbg.declare(metadata %struct.A* addrspace(5)* %p
// CHECK: store %struct.A* %[[RET_CAST]], %struct.A** %p.ascast
__device__ A func1() {
  A a;
  int x[100];
  callee(&a);
  __builtin_memcpy(&a, x, 400);
  A *p = &a;
  return a;
}

// CHECK-LABEL: @_Z6func1av(%struct.B addrspace(5)* noalias sret(%struct.B) align 4 
__device__ B func1a() {
  B b;
  return b;
}

// Check returning the return value again.

// CHECK-LABEL: @_Z5func2v(
// CHECK-SAME: %struct.A addrspace(5)* noalias sret(%struct.A) align 4 %[[RET:.*]])
// CHECK: %[[CAST1:.*]] = addrspacecast %struct.A addrspace(5)* %[[RET]] to %struct.A*
// CHECK: %[[CAST2:.*]] = addrspacecast %struct.A* %[[CAST1]] to %struct.A addrspace(5)*
// CHECK: call void @_Z5func1v(%struct.A addrspace(5)* sret(%struct.A) align 4 %[[CAST2]])
__device__ A func2() {
  A a = func1();
  return a;
}

// Check assigning the return value to a global variable.

// CHECK-LABEL: @_Z5func3v(
// CHECK: %[[RET:.*]] = alloca %struct.B, align 4, addrspace(5)
// CHECK: %[[CAST1:.*]] = addrspacecast %struct.B addrspace(5)* %[[RET]] to %struct.B*
// CHECK: %[[CAST2:.*]] = addrspacecast %struct.B* %[[CAST1]] to %struct.B addrspace(5)*
// CHECK: call void @_Z6func1av(%struct.B addrspace(5)* sret(%struct.B) align 4 %[[CAST2]]
// CHECK: %[[CAST3:.*]] = bitcast %struct.B* %[[CAST1]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64{{.*}}@b{{.*}}%[[CAST3]]
__device__ void func3() {
  b = func1a();
}

// Check assigning the return value to a temporary variable.

// CHECK-LABEL: @_Z5func4v(
// CHECK: %[[TMP:.*]] = alloca %struct.A, align 4, addrspace(5)
// CHECK: %[[TMP_CAST1:.*]] = addrspacecast %struct.A addrspace(5)* %[[TMP]] to %struct.A*
// CHECK: %[[RET:.*]] = alloca %struct.A, align 4, addrspace(5)
// CHECK: %[[RET_CAST1:.*]] = addrspacecast %struct.A addrspace(5)* %[[RET]] to %struct.A*
// CHECK: call void @_ZN1AC1Ev(%struct.A* nonnull dereferenceable(400) %[[TMP_CAST1]])
// CHECK: %[[RET_CAST2:.*]] = addrspacecast %struct.A* %[[RET_CAST1]] to %struct.A addrspace(5)*
// CHECK: call void @_Z5func1v(%struct.A addrspace(5)* sret(%struct.A) align 4 %[[RET_CAST2]]
// CHECK: %[[TMP_CAST2:.*]] = bitcast %struct.A* %[[TMP_CAST1]] to i8*
// CHECK: %[[RET_CAST3:.*]] = bitcast %struct.A* %[[RET_CAST1]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64{{.*}}%[[TMP_CAST2]]{{.*}}%[[RET_CAST3]]
__device__ void func4() {
  A a;
  a = func1();
}
