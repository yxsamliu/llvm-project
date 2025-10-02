// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1300 -x hip -emit-llvm -fcuda-is-device \
// RUN:   -o - %s | FileCheck %s

// Check that the amdgpu-asymmetric-cluster-clamp
// IR attributes are generated correctly.

#define __global__ __attribute__((global))
#define __exp_amd_asymmetric_cluster_clamp __attribute__((amdgpu_asymmetric_cluster_clamp_kernel))

__exp_amd_asymmetric_cluster_clamp
__global__
// CHECK-LABEL: define dso_local amdgpu_kernel void @_Z3foov(
// CHECK-SAME: ) #[[ATTR0:[0-9]+]] {
void foo() {
}

// CHECK: attributes #[[ATTR0]] =
// CHECK-SAME: "amdgpu-asymmetric-cluster-clamp"
