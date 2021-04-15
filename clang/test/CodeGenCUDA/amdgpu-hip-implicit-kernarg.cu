// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 -fcuda-is-device -emit-llvm -x hip -o - %s | FileCheck %s
#include "Inputs/cuda.h"

__global__ void hip_kernel_temp() {
}

// CHECK: attributes {{.*}} = {{.*}} "amdgpu-implicitarg-num-bytes"="56"
