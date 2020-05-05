// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   -target-cpu gfx908 -target-feature +xnack \
// RUN:   -target-feature -sram-ecc \
// RUN:   -emit-llvm -o - %s | FileCheck %s

#include "Inputs/cuda.h"

// CHECK: !{i32 8, !"target-id", !"amdgcn-amd-amdhsa-gfx908:xnack+:sram-ecc-"}
__global__ void foo() {}
