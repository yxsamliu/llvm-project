// RUN: %clang_cc1 -target-sdk-version=8.0 -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=CUDA-OLD,CHECK,COMMON
// RUN: %clang_cc1 -target-sdk-version=9.2  -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=CUDA-NEW,CHECK,COMMON
// RUN: %clang_cc1 -x hip -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=HIP-OLD,CHECK,COMMON
// RUN: %clang_cc1 -fhip-new-launch-api -x hip -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=HIP-NEW

#include "Inputs/cuda.h"

// CHECK-LABEL: define{{.*}}g1
// HIP-OLD: call{{.*}}hipSetupArgument
// HIP-OLD: call{{.*}}hipLaunchByPtr
// HIP-NEW-NOT: call{{.*}}__hipPopCallConfiguration
// HIP-NEW-NOT: call{{.*}}hipLaunchKernel
// CUDA-OLD: call{{.*}}cudaSetupArgument
// CUDA-OLD: call{{.*}}cudaLaunch
// CUDA-NEW: call{{.*}}__cudaPopCallConfiguration
// CUDA-NEW: call{{.*}}cudaLaunchKernel
__global__ void g1(int x) {}

// CHECK-LABEL: define{{.*}}main
int main(void) {
  // HIP-OLD: call{{.*}}hipConfigureCall
  // HIP-NEW-NOT: call{{.*}}__hipPushCallConfiguration
  // HIP-NEW: call{{.*}}hipLaunchKernel
  // CUDA-OLD: call{{.*}}cudaConfigureCall
  // CUDA-NEW: call{{.*}}__cudaPushCallConfiguration
  // COMMON: icmp
  // COMMON: br
  // COMMON: call{{.*}}g1
  g1<<<1, 1>>>(42);
}
