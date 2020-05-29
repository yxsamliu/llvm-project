// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:    -emit-llvm -o - -x hip %s | FileCheck -check-prefix=INT-DEV %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux \
// RUN:    -emit-llvm -o - -x hip %s | FileCheck -check-prefix=INT-HOST %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -hip-cuid=123 \
// RUN:    -emit-llvm -o - -x hip %s | FileCheck -check-prefix=EXT-DEV %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux -hip-cuid=123 \
// RUN:    -emit-llvm -o - -x hip %s | FileCheck -check-prefix=EXT-HOST %s

#include "Inputs/cuda.h"

// Test normal static device variables
// INT-DEV: @_ZL1x = internal addrspace(1) global i32 0
// INT-HOST-DAG: @_ZL1x = internal global i32 undef
// INT-HOST-DAG: @[[DEVNAMEX:[0-9]+]] = {{.*}}c"_ZL1x\00"

// Test externalized static device variables
// EXT-DEV: @_ZL1x.hip.static.123 = addrspace(1) externally_initialized global i32 0
// EXT-HOST-DAG: @_ZL1x.hip.static.123 = internal global i32 undef
// EXT-HOST-DAG: @[[DEVNAMEX:[0-9]+]] = {{.*}}c"_ZL1x.hip.static.123\00"

static __device__ int x;

// Test normal static device variables
// INT-DEV: @_ZL1y = internal addrspace(4) global i32 0
// INT-HOST-DAG: @_ZL1y = internal global i32 undef
// INT-HOST-DAG: @[[DEVNAMEY:[0-9]+]] = {{.*}}c"_ZL1y\00"

// Test externalized static device variables
// EXT-DEV: @_ZL1y.hip.static.123 = addrspace(4) externally_initialized global i32 0
// EXT-HOST-DAG: @_ZL1y.hip.static.123 = internal global i32 undef
// EXT-HOST-DAG: @[[DEVNAMEY:[0-9]+]] = {{.*}}c"_ZL1y.hip.static.123\00"

static __constant__ int y;

__global__ void kernel(int *a) {
  a[0] = x;
  a[1] = y;
}

int* getDeviceSymbol(int *x);

void foo() {
  getDeviceSymbol(&x);
  getDeviceSymbol(&y);
}

// INT-HOST: __hipRegisterVar({{.*}}@_ZL1x{{.*}}@[[DEVNAMEX]]
// INT-HOST: __hipRegisterVar({{.*}}@_ZL1y{{.*}}@[[DEVNAMEY]]
// EXT-HOST: __hipRegisterVar({{.*}}@_ZL1x.hip.static.123{{.*}}@[[DEVNAMEX]]
// EXT-HOST: __hipRegisterVar({{.*}}@_ZL1y.hip.static.123{{.*}}@[[DEVNAMEY]]
