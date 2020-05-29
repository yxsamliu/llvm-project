// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=DEV,INT-DEV %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=HOST,INT-HOST %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -cuid=123abc \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=DEV,EXT-DEV %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux -cuid=123abc \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=HOST,EXT-HOST %s

#include "Inputs/cuda.h"

// Test function scope static device variable, which should not be externalized.
// DEV-DAG: @_ZZ6kernelPiPPKiE1w = internal addrspace(4) constant i32 1

// Test normal static device variables
// INT-DEV-DAG: @_ZL1x = internal addrspace(1) global i32 0
// INT-HOST-DAG: @_ZL1x = internal global i32 undef
// INT-HOST-DAG: @[[DEVNAMEX:[0-9]+]] = {{.*}}c"_ZL1x\00"

// Test externalized static device variables
// EXT-DEV-DAG: @_ZL1x.static.123abc = addrspace(1) externally_initialized global i32 0
// EXT-HOST-DAG: @_ZL1x.static.123abc = internal global i32 undef
// EXT-HOST-DAG: @[[DEVNAMEX:[0-9]+]] = {{.*}}c"_ZL1x.static.123abc\00"

static __device__ int x;

// Test normal static device variables
// INT-DEV-DAG: @_ZL1y = internal addrspace(4) global i32 0
// INT-HOST-DAG: @_ZL1y = internal global i32 undef
// INT-HOST-DAG: @[[DEVNAMEY:[0-9]+]] = {{.*}}c"_ZL1y\00"

// Test externalized static device variables
// EXT-DEV-DAG: @_ZL1y.static.123abc = addrspace(4) externally_initialized global i32 0
// EXT-HOST-DAG: @_ZL1y.static.123abc = internal global i32 undef
// EXT-HOST-DAG: @[[DEVNAMEY:[0-9]+]] = {{.*}}c"_ZL1y.static.123abc\00"

static __constant__ int y;

// Test static host variable, which should not be externalized nor registered.
// HOST-DAG: @_ZL1z = internal global i32 0
// DEV-NOT: @_ZL1z
static int z;

// Test static device variable in inline function, which should not be
// externalized nor registered.
// DEV-DAG: @_ZZ6devfunPPKiE1p = linkonce_odr addrspace(4) constant i32 2, comdat

inline __device__ void devfun(const int ** b) {
  const static int p = 2;
  b[0] = &p;
}

__global__ void kernel(int *a, const int **b) {
  const static int w = 1;
  a[0] = x;
  a[1] = y;
  b[0] = &w;
  devfun(b);
}

int* getDeviceSymbol(int *x);

void foo() {
  getDeviceSymbol(&x);
  getDeviceSymbol(&y);
  z = 123;
}

// INT-HOST: __hipRegisterVar({{.*}}@_ZL1x{{.*}}@[[DEVNAMEX]]
// INT-HOST: __hipRegisterVar({{.*}}@_ZL1y{{.*}}@[[DEVNAMEY]]
// EXT-HOST: __hipRegisterVar({{.*}}@_ZL1x.static.123abc{{.*}}@[[DEVNAMEX]]
// EXT-HOST: __hipRegisterVar({{.*}}@_ZL1y.static.123abc{{.*}}@[[DEVNAMEY]]
// HOST-NOT: __hipRegisterVar({{.*}}@_ZZ6kernelPiPPKiE1w
// HOST-NOT: __hipRegisterVar({{.*}}@_ZZ6devfunPPKiE1p
