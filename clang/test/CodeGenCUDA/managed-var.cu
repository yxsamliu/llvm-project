// REQUIRES: x86-registered-target, amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -std=c++11 \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=DEV %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux -std=c++11 \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=HOST %s

#include "Inputs/cuda.h"


// Check a static device variable referenced by host function is externalized.
// DEV-DAG: @x = addrspace(1) externally_initialized global i32 undef
// HOST-DAG: @x = internal global i32 0
// HOST-DAG: @[[DEVNAMEX:[0-9]+]] = {{.*}}c"x\00"

__managed__ int x;
//__device__ extern int y;

__global__ void foo(int *z) {
  *z = x;
  //*z = y;
}

int load() {
  return x;
}
// HOST: __hipRegisterVar({{.*}}@x {{.*}}@[[DEVNAMEX]]{{.*}}, i32 0, i64 4, i32 0, i32 1)
