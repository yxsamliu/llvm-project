// REQUIRES: x86-registered-target, amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -std=c++11 \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=DEV %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux -std=c++11 \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=HOST %s

#include "Inputs/cuda.h"


// Check a static device variable referenced by host function is externalized.
// DEV-DAG: @x = dso_local addrspace(1) externally_initialized global i32 undef
// HOST-DAG: @x = internal global i32 123
// HOST-DAG: @x.managed = external global i32*
// HOST-DAG: @[[DEVNAMEX:[0-9]+]] = {{.*}}c"x\00"

__managed__ int x = 123;

__global__ void foo(int *z) {
  *z = x;
}

// HOST-LABEL: define {{.*}}@_Z4loadv()
// HOST:  %ld.managed = load i32*, i32** @x.managed, align 4
// HOST:  %0 = load i32, i32* %ld.managed, align 4
// HOST:  ret i32 %0
int load() {
  return x;
}

// HOST-LABEL: define {{.*}}@_Z5storev()
// HOST:  %ld.managed = load i32*, i32** @x.managed, align 4
// HOST:  store i32 456, i32* %ld.managed, align 4
void store() {
  x = 456;
}

// HOST-DAG: __hipRegisterManagedVar({{.*}}@x.managed {{.*}}@x {{.*}}@[[DEVNAMEX]]{{.*}}, i64 4, i32 4)
// HOST-DAG: declare void @__hipRegisterManagedVar(i8**, i8*, i8*, i8*, i64, i32)
