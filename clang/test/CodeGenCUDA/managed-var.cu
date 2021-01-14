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
// HOST-DAG: @x = internal global i32 1
// HOST-DAG: @x.managed = external global i32*
// HOST-DAG: @[[DEVNAMEX:[0-9]+]] = {{.*}}c"x\00"

struct vec {
  float x,y,z;
};

__managed__ int x = 1;
__managed__ vec v[100];

__global__ void foo(int *z) {
  *z = x;
  v[1].x = 2;
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
// HOST:  store i32 2, i32* %ld.managed, align 4
void store() {
  x = 2;
}

// HOST-LABEL: define {{.*}}@_Z10addr_takenv()
// HOST:  %ld.managed = load i32*, i32** @x.managed, align 4
// HOST:  store i32* %ld.managed, i32** %p, align 8
// HOST:  %0 = load i32*, i32** %p, align 8
// HOST:  store i32 3, i32* %0, align 4
void addr_taken() {
  int *p = &x;
  *p = 3;
}

// HOST-LABEL: define {{.*}}@_Z5load2v()
// HOST: %ld.managed = load [100 x %struct.vec]*, [100 x %struct.vec]** @v.managed, align 16
// HOST:  %0 = getelementptr inbounds [100 x %struct.vec], [100 x %struct.vec]* %ld.managed, i64 0, i64 1, i32 0
// HOST:  %1 = load float, float* %0, align 4
// HOST:  ret float %1
float load2() {
  return v[1].x;
}

// HOST-DAG: __hipRegisterManagedVar({{.*}}@x.managed {{.*}}@x {{.*}}@[[DEVNAMEX]]{{.*}}, i64 4, i32 4)
// HOST-DAG: declare void @__hipRegisterManagedVar(i8**, i8*, i8*, i8*, i64, i32)
