// RUN: %clang_cc1 -std=c++11 -triple amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device -emit-llvm -o - -x hip %s | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -triple amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device -emit-llvm -o - -x hip %s | FileCheck -check-prefix=NEG %s

#include "Inputs/cuda.h"

struct A {
  int x;
};

constexpr int constexpr_var = 1;
constexpr A constexpr_struct{2};
constexpr A constexpr_array[4] = {0, 0, 0, 3};
constexpr char constexpr_str[] = "abcd";
const int const_var = 4;
const A const_struct{5};
const A const_array[] = {0, 0, 0, 6};
const char const_str[] = "xyz";

// CHECK-DAG: @_ZL13constexpr_str.const = private unnamed_addr addrspace(4) constant [5 x i8] c"abcd\00"
// CHECK-DAG: @_ZL13constexpr_var = internal addrspace(4) constant i32 1
// CHECK-DAG: @_ZL16constexpr_struct = internal addrspace(4) constant %struct.A { i32 2 }
// CHECK-DAG: @_ZL15constexpr_array = internal addrspace(4) constant [4 x %struct.A] [%struct.A zeroinitializer, %struct.A zeroinitializer, %struct.A zeroinitializer, %struct.A { i32 3 }]
// CHECK-DAG: @_ZL9const_var = internal addrspace(4) constant i32 4
// CHECK-DAG: @_ZL12const_struct = internal addrspace(4) constant %struct.A { i32 5 }
// CHECK-DAG: @_ZL11const_array = internal addrspace(4) constant [4 x %struct.A] [%struct.A zeroinitializer, %struct.A zeroinitializer, %struct.A zeroinitializer, %struct.A { i32 6 }]
// CHECK-DAG: @_ZL9const_str = internal addrspace(4) constant [4 x i8] c"xyz\00"

// NEG-NOT: external

// CHECK-LABEL: define{{.*}}@_Z7dev_funPiPPKi
// CHECK: store i32 1
// CHECK: store i32 2
// CHECK: store i32 3
// CHECK: load i8, i8* getelementptr {{.*}} @_ZL13constexpr_str.const
// CHECK: store i32 4
// CHECK: store i32 5
// CHECK: store i32 6
// CHECK: load i8, i8* getelementptr {{.*}} @_ZL9const_str
// CHECK: store i32* {{.*}}@_ZL13constexpr_var
// CHECK: store i32* getelementptr {{.*}} @_ZL16constexpr_struct
// CHECK: store i32* getelementptr {{.*}} @_ZL15constexpr_array
// CHECK: store i32* {{.*}}@_ZL9const_var
// CHECK: store i32* getelementptr {{.*}} @_ZL12const_struct
// CHECK: store i32* getelementptr {{.*}} @_ZL11const_array
__device__ void dev_fun(int *out, const int **out2) {
  *out = constexpr_var;
  *out = constexpr_struct.x;
  *out = constexpr_array[3].x;
  *out = constexpr_str[3];
  *out = const_var;
  *out = const_struct.x;
  *out = const_array[3].x;
  *out = const_str[3];
  *out2 = &constexpr_var;
  *out2 = &constexpr_struct.x;
  *out2 = &constexpr_array[3].x;
  *out2 = &const_var;
  *out2 = &const_struct.x;
  *out2 = &const_array[3].x;
}
