// RUN: %clang_cc1 -std=c++11 -fcuda-is-device -emit-llvm -o - %s \
// RUN:   |FileCheck %s

#include "Inputs/cuda.h"

struct A {
  int x;
};

constexpr int constexpr_var = 1;
constexpr A constexpr_struct{2};
constexpr A constexpr_array[4] = {0, 0, 0, 3};
constexpr char constexpr_str[] = "abcd";
const int const_var = 4;

// CHECK: @_ZL13constexpr_str.const = private unnamed_addr constant [5 x i8] c"abcd\00"
// CHECK: @_ZL13constexpr_var = internal constant i32 1
// CHECK: @_ZL16constexpr_struct = internal constant %struct.A { i32 2 }
// CHECK: @_ZL15constexpr_array = internal constant [4 x %struct.A] [%struct.A zeroinitializer, %struct.A zeroinitializer, %struct.A zeroinitializer, %struct.A { i32 3 }]
// CHECK-NOT: external

// CHECK: store i32 1
// CHECK: store i32 2
// CHECK: store i32 3
// CHECK: store i32 4
// CHECK: load i8, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @_ZL13constexpr_str.const, i64 0, i64 3)
// CHECK: store i32* @_ZL13constexpr_var
// CHECK: store i32* getelementptr inbounds (%struct.A, %struct.A* @_ZL16constexpr_struct, i32 0, i32 0)
// CHECK: store i32* getelementptr inbounds ([4 x %struct.A], [4 x %struct.A]* @_ZL15constexpr_array, i64 0, i64 3, i32 0)
__device__ void dev_fun(int *out, const int **out2) {
  *out = constexpr_var;
  *out = constexpr_struct.x;
  *out = constexpr_array[3].x;
  *out = const_var;
  *out = constexpr_str[3];
  *out2 = &constexpr_var;
  *out2 = &constexpr_struct.x;
  *out2 = &constexpr_array[3].x;
}
