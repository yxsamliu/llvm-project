// RUN: %clang_cc1 -emit-llvm %s -O3 -o - -fcuda-is-device \
// RUN:   -triple nvptx64-unknown-unknown | FileCheck %s

#include "Inputs/cuda.h"

// Make sure bool loaded from memory is truncated and
// range metadata is not emitted.

// CHECK:  %0 = load i8, i8* %x
// CHECK:  %1 = and i8 %0, 1
// CHECK:  store i8 %1, i8* %y
// CHECK-NOT: !range
__global__ void test1(bool *x, bool *y) {
  *y = *x != false;
}
