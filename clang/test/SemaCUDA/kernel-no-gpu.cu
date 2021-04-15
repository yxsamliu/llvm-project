// RUN: %clang_cc1 -fcuda-is-device -verify=hip -x hip %s
// RUN: %clang_cc1 -fcuda-is-device -verify=cuda %s
// cuda-no-diagnostics

#include "Inputs/cuda.h"

__global__ void kern1() {}
// hip-error@-1 {{compile HIP kernel without specifying offload arch is not allowed}}

// Make sure the error is emitted once.
__global__ void kern2() {}
