// RUN: %clang_cc1 -fcuda-is-device -verify=hip -x hip %s
// RUN: %clang_cc1 -fcuda-is-device -verify=cuda %s
// cuda-no-diagnostics

#include "Inputs/cuda.h"

__global__ void kern1() {}
// hip-error@-1 {{compiling a HIP kernel without specifying an offload arch is not allowed}}

// Make sure the error is emitted once.
__global__ void kern2() {}
