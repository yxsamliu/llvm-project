// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:   -target-cpu gfx908 -target-feature +xnack \
// RUN:   -target-feature -sram-ecc \
// RUN:   -emit-llvm -o - %s | FileCheck -check-prefix=ID1 %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:   -target-cpu fiji \
// RUN:   -emit-llvm -o - %s | FileCheck -check-prefix=ID2 %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:   -emit-llvm -o - %s | FileCheck -check-prefix=NONE %s

// ID1: !{i32 8, !"target-id", !"amdgcn-amd-amdhsa-gfx908:sram-ecc-:xnack+"}
// ID2: !{i32 8, !"target-id", !"amdgcn-amd-amdhsa-gfx803"}
// NONE: !{i32 8, !"target-id", !""}

kernel void foo() {}
