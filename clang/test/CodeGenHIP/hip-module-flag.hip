// Verify clang emits a "hip" module flag for HIP compilations on both host
// and device sides, and does not emit it for non-HIP compilations.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -emit-llvm \
// RUN:   -o - -x hip %s | FileCheck --check-prefix=HIP %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
// RUN:   -o - -x hip %s | FileCheck --check-prefix=HIP %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
// RUN:   -o - -x c++ /dev/null | FileCheck --check-prefix=NOHIP %s

// HIP: !{{.*}} = !{i32 1, !"hip", i32 1}
// NOHIP-NOT: !"hip"
