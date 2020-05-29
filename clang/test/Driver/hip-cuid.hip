// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -x hip \
// RUN:   -target x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=gfx900 \
// RUN:   --offload-arch=gfx906 \
// RUN:   -c -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s

// CHECK: "{{.*}}clang{{.*}}" "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-hip-cuid=[[CUID:[^"]+]]"

// CHECK: "{{.*}}clang{{.*}}" "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-hip-cuid=[[CUID]]"

// CHECK: "{{.*}}clang{{.*}}" "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-hip-cuid=[[CUID]]"
