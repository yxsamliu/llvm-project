// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -x hip \
// RUN:   -target x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=gfx900 \
// RUN:   --offload-arch=gfx906 \
// RUN:   -c -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck -check-prefixes=COMMON,RANDOM %s

// RUN: %clang -### -x hip \
// RUN:   -target x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=gfx900 \
// RUN:   --offload-arch=gfx906 \
// RUN:   -c -nogpulib -cuid=abcd \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck -check-prefixes=COMMON,FIXED %s

// COMMON: "{{.*}}clang{{.*}}" "-cc1" "-triple" "amdgcn-amd-amdhsa"
// RANDOM-SAME: "-cuid=[[CUID:[0-9]+]]"
// FIXED-SAME: "-cuid=[[CUID:abcd]]"

// COMMON: "{{.*}}clang{{.*}}" "-cc1" "-triple" "amdgcn-amd-amdhsa"
// COMMON-SAME: "-cuid=[[CUID]]"

// COMMON: "{{.*}}clang{{.*}}" "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// COMMON-SAME: "-cuid=[[CUID]]"
