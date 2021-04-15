// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -c %s 2>&1 | FileCheck %s

// CHECK-NOT: {{.*}}clang{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu"
// CHECK: {{.*}}clang{{.*}}"-triple" "amdgcn-amd-amdhsa"
