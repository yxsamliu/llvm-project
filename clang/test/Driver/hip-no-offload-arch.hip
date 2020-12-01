// REQUIRES: clang-driver, x86-registered-target, amdgpu-registered-target, nvptx-registered-target

// RUN: %clang -### -target x86_64-linux-gnu -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=WARN %s
// WARN: warning: no offload arch is specified. Assuming gfx803. Use --offload-arch=<arch> to specify offload arch [-Whip-option-missing]

// RUN: %clang -### -target x86_64-linux-gnu -nogpuinc -nogpulib \
// RUN:   -x cuda %s 2>&1 | FileCheck -check-prefix=NOWARN %s
// NOWARN-NOT: warning
