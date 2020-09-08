// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -ggdb -gz=zlib 2>&1 | FileCheck %s

// RUN: %clang -### -fgpu-rdc --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -ggdb -gz=zlib 2>&1 | FileCheck %s

// CHECK: {{".*clang.*" .* "--compress-debug-sections=zlib"}}
// CHECK-DAG: {{".*lld.*" .* "--compress-debug-sections=zlib"}}
// CHECK-DAG: {{".*clang.*" .* "--compress-debug-sections=zlib"}}
// CHECK: {{".*ld.*" .* "--compress-debug-sections=zlib"}}

