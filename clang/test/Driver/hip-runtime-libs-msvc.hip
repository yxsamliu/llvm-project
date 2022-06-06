// REQUIRES: system-windows

// RUN:  touch %t.o

// Get compiler-rt library full path.
// RUN: %clang -target x86_64-pc-windows-msvc -rtlib=compiler-rt \
// RUN:   -print-libgcc-file-name >%t.1

// Test HIP runtime lib args specified by --rocm-path.
// RUN: %clang -### --hip-link -target x86_64-pc-windows-msvc \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o >%t.2 2>&1
// RUN: cat %t.1 %t.2 | FileCheck %s

// CHECK: [[COMPILER_RT:.*clang_rt\.builtins.*]]
// CHECK: "-libpath:{{.*Inputs.*rocm.*lib}}" "amdhip64.lib"
// CHECK-SAME: "[[COMPILER_RT]]"
