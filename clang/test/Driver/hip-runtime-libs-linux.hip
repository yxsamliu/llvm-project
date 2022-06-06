// UNSUPPORTED: system-windows

// RUN:  touch %t.o

// Get compiler-rt library full path.
// RUN: %clang -target x86_64-linux-gnu -rtlib=compiler-rt \
// RUN:   -print-libgcc-file-name >%t.1

// Test HIP runtime lib args specified by --rocm-path.
// RUN: %clang -### --hip-link -target x86_64-linux-gnu \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o >%t.2 2>&1
// RUN: cat %t.1 %t.2 | FileCheck -check-prefixes=ROCM-PATH,COMMON %s

// Test HIP runtime lib args specified by environment variable ROCM_PATH.
// RUN: env ROCM_PATH=%S/Inputs/rocm %clang -### --hip-link \
// RUN:   -target x86_64-linux-gnu %t.o >%t.2 2>&1
// RUN: cat %t.1 %t.2 | FileCheck -check-prefixes=ROCM-PATH,COMMON %s

// Test detecting latest /opt/rocm-{release} directory.
// RUN: rm -rf %T/opt
// RUN: mkdir -p %T/opt
// RUN: cp -r %S/Inputs/rocm %T/opt/rocm-3.9.0-1234
// RUN: cp -r %S/Inputs/rocm %T/opt/rocm-3.10.0
// RUN: %clang -### --hip-link -target x86_64-linux-gnu \
// RUN:   --sysroot=%T %t.o >%t.2 2>&1
// RUN: cat %t.1 %t.2 | FileCheck -check-prefixes=ROCM-REL,COMMON %s

// Test HIP runtime lib is not linked without --hip-link.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOHIPRT %s

// Test HIP runtime lib is not linked with -nostdlib.
// RUN: %clang -### --hip-link -nostdlib -target x86_64-linux-gnu \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOHIPRT %s

// Test HIP runtime lib is not linked with -no-hip-rt.
// RUN: %clang -### --hip-link -no-hip-rt -target x86_64-linux-gnu \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOHIPRT %s

// COMMON: [[COMPILER_RT:/.*/libclang_rt\.builtins.*]]
// ROCM-PATH: "-L[[HIPRT:.*/Inputs/rocm/lib]]" "-rpath" "[[HIPRT]]" "-lamdhip64"
// ROCM-REL: "-L[[HIPRT:.*/opt/rocm-3.10.0/lib]]" "-rpath" "[[HIPRT]]" "-lamdhip64"
// COMMON-SAME: "[[COMPILER_RT]]"

// NOHIPRT-NOT: "-L{{.*/Inputs/rocm/lib}}"
// NOHIPRT-NOT: "-rpath" "{{.*/Inputs/rocm/lib}}"
// NOHIPRT-NOT: "-lamdhip64"
// NOHIPRT-NOT: "{{/[^"]*clang_rt\.builtins[^"]*}}"
