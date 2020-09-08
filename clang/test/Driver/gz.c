// RUN: %clang -### -target x86_64-unknown-linux-gnu -g -gz=none %s 2>&1 | \
// RUN:    FileCheck -check-prefix=NONE %s
// RUN: %clang -### -target amdgcn-amd-amdhsa -nogpulib -g -gz=none %s 2>&1 | \
// RUN:    FileCheck -check-prefix=NONE %s

// NONE: {{".*clang.*".* "--compress-debug-sections=none"}}
// NONE: {{".*ld.*".* "--compress-debug-sections=none"}}

// RUN: %clang -### -target x86_64-unknown-linux-gnu -g -gz=zlib %s 2>&1 | \
// RUN:    FileCheck -check-prefix=ZLIB %s
// RUN: %clang -### -target amdgcn-amd-amdhsa -nogpulib -g -gz=zlib %s 2>&1 | \
// RUN:    FileCheck -check-prefix=ZLIB %s

// ZLIB: {{".*clang.*".* "--compress-debug-sections=zlib"}}
// ZLIB: {{".*ld.*".* "--compress-debug-sections=zlib"}}

// RUN: %clang -### -target x86_64-unknown-linux-gnu -g -gz=zlib-gnu %s 2>&1 | \
// RUN:    FileCheck -check-prefix=ZLIB-GNU %s
// RUN: %clang -### -target amdgcn-amd-amdhsa -nogpulib -g -gz=zlib-gnu %s 2>&1 | \
// RUN:    FileCheck -check-prefix=ZLIB-GNU %s

// ZLIB-GNU: {{".*clang.*".* "--compress-debug-sections=zlib-gnu"}}
// ZLIB-GNU: {{".*ld.*".* "--compress-debug-sections=zlib-gnu"}}
