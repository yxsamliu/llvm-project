// REQUIRES: x86-registered-target, amdgpu-registered-target
// UNSUPPORTED: system-windows

// Build a fake resource dir containing both the base profile runtime and the
// ROCm device-profile runtime so the driver's existence check passes.
// RUN: rm -rf %t && mkdir -p %t/lib/x86_64-unknown-linux
// RUN: touch %t/lib/x86_64-unknown-linux/libclang_rt.profile.a
// RUN: touch %t/lib/x86_64-unknown-linux/libclang_rt.profile_rocm.a
// RUN: mkdir -p %t/lib/x86_64-pc-windows-msvc
// RUN: touch %t/lib/x86_64-pc-windows-msvc/clang_rt.profile_rocm.lib
// RUN: mkdir -p %t/sycl/bin %t/sycl/lib/x86_64-unknown-linux
// RUN: touch %t/sycl/lib/x86_64-unknown-linux/libLLVMSYCL.so
// RUN: ln -s %clang %t/sycl/bin/clang
// RUN: touch %t.o

// HIP host link with PGO links clang_rt.profile_rocm.
// RUN: %clang -### --hip-link --target=x86_64-unknown-linux \
// RUN:   -fprofile-instr-generate -resource-dir=%t \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=HIP-PGO %s
// profile_rocm must precede the base profile so the base archive stays inert.
// HIP-PGO: "{{.*}}libclang_rt.profile_rocm.a"
// The constructor-only hipModuleLoad* interceptor object is force-linked.
// HIP-PGO: "-u" "__llvm_profile_offload_register_dynamic_module"
// HIP-PGO: "{{.*}}libclang_rt.profile.a"

// An explicit request on a HIP PGO link does not add the ROCm runtime twice.
// RUN: %clang -### --hip-link --target=x86_64-unknown-linux \
// RUN:   -fprofile-instr-generate --hip-link-profile-runtime \
// RUN:   -resource-dir=%t --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=HIP-PGO-EXPLICIT %s
// HIP-PGO-EXPLICIT-NOT: "-u__llvm_profile_runtime"
// HIP-PGO-EXPLICIT: "{{.*}}libclang_rt.profile_rocm.a"
// HIP-PGO-EXPLICIT-NOT: libclang_rt.profile_rocm.a
// HIP-PGO-EXPLICIT: "{{.*}}libamdhip64.so"
// HIP-PGO-EXPLICIT: "-u__llvm_profile_runtime"
// HIP-PGO-EXPLICIT-NOT: "-u__llvm_profile_runtime"
// HIP-PGO-EXPLICIT: "{{.*}}libclang_rt.profile.a"
// HIP-PGO-EXPLICIT-NOT: libclang_rt.profile_rocm.a
// HIP-PGO-EXPLICIT-NOT: "-u__llvm_profile_runtime"

// Without profiling, the ROCm device-profile runtime is not linked.
// RUN: %clang -### --hip-link --target=x86_64-unknown-linux \
// RUN:   -resource-dir=%t --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=HIP-NOPGO %s
// HIP-NOPGO-NOT: libclang_rt.profile_rocm.a

// A non-HIP host link with PGO does not link the ROCm device-profile runtime.
// RUN: %clang -### --target=x86_64-unknown-linux \
// RUN:   -fprofile-instr-generate -resource-dir=%t %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=HOST-PGO %s
// HOST-PGO: "{{.*}}libclang_rt.profile.a"
// HOST-PGO-NOT: libclang_rt.profile_rocm.a

// A plain C++ host link can explicitly request the ROCm profile runtime.
// RUN: %clang -### --target=x86_64-unknown-linux -Wl,--as-needed \
// RUN:   --hip-link-profile-runtime -resource-dir=%t \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=HOST-HIP-PGO %s
// HOST-HIP-PGO-NOT: clang-linker-wrapper
// HOST-HIP-PGO: "-u__llvm_profile_runtime"
// HOST-HIP-PGO: "{{.*}}libclang_rt.profile_rocm.a"
// HOST-HIP-PGO: "-u" "__llvm_profile_offload_register_dynamic_module"
// HOST-HIP-PGO: "{{.*}}libamdhip64.so"
// HOST-HIP-PGO-NOT: libclang_rt.profile.a

// A HIP source still links the HIP runtime only once.
// RUN: %clang -### --target=x86_64-unknown-linux -nogpuinc \
// RUN:   --offload-arch=gfx900 --hip-link-profile-runtime -resource-dir=%t \
// RUN:   --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefix=HIP-SOURCE %s
// HIP-SOURCE: "{{.*}}libamdhip64.so"
// HIP-SOURCE-NOT: libamdhip64.so

// The explicit option preserves the HIP runtime rpath behavior.
// RUN: %clang -### --target=x86_64-unknown-linux \
// RUN:   --hip-link-profile-runtime -frtlib-add-rpath -resource-dir=%t \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=HOST-HIP-PGO-RPATH %s
// HOST-HIP-PGO-RPATH: "-rpath" "{{.*}}Inputs/rocm/lib"

// The explicit option does not replace another active offload runtime.
// RUN: %t/sycl/bin/clang -### -no-canonical-prefixes \
// RUN:   --target=x86_64-unknown-linux -fsycl --hip-link-profile-runtime \
// RUN:   -resource-dir=%t --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=SYCL-HIP-PGO %s
// SYCL-HIP-PGO: libclang_rt.profile_rocm.a
// SYCL-HIP-PGO: libamdhip64.so
// SYCL-HIP-PGO: libLLVMSYCL.so

// --no-offloadlib suppresses both runtimes requested by the explicit option.
// RUN: %clang -### --target=x86_64-unknown-linux --no-offloadlib \
// RUN:   --hip-link-profile-runtime -resource-dir=%t \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=NO-OFFLOADLIB %s
// NO-OFFLOADLIB-NOT: libamdhip64.so
// NO-OFFLOADLIB-NOT: libclang_rt.profile_rocm.a

// Options that suppress the requested runtimes suppress both of them.
// RUN: %clang -### --target=x86_64-unknown-linux -no-hip-rt \
// RUN:   --hip-link-profile-runtime -resource-dir=%t \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=SUPPRESS-RUNTIMES %s
// RUN: %clang -### --target=x86_64-unknown-linux -nostdlib \
// RUN:   --hip-link-profile-runtime -resource-dir=%t \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=SUPPRESS-RUNTIMES %s
// RUN: %clang -### --target=x86_64-unknown-linux -r \
// RUN:   --hip-link-profile-runtime -resource-dir=%t \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=SUPPRESS-RUNTIMES %s
// RUN: %clang -### --target=x86_64-unknown-linux -noprofilelib \
// RUN:   --hip-link-profile-runtime -resource-dir=%t \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=SUPPRESS-RUNTIMES %s
// SUPPRESS-RUNTIMES-NOT: libamdhip64.so
// SUPPRESS-RUNTIMES-NOT: libclang_rt.profile_rocm.a

// RUN: %clang -### --target=x86_64-pc-windows-msvc \
// RUN:   --hip-link-profile-runtime -resource-dir=%t \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=WINDOWS-HOST-HIP-PGO %s
// WINDOWS-HOST-HIP-PGO-NOT: clang-linker-wrapper
// WINDOWS-HOST-HIP-PGO: "-include:__llvm_profile_runtime"
// WINDOWS-HOST-HIP-PGO: "{{.*}}clang_rt.profile_rocm.lib"
// WINDOWS-HOST-HIP-PGO: "-include:__llvm_profile_offload_register_dynamic_module"
// WINDOWS-HOST-HIP-PGO: "amdhip64.lib"
// WINDOWS-HOST-HIP-PGO-NOT: clang_rt.profile.lib

// An explicit request diagnoses missing required runtimes.
// RUN: not %clang -### --target=x86_64-unknown-linux \
// RUN:   --hip-link-profile-runtime -resource-dir=%t/missing-profile \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=MISSING-PROFILE %s
// MISSING-PROFILE: error: cannot find ROCm profile runtime required by '--hip-link-profile-runtime': '{{.*}}libclang_rt.profile_rocm.a'

// RUN: not %clang -### --target=x86_64-unknown-linux \
// RUN:   --hip-link-profile-runtime -resource-dir=%t \
// RUN:   --rocm-path=%t/missing-rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=MISSING-HIP %s
// MISSING-HIP: error: cannot find HIP runtime required by '--hip-link-profile-runtime'; provide its path via '--rocm-path'

// RUN: not %clang -### --target=x86_64-pc-windows-msvc \
// RUN:   --hip-link-profile-runtime -resource-dir=%t/missing-profile \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=WINDOWS-MISSING-PROFILE %s
// WINDOWS-MISSING-PROFILE: error: cannot find ROCm profile runtime required by '--hip-link-profile-runtime': '{{.*}}clang_rt.profile_rocm.lib'

// RUN: not %clang -### --target=x86_64-pc-windows-msvc \
// RUN:   --hip-link-profile-runtime -resource-dir=%t \
// RUN:   --rocm-path=%t/missing-rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=WINDOWS-MISSING-HIP %s
// WINDOWS-MISSING-HIP: error: cannot find HIP runtime required by '--hip-link-profile-runtime'; provide its path via '--rocm-path'

// The explicit option is rejected by unsupported host toolchains.
// RUN: not %clang -### --target=x86_64-apple-darwin \
// RUN:   --hip-link-profile-runtime %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=UNSUPPORTED-TARGET %s
// UNSUPPORTED-TARGET: error: unsupported option '--hip-link-profile-runtime' for target 'x86_64-apple-darwin'
