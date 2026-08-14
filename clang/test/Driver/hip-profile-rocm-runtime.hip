// REQUIRES: x86-registered-target, amdgpu-registered-target
// UNSUPPORTED: system-windows

// Build a fake resource dir containing both the base profile runtime and the
// ROCm device-profile runtime so the driver's existence check passes.
// RUN: rm -rf %t && mkdir -p %t/lib/x86_64-unknown-linux %t/lib/amdgcn-amd-amdhsa
// RUN: touch %t/lib/x86_64-unknown-linux/libclang_rt.profile.a
// RUN: touch %t/lib/x86_64-unknown-linux/libclang_rt.profile_rocm.a
// RUN: touch %t/lib/amdgcn-amd-amdhsa/libclang_rt.profile.a
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

// HIP device-only code object builds use the linker wrapper so its inner Clang
// invocation links the AMDGPU profile runtime.
// RUN: %clang -### -x hip --cuda-device-only \
// RUN:   --target=x86_64-unknown-linux --offload-arch=gfx906 \
// RUN:   -fprofile-generate -resource-dir=%t -nogpuinc -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefix=HIP-DEVICE-PGO %s
// HIP-DEVICE-PGO-DAG: "-emit-llvm-bc"
// HIP-DEVICE-PGO-DAG: "-flto=full"
// HIP-DEVICE-PGO: "{{.*}}llvm-offload-binary"
// HIP-DEVICE-PGO: "{{.*}}clang-linker-wrapper"
// HIP-DEVICE-PGO-SAME: "--device-compiler=amdgcn-amd-amdhsa=-fprofile-generate"
// HIP-DEVICE-PGO-SAME: "--emit-fatbin-only"

// Unbundled code object output uses the linker wrapper's fat binary mode, then
// unbundles that fat binary into raw images.
// RUN: %clang -### -x hip --cuda-device-only --no-gpu-bundle-output \
// RUN:   --target=x86_64-unknown-linux --offload-arch=gfx906 \
// RUN:   -fprofile-generate -resource-dir=%t -nogpuinc -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefix=HIP-DEVICE-PGO-RAW %s
// HIP-DEVICE-PGO-RAW: "{{.*}}llvm-offload-binary"
// HIP-DEVICE-PGO-RAW: "{{.*}}clang-linker-wrapper"
// HIP-DEVICE-PGO-RAW-SAME: "--device-compiler=amdgcn-amd-amdhsa=-fprofile-generate"
// HIP-DEVICE-PGO-RAW-SAME: "--emit-fatbin-only"
// HIP-DEVICE-PGO-RAW: "{{.*}}clang-offload-bundler"
// HIP-DEVICE-PGO-RAW-SAME: "-type=hipfb"
// HIP-DEVICE-PGO-RAW-SAME: "-output=a.out-hip-amdgcn-amd-amdhsa-gfx906"
// HIP-DEVICE-PGO-RAW-SAME: "-unbundle"

// Other profile generation modes are also forwarded to the linker wrapper.
// RUN: %clang -### -x hip --cuda-device-only \
// RUN:   --target=x86_64-unknown-linux --offload-arch=gfx906 \
// RUN:   -fcs-profile-generate -resource-dir=%t -nogpuinc -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefix=HIP-DEVICE-CSPGO %s
// HIP-DEVICE-CSPGO: "{{.*}}clang-linker-wrapper"
// HIP-DEVICE-CSPGO-SAME: "--device-compiler=amdgcn-amd-amdhsa=-fcs-profile-generate"

// RUN: %clang -### -x hip --cuda-device-only \
// RUN:   --target=x86_64-unknown-linux --offload-arch=gfx906 \
// RUN:   -fcreate-profile -resource-dir=%t -nogpuinc -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefix=HIP-DEVICE-CREATE-PROFILE %s
// HIP-DEVICE-CREATE-PROFILE: "{{.*}}clang-linker-wrapper"
// HIP-DEVICE-CREATE-PROFILE-SAME: "--device-compiler=amdgcn-amd-amdhsa=-fcreate-profile"

// RUN: %clang -### -x hip --cuda-device-only \
// RUN:   --target=x86_64-unknown-linux --offload-arch=gfx906 \
// RUN:   -fprofile-generate-cold-function-coverage -resource-dir=%t \
// RUN:   -nogpuinc -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefix=HIP-DEVICE-COLD-PROFILE %s
// HIP-DEVICE-COLD-PROFILE: "{{.*}}clang-linker-wrapper"
// HIP-DEVICE-COLD-PROFILE-SAME: "--device-compiler=amdgcn-amd-amdhsa=-fprofile-generate-cold-function-coverage"

// -noprofilelib must reach the inner Clang together with the profile option.
// RUN: rm %t/lib/amdgcn-amd-amdhsa/libclang_rt.profile.a
// RUN: %clang -### -x hip --cuda-device-only \
// RUN:   --target=x86_64-unknown-linux --offload-arch=gfx906 \
// RUN:   -fprofile-generate -noprofilelib -resource-dir=%t \
// RUN:   -nogpuinc -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefix=HIP-DEVICE-NOPROFILELIB %s
// HIP-DEVICE-NOPROFILELIB: "{{.*}}clang-linker-wrapper"
// HIP-DEVICE-NOPROFILELIB-SAME: "--device-compiler=amdgcn-amd-amdhsa=-fprofile-generate"
// HIP-DEVICE-NOPROFILELIB-SAME: "--device-compiler=amdgcn-amd-amdhsa=-noprofilelib"
