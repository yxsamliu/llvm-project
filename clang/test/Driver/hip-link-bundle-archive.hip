// REQUIRES: clang-driver, x86-registered-target, amdgpu-registered-target

// RUN: touch %T/libhipBundled.a

// Check in -fgpu-rdc mode clang unbundle the archive and link them by lld.

// RUN: %clang -### --offload-arch=gfx906 --offload-arch=gfx1030 \
// RUN:   -nogpulib %s -fgpu-rdc -L%T -lhipBundled \
// RUN:   2>&1 | FileCheck -check-prefix=RDC %s

// RDC: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-inputs={{.*}}libhipBundled.a" "-targets=hip-amdgcn-amd-amdhsa-gfx1030" "-outputs=[[A1030:.*\.a]]" "-allow-missing-bundles"
// RDC: "{{.*}}lld" {{.*}}"-plugin-opt=mcpu=gfx1030" {{.*}} "[[A1030]]"
// RDC: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-inputs={{.*}}libhipBundled.a" "-targets=hip-amdgcn-amd-amdhsa-gfx906" "-outputs=[[A906:.*\.a]]" "-allow-missing-bundles"
// RDC: "{{.*}}lld" {{.*}}"-plugin-opt=mcpu=gfx906" {{.*}} "[[A906]]"

// Check in default -fno-gpu-rdc mode clang unbundle the archive and link them by -mlink-builtin-bitcode.

// RUN: %clang -### --offload-arch=gfx906 --offload-arch=gfx1030 \
// RUN:   -nogpulib %s -L%T -lhipBundled \
// RUN:   2>&1 | FileCheck -check-prefix=NORDC %s

// NORDC: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-inputs={{.*}}libhipBundled.a" "-targets=hip-amdgcn-amd-amdhsa-gfx1030" "-outputs=[[A1030:.*\.a]]" "-allow-missing-bundles"
// NORDC: "{{.*}}clang{{.*}}" "-cc1" {{.*}}"-mlink-builtin-bitcode" "[[A1030]]" {{.*}}"-target-cpu" "gfx1030"
// NORDC: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-inputs={{.*}}libhipBundled.a" "-targets=hip-amdgcn-amd-amdhsa-gfx906" "-outputs=[[A906:.*\.a]]" "-allow-missing-bundles"
// NORDC: "{{.*}}clang{{.*}}" "-cc1" {{.*}}"-mlink-builtin-bitcode" "[[A906]]" {{.*}}"-target-cpu" "gfx906"
