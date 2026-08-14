// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx803 --offload-arch=gfx900 \
// RUN:   --cuda-device-only -nogpuinc -nogpulib -c \
// RUN:   %s 2>&1 | FileCheck %s

// CHECK-NOT: error:

// CHECK: [[CLANG:".*clang.*"]] "-cc1"{{.*}} "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: "-fcuda-is-device"
// CHECK-SAME: "-target-cpu" "gfx803"
// CHECK-SAME: {{.*}} "-o" "[[OBJ_DEV_A_803:[^"]+\.o]]" "-x" "hip"

// CHECK: [[CLANG:".*clang.*"]] "-cc1"{{.*}} "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: "-fcuda-is-device"
// CHECK-SAME: "-target-cpu" "gfx900"
// CHECK-SAME: {{.*}} "-o" "[[OBJ_DEV_A_900:[^"]+\.o]]" "-x" "hip"

// CHECK: [[PACKAGER:"[^"]*llvm-offload-binary[^"]*"]] "-o" "[[PACKAGE:[^"]+]]"
// CHECK-SAME: "--image=file=[[OBJ_DEV_A_803]],triple=amdgcn-amd-amdhsa,arch=gfx803,kind=hip"
// CHECK-SAME: "--image=file=[[OBJ_DEV_A_900]],triple=amdgcn-amd-amdhsa,arch=gfx900,kind=hip"

// CHECK: [[WRAPPER:"[^"]*clang-linker-wrapper[^"]*"]]
// CHECK-SAME: "--should-extract=gfx803" "--should-extract=gfx900"
// CHECK-SAME: "--host-triple=x86_64-unknown-linux-gnu"
// CHECK-SAME: "--emit-fatbin-only" "-o" "{{[^"]+}}" "[[PACKAGE]]"
