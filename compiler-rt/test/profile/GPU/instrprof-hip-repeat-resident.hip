// NOTE: Do not autogenerate. Check cumulative records and resident-image coverage.
// Repeated collection refreshes earlier images and discovers a later-loaded
// image. Cumulative snapshots replace their files, while an unused image stays
// unloaded and contributes no profile.
// REQUIRES: hip, amdgpu
// UNSUPPORTED: windows
// RUN: split-file %s %t.dir
// RUN: %clang -x hip --offload-arch=%amdgpu_arch -O1 -fno-gpu-rdc -fprofile-instr-generate -c %t.dir/hot.hip -o %t.dir/hot.o
// RUN: %clang -x hip --offload-arch=%amdgpu_arch -O1 -fno-gpu-rdc -fprofile-instr-generate -c %t.dir/late.hip -o %t.dir/late.o
// RUN: %clang -x hip --offload-arch=%amdgpu_arch -O1 -fno-gpu-rdc -fprofile-instr-generate -c %t.dir/cold.hip -o %t.dir/cold.o
// RUN: %clang --hip-link --offload-arch=%amdgpu_arch -fprofile-instr-generate %t.dir/hot.o %t.dir/late.o %t.dir/cold.o -o %t.dir/bench -L%hip_lib_path -lamdhip64
// RUN: env LLVM_PROFILE_FILE=%t.dir/profile.%%p.profraw LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH %t.dir/bench | FileCheck %s --check-prefix=OUTPUT
// RUN: /bin/sh -c 'test "$(find %t.dir -name "gfx*.profraw" | wc -l)" -eq 2'
// RUN: llvm-profdata merge %t.dir/gfx*.profraw -o %t.dir/device.profdata
// RUN: llvm-profdata show --all-functions --counts %t.dir/device.profdata | FileCheck %s --check-prefix=ALL --implicit-check-not=cold_kernel
// RUN: llvm-profdata show --function=hot_kernel --counts %t.dir/device.profdata | FileCheck %s --check-prefix=HOT
// RUN: llvm-profdata show --function=late_kernel --counts %t.dir/device.profdata | FileCheck %s --check-prefix=LATE
// OUTPUT: PASS
// ALL: Functions shown: 2
// HOT: hot_kernel:
// HOT: Function count: 128
// LATE: late_kernel:
// LATE: Function count: 64

//--- hot.hip
#include <cstdio>
#include <hip/hip_runtime.h>
extern "C" int __llvm_profile_hip_collect_device_data();
extern "C" void launch_late(int *);
extern "C" void cold_anchor();
extern "C" __global__ void hot_kernel(int *p) { ++p[threadIdx.x]; }
static bool check(int *p, int expected) {
  int h[64];
  if (hipMemcpy(h, p, sizeof(h), hipMemcpyDeviceToHost) != hipSuccess)
    return false;
  for (int x : h)
    if (x != expected)
      return false;
  return true;
}
int main() {
  cold_anchor();
  int *p;
  if (hipMalloc(&p, 64 * sizeof(int)) != hipSuccess ||
      hipMemset(p, 0, 64 * sizeof(int)) != hipSuccess)
    return 1;
  hot_kernel<<<1, 64>>>(p);
  if (!check(p, 1) || __llvm_profile_hip_collect_device_data())
    return 2;
  launch_late(p);
  hot_kernel<<<1, 64>>>(p);
  if (!check(p, 3) || __llvm_profile_hip_collect_device_data())
    return 3;
  if (hipFree(p) != hipSuccess)
    return 4;
  std::puts("PASS");
}

//--- late.hip
#include <hip/hip_runtime.h>
extern "C" __global__ void late_kernel(int *p) { ++p[threadIdx.x]; }
extern "C" void launch_late(int *p) { late_kernel<<<1, 64>>>(p); }

//--- cold.hip
#include <hip/hip_runtime.h>
extern "C" __global__ void cold_kernel(int *p) { ++p[threadIdx.x]; }
extern "C" void cold_anchor() {}
