// An independently compiled loader must preserve each module's profile kind
// and layout, including when only the supplemental HSA collector finds it.
// REQUIRES: hip, amdgpu
// UNSUPPORTED: windows
//
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %clang -x hip -O2 -Xarch_device -nogpulib -Xarch_device -nogpuinc \
// RUN:   --offload-arch=%amdgpu_arch \
// RUN:   --rocm-path=%hip_lib_path/.. -fprofile-generate \
// RUN:   -mllvm -pgo-instrument-entry -mllvm -pgo-instrument-loop-entries \
// RUN:   -DBUILD_MODULE %s -L%hip_lib_path -lamdhip64 -o %t.dir/module
// RUN: cd %t.dir && llvm-objdump --offloading module
// RUN: mv %t.dir/module.0.hip-amdgcn-amd-amdhsa--*gfx* %t.dir/module.co
// RUN: %clang -x c++ -O2 -c %s -D__HIP_PLATFORM_AMD__ \
// RUN:   -fprofile-instr-generate -I%hip_lib_path/../include -o %t.dir/loader.o
// RUN: %clang %t.dir/loader.o --hip-link --rocm-path=%hip_lib_path/.. \
// RUN:   -fprofile-instr-generate -L%hip_lib_path -lamdhip64 -o %t.dir/loader
// RUN: mkdir %t.dir/intercepted %t.dir/hsa
// RUN: env LLVM_PROFILE_FILE=%t.dir/intercepted/host.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH HIP_VISIBLE_DEVICES=0 \
// RUN:   %run %t.dir/loader %t.dir/module.co
// RUN: env LLVM_PROFILE_FILE=%t.dir/hsa/host.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH HIP_VISIBLE_DEVICES=0 \
// RUN:   %run %t.dir/loader %t.dir/module.co bypass
// RUN: llvm-profdata show %t.dir/intercepted/gfx*.profraw | FileCheck %s
// RUN: llvm-profdata show %t.dir/hsa/gfx*.profraw | FileCheck %s
// RUN: llvm-profdata show %t.dir/intercepted/host.profraw | FileCheck %s --check-prefix=HOST
// RUN: llvm-profdata show %t.dir/hsa/host.profraw | FileCheck %s --check-prefix=HOST
// RUN: llvm-profdata merge %t.dir/intercepted/gfx*.profraw %t.dir/hsa/gfx*.profraw -o %t.dir/device.profdata
// RUN: %clang -x hip -O2 -Xarch_device -nogpulib -Xarch_device -nogpuinc \
// RUN:   --offload-arch=%amdgpu_arch --offload-device-only \
// RUN:   --rocm-path=%hip_lib_path/.. -DBUILD_MODULE \
// RUN:   -fprofile-use=%t.dir/device.profdata \
// RUN:   -Werror=profile-instr-out-of-date -Werror=profile-instr-unprofiled \
// RUN:   -S -emit-llvm %s -o - | FileCheck %s --check-prefix=USE
//
// CHECK: Instrumentation level: IR  entry_first = 1  instrument_loop_entries = 1
// HOST: Instrumentation level: Front-end
// USE-LABEL: define{{.*}} @layout_kernel(
// USE-SAME: !wave.profile ![[WAVES:[0-9]+]]

#ifdef __HIP_DEVICE_COMPILE__
#  define __global__ __attribute__((global))
#else
#  include <hip/hip_runtime.h>
#endif

#ifdef BUILD_MODULE

extern "C" __global__ void layout_kernel(int *Out, int N) {
  unsigned I = 0;
#  ifdef __HIP_DEVICE_COMPILE__
  I = __builtin_amdgcn_workitem_id_x();
#  endif
  Out[I] = (I & 1) ? N : 2 * N;
}
int main() { return 0; }

#else

#  include <dlfcn.h>

int main(int Argc, char **Argv) {
  if (Argc != 2 && Argc != 3)
    return 1;
  bool Bypass = Argc == 3;
  auto Load = hipModuleLoad;
  if (Bypass) {
    void *Lib = dlopen("libamdhip64.so", RTLD_NOW);
    if (!Lib)
      return 2;
    Load = reinterpret_cast<decltype(Load)>(dlsym(Lib, "hipModuleLoad"));
    if (!Load)
      return 3;
  }
  hipModule_t Module;
  if (Load(&Module, Argv[1]) != hipSuccess)
    return 4;
  hipFunction_t Kernel;
  if (hipModuleGetFunction(&Kernel, Module, "layout_kernel") != hipSuccess)
    return 5;
  int *Dev, Host[64], N = 10;
  if (hipMalloc(&Dev, sizeof(Host)) != hipSuccess)
    return 6;
  void *Args[] = {&Dev, &N};
  if (hipModuleLaunchKernel(Kernel, 1, 1, 1, 64, 1, 1, 0, nullptr, Args,
                            nullptr) != hipSuccess ||
      hipDeviceSynchronize() != hipSuccess ||
      hipMemcpy(Host, Dev, sizeof(Host), hipMemcpyDeviceToHost) != hipSuccess)
    return 7;
  for (int I = 0; I < 64; ++I)
    if (Host[I] != ((I & 1) ? 10 : 20))
      return 8;
  if (hipFree(Dev) != hipSuccess)
    return 9;
  // Keep the bypassed module loaded for the exit-time HSA walk.
  return !Bypass && hipModuleUnload(Module) != hipSuccess;
}
#endif
