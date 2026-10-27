// NOTE: Do not autogenerate. Check runtime counts and application output.
// Host kernel registrations omit instrumented helpers and can also refer to
// kernels with excluded instrumentation. Linux collection must follow emitted
// sections. Shared RDC section bounds must be written once per collection,
// and a single exit collection must include both kernel launches.
// REQUIRES: hip, amdgpu
// UNSUPPORTED: windows
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang -x hip --offload-arch=%amdgpu_arch -O1 -fprofile-instr-generate -fcoverage-mapping %t/complete.hip -L%hip_lib_path -lamdhip64 -o %t/single
// RUN: env LLVM_PROFILE_FILE=%t/single.%%p.profraw LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH %run %t/single
// RUN: llvm-profdata merge %t/*single.*.profraw -o %t/single.profdata
// RUN: llvm-profdata show --function=_Z11live_helperi --counts %t/single.profdata | FileCheck %s
// RUN: %clang -x hip --offload-arch=%amdgpu_arch -O1 -fgpu-rdc -fprofile-instr-generate -fcoverage-mapping %t/main.hip %t/other.hip %t/empty.hip -L%hip_lib_path -lamdhip64 -o %t/rdc
// RUN: env LLVM_PROFILE_FILE=%t/rdc.%%p.profraw LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH %run %t/rdc
// RUN: llvm-profdata merge %t/*rdc.*.profraw -o %t/rdc.profdata
// RUN: llvm-profdata show --function=_Z11live_helperi --counts %t/rdc.profdata | FileCheck %s
// RUN: %clang -x hip --offload-arch=%amdgpu_arch -O1 -DEXCLUDE -fprofile-instr-generate -fcoverage-mapping %t/complete.hip -L%hip_lib_path -lamdhip64 -o %t/excluded
// RUN: env LLVM_PROFILE_FILE=%t/excluded.%%p.profraw LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH %run %t/excluded
// RUN: llvm-profdata merge %t/*excluded.*.profraw -o %t/excluded.profdata
// RUN: llvm-profdata show --function=_Z11live_helperi --counts %t/excluded.profdata | FileCheck %s
// Exercise the host-registration fallback when HSA introspection is unavailable.
// The fixture is opened by the collector as libhsa-runtime64.so. HIP keeps using
// its versioned libhsa-runtime64.so.1 dependency to execute the kernels.
// RUN: mkdir %t/fallback
// RUN: %clang -shared -fPIC %t/hsa-unavailable.c -o %t/fallback/libhsa-runtime64.so
// RUN: env LLVM_PROFILE_FILE=%t/fallback-single.%%p.profraw LLVM_PROFILE_VERBOSE=1 \
// RUN:   LD_LIBRARY_PATH=%t/fallback:%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   %run %t/single > %t/fallback-single.log 2>&1
// RUN: FileCheck %s --check-prefix=FALLBACK < %t/fallback-single.log
// RUN: llvm-profdata merge %t/*fallback-single.*.profraw -o %t/fallback-single.profdata
// RUN: llvm-profdata show --function=_Z11live_helperi --counts %t/fallback-single.profdata | FileCheck %s
// RUN: env LLVM_PROFILE_FILE=%t/fallback-rdc.%%p.profraw LLVM_PROFILE_VERBOSE=1 \
// RUN:   LD_LIBRARY_PATH=%t/fallback:%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   %run %t/rdc > %t/fallback-rdc.log 2>&1
// RUN: FileCheck %s --check-prefix=FALLBACK < %t/fallback-rdc.log
// RUN: llvm-profdata merge %t/*fallback-rdc.*.profraw -o %t/fallback-rdc.profdata
// RUN: llvm-profdata show --function=_Z11live_helperi --counts %t/fallback-rdc.profdata | FileCheck %s
// RUN: env LLVM_PROFILE_FILE=%t/fallback-excluded.%%p.profraw LLVM_PROFILE_VERBOSE=1 \
// RUN:   LD_LIBRARY_PATH=%t/fallback:%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   %run %t/excluded > %t/fallback-excluded.log 2>&1
// RUN: FileCheck %s --check-prefix=FALLBACK < %t/fallback-excluded.log
// RUN: llvm-profdata merge %t/*fallback-excluded.*.profraw -o %t/fallback-excluded.profdata
// RUN: llvm-profdata show --function=_Z11live_helperi --counts %t/fallback-excluded.profdata | FileCheck %s
// FALLBACK: required HSA symbols missing - HSA device profiling disabled
// CHECK: _Z11live_helperi:
// CHECK: Function count: 128
// CHECK: Block counts: [64]
// CHECK: Block uniformity: [D]
// CHECK: Functions shown: 1

//--- complete.hip
#include "main.hip"
#include "other.hip"

//--- main.hip
#include <hip/hip_runtime.h>
__device__ int live_helper(int);
__global__ void second(int *);
__global__ void kernel(int *out) {
  out[threadIdx.x] = live_helper(threadIdx.x);
}
#ifdef EXCLUDE
__global__
    __attribute__((no_profile_instrument_function)) void excluded(int *p) {
  p[threadIdx.x] = 0;
}
#endif
int main() {
  int *p;
  if (hipMalloc(&p, 64 * sizeof(int)) != hipSuccess)
    return 1;
  for (int rep = 0; rep < 2; ++rep) {
    kernel<<<1, 64>>>(p);
    int h[64];
    if (hipMemcpy(h, p, sizeof(h), hipMemcpyDeviceToHost) != hipSuccess)
      return 2;
    for (int i = 0; i < 64; ++i)
      if (h[i] != ((i & 1) ? 7 : 3))
        return 3;
    second<<<1, 64>>>(p);
    if (hipMemcpy(h, p, sizeof(h), hipMemcpyDeviceToHost) != hipSuccess)
      return 4;
    for (int i = 0; i < 64; ++i)
      if (h[i] != 11)
        return 5;
  }
#ifdef EXCLUDE
  excluded<<<1, 64>>>(p);
  if (hipDeviceSynchronize() != hipSuccess)
    return 7;
#endif
  return hipFree(p) != hipSuccess;
}

//--- other.hip
#include <hip/hip_runtime.h>
__device__ __attribute__((noinline)) int live_helper(int i) {
  if (i & 1)
    return 7;
  return 3;
}
__global__ void second(int *p) { p[threadIdx.x] = 11; }

//--- empty.hip
int host_only_function() { return 42; }

//--- hsa-unavailable.c
void unavailable_hsa_fixture(void) {}
