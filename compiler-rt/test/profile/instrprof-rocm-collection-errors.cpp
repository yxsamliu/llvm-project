// REQUIRES: profile-rocm
// RUN: split-file %s %t
// RUN: %clang -shared -fPIC %t/hip.c -o %t/libamdhip64.so
// RUN: %clang -shared -fPIC %t/hsa.c -o %t/libhsa-runtime64.so
// RUN: %clangxx -std=c++20 -I%S/../../lib/profile %t/main.cpp -x none \
// RUN:   %profile_rocm -L%t -lamdhip64 -Wl,-rpath,%t -Wl,--wrap=realloc \
// RUN:   -pthread -o %t/test
// RUN: mkdir %t/normal %t/oom %t/copy
// RUN: env LD_LIBRARY_PATH=%t %run %t/test normal %t/normal/out.profraw | \
// RUN:   FileCheck %s --check-prefix=NORMAL
// RUN: llvm-profdata merge %t/normal/*.profraw -o %t/normal.profdata
// RUN: llvm-profdata show --all-functions --counts %t/normal.profdata | \
// RUN:   FileCheck %s --check-prefix=COUNT
// RUN: env LD_LIBRARY_PATH=%t %run %t/test oom %t/oom/out.profraw | \
// RUN:   FileCheck %s --check-prefix=OOM
// RUN: llvm-profdata merge %t/oom/*.profraw -o %t/oom.profdata
// RUN: llvm-profdata show --all-functions --counts %t/oom.profdata | \
// RUN:   FileCheck %s --check-prefix=COUNT
// RUN: env LD_LIBRARY_PATH=%t %run %t/test copy %t/copy/out.profraw | \
// RUN:   FileCheck %s --check-prefix=COPY
// RUN: not ls %t/copy/amdgpu.*.profraw

// The mock exposes the same sections through every shadow lookup. Exercise the
// real static collection loops, including two aliases and two used devices.
// A failure must stop collection and restore the original device. If recording
// the first output fails, no later alias may write a duplicate profile. Merge
// every output to catch duplicate snapshots.

// NORMAL: first=0 lookups=4 allocation_failures=0 device=1
// OOM: first=-1 lookups=1 allocation_failures=1 device=1
// COPY: first=-1 lookups=1 allocation_failures=0 device=1
// COUNT: foo:
// COUNT: Function count: 64

//--- main.cpp
extern "C" {
#include "InstrProfiling.h"
void *__real_realloc(void *, size_t);
void __llvm_profile_offload_register_shadow_variable(void *);
void __llvm_profile_offload_register_section_shadow_variable(void *);
int __llvm_profile_hip_collect_device_data();
int hipGetDevice(int *);
int hipSetDevice(int);
struct Dim3 {
  unsigned x, y, z;
};
int hipLaunchKernel(const void *, Dim3, Dim3, void **, size_t, void *);
extern const void *FailCopy;
extern unsigned Lookups;
}
#include "InstrProfilingPlatformROCmInternal.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool FailGrowth;
static unsigned AllocationFailures;
extern "C" void *__wrap_realloc(void *Ptr, size_t Size) {
  if (FailGrowth && Size == 64 * sizeof(__prof_rocm::ProfBoundsSet::Tuple)) {
    FailGrowth = false;
    ++AllocationFailures;
    return nullptr;
  }
  return __real_realloc(Ptr, Size);
}

int main(int Argc, char **Argv) {
  assert(Argc == 3);
  __llvm_profile_set_filename(Argv[2]);
  // One valid raw profile record, with an uncompressed name and a relative
  // counter pointer. The name reference is the profile hash of "foo".
  static struct {
    __llvm_profile_data Data;
    uint64_t Counter;
    char Names[5];
  } Sections = {{.NameRef = 6699318081062747564ULL,
                 .FuncHash = 0,
                 .CounterPtr = (void *)sizeof(__llvm_profile_data),
                 .NumCounters = 1},
                64,
                {3, 0, 'f', 'o', 'o'}};
  static uint64_t Version = INSTR_PROF_RAW_VERSION;
  static __llvm_profile_gpu_sections Table = {};
  Table.DataStart = &Sections.Data;
  Table.DataStop = &Sections.Data + 1;
  Table.CountersStart = (char *)&Sections.Counter;
  Table.CountersStop = (char *)(&Sections.Counter + 1);
  Table.NamesStart = Sections.Names;
  Table.NamesStop = Sections.Names + sizeof(Sections.Names);
  Table.VersionVar = &Version;
  Table.UniformCountersStart = Table.CountersStart;
  Table.UniformCountersStop = Table.CountersStart;
  for (int I = 0; I != 2; ++I) {
    __llvm_profile_offload_register_shadow_variable(&Table);
    __llvm_profile_offload_register_section_shadow_variable(&Sections.Data);
    __llvm_profile_offload_register_section_shadow_variable(&Sections.Counter);
    __llvm_profile_offload_register_section_shadow_variable(&Sections.Counter);
    __llvm_profile_offload_register_section_shadow_variable(Sections.Names);
    // Let the real launch interceptor mark both mock devices as used.
    assert(hipSetDevice(I) == 0);
    assert(hipLaunchKernel(nullptr, {1, 1, 1}, {1, 1, 1}, nullptr, 0,
                           nullptr) == 0);
  }

  FailGrowth = strcmp(Argv[1], "oom") == 0;
  if (strcmp(Argv[1], "copy") == 0)
    FailCopy = &Sections.Counter;
  int First = __llvm_profile_hip_collect_device_data();
  int Device = -1;
  assert(hipGetDevice(&Device) == 0);
  printf("first=%d lookups=%u allocation_failures=%u device=%d\n", First,
         Lookups, AllocationFailures, Device);
  return 0;
}

//--- hip.c
#include <stddef.h>
#include <string.h>

const void *FailCopy;
unsigned Lookups;
static int Device;

int hipMemcpy(void *Dst, const void *Src, size_t Size, int Kind) {
  if (Src == FailCopy) {
    FailCopy = NULL;
    return 1;
  }
  memcpy(Dst, Src, Size);
  return 0;
}
int hipGetSymbolAddress(void **Dst, const void *Src) {
  ++Lookups;
  *Dst = (void *)Src;
  return 0;
}
int hipGetDeviceCount(int *Count) {
  *Count = 2;
  return 0;
}
int hipGetDevice(int *Result) {
  *Result = Device;
  return 0;
}
int hipSetDevice(int Value) {
  Device = Value;
  return 0;
}
int hipGetDevicePropertiesR0600(void *Properties, int Value) { return -1; }
struct Dim3 {
  unsigned x, y, z;
};
int hipLaunchKernel(const void *Function, struct Dim3 Grid, struct Dim3 Block,
                    void **Args, size_t Shared, void *Stream) {
  return 0;
}

//--- hsa.c
// Force the static fallback even on hosts with HSA installed.
void mock_hsa(void) {}
