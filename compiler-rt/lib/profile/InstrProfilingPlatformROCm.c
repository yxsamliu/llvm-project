//===- InstrProfilingPlatformROCm.cpp - Profile data ROCm platform ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InstrProfiling.h"
#include "InstrProfilingInternal.h"
#include <stdio.h>  /* For printf */
#include <stdlib.h> /* For malloc and free */
#include <string.h> /* For memcpy */

/* Forward declare the HIP API functions. */
int hipMemcpyFromSymbol(void *dst, const void *symbol, size_t sizeBytes,
                        size_t offset, int kind);
int hipGetSymbolAddress(void **devPtr, const void *symbol);
int hipMemcpy(void *dest, void *src, size_t len, int kind /*2=DToH*/);

/* Declare the shadow variables for device profile data. */
/* These are defined in the host-side IR by the instrumentation pass. */
extern char __start___llvm_prf_cnts_offload;
extern char __stop___llvm_prf_cnts_offload;
extern char __start___llvm_prf_data_offload;
extern char __stop___llvm_prf_data_offload;
extern char __start___llvm_prf_names_offload;
extern char __stop___llvm_prf_names_offload;
extern char xxx;
extern char __llvm_offload_prf;

/* Collects the device-side profile data and writes it to a file. */
int __llvm_profile_hip_collect_device_data(void) {
  printf("DEBUG: __llvm_profile_hip_collect_device_data called\n");
  void *dev_llvm_offload_prf = NULL;

  if (hipGetSymbolAddress(&dev_llvm_offload_prf, &__llvm_offload_prf) != 0) {
    printf("DEBUG: Failed to get __llvm_offload_prf\n");
    return -1;
  }

  void *host_offload_prf[6];
  if (hipMemcpy(host_offload_prf, dev_llvm_offload_prf,
                sizeof(host_offload_prf), 2 /*DToH*/) != 0) {
    printf("DEBUG: Failed to copy __llvm_offload_prf structure from device\n");
    return -1;
  }

  void *dev_cnts_begin = host_offload_prf[0];
  void *dev_data_begin = host_offload_prf[1];
  void *dev_names_begin = host_offload_prf[2];
  void *dev_cnts_end = host_offload_prf[3];
  void *dev_data_end = host_offload_prf[4];
  void *dev_names_end = host_offload_prf[5];

  size_t CountersSize = (char *)dev_cnts_end - (char *)dev_cnts_begin;
  size_t DataSize = (char *)dev_data_end - (char *)dev_data_begin;
  size_t NamesSize = (char *)dev_names_end - (char *)dev_names_begin;

  if (CountersSize == 0 || DataSize == 0) {
    printf("DEBUG: Counters or Data section has zero size. No profile data to "
           "collect.\n");
    return 0;
  }

  // Allocate one contiguous buffer for all sections, mimicking the layout
  // expected by the profile writer. Layout: [Counters | Data | Names]
  size_t TotalSize = CountersSize + DataSize + NamesSize;
  char *Buffer = (char *)malloc(TotalSize);
  if (!Buffer) {
    printf("DEBUG: Failed to allocate host memory for contiguous buffer\n");
    return -1;
  }

  // Set up pointers to the regions within the contiguous buffer.
  char *HostCountersBegin = Buffer;
  char *HostDataBegin = HostCountersBegin + CountersSize;
  char *HostNamesBegin = HostDataBegin + DataSize;

  // Copy the raw data from the device into the correct regions of our buffer.
  if (hipMemcpy(HostCountersBegin, dev_cnts_begin, CountersSize, 2 /*DToH*/) !=
      0) {
    printf("DEBUG: Failed to copy counters from device\n");
    free(Buffer);
    return -1;
  }
  if (hipMemcpy(HostDataBegin, dev_data_begin, DataSize, 2 /*DToH*/) != 0) {
    printf("DEBUG: Failed to copy data from device\n");
    free(Buffer);
    return -1;
  }
  if (NamesSize > 0 &&
      hipMemcpy(HostNamesBegin, dev_names_begin, NamesSize, 2 /*DToH*/) != 0) {
    printf("DEBUG: Failed to copy names from device\n");
    free(Buffer);
    return -1;
  }

  // The data is now in the contiguous buffer. We need to relocate the pointers.
  __llvm_profile_data *RelocatedData = (__llvm_profile_data *)HostDataBegin;
  uint64_t NumData = DataSize / sizeof(__llvm_profile_data);

  for (uint64_t i = 0; i < NumData; ++i) {
    if (RelocatedData[i].CounterPtr) {
      // This is the relative offset stored on the device.
      ptrdiff_t DeviceCounterPtrOffset = (ptrdiff_t)RelocatedData[i].CounterPtr;
      // This is the absolute address of the data struct on the device.
      void *DeviceDataStructAddr =
          (char *)dev_data_begin + (i * sizeof(__llvm_profile_data));
      // This is the absolute address of the counters for this function on the
      // device.
      void *DeviceCountersAddr =
          (char *)DeviceDataStructAddr + DeviceCounterPtrOffset;
      // This is the offset of the function's counters from the start of the
      // global device counter section.
      ptrdiff_t OffsetIntoCountersSection =
          (char *)DeviceCountersAddr - (char *)dev_cnts_begin;

      // The writer expects an absolute pointer *within the contiguous buffer*.
      void *AbsoluteHostCounterPtr =
          (void *)(HostCountersBegin + OffsetIntoCountersSection);

      *((IntPtrT *)&RelocatedData[i].CounterPtr) =
          (IntPtrT)AbsoluteHostCounterPtr;
    }

    // Null out pointers that are not used or not collected.
    *((IntPtrT *)&RelocatedData[i].BitmapPtr) = (IntPtrT)NULL;
    *((IntPtrT *)&RelocatedData[i].FunctionPointer) = (IntPtrT)NULL;
    *((IntPtrT *)&RelocatedData[i].Values) = (IntPtrT)NULL;
  }

  // Get the target triple from the environment (a simplification).
  const char *TargetTriple = getenv("LLVM_TARGET_TRIPLE");
  if (!TargetTriple) {
    TargetTriple = "amdgcn-amd-amdhsa";
  }

  // The version is usually read from the device, but we'll use the default for
  // now.
  uint64_t Version = __llvm_profile_get_version();

  // Invoke the writer with the pointers to our contiguous buffer.
  int result = __llvm_write_custom_profile(
      TargetTriple, (const __llvm_profile_data *)HostDataBegin,
      (const __llvm_profile_data *)(HostDataBegin + DataSize),
      (const char *)HostCountersBegin,
      (const char *)(HostCountersBegin + CountersSize),
      (const char *)HostNamesBegin, (const char *)(HostNamesBegin + NamesSize),
      &Version);

  printf("DEBUG: __llvm_write_custom_profile returned %d\n", result);

  free(Buffer);
  return result;
}
