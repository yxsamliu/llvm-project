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

  printf("DEBUG: Getting symbol address for __llvm_offload_prf...\n");
  if (hipGetSymbolAddress(&dev_llvm_offload_prf, &__llvm_offload_prf) != 0) {
    printf("DEBUG: Failed to get __llvm_offload_prf\n");
    return -1;
  }
  printf("DEBUG: Device __llvm_offload_prf address: %p\n",
         dev_llvm_offload_prf);

  // Copy the unified structure from device to host to get start/stop addresses
  // Structure layout: [start_cnts, start_data, start_names, stop_cnts,
  // stop_data, stop_names]
  void *host_offload_prf[6];
  if (hipMemcpy(host_offload_prf, dev_llvm_offload_prf,
                sizeof(host_offload_prf), 2 /*DToH*/) != 0) {
    printf("DEBUG: Failed to copy __llvm_offload_prf structure from device\n");
    return -1;
  }

  // Extract start/stop addresses directly from the copied structure
  // These are direct addresses, not pointers to addresses
  void *dev_cnts_begin = host_offload_prf[0];
  void *dev_data_begin = host_offload_prf[1];
  void *dev_names_begin = host_offload_prf[2];
  void *dev_cnts_end = host_offload_prf[3];
  void *dev_data_end = host_offload_prf[4];
  void *dev_names_end = host_offload_prf[5];

  printf("DEBUG: Device symbol addresses from unified structure:\n");
  printf("DEBUG:   cnts: %p - %p\n", dev_cnts_begin, dev_cnts_end);
  printf("DEBUG:   data: %p - %p\n", dev_data_begin, dev_data_end);
  printf("DEBUG:   names: %p - %p\n", dev_names_begin, dev_names_end);

  printf("DEBUG: Expected addresses based on binary analysis:\n");
  printf("DEBUG:   cnts: 0x4f98 - 0x4fe0 (should match above)\n");
  printf("DEBUG:   data: 0x4fe0 - 0x50a0 (should match above)\n");
  printf("DEBUG:   names: 0x17c0 - 0x17f8 (should match above)\n");

  // Calculate sizes
  size_t CountersSize = (char *)dev_cnts_end - (char *)dev_cnts_begin;
  size_t DataSize = (char *)dev_data_end - (char *)dev_data_begin;
  size_t NamesSize = (char *)dev_names_end - (char *)dev_names_begin;

  printf("DEBUG: Section sizes:\n");
  printf("DEBUG:   Counters: %zu bytes\n", CountersSize);
  printf("DEBUG:   Data: %zu bytes\n", DataSize);
  printf("DEBUG:   Names: %zu bytes\n", NamesSize);

  if (CountersSize == 0 || DataSize == 0 || NamesSize == 0) {
    printf("DEBUG: One or more sections have zero size\n");
    return -1;
  }

  // Allocate host memory for the profile data
  char *Counters = (char *)malloc(CountersSize);
  __llvm_profile_data *OriginalData = (__llvm_profile_data *)malloc(DataSize);
  char *Names = (char *)malloc(NamesSize);

  // Allocate a new array for relocated profile data
  __llvm_profile_data *RelocatedData = (__llvm_profile_data *)malloc(DataSize);

  if (!Counters || !OriginalData || !Names || !RelocatedData) {
    printf("DEBUG: Failed to allocate host memory\n");
    free(Counters);
    free(OriginalData);
    free(Names);
    free(RelocatedData);
    return -1;
  }

  printf("DEBUG: Allocated host memory:\n");
  printf("DEBUG:   Counters: %p (%zu bytes)\n", Counters, CountersSize);
  printf("DEBUG:   Data: %p (%zu bytes)\n", OriginalData, DataSize);
  printf("DEBUG:   Names: %p (%zu bytes)\n", Names, NamesSize);

  // Copy profile data from device to host using hipMemcpy with device addresses
  printf(
      "DEBUG: Copying profile data from device to host using hipMemcpy...\n");
  if (hipMemcpy(Counters, dev_cnts_begin, CountersSize, 2 /*DToH*/) != 0) {
    printf("DEBUG: Failed to copy counters from device\n");
    free(Counters);
    free(OriginalData);
    free(Names);
    free(RelocatedData);
    return -1;
  }

  if (hipMemcpy(OriginalData, dev_data_begin, DataSize, 2 /*DToH*/) != 0) {
    printf("DEBUG: Failed to copy data from device\n");
    free(Counters);
    free(OriginalData);
    free(Names);
    free(RelocatedData);
    return -1;
  }

  if (hipMemcpy(Names, dev_names_begin, NamesSize, 2 /*DToH*/) != 0) {
    printf("DEBUG: Failed to copy names from device\n");
    free(Counters);
    free(OriginalData);
    free(Names);
    free(RelocatedData);
    return -1;
  }

  printf("DEBUG: Successfully copied all profile data from device\n");

  // Calculate the number of data entries
  uint64_t NumData = DataSize / sizeof(__llvm_profile_data);

  printf("DEBUG: Data entry size: %zu, Number of entries: %lu\n",
         sizeof(__llvm_profile_data), NumData);

  if (NumData == 0) {
    printf("DEBUG: No profile data entries found\n");
    free(Counters);
    free(OriginalData);
    free(Names);
    free(RelocatedData);
    return -1;
  }

  printf("DEBUG: Performing pointer relocations...\n");

  // Relocate pointers within the copied __llvm_profile_data structures.
  for (uint64_t i = 0; i < NumData; ++i) {
    // Copy the original data to the new buffer
    memcpy(&RelocatedData[i], &OriginalData[i], sizeof(__llvm_profile_data));

    // The CounterPtr from the device is a relative offset from the __llvm_profd
    // variable. We need to reconstruct the absolute address on the host.
    if (OriginalData[i].CounterPtr) {
      // This is the relative offset stored on the device.
      ptrdiff_t DeviceCounterPtrOffset = (ptrdiff_t)OriginalData[i].CounterPtr;

      // This is the absolute address of the data struct on the device.
      void *DeviceDataStructAddr =
          (char *)dev_data_begin + (i * sizeof(__llvm_profile_data));

      // This is the absolute address of the counters for this function on the
      // device.
      void *DeviceCountersAddr =
          (char *)DeviceDataStructAddr + DeviceCounterPtrOffset;

      // This is the offset of the function's counters from the start of the
      // global counter section.
      ptrdiff_t OffsetIntoCountersSection =
          (char *)DeviceCountersAddr - (char *)dev_cnts_begin;

      if (OffsetIntoCountersSection < 0 ||
          (size_t)OffsetIntoCountersSection >= CountersSize) {
        printf("DEBUG: FATAL: Invalid counter offset %td for func %lu\n",
               OffsetIntoCountersSection, i);
      }

      // The profraw format expects the CounterPtr to be the offset relative to
      // the start of the counters section.
      *((IntPtrT *)&RelocatedData[i].CounterPtr) =
          (IntPtrT)(OffsetIntoCountersSection);

      printf("DEBUG: Relocated CounterPtr[%lu]: final_offset=%td, "
             "stored_value=%p\n",
             i, OffsetIntoCountersSection, (void *)RelocatedData[i].CounterPtr);
    }

    // The bitmap section is not currently collected from the device, so we
    // cannot relocate this pointer. It must be nulled out to prevent the
    // profile writer from looking for bitmap data that doesn't exist.
    *((IntPtrT *)&RelocatedData[i].BitmapPtr) = (IntPtrT)NULL;

    // NULL out other pointers that are not being relocated to avoid confusion.
    // Their raw device values are invalid on the host.
    *((IntPtrT *)&RelocatedData[i].FunctionPointer) = (IntPtrT)NULL;
    *((IntPtrT *)&RelocatedData[i].Values) = (IntPtrT)NULL;

    // Note: NameRef is a hash, not a pointer, so no relocation needed
  }

  printf("DEBUG: Pointer relocations completed\n");

  // Write the profile data to a file
  printf("DEBUG: Writing profile data to file...\n");
  int result = __llvm_write_custom_profile(
      "amdgcn", RelocatedData, RelocatedData + NumData, Counters,
      Counters + CountersSize, Names, Names + NamesSize, NULL);
  printf("DEBUG: __llvm_write_custom_profile returned %d\n", result);

  // Free the host memory.
  free(Counters);
  free(OriginalData);
  free(Names);
  free(RelocatedData);

  return 0;
}
