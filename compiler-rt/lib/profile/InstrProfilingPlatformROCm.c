//===- InstrProfilingPlatformROCm.cpp - Profile data ROCm platform ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InstrProfiling.h"
#include "InstrProfilingInternal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward declare the HIP API functions. */
int hipMemcpyFromSymbol(void *dst, const void *symbol, size_t sizeBytes,
                        size_t offset, int kind);
int hipGetSymbolAddress(void **devPtr, const void *symbol);
int hipMemcpy(void *dest, void *src, size_t len, int kind /*2=DToH*/);

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

  // Allocate host memory for the device sections
  char *HostCountersBegin = (char *)malloc(CountersSize);
  char *HostDataBegin = (char *)malloc(DataSize);
  char *HostNamesBegin = (char *)malloc(NamesSize);

  if (!HostCountersBegin || !HostDataBegin ||
      (NamesSize > 0 && !HostNamesBegin)) {
    printf("DEBUG: Failed to allocate host memory for device sections\n");
    free(HostCountersBegin);
    free(HostDataBegin);
    free(HostNamesBegin);
    return -1;
  }

  // Copy data from device to host
  if (hipMemcpy(HostCountersBegin, dev_cnts_begin, CountersSize, 2) != 0 ||
      hipMemcpy(HostDataBegin, dev_data_begin, DataSize, 2) != 0 ||
      (NamesSize > 0 &&
       hipMemcpy(HostNamesBegin, dev_names_begin, NamesSize, 2) != 0)) {
    printf("DEBUG: Failed to copy profile sections from device\n");
    free(HostCountersBegin);
    free(HostDataBegin);
    free(HostNamesBegin);
    return -1;
  }

  // Construct the device-specific filename
  const char *BaseFilename = __llvm_profile_get_filename();
  if (!BaseFilename) {
    printf("DEBUG: Failed to get base profile filename\n");
    free(HostCountersBegin);
    free(HostDataBegin);
    free(HostNamesBegin);
    return -1;
  }

  const char *TargetInfix = "amdgcn-amd-amdhsa";
  char *DeviceFilename = NULL;
  const char *Extension = strrchr(BaseFilename, '.');

  if (Extension) {
    size_t BaseLen = Extension - BaseFilename;
    size_t InfixLen = strlen(TargetInfix);
    size_t ExtLen = strlen(Extension);
    DeviceFilename = (char *)malloc(BaseLen + 1 + InfixLen + ExtLen + 1);
    strncpy(DeviceFilename, BaseFilename, BaseLen);
    DeviceFilename[BaseLen] = '\0';
    strcat(DeviceFilename, ".");
    strcat(DeviceFilename, TargetInfix);
    strcat(DeviceFilename, Extension);
  } else {
    DeviceFilename =
        (char *)malloc(strlen(BaseFilename) + 1 + strlen(TargetInfix) + 1);
    strcpy(DeviceFilename, BaseFilename);
    strcat(DeviceFilename, ".");
    strcat(DeviceFilename, TargetInfix);
  }
  free((void *)BaseFilename);

  // Manually write the profile data with a proper header
  FILE *File = fopen(DeviceFilename, "w");
  if (!File) {
    printf("DEBUG: Failed to open %s for writing\n", DeviceFilename);
    free(DeviceFilename);
    free(HostCountersBegin);
    free(HostDataBegin);
    free(HostNamesBegin);
    return -1;
  }

  __llvm_profile_header Header;
  const uint64_t NumData = DataSize / sizeof(__llvm_profile_data);
  const uint64_t NumCounters = CountersSize / sizeof(uint64_t);
  const uint64_t NumBitmapBytes = 0;
  const uint64_t VTableSectionSize = 0;
  const uint64_t VNamesSize = 0;
  uint64_t PaddingBytesBeforeCounters, PaddingBytesAfterCounters,
      PaddingBytesAfterBitmapBytes, PaddingBytesAfterNames,
      PaddingBytesAfterVTable, PaddingBytesAfterVNames;

  if (__llvm_profile_get_padding_sizes_for_counters(
          DataSize, CountersSize, NumBitmapBytes, NamesSize, VTableSectionSize,
          VNamesSize, &PaddingBytesBeforeCounters, &PaddingBytesAfterCounters,
          &PaddingBytesAfterBitmapBytes, &PaddingBytesAfterNames,
          &PaddingBytesAfterVTable, &PaddingBytesAfterVNames) != 0) {
    printf("DEBUG: Failed to get padding sizes\n");
    fclose(File);
    free(DeviceFilename);
    free(HostCountersBegin);
    free(HostDataBegin);
    free(HostNamesBegin);
    return -1;
  }

  // Relocate pointers
  __llvm_profile_data *RelocatedData = (__llvm_profile_data *)HostDataBegin;
  for (uint64_t i = 0; i < NumData; ++i) {
    if (RelocatedData[i].CounterPtr) {
      ptrdiff_t DeviceCounterPtrOffset = (ptrdiff_t)RelocatedData[i].CounterPtr;
      void *DeviceDataStructAddr =
          (char *)dev_data_begin + (i * sizeof(__llvm_profile_data));
      void *DeviceCountersAddr =
          (char *)DeviceDataStructAddr + DeviceCounterPtrOffset;
      ptrdiff_t OffsetIntoCountersSection =
          (char *)DeviceCountersAddr - (char *)dev_cnts_begin;

      ptrdiff_t NewRelativeOffset = DataSize + PaddingBytesBeforeCounters +
                                    OffsetIntoCountersSection -
                                    (i * sizeof(__llvm_profile_data));
      *((uint64_t *)&RelocatedData[i].CounterPtr) = NewRelativeOffset;
    }
    *((uint64_t *)&RelocatedData[i].BitmapPtr) = 0;
    *((uint64_t *)&RelocatedData[i].FunctionPointer) = 0;
    *((uint64_t *)&RelocatedData[i].Values) = 0;
  }

  // Populate header
  Header.Magic = __llvm_profile_get_magic();
  Header.Version = __llvm_profile_get_version();
  Header.BinaryIdsSize = 0; // Not supported for device PGO yet
  Header.NumData = NumData;
  Header.PaddingBytesBeforeCounters = PaddingBytesBeforeCounters;
  Header.NumCounters = NumCounters;
  Header.PaddingBytesAfterCounters = PaddingBytesAfterCounters;
  Header.NumBitmapBytes = NumBitmapBytes;
  Header.PaddingBytesAfterBitmapBytes = PaddingBytesAfterBitmapBytes;
  Header.NamesSize = NamesSize;
  Header.CountersDelta = DataSize + PaddingBytesBeforeCounters;
  Header.BitmapDelta =
      Header.CountersDelta + CountersSize + PaddingBytesAfterCounters;
  Header.NamesDelta =
      Header.BitmapDelta + NumBitmapBytes + PaddingBytesAfterBitmapBytes;
  Header.NumVTables = 0;
  Header.VNamesSize = 0;
  Header.ValueKindLast = 0; // No value profiling

  // Write header and data
  if (fwrite(&Header, sizeof(__llvm_profile_header), 1, File) != 1)
    goto write_error_close;
  if (fwrite(HostDataBegin, 1, DataSize, File) != DataSize)
    goto write_error_close;
  if (PaddingBytesBeforeCounters > 0 &&
      fseek(File, PaddingBytesBeforeCounters, SEEK_CUR) != 0)
    goto write_error_close;
  if (fwrite(HostCountersBegin, 1, CountersSize, File) != CountersSize)
    goto write_error_close;
  if (PaddingBytesAfterCounters > 0 &&
      fseek(File, PaddingBytesAfterCounters, SEEK_CUR) != 0)
    goto write_error_close;
  if (fwrite(HostNamesBegin, 1, NamesSize, File) != NamesSize)
    goto write_error_close;

  fclose(File);
  free(DeviceFilename);
  free(HostCountersBegin);
  free(HostDataBegin);
  free(HostNamesBegin);
  printf("DEBUG: Successfully wrote profile data to %s\n", DeviceFilename);
  return 0;

write_error_close:
  printf("DEBUG: Failed to write to %s\n", DeviceFilename);
  fclose(File);
  free(DeviceFilename);
  free(HostCountersBegin);
  free(HostDataBegin);
  free(HostNamesBegin);
  return -1;
}
