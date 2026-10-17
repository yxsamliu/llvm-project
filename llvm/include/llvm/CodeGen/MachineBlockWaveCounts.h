//===- MachineBlockWaveCounts.h - Mapped wave visits ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEBLOCKWAVECOUNTS_H
#define LLVM_CODEGEN_MACHINEBLOCKWAVECOUNTS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>

namespace llvm {

class MachineBasicBlock;
class MachineFunction;

/// A snapshot of direct wave visits with unambiguous machine CFG mappings.
/// Recompute after changing either CFG. Raw counts include measured zero and
/// saturated values; callers choose their policy for saturated measurements.
struct MachineBlockWaveCounts {
  DenseMap<const MachineBasicBlock *, uint64_t> Counts;
  uint64_t EntryCount = 0;
  unsigned RejectedBlocks = 0;
};

/// Clear Result and extract the current valid IR-to-machine associations.
/// Return false for an unavailable IR profile. EntryCount is the original
/// invocation count, not necessarily the count of the current entry block.
/// Missing counts remain absent; topology mismatches increment RejectedBlocks.
LLVM_ABI bool extractMappedBlockWaveCounts(const MachineFunction &MF,
                                           MachineBlockWaveCounts &Result);

/// Express a wave count in the supplied entry-frequency units. Both entry
/// values must be nonzero. A positive floor satisfies spill cost invariants;
/// it does not change the underlying measured count.
LLVM_ABI BlockFrequency scaleWaveCount(uint64_t Count, uint64_t EntryCount,
                                       BlockFrequency EntryFrequency);

} // namespace llvm

#endif
