//===- MachineBlockWaveCounts.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineBlockWaveCounts.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/Support/ScaledNumber.h"
#include <algorithm>
#include <cassert>

using namespace llvm;

static bool hasMatchingWaveProfileBlock(
    const MachineBasicBlock &MBB,
    const DenseMap<const BasicBlock *, const MachineBasicBlock *> &UniqueMBBs) {
  const BasicBlock *BB = MBB.getBasicBlock();
  auto HasMatchingEdges = [&](auto IRBlocks, auto MachineBlocks) {
    SmallPtrSet<const BasicBlock *, 4> Expected;
    for (const BasicBlock *IRBlock : IRBlocks)
      Expected.insert(IRBlock);
    for (const MachineBasicBlock *MachineBlock : MachineBlocks) {
      const BasicBlock *IRBlock = MachineBlock->getBasicBlock();
      if (!IRBlock || UniqueMBBs.lookup(IRBlock) != MachineBlock ||
          !Expected.erase(IRBlock)) {
        return false;
      }
    }
    return Expected.empty();
  };
  return HasMatchingEdges(predecessors(BB), MBB.predecessors()) &&
         HasMatchingEdges(successors(BB), MBB.successors());
}

bool llvm::extractMappedBlockWaveCounts(const MachineFunction &MF,
                                        MachineBlockWaveCounts &Result) {
  Result = {};
  SmallVector<uint64_t> Counts;
  BitVector HasCount;
  if (!extractMappedBlockWaveCounts(MF.getFunction(), Counts, HasCount,
                                    Result.EntryCount)) {
    return false;
  }

  DenseMap<const BasicBlock *, const MachineBasicBlock *> UniqueMBBs;
  for (const MachineBasicBlock &MBB : MF) {
    const BasicBlock *BB = MBB.getBasicBlock();
    if (!BB || BB->getParent() != &MF.getFunction())
      continue;
    auto [It, Inserted] = UniqueMBBs.try_emplace(BB, &MBB);
    if (!Inserted)
      It->second = nullptr;
  }

  unsigned Index = 0;
  for (const BasicBlock &BB : MF.getFunction()) {
    unsigned I = Index++;
    if (!HasCount[I])
      continue;
    const MachineBasicBlock *MBB = UniqueMBBs.lookup(&BB);
    if (!MBB || !hasMatchingWaveProfileBlock(*MBB, UniqueMBBs)) {
      ++Result.RejectedBlocks;
      continue;
    }
    Result.Counts[MBB] = Counts[I];
  }
  return true;
}

BlockFrequency llvm::scaleWaveCount(uint64_t Count, uint64_t EntryCount,
                                    BlockFrequency EntryFrequency) {
  assert(EntryCount && EntryFrequency.getFrequency() &&
         "wave normalization requires nonzero entry values");
  uint64_t Frequency = ScaledNumber<uint64_t>::getFraction(Count, EntryCount)
                           .scale(EntryFrequency.getFrequency());
  return BlockFrequency(std::max(Frequency, uint64_t(1)));
}
