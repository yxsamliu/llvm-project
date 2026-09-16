//===- BlockUniformityProfile.cpp - Block uniformity from PGO -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/BlockUniformityProfile.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static bool hasIRBlockUniformityProfile(const BasicBlock &BB) {
  return BB.getTerminator()->getMetadata(
      LLVMContext::MD_block_uniformity_profile);
}

// Conservatively determine whether BB may execute under divergent control.
// Profiled-uniform annotations override this fallback. This first experiment
// intentionally does not model reconvergence: if any predecessor ancestry has
// a divergent terminator, the block is treated as potentially divergent.
static bool mayBeDivergentlyReached(const BasicBlock &BB,
                                    const UniformityInfo &UI) {
  SmallVector<const BasicBlock *, 8> Worklist;
  SmallPtrSet<const BasicBlock *, 16> Visited;
  Visited.insert(&BB);
  for (const BasicBlock *Pred : predecessors(&BB))
    Worklist.push_back(Pred);

  while (!Worklist.empty()) {
    const BasicBlock *Current = Worklist.pop_back_val();
    if (!Visited.insert(Current).second)
      continue;
    if (UI.isDivergentTerminator(Current->getTerminator()))
      return true;
    for (const BasicBlock *Pred : predecessors(Current))
      Worklist.push_back(Pred);
  }
  return false;
}

void BlockUniformityProfile::compute(const MachineFunction &MF,
                                     const UniformityInfo &UI) {
  HasProfile = MF.getFunction().getMetadata(
      LLVMContext::MD_uniformity_profile);
  NumBlockIDs = MF.getNumBlockIDs();
  DivergentBlocks.clear();
  DivergentBlocks.resize(NumBlockIDs);

  for (const MachineBasicBlock &MBB : MF) {
    const unsigned Num = MBB.getNumber();
    if (Num >= DivergentBlocks.size())
      continue;

    const BasicBlock *BB = MBB.getBasicBlock();
    if (!BB) {
      DivergentBlocks.set(Num);
      continue;
    }
    if (!hasIRBlockUniformityProfile(*BB) &&
        mayBeDivergentlyReached(*BB, UI))
      DivergentBlocks.set(Num);
  }
}

void BlockUniformityProfile::print(raw_ostream &OS,
                                   const MachineFunction &MF) const {
  OS << "BlockUniformityProfile for function: ";
  MF.getFunction().printAsOperand(OS, /*PrintType=*/false);
  OS << '\n';
  OS << "HasProfile: " << (HasProfile ? "true" : "false") << '\n';
  if (!HasProfile)
    return;

  for (const MachineBasicBlock &MBB : MF) {
    const BasicBlock *BB = MBB.getBasicBlock();
    if (!BB)
      continue;
    OS << "  " << printMBBReference(MBB);
    if (BB->hasName())
      OS << " (%" << BB->getName() << ")";
    if (hasIRBlockUniformityProfile(*BB)) {
      OS << ": uniform\n";
      continue;
    }
    if (isDivergent(MBB)) {
      OS << ": no PGO annotation (statically may be divergent)\n";
      continue;
    }
    OS << ": no PGO annotation (statically uniformly reached)\n";
  }
}

bool BlockUniformityProfile::isDivergent(const MachineBasicBlock &MBB) const {
  if (!HasProfile)
    return false;
  assert(MBB.getParent()->getNumBlockIDs() == NumBlockIDs &&
         "MachineFunction was modified without invalidating "
         "BlockUniformityProfile");
  const unsigned Num = MBB.getNumber();
  assert(Num < DivergentBlocks.size() && "Block number out of range");
  return DivergentBlocks.test(Num);
}

AnalysisKey BlockUniformityProfileProxy::Key;

BlockUniformityProfileProxy::Result
BlockUniformityProfileProxy::run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM) {
  FunctionAnalysisManager &FAM =
      MFAM.getResult<FunctionAnalysisManagerMachineFunctionProxy>(MF)
          .getManager();
  Function &F = MF.getFunction();
  const UniformityInfo &UI = FAM.getResult<UniformityInfoAnalysis>(F);
  BlockUniformityProfile Profile;
  Profile.compute(MF, UI);
  return Profile;
}

PreservedAnalyses
BlockUniformityProfilePrinterPass::run(MachineFunction &MF,
                                       MachineFunctionAnalysisManager &MFAM) {
  auto &Profile = MFAM.getResult<BlockUniformityProfileProxy>(MF);
  Profile.print(OS, MF);
  return PreservedAnalyses::all();
}
