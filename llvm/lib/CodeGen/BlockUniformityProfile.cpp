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
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace llvm;

static cl::opt<bool> DisableStaticUniformityFallback(
    "disable-static-uniformity-fallback", cl::Hidden, cl::init(false),
    cl::desc("Treat blocks without uniformity metadata as unclassified"));

static cl::opt<bool> SpillUseBranchUnanimityPrototype(
    "spill-use-branch-unanimity-prototype", cl::Hidden, cl::init(false),
    cl::desc("Use unanimous direct branch votes in the GPU spill fallback"));

static bool hasIRBlockUniformityProfile(const BasicBlock &BB) {
  return BB.getTerminator()->getMetadata(
      LLVMContext::MD_block_uniformity_profile);
}

static bool hasUnanimousBranchVotes(const Instruction &Term) {
  const auto *Branch = dyn_cast<CondBrInst>(&Term);
  if (!Branch)
    return false;
  const MDNode *MD = Branch->getMetadata("branch.unanimity.prototype");
  if (!MD || MD->getNumOperands() != 2)
    return false;

  const auto *TotalMD = mdconst::dyn_extract<ConstantInt>(MD->getOperand(0));
  const auto *UnanimousMD =
      mdconst::dyn_extract<ConstantInt>(MD->getOperand(1));
  if (!TotalMD || !UnanimousMD || !TotalMD->getType()->isIntegerTy(64) ||
      !UnanimousMD->getType()->isIntegerTy(64))
    return false;

  uint64_t Total = TotalMD->getZExtValue();
  return Total >= 100 && UnanimousMD->getZExtValue() == Total;
}

// Conservatively determine whether BB may execute under divergent control.
// Profiled-uniform annotations override this fallback. The opt-in direct-vote
// hint can suppress a divergent branch for the spill cost heuristic only; it
// does not prove full-wave block uniformity. This first experiment
// intentionally does not model reconvergence: if any predecessor ancestry has
// an uncovered divergent terminator, the block is treated as potentially
// divergent.
static bool mayBeDivergentlyReached(const BasicBlock &BB,
                                    const UniformityInfo &UI,
                                    bool UseBranchVotes) {
  SmallVector<const BasicBlock *, 8> Worklist;
  SmallPtrSet<const BasicBlock *, 16> Visited;
  Visited.insert(&BB);
  for (const BasicBlock *Pred : predecessors(&BB))
    Worklist.push_back(Pred);

  while (!Worklist.empty()) {
    const BasicBlock *Current = Worklist.pop_back_val();
    if (!Visited.insert(Current).second)
      continue;
    const Instruction *Term = Current->getTerminator();
    if (UI.isDivergentTerminator(Term) &&
        !(UseBranchVotes && hasUnanimousBranchVotes(*Term)))
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
  const bool UseBranchVotes =
      HasProfile && SpillUseBranchUnanimityPrototype &&
      MF.getFunction().getParent()->getTargetTriple().isAMDGPU();
  NumBlockIDs = MF.getNumBlockIDs();
  DivergentBlocks.clear();
  DivergentBlocks.resize(NumBlockIDs);

  if (DisableStaticUniformityFallback)
    return;

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
        mayBeDivergentlyReached(*BB, UI, UseBranchVotes))
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
    if (SpillUseBranchUnanimityPrototype)
      OS << ": no PGO annotation (not classified as divergent)\n";
    else
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
