//===- AMDGPUBundleIdxLdSt.cpp - Bundle indexed load/store with uses    ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Form Bundles with VALU instructions and the V_LOAD/STORE_IDX that are used
/// to index the operands. If the V_LOAD_IDX or VALU instruction are in a
/// different basic block, try to sink them to the their uses so that we are
/// able to form bundles (this pre-bundling sinking phase adapts some of the
/// methods from the generic MachineSink phase). Most bundles can be lowered to
/// a single VALU in the AMDGPULowerVGPREncoding pass (with the exception of
/// data movement bundles containing only loads and stores). Replace the
/// V_LOAD/STORE_IDX data operands with staging registers.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUMachineInstrs.h"
#include "AMDGPUResourceUsageAnalysis.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/InitializePasses.h"
#include <unordered_set>

using namespace llvm;

#define DEBUG_TYPE "bundle-indexed-load-store"

namespace {

/// Representation of a candidate operand of the core MI for bundling.
struct Operand {
  /// The operand of the core MI.
  MachineOperand *Op = nullptr;

  /// The V_{LOAD,STORE}_IDX instruction.
  AMDGPUMI::VLoadStoreIdxInst *LoadStore = nullptr;

  /// Size of the operand in bytes.
  unsigned NumBytes = 0;

  MachineOperand &getDataOperand() const { return LoadStore->getDataOp(); }

  Register getDataReg() const { return getDataOperand().getReg(); }

  MachineOperand &getIndexOperand() const { return LoadStore->getIdxOp(); }

  Register getIndexReg() const { return getIndexOperand().getReg(); }

  unsigned getOffset() const { return LoadStore->getOffsetOp().getImm(); }
};

/// Helper class for hoisting instructions within a basic block.
///
/// The user of this class desires to hoist various instructions to (before) a
/// common *target* iterator within the basic block. Hoisted instructions have
/// two kinds of use operands:
///
///  1. Use operands that are ignored by this class. The class assumes that
///     their register definition is already safely before the target iterator.
///  2. Use operands that must be fully defined before a common *prolog*
///     iterator. The class attempts to hoist intermediate instructions to
///     satisfy this condition.
///
/// In the use case here, we have:
///
///     <previous instruction>
///     V_CORE                       <-- prolog iterator
///     <next instruction>           <-- target iterator
///     ...
///     S_ADD_U32                    <-- index computation
///     ...
///     V_STORE_IDX_Bnn              <-- candidate instruction for hoisting
///
/// ... which will be hoisted as:
///
///     <previous instruction>
///     S_ADD_U32                    <-- index computation
///     V_CORE                       <-- prolog iterator
///     V_STORE_IDX_Bnn
///     <next instruction>           <-- target iterator
///
/// A common case is that the instruction pointed to by the original target
/// iterator is hoisted, so that:
///
///     <previous instruction>
///     V_CORE                       <-- prolog iterator
///     S_ADD_U32                    <-- target iterator; index computation
///     V_STORE_IDX_Bnn              <-- candidate instruction for hoisting
///     <next instruction>
///
/// is hoisted to:
///
///     <previous instruction>
///     S_ADD_U32                    <-- index computation
///     V_CORE                       <-- prolog iterator
///     V_STORE_IDX_Bnn
///     <next instruction>           <-- updated target iterator
///
/// This class allows a two-phase approach to hoisting multiple instructions. In
/// the first phase, candidate instructions for hoisting are checked. Finally,
/// a hoisting of a subset of the candidates is committed in the second phase.
///
/// Prolog and Target must be equivalent iterators for each call to methods of
/// this class.
class InstHoisting {
private:
  struct InstrInfo {
    bool IsRoot = false;
    bool CheckComplete = false;
    bool CanHoist = false;
    unsigned Id = -1;
    SmallBitVector Dependencies;
  };

  MapVector<MachineInstr *, InstrInfo> InstrInfos;

  // Analyzed instructions in reverse basic block order.
  SmallVector<MachineInstr *> Instrs;

  const DenseSet<std::pair<MachineInstr *, MachineInstr *>> *CommutableInstrs =
      nullptr;

public:
  void setCommutableInstrs(
      const DenseSet<std::pair<MachineInstr *, MachineInstr *>> *Commutable) {
    CommutableInstrs = Commutable;
  }

  bool check(MachineBasicBlock::iterator Prolog,
             MachineBasicBlock::iterator Target, MachineInstr *MI,
             ArrayRef<MachineOperand *> CheckedUses, AAResults *AA,
             unsigned MaxWorklist = 8);

  MachineBasicBlock::instr_iterator
  commit(MachineBasicBlock::instr_iterator Prolog,
         MachineBasicBlock::instr_iterator Target,
         ArrayRef<MachineInstr *> MIs);
};

class AMDGPUBundleIdxLdSt : public MachineFunctionPass {
  struct BundlingInfo {
    MachineInstr *MI = nullptr;
    InstHoisting StoreHoisting;

    // Operands of MI that are to be bundled, in no particular order.
    SmallVector<Operand, 8> BundledOps;
    // Operands of MI that need to be marked as not killed.
    SmallVector<MachineOperand *, 8> OpsToUnmarkKill;
  };

public:
  static char ID;

  AMDGPUBundleIdxLdSt() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "Bundle indexed load/store with uses";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<MachineCycleInfoWrapperPass>();
    AU.addPreserved<MachineCycleInfoWrapperPass>();
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  bool bundleIdxLdSt(MachineInstr *MI);
  bool analyze(BundlingInfo &BI);
  void reject(MachineOperand &MO, const Twine &Reason = {});
  MachineInstr *convertInstTo3Addr(MachineInstr *MI);
  bool sinkInstruction(MachineInstr &MI, bool &SawStore);
  bool sinkLoadsAndCoreMIs(MachineFunction &MF);
  void lowerLanesharedPseudoInst(MachineInstr &MI);
  void lowerBlockLoadMCastLanesharedPseudoInst(MachineInstr &MI);
  void lowerLoadIdxBits(MachineInstr &MI, bool IsD16);
  void lowerStoreIdxBits(MachineInstr &MI, bool IsD16);
  bool expandPseudoInstructions(MachineFunction &MF, bool &HaveLoadStoreIdx);
  SmallVector<std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>, 4>
  findSuccsToSinkTo(MachineInstr &MI, MachineBasicBlock *MBB);
  bool hasConflictBetween(MachineBasicBlock *From, MachineBasicBlock *To,
                          MachineInstr &MI);
  bool blockPrologueInterferes(const MachineBasicBlock *BB,
                               MachineBasicBlock::const_iterator End,
                               const MachineInstr &MI);
  void findAllPaths(MachineBasicBlock *Start, MachineBasicBlock *End,
                    SmallVector<SmallVector<MachineBasicBlock *, 8>, 8> &Paths,
                    SmallVector<MachineBasicBlock *, 8> &CurrentPath,
                    DenseSet<MachineBasicBlock *> &Visited);
  SmallVector<SmallVector<MachineBasicBlock *, 8>, 8>
  getAllPathsBetweenBlocks(MachineBasicBlock *Start, MachineBasicBlock *End);

  DenseSet<Register> RegsToClearKillFlags;

  DenseMap<std::pair<MachineBasicBlock *, MachineBasicBlock *>,
           SmallVector<MachineInstr *>>
      ConflictInstrCache;

  DenseMap<std::pair<MachineBasicBlock *, MachineBasicBlock *>, bool>
      HasConflictCache;

  DenseMap<std::pair<MachineBasicBlock *, MachineBasicBlock *>,
           SmallVector<SmallVector<MachineBasicBlock *, 8>, 8>>
      PathsCache;

  /// Pairs of store instructions whose order can be exchanged regardless of
  /// what a standard alias analysis would say.
  DenseSet<std::pair<MachineInstr *, MachineInstr *>> CommutableStores;

  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  const GCNSubtarget *ST = nullptr;
  const MCSubtargetInfo *MCSTI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  AliasAnalysis *AA = nullptr;
  MachineCycleInfo *CI = nullptr;

  // (original idx reg) to new-idx-reg mapping
  // Private-in-vgpr objects need a new idx reg that is calculated with idx0 as
  // the offset.
  DenseMap<unsigned, unsigned> PrivateObjectNewRegs;
};

bool sideEffectConflict(MachineInstr &MIa, MachineInstr &MIb) {
  return MIa.hasUnmodeledSideEffects() && MIb.hasUnmodeledSideEffects();
}

// Sink an instruction MI to it's position InsertPos in SuccToSinkTo.
void performSink(MachineInstr &MI, MachineBasicBlock &SuccToSinkTo,
                 MachineBasicBlock::iterator InsertPos) {
  // If we cannot find a location to use (merge with), then we erase the debug
  // location to prevent debug-info driven tools from potentially reporting
  // wrong location information.
  if (!SuccToSinkTo.empty() && InsertPos != SuccToSinkTo.end())
    MI.setDebugLoc(DILocation::getMergedLocation(MI.getDebugLoc(),
                                                 InsertPos->getDebugLoc()));
  else
    MI.setDebugLoc(DebugLoc());

  // Move the instruction.
  MachineBasicBlock *ParentBlock = MI.getParent();
  SuccToSinkTo.splice(InsertPos, ParentBlock, MI,
                      ++MachineBasicBlock::iterator(MI));
}
} // End anonymous namespace.

/// Check whether MI can be hoisted to (before) Target while making sure that
/// any registers used by operands in CheckedUses are or can be defined before
/// the Prolog iterator (if necessary, by hoisting intermediate instructions).
///
/// This method also records dependency information that is later used by the
/// commit() method.
///
/// Also refer to the class comment for more context.
bool InstHoisting::check(MachineBasicBlock::iterator Prolog,
                         MachineBasicBlock::iterator Target, MachineInstr *MI,
                         ArrayRef<MachineOperand *> CheckedUses, AAResults *AA,
                         unsigned MaxWorklist) {
  assert(Prolog != MI->getParent()->end());
  assert(Target != MI->getParent()->end());
  assert(Prolog->getParent() == MI->getParent());
  assert(Target->getParent() == MI->getParent());
  assert(!InstrInfos.contains(MI));

  for (const auto &Def : MI->all_defs()) {
    if (Def.getReg().isPhysical())
      return false;
  }

  unsigned RootId = InstrInfos.size();
  {
    auto Insert = InstrInfos.try_emplace(MI);
    assert(Insert.second);
    InstrInfo &II = Insert.first->second;
    II.IsRoot = true;
    II.Id = RootId;
    Instrs.push_back(MI);
  }

  // First phase: Scan backwards from MI to determine dependencies and find
  // blockers against hoisting MI and its dependencies.

  // Map registers to instructions that use them.
  DenseMap<Register, DenseSet<MachineInstr *>> Regs;
  SmallVector<MachineInstr *> AllInstrs;

  AllInstrs.push_back(MI);

  for (const auto &Use : CheckedUses) {
    assert(Use->getParent() == MI);
    if (!Use->isReg() || !Use->readsReg())
      continue;
    if (Use->getSubReg() != 0) {
      InstrInfo &II = InstrInfos.find(MI)->second;
      II.CheckComplete = true;
      return false;
    }
    Regs[Use->getReg()].insert(MI);
  }

  bool ScanComplete = true;
  bool InProlog = false;
  for (MachineBasicBlock::iterator II = MI->getIterator(); II != Prolog;) {
    if (II == Target)
      InProlog = true;
    --II;

    if (!InProlog) {
      bool Commutable = false;

      if (CommutableInstrs && !CommutableInstrs->empty()) {
        if (CommutableInstrs->contains({MI, &*II}) ||
            CommutableInstrs->contains({&*II, MI}))
          Commutable = true;
      }

      if (!Commutable && MI->mayAlias(AA, *II, true)) {
        InstrInfo &Info = InstrInfos.find(MI)->second;
        Info.CheckComplete = true;
        ScanComplete = false;
        break;
      }
    }

    unsigned IIId = ~0u;
    bool CanHoist = AllInstrs.size() < MaxWorklist;

    // For simplicity, never hoist a memory access.
    if (II->mayLoadOrStore())
      CanHoist = false;

    // Check if this instruction defines any registers used by a relevant
    // later instruction, and record dependencies if so.
    for (const auto &Def : II->all_defs()) {
      // For simplicity, never hoist a physical register def.
      assert(!Def.getSubReg() && "no subregister defs allowed in SSA form");
      if (InProlog || (Def.getReg().isPhysical() && !Def.isDead()))
        CanHoist = false;

      auto RegIt = Regs.find(Def.getReg());
      if (RegIt == Regs.end())
        continue;

      // The current instruction (II) defines a register that is used by one of
      // the previously found candidates for hoisting. That means that II itself
      // becomes a candidate for hoisting that we add to our data structures.
      if (IIId == ~0u) {
        auto [It, Inserted] = InstrInfos.try_emplace(&*II);
        if (Inserted) {
          It->second.Id = InstrInfos.size() - 1;
          Instrs.push_back(&*II);
        }
        IIId = It->second.Id;
      }

      // Add II as a dependency to all previously found candidates for hoisting
      // that use this register defined by II.
      for (MachineInstr *DepMI : RegIt->second) {
        auto &DepMIInfo = InstrInfos.find(DepMI)->second;
        if (IIId >= DepMIInfo.Dependencies.size())
          DepMIInfo.Dependencies.resize(IIId + 1);
        DepMIInfo.Dependencies.set(IIId);
      }

      Regs.erase(RegIt);
    }

    // If this instruction has a later dependency, see if we can hoist it.
    if (IIId != ~0u && CanHoist) {
      auto &IIInfo = InstrInfos.find(&*II)->second;

      if (!IIInfo.CheckComplete) {
        for (const auto &Use : II->all_uses()) {
          if (!Use.isReg() || !Use.readsReg())
            continue;

          if (Use.getSubReg() != 0) {
            IIInfo.CheckComplete = true;
            IIInfo.CanHoist = false;
            break;
          }

          Regs[Use.getReg()].insert(&*II);
        }

        AllInstrs.push_back(&*II);
      }
    }
  }

  // Second phase: Propagate any completed checks backwards.
  for (MachineInstr *MI : reverse(AllInstrs)) {
    InstrInfo &Info = InstrInfos.find(MI)->second;
    if (Info.CheckComplete)
      continue;

    bool AllDependenciesComplete = true;
    for (unsigned Id : Info.Dependencies.set_bits()) {
      InstrInfo &DepInfo = InstrInfos.find(Instrs[Id])->second;
      if (DepInfo.CheckComplete) {
        if (!DepInfo.CanHoist) {
          Info.CheckComplete = true;
          Info.CanHoist = false;
          break;
        }
      } else {
        AllDependenciesComplete = false;
      }
    }

    if (!Info.CheckComplete && AllDependenciesComplete && ScanComplete) {
      Info.CheckComplete = true;
      Info.CanHoist = true;
    }
  }

  return InstrInfos.find(MI)->second.CanHoist;
}

/// Commit the hoisting of the given subset of previously checked instructions.
///
/// Returns the updated target iterator (this differs from the input target if
/// the target instruction itself was hoisted).
MachineBasicBlock::instr_iterator
InstHoisting::commit(MachineBasicBlock::instr_iterator Prolog,
                     MachineBasicBlock::instr_iterator Target,
                     ArrayRef<MachineInstr *> MIs) {
  if (MIs.empty())
    return Target;

  MachineBasicBlock *MBB = MIs[0]->getParent();

  SmallBitVector Committing;
  Committing.resize(InstrInfos.size());
  for (MachineInstr *MI : MIs) {
    const InstrInfo &II = InstrInfos.find(MI)->second;
    assert(II.IsRoot);
    Committing.set(II.Id);
  }

  for (MachineInstr *MI : Instrs) {
    const InstrInfo &II = InstrInfos.find(MI)->second;
    if (!Committing.test(II.Id))
      continue;

    if (II.IsRoot) {
      if (Target != MI->getIterator()) {
        MI->removeFromParent();
        MBB->insert(Target, MI);
      } else {
        Target = std::next(MI->getIterator());
      }
    } else {
      assert(Prolog != MI->getIterator());

      if (Target == MI->getIterator())
        Target = std::next(MI->getIterator());

      MI->removeFromParent();
      MBB->insert(Prolog, MI);
      Prolog = MI->getIterator();
    }

    Committing |= II.Dependencies;
  }

  return Target;
}

// Return true if a target defined block prologue instruction interferes
// with a sink candidate.
bool AMDGPUBundleIdxLdSt::blockPrologueInterferes(
    const MachineBasicBlock *BB, MachineBasicBlock::const_iterator End,
    const MachineInstr &MI) {
  for (MachineBasicBlock::const_iterator PI = BB->getFirstNonPHI(); PI != End;
       ++PI) {
    // Only check target defined prologue instructions
    if (!TII->isBasicBlockPrologue(*PI))
      continue;
    for (auto &MO : MI.operands()) {
      if (!MO.isReg())
        continue;
      Register Reg = MO.getReg();
      if (!Reg)
        continue;
      if (MO.isUse()) {
        if (Reg.isPhysical() &&
            (TII->isIgnorableUse(MO) || (MRI && MRI->isConstantPhysReg(Reg))))
          continue;
        if (PI->modifiesRegister(Reg, TRI))
          return true;
      } else {
        if (PI->readsRegister(Reg, TRI))
          return true;
        // Check for interference with non-dead defs
        auto *DefOp = PI->findRegisterDefOperand(Reg, TRI, false, true);
        if (DefOp && !DefOp->isDead())
          return true;
      }
    }
  }
  return false;
}

// Find all paths between a given Start and End block.
void AMDGPUBundleIdxLdSt::findAllPaths(
    MachineBasicBlock *Start, MachineBasicBlock *End,
    SmallVector<SmallVector<MachineBasicBlock *, 8>, 8> &Paths,
    SmallVector<MachineBasicBlock *, 8> &CurrentPath,
    DenseSet<MachineBasicBlock *> &Visited) {
  if (Start == End) {
    Paths.push_back(CurrentPath);
    return;
  }

  Visited.insert(Start);
  for (MachineBasicBlock *Succ : Start->successors()) {
    if (Visited.count(Succ) == 0) { // Avoid loops.
      CurrentPath.push_back(Succ);
      findAllPaths(Succ, End, Paths, CurrentPath, Visited);
      CurrentPath.pop_back();
    }
  }
  Visited.erase(Start);
}

// Wraps the recursion and uses a cache for already seen Start/End pairs
SmallVector<SmallVector<MachineBasicBlock *, 8>, 8>
AMDGPUBundleIdxLdSt::getAllPathsBetweenBlocks(MachineBasicBlock *Start,
                                              MachineBasicBlock *End) {

  // Check cache to see if we've already computed these paths.
  auto BlockPair = std::make_pair(Start, End);
  if (auto It = PathsCache.find(BlockPair); It != PathsCache.end())
    return It->second;

  SmallVector<SmallVector<MachineBasicBlock *, 8>, 8> Paths;
  SmallVector<MachineBasicBlock *, 8> CurrentPath;
  DenseSet<MachineBasicBlock *> Visited;
  CurrentPath.push_back(Start);
  findAllPaths(Start, End, Paths, CurrentPath, Visited);

  PathsCache[BlockPair] = Paths;

  return Paths;
}

// Find successors to sink this instruction to, and their insertion points.
// This function uses an all-or-nothing strategy: if we can't sink
// to all basic blocks that have a use, then don't sink at all.
SmallVector<std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>, 4>
AMDGPUBundleIdxLdSt::findSuccsToSinkTo(MachineInstr &MI,
                                       MachineBasicBlock *MBB) {

  SmallVector<std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>, 4>
      Candidates;
  bool IsCoreMI = false;
  bool IsLoadMI = isa<AMDGPUMI::VLoadIdxInst>(MI);

  // Loop over all the Defs of the instr, and collect the candidates to sink to.
  size_t TotalUses = 0;
  for (auto &Def : MI.defs()) {
    if (!Def.isReg() || Def.getReg() == 0)
      continue;
    Register DefReg = Def.getReg();

    for (auto U = MRI->use_begin(DefReg); U != MRI->use_end(); U++) {
      assert(U->isReg() && "Expected Use to be reg if Def was reg.");
      TotalUses++;
      MachineInstr *UseMI = U->getParent();
      MachineBasicBlock *UseMBB = UseMI->getParent();

      // If there's a meta/debug use, we wouldn't be able to bundle all uses.
      if (UseMI->isMetaInstruction() || UseMI->isCopy() ||
          UseMI->isDebugOrPseudoInstr() || UseMI->isFakeUse())
        return {};
      // TODO-GFX13 Update TwoAddressInstructionPass to handle Bundles
      if (UseMI->isRegSequence() || UseMI->isInsertSubreg())
        return {};
      // TODO-GFX13 Handle phis.
      if (UseMI->isPHI())
        return {};

      // Determine if this is CoreMI.
      if (!IsLoadMI) {
        if (auto *St = dyn_cast<AMDGPUMI::VStoreIdxInst>(UseMI)) {
          if (St->getDataOp().getReg() == DefReg)
            IsCoreMI = true;
        }
      }
      assert(!(IsCoreMI && IsLoadMI) &&
             "MI can't be both a CoreMI and V_LOAD_IDX.");
      if (!IsLoadMI && !IsCoreMI)
        return {};

      // Check safety of sinking MI to U.
      bool Conflict =
          MI.mayLoad() ? hasConflictBetween(MI.getParent(), UseMBB, MI) : false;
      if (!MI.isSafeToMove(Conflict))
        return {};
      if (!TII->isSafeToSink(MI, UseMBB, CI))
        return {};

      // If the instruction to move defines a dead physical register which is
      // live when leaving the basic block, don't move it because it could turn
      // into a "zombie" define of that phys reg.
      for (const MachineOperand &MO : MI.all_defs()) {
        Register Reg = MO.getReg();
        if (Reg == 0 || !Reg.isPhysical())
          continue;
        if (UseMBB->isLiveIn(Reg))
          return {};
      }

      // Don't move a CoreMI into a cycle.
      if (IsCoreMI && CI->getCycleDepth(UseMBB) > CI->getCycleDepth(MBB)) {
        LLVM_DEBUG(dbgs() << " *** CoreMI sinking to larger cycle depth is "
                             "not profitable\n");
        return {};
      }

      // Determine where to insert into. Skip phi nodes.
      MachineBasicBlock::iterator InsertPos =
          UseMBB->SkipPHIsAndLabels(UseMBB->begin());
      if (blockPrologueInterferes(UseMBB, InsertPos, MI)) {
        LLVM_DEBUG(dbgs() << " *** Not sinking: prologue interference\n");
        return {};
      }

      auto Item = std::make_pair(UseMBB, InsertPos);
      Candidates.push_back(Item);

      // Duplicating CoreMI won't generally be profitable.
      if (IsCoreMI && TotalUses > 1) {
        LLVM_DEBUG(dbgs() << " *** CoreMI has multiple uses; duplicating isn't "
                             "profitable.\n");
        return {};
      }
    }
  }

  return Candidates;
}

// Check if any instruction conflicts with MI between From and To, where a
// conflict is defined as either an alias conflict or both having unmodeled side
// effects. Two caches are used. HasConflictCache is a coarse cache which
// returns true if the pair contains some case we want to treat conservatively
// for all MI (eg. a function call), and returns false if there are no stores at
// all. ConflictInstrCache is used to cache and check the potentially
// conflicting instructions against MI.
bool AMDGPUBundleIdxLdSt::hasConflictBetween(MachineBasicBlock *From,
                                             MachineBasicBlock *To,
                                             MachineInstr &MI) {

  auto BlockPair = std::make_pair(From, To);

  if (auto It = HasConflictCache.find(BlockPair); It != HasConflictCache.end())
    return It->second;

  if (auto It = ConflictInstrCache.find(BlockPair);
      It != ConflictInstrCache.end())
    return llvm::any_of(It->second, [&](MachineInstr *I) {
      bool MayAlias = I->mayAlias(AA, MI, false);
      LLVM_DEBUG(if (MayAlias) {
        dbgs() << " *** Alias conflict with ";
        I->print(dbgs());
      });
      bool SideEffectHazard =
          MI.hasUnmodeledSideEffects() && I->hasUnmodeledSideEffects();
      LLVM_DEBUG(if (SideEffectHazard) {
        dbgs() << " *** Side effect hazard with ";
        I->print(dbgs());
      });
      return SideEffectHazard || MayAlias;
    });

  unsigned int MaxBasicBlockSize = 2000;
  unsigned int MaxPaths = 20;
  unsigned int MaxPathLength = 20;
  bool SawPotentialConflict = false;
  bool HasConflict = false;
  DenseSet<MachineBasicBlock *> HandledBlocks;

  SmallVector<SmallVector<MachineBasicBlock *, 8>, 8> AllPaths =
      getAllPathsBetweenBlocks(From, To);

  // If there are too many paths, treat conservatively to save compile time.
  if (AllPaths.size() > MaxPaths) {
    HasConflictCache[BlockPair] = true;
    return true;
  }

  // Go through all reachable blocks from From.
  for (auto Path : AllPaths) {
    // If any given path is too long, save compiling time.
    if (Path.size() > MaxPathLength) {
      HasConflictCache[BlockPair] = true;
      return true;
    }
    for (auto BB : Path) {
      // We insert the instruction at the start of block To, so no need to
      // worry about conflicts inside To. Conflicts in block From should be
      // already considered when just enter function sinkInstruction.
      if (BB == To || BB == From)
        continue;

      // We already handle this BB in previous iteration.
      if (HandledBlocks.count(BB))
        continue;

      HandledBlocks.insert(BB);

      // If this BB is too big stop searching to save compiling time.
      if (BB->sizeWithoutDebugLargerThan(MaxBasicBlockSize)) {
        HasConflictCache[BlockPair] = true;
        return true;
      }

      for (MachineInstr &I : *BB) {
        if (I.isCall() || I.hasOrderedMemoryRef()) {
          HasConflictCache[BlockPair] = true;
          return true;
        }

        if (I.mayStore() || I.hasUnmodeledSideEffects()) {
          SawPotentialConflict = true;
          // We still have chance to sink MI if all stores between are not
          // aliased to MI, and neither have side effects.
          // Cache all conflicts, so that we don't need to go through
          // all From reachable blocks for next load instruction.
          if (sideEffectConflict(MI, I) || I.mayAlias(AA, MI, false)) {
            LLVM_DEBUG(dbgs() << " *** Conflict with "; I.print(dbgs()));
            HasConflict = true;
          }
          ConflictInstrCache[BlockPair].push_back(&I);
        }
      }
    }
  }
  // If there is no conflict at all, cache the result.
  if (!SawPotentialConflict)
    HasConflictCache[BlockPair] = false;
  return HasConflict;
}

bool AMDGPUBundleIdxLdSt::sinkInstruction(MachineInstr &MI, bool &SawStore) {

  // Don't sink instructions that the target prefers not to sink.
  if (!TII->shouldSink(MI))
    return false;

  // Check if it's safe to move the instruction.
  if (!MI.isSafeToMove(SawStore))
    return false;

  // Convergent operations may not be made control-dependent on additional
  // values.
  if (MI.isConvergent())
    return false;

  MachineBasicBlock *ParentBlock = MI.getParent();
  SmallVector<std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>, 4>
      SuccsToSinkTo = findSuccsToSinkTo(MI, ParentBlock);

  size_t SinksRemaining = SuccsToSinkTo.size();
  if (SinksRemaining == 0)
    return false;

  LLVM_DEBUG(dbgs() << " *** Found " << SinksRemaining << " use(s)\n");
  for (auto Pair : SuccsToSinkTo) {
    auto Succ = Pair.first;
    auto InsertPos = Pair.second;
    // Note that if we previously encountered Succ == MI.getParent(), we'll
    // have an extra sink remaining, which is need for the remaining local use.
    if (Succ == MI.getParent()) {
      LLVM_DEBUG(
          dbgs()
          << " *** Use is in MI's current block. Leaving a copy in block "
          << Succ->getNumber() << "\n");
      continue;
    }

    if (SinksRemaining > 1) {
      assert(isa<AMDGPUMI::VLoadIdxInst>(MI));
      LLVM_DEBUG(dbgs() << "\t *** Duplicating MI and sinking to block "
                        << Succ->getNumber() << "\n");
      MachineInstr *DupLoad =
          MI.getParent()->getParent()->CloneMachineInstr(&MI);
      MI.getParent()->insert(MI, DupLoad);

      // When we duplicate, we must assign to a new register because the
      // bundling phase requires searching for an inst's def, of which there can
      // only be one.
      Register OldDefReg = DupLoad->getOperand(0).getReg();
      auto *RC = MRI->getRegClass(OldDefReg);
      Register NewDefReg = MRI->createVirtualRegister(RC);
      for (auto &UseInSucc : MRI->use_nodbg_operands(OldDefReg)) {
        if (UseInSucc.getParent()->getParent() != Succ || !UseInSucc.isReg() ||
            UseInSucc.getReg() != OldDefReg)
          continue;
        UseInSucc.setReg(NewDefReg);
      }
      DupLoad->getOperand(0).setReg(NewDefReg);
      performSink(*DupLoad, *Succ, InsertPos);
    } else {
      LLVM_DEBUG(dbgs() << "\t *** Sinking MI to block " << Succ->getNumber()
                        << "\n");
      performSink(MI, *Succ, InsertPos);
    }
    SinksRemaining--;
  }

  return true;
}

bool AMDGPUBundleIdxLdSt::sinkLoadsAndCoreMIs(MachineFunction &MF) {
  bool MadeChange = false;
  bool IsConflict = false;
  for (auto &MBB : ReversePostOrderTraversal<MachineFunction *>(&MF)) {

    // Walk the basic block bottom-up.
    SmallVector<MachineInstr *, 8> Conflicts;
    for (auto &I : make_early_inc_range(llvm::reverse(*MBB))) {
      MachineInstr &MI = I; // MI is the instruction to sink.

      // Check if MI conflicts with any of the previously seen instructions in
      // this block
      IsConflict = false;
      for (auto C : Conflicts)
        if (MI.mayAlias(AA, *C, false) || sideEffectConflict(MI, I))
          IsConflict = true;

      if (MI.mayStore() || sideEffectConflict(MI, I))
        Conflicts.push_back(&MI);

      LLVM_DEBUG(dbgs() << "BB." << MBB->getNumber() << " :: ";
                 MI.print(dbgs()));

      if (sinkInstruction(MI, IsConflict))
        MadeChange = true;
    }
  }

  // Now clear any kill flags for recorded registers.
  LLVM_DEBUG(dbgs() << "\n");
  for (auto I : RegsToClearKillFlags)
    MRI->clearKillFlags(I);
  RegsToClearKillFlags.clear();

  return MadeChange;
}

// This lowering puts the value into the lo16 bits of a private VGPR.
// For D16, extract a 16-bit register
void AMDGPUBundleIdxLdSt::lowerLoadIdxBits(MachineInstr &MI, bool IsD16) {
  MachineBasicBlock *MBB = MI.getParent();

  const bool IsSigned = MI.getOperand(5).getImm() != 0;
  const MCInstrDesc &II =
      TII->get(IsSigned ? AMDGPU::V_BFE_I32_e64 : AMDGPU::V_BFE_U32_e64);
  Register ReadReg = MRI->createVirtualRegister(
      TRI->getAllocatableClass(TII->getRegClass(II, 0)));

  auto LoadMIB = BuildMI(*MBB, MI, MI.getDebugLoc(),
                         TII->get(AMDGPU::V_LOAD_IDX_B32), ReadReg)
                     .add(MI.getOperand(1))  // idx
                     .add(MI.getOperand(2)); // offset
  auto *LoadMMO = *MI.memoperands_begin();
  LoadMIB.addMemOperand(LoadMMO);
  Register DataReg = MI.getOperand(0).getReg();

  if (IsD16) {
    assert(TRI->getRegSizeInBits(DataReg, *MRI) == 16 &&
           "Expected 16-bit data register");
    Register TmpReg = MRI->createVirtualRegister(
        TRI->getAllocatableClass(TII->getRegClass(II, 0)));
    BuildMI(*MBB, MI, MI.getDebugLoc(), II, TmpReg)
        .addReg(ReadReg)
        .add(MI.getOperand(4))  // bitoffset
        .add(MI.getOperand(3)); // bitsize
    BuildMI(*MBB, MI, MI.getDebugLoc(), TII->get(AMDGPU::COPY), DataReg)
        .addReg(TmpReg, {}, AMDGPU::lo16);
  } else {
    BuildMI(*MBB, MI, MI.getDebugLoc(), II, DataReg)
        .addReg(ReadReg)
        .add(MI.getOperand(4))  // bitoffset
        .add(MI.getOperand(3)); // bitsize
  }

  LLVM_DEBUG(dbgs() << " *** Expanded pseudo: "; MI.print(dbgs()));
  MI.eraseFromParent();
}

void AMDGPUBundleIdxLdSt::lowerStoreIdxBits(MachineInstr &MI, bool IsD16) {
  MachineBasicBlock *MBB = MI.getParent();
  MachineFunction *MF = MBB->getParent();

  // BFM
  const MCInstrDesc &BFMII = TII->get(AMDGPU::V_BFM_B32_e64);
  Register MaskReg = MRI->createVirtualRegister(
      TRI->getAllocatableClass(TII->getRegClass(BFMII, 0)));
  BuildMI(*MBB, MI, MI.getDebugLoc(), BFMII, MaskReg)
      .add(MI.getOperand(3))  // bitsize
      .add(MI.getOperand(4)); // bitoffset

  const MCInstrDesc &II = TII->get(AMDGPU::V_BFI_B32_e64);
  Register ReadReg = MRI->createVirtualRegister(
      TRI->getAllocatableClass(TII->getRegClass(II, 0)));

  auto LoadMIB = BuildMI(*MBB, MI, MI.getDebugLoc(),
                         TII->get(AMDGPU::V_LOAD_IDX_B32), ReadReg)
                     .add(MI.getOperand(1))  // idx
                     .add(MI.getOperand(2)); // offset
  auto *StoreMMO = *MI.memoperands_begin();
  // Synthesize MMO for V_LOAD_IDX.
  auto NewFlags = MachineMemOperand::MOLoad;
  NewFlags |= StoreMMO->getFlags() & ~MachineMemOperand::MOStore;
  MachineMemOperand *LoadMMO = MF->getMachineMemOperand(StoreMMO, NewFlags);
  LoadMIB.addMemOperand(LoadMMO);

  Register DataReg = MI.getOperand(0).getReg();
  Register ExpandReg = DataReg;
  if (IsD16) {
    // Put the 16-bit data_op in a 32-bit register.
    assert(TRI->getRegSizeInBits(DataReg, *MRI) == 16 &&
           "Expected 16-bit data register");
    ExpandReg = MRI->createVirtualRegister(
        TRI->getAllocatableClass(TII->getRegClass(II, 2)));
    Register Undef = MRI->createVirtualRegister(&AMDGPU::VGPR_16RegClass);
    BuildMI(*MBB, MI, MI.getDebugLoc(), TII->get(AMDGPU::REG_SEQUENCE),
            ExpandReg)
        .addReg(DataReg)
        .addImm(AMDGPU::lo16)
        .addReg(Undef, RegState::Undef)
        .addImm(AMDGPU::hi16);
  }

  Register WriteReg = MRI->createVirtualRegister(
      TRI->getAllocatableClass(TII->getRegClass(II, 0)));
  // BFI
  auto CoreMIB = BuildMI(*MBB, MI, MI.getDebugLoc(), II, WriteReg);
  CoreMIB.addReg(MaskReg);
  CoreMIB.addReg(ExpandReg);
  CoreMIB.addReg(ReadReg);

  // V_STORE_IDX
  auto StoreMIB = BuildMI(*MBB, MI, MI.getDebugLoc(),
                          TII->get(AMDGPU::V_STORE_IDX_B32))
                      .addReg(WriteReg)
                      .add(MI.getOperand(1))  // idx
                      .add(MI.getOperand(2)); // offset
  StoreMIB.addMemOperand(StoreMMO);

  LLVM_DEBUG(dbgs() << " *** Expanded pseudo: "; MI.print(dbgs()));
  MI.eraseFromParent();
}

// Lower block load mcast pseudo instruction to another pseudo and V_STORE_IDX
void
AMDGPUBundleIdxLdSt::lowerBlockLoadMCastLanesharedPseudoInst(MachineInstr &MI) {
  MachineBasicBlock *MBB = MI.getParent();
  MachineFunction *MF = MBB->getParent();

  unsigned Opc, DstSize;
  switch (MI.getOpcode()) {
  case AMDGPU::DS_BLOCK_LOAD_MCAST_B128_LANESHARED:
    DstSize = 128;
    Opc = AMDGPU::DS_BLOCK_LOAD_MCAST_B128;
    break;
  case AMDGPU::DS_BLOCK_LOAD_MCAST_B256_LANESHARED:
    DstSize = 256;
    Opc = AMDGPU::DS_BLOCK_LOAD_MCAST_B256;
    break;
  case AMDGPU::DS_BLOCK_LOAD_MCAST_B512_LANESHARED:
    DstSize = 512;
    Opc = AMDGPU::DS_BLOCK_LOAD_MCAST_B512;
    break;
  case AMDGPU::DS_BLOCK_LOAD_MCAST_B1024_LANESHARED:
    DstSize = 1024;
    Opc = AMDGPU::DS_BLOCK_LOAD_MCAST_B1024;
    break;
  default:
    llvm_unreachable("Unknown DS block load mcast instruction!");
  }

  const MCInstrDesc &II = TII->get(Opc);
  Register DstReg = MRI->createVirtualRegister(
      TRI->getAllocatableClass(TII->getRegClass(II, 0)));
  BuildMI(*MBB, MI, MI.getDebugLoc(), II, DstReg)
      .add(MI.getOperand(0))  // L#
      .add(MI.getOperand(1))  // LDS address offset
      .add(MI.getOperand(2)); // gds bit

  MachinePointerInfo StorePtrI(AMDGPUAS::LANE_SHARED);
  MachineMemOperand *StoreMMO =
      MF->getMachineMemOperand(StorePtrI, MachineMemOperand::MOStore,
                               LocationSize::precise(4), Align(4));
  unsigned StIdxOpc = AMDGPUMI::VStoreIdxInst::getOpcodeForBitWidth(DstSize);
  BuildMI(*MBB, MI, MI.getDebugLoc(), TII->get(StIdxOpc))
      .addReg(DstReg)        // data
      .add(MI.getOperand(3)) // idx
      .add(MI.getOperand(4)) // offset
      .addMemOperand(StoreMMO);

  LLVM_DEBUG(dbgs() << " *** Expanded pseudo: "; MI.print(dbgs()));
  MI.eraseFromParent();
}

// Lower the pseudo instruction to another pseudo and V_STORE_IDX
void AMDGPUBundleIdxLdSt::lowerLanesharedPseudoInst(MachineInstr &MI) {
  MachineBasicBlock *MBB = MI.getParent();
  MachineFunction *MF = MBB->getParent();
  MachineMemOperand *StoreMMO = nullptr;
  unsigned Opc = 0;
  bool Loads = true;
  unsigned NumStores = 1;
  switch (MI.getOpcode()) {
  case AMDGPU::V_SEND_VGPR_NEXT_B32_LANESHARED: {
    Opc = AMDGPU::V_SEND_VGPR_NEXT_B32;
    NumStores = 2;
    Loads = false;
    StoreMMO = *MI.memoperands_begin();
    break;
  }
  case AMDGPU::V_SEND_VGPR_PREV_B32_LANESHARED: {
    Opc = AMDGPU::V_SEND_VGPR_PREV_B32;
    NumStores = 2;
    Loads = false;
    StoreMMO = *MI.memoperands_begin();
    break;
  }
  // TODO-GFX13: Add pre-existing StoreMMOs
  case AMDGPU::CLUSTER_LOAD_B32_LANESHARED:
    Opc = AMDGPU::CLUSTER_LOAD_B32;
    break;
  case AMDGPU::CLUSTER_LOAD_B32_LANESHARED_SADDR:
    Opc = AMDGPU::CLUSTER_LOAD_B32_SADDR;
    break;
  case AMDGPU::CLUSTER_LOAD_B64_LANESHARED:
    Opc = AMDGPU::CLUSTER_LOAD_B64;
    break;
  case AMDGPU::CLUSTER_LOAD_B64_LANESHARED_SADDR:
    Opc = AMDGPU::CLUSTER_LOAD_B64_SADDR;
    break;
  case AMDGPU::CLUSTER_LOAD_B128_LANESHARED:
    Opc = AMDGPU::CLUSTER_LOAD_B128;
    break;
  case AMDGPU::CLUSTER_LOAD_B128_LANESHARED_SADDR:
    Opc = AMDGPU::CLUSTER_LOAD_B128_SADDR;
    break;
  case AMDGPU::DDS_LOAD_MCAST_B32_LANESHARED:
    Opc = AMDGPU::DDS_LOAD_MCAST_B32;
    break;
  case AMDGPU::DDS_LOAD_MCAST_B32_LANESHARED_SADDR:
    Opc = AMDGPU::DDS_LOAD_MCAST_B32_SADDR;
    break;
  case AMDGPU::DDS_LOAD_MCAST_B64_LANESHARED:
    Opc = AMDGPU::DDS_LOAD_MCAST_B64;
    break;
  case AMDGPU::DDS_LOAD_MCAST_B64_LANESHARED_SADDR:
    Opc = AMDGPU::DDS_LOAD_MCAST_B64_SADDR;
    break;
  case AMDGPU::DDS_LOAD_MCAST_B128_LANESHARED:
    Opc = AMDGPU::DDS_LOAD_MCAST_B128;
    break;
  case AMDGPU::DDS_LOAD_MCAST_B128_LANESHARED_SADDR:
    Opc = AMDGPU::DDS_LOAD_MCAST_B128_SADDR;
    break;
  case AMDGPU::DS_LOAD_MCAST_B32_LANESHARED:
    Opc = AMDGPU::DS_LOAD_MCAST_B32;
    break;
  case AMDGPU::DS_LOAD_MCAST_B64_LANESHARED:
    Opc = AMDGPU::DS_LOAD_MCAST_B64;
    break;
  case AMDGPU::DS_LOAD_MCAST_B128_LANESHARED:
    Opc = AMDGPU::DS_LOAD_MCAST_B128;
    break;
  case AMDGPU::DS_BLOCK_LOAD_MCAST_B128_LANESHARED:
  case AMDGPU::DS_BLOCK_LOAD_MCAST_B256_LANESHARED:
  case AMDGPU::DS_BLOCK_LOAD_MCAST_B512_LANESHARED:
  case AMDGPU::DS_BLOCK_LOAD_MCAST_B1024_LANESHARED:
    lowerBlockLoadMCastLanesharedPseudoInst(MI);
    return;
  default:
    return;
  }
  const MCInstrDesc &II = TII->get(Opc);
  auto CoreMIB = BuildMI(*MBB, MI, MI.getDebugLoc(), II);
  SmallVector<Register, 4> DataRegs;
  for (unsigned I = 0; I < NumStores; I++) {
    Register DataReg = MRI->createVirtualRegister(
        TRI->getAllocatableClass(TII->getRegClass(II, 0)));
    CoreMIB.addDef(DataReg);
    DataRegs.push_back(DataReg);
  }
  unsigned OpsToCopy = II.getNumOperands() - NumStores;
  unsigned NumMIOps = MI.getNumExplicitOperands();
  assert(OpsToCopy + 2 * NumStores == NumMIOps &&
         "Unexpected number of operands in laneshared pseudo");
  for (unsigned I = 0, E = OpsToCopy; I < E; ++I) {
    CoreMIB.add(MI.getOperand(I));
  }
  auto StoreFlags = MachineMemOperand::MOStore;
  LocationSize Size = LocationSize::precise(4);
  Align BaseAlign = Align(4);
  if (Loads) {
    auto *LoadMMO = *MI.memoperands_begin();
    CoreMIB.addMemOperand(LoadMMO);
    StoreFlags |= LoadMMO->getFlags() & ~MachineMemOperand::MOLoad;
    Size = LoadMMO->getSize();
    BaseAlign = LoadMMO->getBaseAlign();
  }

  SmallVector<MachineInstr *, 2> Stores;
  for (unsigned I = 0; I < NumStores; ++I) {
    // DataRegs is in reverse order of V_STORE_IDXs
    auto StoreMIB =
        BuildMI(*MBB, MI, MI.getDebugLoc(),
                TII->get(AMDGPUMI::VStoreIdxInst::getOpcodeForBitWidth(
                    Size.getValue() * 8)))
            .addReg(DataRegs[NumStores - I - 1])         // data
            .add(MI.getOperand(NumMIOps - (2 + 2 * I)))  // idx
            .add(MI.getOperand(NumMIOps - (1 + 2 * I))); // offset

    // Ideally use a pre-existing StoreMMO, if not synthesize here.
    if (!StoreMMO) {
      MachinePointerInfo StorePtrI(AMDGPUAS::LANE_SHARED);
      StoreMMO =
          MF->getMachineMemOperand(StorePtrI, StoreFlags, Size, BaseAlign);
    }

    StoreMIB.addMemOperand(StoreMMO);
    Stores.push_back(StoreMIB);
  }

  if (Stores.size() == 2)
    CommutableStores.insert(std::make_pair(Stores[0], Stores[1]));

  LLVM_DEBUG(dbgs() << " *** Expanded pseudo: "; MI.print(dbgs()));
  MI.eraseFromParent();
}

bool AMDGPUBundleIdxLdSt::expandPseudoInstructions(MachineFunction &MF,
                                                   bool &HaveLoadStoreIdx) {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      if (SIInstrInfo::mustHaveLanesharedResult(MI)) {
        // To fulfill the programming model, these instructions must not fail to
        // map their destination into laneshared VGPRs. Therefore we expand
        // their temporary pseudos into bundles as late as possible
        Changed = true;
        lowerLanesharedPseudoInst(MI);
        continue;
      }
      if (MI.getOpcode() == AMDGPU::V_LOAD_IDX_BITS) {
        Changed = true;
        lowerLoadIdxBits(MI, false);
        continue;
      }
      if (MI.getOpcode() == AMDGPU::V_LOAD_IDX_BITS_D16) {
        Changed = true;
        lowerLoadIdxBits(MI, true);
        continue;
      }
      if (MI.getOpcode() == AMDGPU::V_STORE_IDX_BITS) {
        Changed = true;
        lowerStoreIdxBits(MI, false);
        continue;
      }
      if (MI.getOpcode() == AMDGPU::V_STORE_IDX_BITS_D16) {
        Changed = true;
        lowerStoreIdxBits(MI, true);
        continue;
      }
      if (isa<AMDGPUMI::VLoadStoreIdxInst>(MI))
        HaveLoadStoreIdx = true;
    }
  }
  HaveLoadStoreIdx |= Changed;
  return Changed;
}

/// Convert the specified two-address instruction into a three address one.
/// Return the new instruction if this transformation was successful.
///
/// TODO-GFX13: Extract common code from
/// TwoAddressInstructionImpl::convertInstTo3Addr
///             during upstreaming.
MachineInstr *AMDGPUBundleIdxLdSt::convertInstTo3Addr(MachineInstr *MI) {
  MachineBasicBlock *MBB = MI->getParent();
  MachineFunction *MF = MBB->getParent();
  MachineInstr *NewMI = TII->convertToThreeAddress(*MI, nullptr, nullptr);
  if (!NewMI)
    return nullptr;

  LLVM_DEBUG(dbgs() << "2addr: CONVERTING 2-ADDR: " << *MI);
  LLVM_DEBUG(dbgs() << "2addr:         TO 3-ADDR: " << *NewMI);

  // If the old instruction is debug value tracked, an update is required.
  if (auto OldInstrNum = MI->peekDebugInstrNum()) {
    assert(MI->getNumExplicitDefs() == 1);
    assert(NewMI->getNumExplicitDefs() == 1);

    // Find the old and new def location.
    unsigned OldIdx = MI->defs().begin()->getOperandNo();
    unsigned NewIdx = NewMI->defs().begin()->getOperandNo();

    // Record that one def has been replaced by the other.
    unsigned NewInstrNum = NewMI->getDebugInstrNum();
    MF->makeDebugValueSubstitution(std::make_pair(OldInstrNum, OldIdx),
                                   std::make_pair(NewInstrNum, NewIdx));
  }

  MBB->erase(MI); // Nuke the old inst.

  return NewMI;
}

/// Report that a candidate for bundling was rejected.
void AMDGPUBundleIdxLdSt::reject(MachineOperand &MO, const Twine &Reason) {
  LLVM_DEBUG(dbgs() << "  Cannot bundle operand: " << MO << '\n'
                    << "                     in: " << *MO.getParent() << "\n";
             if (!Reason.isTriviallyEmpty()) dbgs()
             << "                 reason: " << Reason << "\n";);
  if (MO.isDef() && SIInstrInfo::mustHaveLanesharedResult(*MO.getParent()))
    report_fatal_error(
        "Failed to bundle instruction that must have laneshared");
}

/// Check whether MI has some indexing operands that should be bundled, and
/// collect information in BI without making any IR changes.
///
/// This is read-only **except** it may convert the core MI into three-address
/// form.
///
/// TODO: It would be great if we could avoid that conversion here, but the
/// code relies to much on being able to reference the correct MachineOperands.
///
/// TODO-GFX13: There is also a guaranteed-to-be-broken mutation of private
/// object pointers here.
bool AMDGPUBundleIdxLdSt::analyze(BundlingInfo &BI) {
  MachineInstr *MI = BI.MI;

  LLVM_DEBUG(dbgs() << "BB." << MI->getParent()->getNumber() << " :: ";
             MI->print(dbgs()));

  if (MI->isMetaInstruction())
    return false;
  if (MI->isRegSequence() || MI->isInsertSubreg())
    return false;
  // COPY would be lowered to v_mov, which is equivalent to not bundling at all,
  // and further optimization of the COPY would be blocked by the BUNDLE, so
  // skip it.
  if (MI->isCopy())
    return false;
  // TODO-GFX13 Handle phis.
  if (MI->isPHI())
    return false;
  // There is no way to safely expand it post RA if bundled.
  if (MI->getOpcode() == AMDGPU::V_MOV_B64_PSEUDO && !ST->hasMovB64())
    return false;

  // Cannot bundle instructions using frame indices because
  // PrologueEpilogueInserter cannot handle them inside bundles
  // during replaceFrameIndicesBackward.
  if (llvm::any_of(MI->operands(),
                   [](const MachineOperand &MO) { return MO.isFI(); }))
    return false;

  BI.StoreHoisting.setCommutableInstrs(&CommutableStores);

  // Step 1: Collect candidate defs.
  MachineBasicBlock *MBB = MI->getParent();

  for (auto &Def : MI->defs()) {
    Register DefReg = Def.getReg();
    assert(Def.getSubReg() == 0 && "unexpected subreg def in SSA form");
    if (!MRI->hasOneNonDBGUse(DefReg))
      continue;
    MachineOperand *UseOfMI = &*MRI->use_nodbg_begin(DefReg);
    if (UseOfMI->getSubReg() != 0)
      continue;
    auto *StoreMI = dyn_cast<AMDGPUMI::VStoreIdxInst>(UseOfMI->getParent());
    if (!StoreMI)
      continue;

    Operand Candidate{&Def, StoreMI};
    if (Candidate.getDataReg() != DefReg)
      continue;

    if (StoreMI->getParent() != MBB) {
      reject(Def, "store in different basic block");
      continue;
    }

    // Check that we can hoist the store to MI
    if (!BI.StoreHoisting.check(MI->getIterator(), std::next(MI->getIterator()),
                                StoreMI, &StoreMI->getIdxOp(), AA)) {
      reject(Def, "cannot hoist store");
      continue;
    }

    BI.BundledOps.push_back(Candidate);
  }

  assert(BI.BundledOps.size() <= 2); // would need more staging registers

  // Step 2: Collect candidate uses.
  for (auto &Use : MI->explicit_uses()) {
    // TODO-GFX13: Skip uses that *cannot* use indexing, e.g. high operands of
    //             image instructions.
    if (!Use.isReg() || Use.getSubReg() != 0)
      continue;
    Register UseReg = Use.getReg();
    if (!UseReg.isVirtual())
      continue;
    auto *LoadMI = dyn_cast<AMDGPUMI::VLoadIdxInst>(MRI->getVRegDef(UseReg));
    if (!LoadMI || LoadMI->getParent() != MBB)
      continue;
    // Unset kill hints since LoadStore MIs could be completely reorderded.
    BI.OpsToUnmarkKill.push_back(&LoadMI->getIdxOp());

    // Check that we can sink the load to MI
    bool MayAlias = false;
    for (auto II = ++MachineBasicBlock::iterator(LoadMI); &*II != MI; ++II) {
      if ((II->hasOrderedMemoryRef() || LoadMI->mayAlias(AA, *II, true))) {
        LLVM_DEBUG(dbgs() << "***Sinking conflict: \n\tLoadMI: ";
                   LoadMI->print(dbgs()); dbgs() << "\twith: ";
                   II->print(dbgs()));
        reject(Use);
        MayAlias = true;
        break;
      }
    }
    if (MayAlias)
      continue;

    Operand Candidate{&Use, LoadMI};
    BI.BundledOps.push_back(Candidate);
  }

  // Step 3: Common checks that apply to both defs and uses.
  BI.BundledOps.erase(
      llvm::remove_if(BI.BundledOps,
                      [&](Operand &Op) {
                        const TargetRegisterClass *RegClass =
                            TRI->getRegClassForReg(*MRI, Op.getDataReg());
                        Op.NumBytes = AMDGPU::getRegBitWidth(*RegClass) / 8;

                        if (ST->needsAlignedVGPRs() && Op.NumBytes > 4) {
                          // Do not bundle instructions with odd offsets to
                          // ensure proper register alignment.
                          //
                          // TODO-GFX13: Should this also check the alignment
                          // in the MMO, considering that the index itself
                          // might not be aligned?
                          if (Op.getOffset() & 1) {
                            reject(*Op.Op);
                            return true;
                          }
                        }

                        if (!TII->canUseVGPRIndexing(*MI,
                                                     Op.Op->getOperandNo()))
                          return true;

                        return false;
                      }),
      BI.BundledOps.end());

  if (BI.BundledOps.empty())
    return false;

  // Step 4: Handle earlyclobber and tied operands.
  for (auto &Def : MI->defs()) {
    if (Def.isEarlyClobber()) {
      auto DefIt = llvm::find_if(BI.BundledOps,
                                 [&](Operand &Op) { return Op.Op == &Def; });
      if (DefIt != BI.BundledOps.end()) {
        unsigned DefIdx = std::distance(BI.BundledOps.begin(), DefIt);
        SmallVector<unsigned> UseIdxs;
        unsigned UseBytes = 0;
        for (unsigned UseIdx = DefIdx + 1; UseIdx != BI.BundledOps.size();
             ++UseIdx) {
          // Earlyclobber does not affect a tied use.
          if (Def.isTied()) {
            unsigned TiedUseOpIdx = MI->findTiedOperandIdx(Def.getOperandNo());
            if (BI.BundledOps[UseIdx].Op == &MI->getOperand(TiedUseOpIdx))
              continue;
          }

          if (DefIt->LoadStore->mayAlias(AA, *BI.BundledOps[UseIdx].LoadStore,
                                         true)) {
            UseIdxs.push_back(UseIdx);
            UseBytes += BI.BundledOps[UseIdx].NumBytes;
          }
        }

        // If the def aliases any uses, refuse bundling either the uses or the
        // defs.
        if (!UseIdxs.empty()) {
          if (SIInstrInfo::mustHaveLanesharedResult(*MI) ||
              DefIt->NumBytes >= UseBytes) {
            for (unsigned UseIdx : reverse(UseIdxs))
              BI.BundledOps.erase(BI.BundledOps.begin() + UseIdx);
          } else {
            BI.BundledOps.erase(DefIt);
          }
        }
      }
    }

    if (Def.isTied()) {
      unsigned TiedUseIdx = MI->findTiedOperandIdx(Def.getOperandNo());
      MachineOperand &TiedUse = MI->getOperand(TiedUseIdx);
      auto DefIt = llvm::find_if(BI.BundledOps,
                                 [&](Operand &Op) { return Op.Op == &Def; });
      auto UseIt = llvm::find_if(
          BI.BundledOps, [&](Operand &Op) { return Op.Op == &TiedUse; });
      bool BundleDef = DefIt != BI.BundledOps.end();
      bool BundleUse = UseIt != BI.BundledOps.end();
      bool Conflict = false;

      if (BundleDef != BundleUse) {
        Conflict = true;
      } else if (BundleDef && BundleUse) {
        Conflict = DefIt->getIndexReg() != UseIt->getIndexReg() ||
                   DefIt->getOffset() != UseIt->getOffset();
      }

      if (Conflict && MI->isConvertibleTo3Addr()) {
        // Convert into 3-address form, unless the def is earlyclobber and we
        // would bundle a partially overlapping tied use.
        if (!BundleDef || !BundleUse || !Def.isEarlyClobber() ||
            !DefIt->LoadStore->mayAlias(AA, *UseIt->LoadStore, true)) {
          if (MachineInstr *NewMI = convertInstTo3Addr(MI)) {
            // The instruction was completely replaced, so we have to re-scan
            // it from the top.
            BI = {};
            BI.MI = NewMI;
            return analyze(BI);
          }
        }
      }

      if (Conflict) {
        // Uses come after defs, so erase the use first to avoid iterator
        // invalidation.
        if (BundleUse)
          BI.BundledOps.erase(UseIt);
        if (BundleDef) {
          BI.BundledOps.erase(DefIt);
        }
      }
    }
  }

  if (BI.BundledOps.empty())
    return false;

  // Step 5: Account for availability of index registers.
  //
  // Map each virtual index register to the total number of bytes accessed
  // through it.
  SmallVector<std::pair<Register, unsigned>> Indices;
  unsigned NumDstIndices = 0;
  bool HasPrivate = false;

  for (auto &Op : MI->operands()) {
    if (!Op.isReg())
      continue;
    const TargetRegisterClass *RegClass =
        TRI->getRegClassForReg(*MRI, Op.getReg());
    if (!RegClass)
      continue; // happens e.g. for (implicit) MODE operands
    if (SIRegisterInfo::hasVGPRs(RegClass)) {
      if (none_of(BI.BundledOps, [&](Operand &O) { return O.Op == &Op; })) {
        HasPrivate = true;
        break;
      }
    }
  }

  for (auto &Op : BI.BundledOps) {
    auto It = find_if(Indices, [&](const auto &Pair) {
      return Pair.first == Op.getIndexReg();
    });
    if (It == Indices.end()) {
      Indices.emplace_back(Op.getIndexReg(), 0);
      It = std::prev(Indices.end());
    }
    It->second += Op.NumBytes;

    if (Op.Op->isDef())
      NumDstIndices = Indices.size();
  }

  assert(NumDstIndices <= 2);

  if (Indices.size() > (HasPrivate ? 3 : 4)) {
    auto Begin = Indices.begin();
    if (SIInstrInfo::mustHaveLanesharedResult(*MI))
      Begin += NumDstIndices;

    std::stable_sort(Begin, Indices.end(), [](const auto &A, const auto &B) {
      return A.second > B.second;
    });

    // Since we failed to bundle everything, we need idx0 for private
    // registers.
    Indices.resize(3);

    BI.BundledOps.erase(
        llvm::remove_if(BI.BundledOps,
                        [&](Operand &Op) {
                          return llvm::none_of(Indices, [&](const auto &Pair) {
                            return Pair.first == Op.getIndexReg();
                          });
                        }),
        BI.BundledOps.end());
  }

  assert(!BI.BundledOps.empty());
  return true;
}

bool AMDGPUBundleIdxLdSt::bundleIdxLdSt(MachineInstr *OrigMI) {
  BundlingInfo BI;
  BI.MI = OrigMI;
  bool WillBundle = analyze(BI);
  // Clear any kill flags that may become incorrect when other bundles get
  // committed.
  for (MachineOperand *&Op : BI.OpsToUnmarkKill)
    Op->setIsKill(false);
  if (!WillBundle)
    return false;

  MachineInstr *MI = BI.MI;
  MachineFunction *MF = MI->getParent()->getParent();
  MachineBasicBlock *MBB = MI->getParent();

  // Turn unbundled tied sub-register uses into full register uses by inserting
  // COPY.
  for (MachineOperand &MO : MI->all_uses()) {
    if (!MO.isReg() || !MO.isTied() || MO.getSubReg() == 0)
      continue;

    assert(none_of(BI.BundledOps, [&](Operand &Op) { return Op.Op == &MO; }));

    const TargetRegisterClass *SuperRC = MRI->getRegClass(MO.getReg());
    const TargetRegisterClass *SubRC =
        TRI->getSubRegisterClass(SuperRC, MO.getSubReg());
    Register Tmp = MRI->createVirtualRegister(SubRC);
    BuildMI(*MBB, MI, MI->getDebugLoc(), TII->get(AMDGPU::COPY), Tmp)
        .addUse(MO.getReg(), {}, MO.getSubReg());
    MO.setReg(Tmp);
    MO.setSubReg(0);
  }

  // Commit the bundle.
  static const Register DstStagingRegsList[] = {AMDGPU::STG_DSTA,
                                                AMDGPU::STG_DSTB};
  static const Register SrcStagingRegsList[] = {
      AMDGPU::STG_SRCA, AMDGPU::STG_SRCB, AMDGPU::STG_SRCC,
      AMDGPU::STG_SRCD, AMDGPU::STG_SRCE, AMDGPU::STG_SRCF};
  auto DstStagingRegs = ArrayRef(DstStagingRegsList);
  auto SrcStagingRegs = ArrayRef(SrcStagingRegsList);

  MachineInstr *FirstMI = MI;

  SmallVector<MachineInstr *, 2> Stores;

  for (auto &Op : BI.BundledOps) {
    if (Op.Op->isDef()) {
      Stores.push_back(Op.LoadStore);
    } else {
      if (!MRI->hasOneNonDBGUse(Op.getDataReg())) {
        LLVM_DEBUG(dbgs() << " *** Duplicating "; Op.LoadStore->print(dbgs()));
        Op.LoadStore = cast<AMDGPUMI::VLoadStoreIdxInst>(
            MF->CloneMachineInstr(Op.LoadStore));
      } else {
        Op.LoadStore->removeFromParent();
      }
    }

    Register Stg;
    MachineOperand *UnderlyingOp = Op.Op;
    if (Op.Op->isUse() && Op.Op->isTied()) {
      unsigned DefIdx = MI->findTiedOperandIdx(Op.Op->getOperandNo());
      UnderlyingOp = &MI->getOperand(DefIdx);
    }
    if (UnderlyingOp->isDef()) {
      Stg = DstStagingRegs[UnderlyingOp->getOperandNo()];
    } else {
      Stg = SrcStagingRegs.consume_front();
    }

    Op.Op->setReg(Stg);
    Op.getDataOperand().setReg(Stg);

    if (!Op.Op->isDef()) {
      MBB->insert(FirstMI->getIterator(), Op.LoadStore);
      FirstMI = Op.LoadStore;
    }
  }

  auto BeginMII = FirstMI->getIterator();
  auto EndMII = std::next(MI->getIterator());
  EndMII = BI.StoreHoisting.commit(BeginMII, EndMII, Stores);

  finalizeBundle(*MBB, BeginMII, EndMII);

  LLVM_DEBUG({
    dbgs() << " *** Created bundle from \n";
    for (MachineInstr &MI : make_range(BeginMII, EndMII))
      dbgs() << "\t" << MI;
  });
  return true;
}

bool AMDGPUBundleIdxLdSt::runOnMachineFunction(MachineFunction &MF) {

  ST = &MF.getSubtarget<GCNSubtarget>();

  if (!ST->hasVGPRIndexingRegisters())
    return false;

  TRI = ST->getRegisterInfo();
  TII = ST->getInstrInfo();
  MRI = &MF.getRegInfo();
  MCSTI = MF.getTarget().getMCSubtargetInfo();

  bool Changed = false;
  bool HaveLoadStoreIdx = false;
  LLVM_DEBUG(
      dbgs()
      << "===== AMDGPUBundleIdxLdSt :: Lower pseudo-Instructions =====\n");
  Changed |= expandPseudoInstructions(MF, HaveLoadStoreIdx);
  if (!HaveLoadStoreIdx) {
    assert(!Changed);
    return false; // early out
  }

  if (auto *AAR = getAnalysisIfAvailable<AAResultsWrapperPass>())
    AA = &AAR->getAAResults();
  CI = &getAnalysis<MachineCycleInfoWrapperPass>().getCycleInfo();

  LLVM_DEBUG(dbgs() << "===== AMDGPUBundleIdxLdSt :: Sinking Phase =====\n");
  Changed |= sinkLoadsAndCoreMIs(MF);

  LLVM_DEBUG(dbgs() << "===== AMDGPUBundleIdxLdSt :: Bundling Phase =====\n");
  PrivateObjectNewRegs.clear();
  for (MachineBasicBlock &MBB : MF) {
    // Instruction iterator stability:
    //  * We use an early incrementing range because MI might be erased and
    //    replaced by its 3-address variant.
    //  * The next instruction might be a V_STORE_IDX that gets bundled with
    //    MI. If that happens, we skip it immediately in the next iteration of
    //    the loop, and the iterator safely skips over the rest of the bundle.
    for (auto &MI : make_early_inc_range(MBB)) {
      if (MI.isBundled())
        continue;
      Changed |= bundleIdxLdSt(&MI);
    }
  }
  return Changed;
}

char AMDGPUBundleIdxLdSt::ID = 0;
char &llvm::AMDGPUBundleIdxLdStID = AMDGPUBundleIdxLdSt::ID;

INITIALIZE_PASS_BEGIN(AMDGPUBundleIdxLdSt, DEBUG_TYPE,
                      "Bundle indexed load/store with uses", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineCycleInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUBundleIdxLdSt, DEBUG_TYPE,
                    "Bundle indexed load/store with uses", false, false)
