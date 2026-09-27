//===- SpillStoreElimination.cpp - Eliminate redundant spill stores
//--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A spill store at a join can be redundant on an incoming edge that reloads
// the same value. Move the store to the other edge, preserving initialization
// on paths that have not stored the value yet. Run after final register
// allocation and stack slot coloring, before frame indices are eliminated.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SpillStoreElimination.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "spill-store-elimination"

static cl::opt<bool>
    EnableSpillStoreElimination("enable-spill-store-elimination", cl::Hidden,
                                cl::init(false),
                                cl::desc("Eliminate partially redundant spill "
                                         "stores after register allocation"));

STATISTIC(NumStores, "Number of partially redundant spill stores eliminated");
STATISTIC(NumReloads, "Number of dead spill reloads eliminated");

namespace {

struct StackAccess {
  Register Reg;
  int FI;
  TypeSize Bytes = TypeSize::getZero();
  unsigned DataOp;
};

class SpillStoreElimination {
  MachineFunction &MF;
  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;
  bool HasDebugInstrs = false;

  bool getAccess(const MachineInstr &MI, bool IsLoad,
                 StackAccess &Access) const;
  bool clobbersInputs(const MachineInstr &MI, const MachineInstr &Store) const;
  MachineInstr *findReload(MachineBasicBlock &Pred, const MachineInstr &Store,
                           const StackAccess &Access) const;
  bool processBlock(MachineBasicBlock &MBB);

public:
  explicit SpillStoreElimination(MachineFunction &MF)
      : MF(MF), TII(*MF.getSubtarget().getInstrInfo()),
        TRI(*MF.getSubtarget().getRegisterInfo()) {}
  bool run();
};

class SpillStoreEliminationLegacy : public MachineFunctionPass {
public:
  static char ID;
  SpillStoreEliminationLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!EnableSpillStoreElimination || skipFunction(MF.getFunction()))
      return false;
    return SpillStoreElimination(MF).run();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }
};

} // namespace

char SpillStoreEliminationLegacy::ID = 0;
char &llvm::SpillStoreEliminationID = SpillStoreEliminationLegacy::ID;

INITIALIZE_PASS(SpillStoreEliminationLegacy, DEBUG_TYPE,
                "Spill Store Elimination", false, false)

bool SpillStoreElimination::getAccess(const MachineInstr &MI, bool IsLoad,
                                      StackAccess &Access) const {
  if (MI.isBundled() || MI.getFlag(MachineInstr::FrameSetup) ||
      MI.getFlag(MachineInstr::FrameDestroy) || MI.hasOrderedMemoryRef() ||
      MI.memoperands().size() != 1 || MI.mayLoad() != IsLoad ||
      MI.mayStore() == IsLoad)
    return false;

  // The target queries certify a pure stack access, even when a spill pseudo
  // carries a conservative unmodeled-side-effects flag.
  Access.Bytes = TypeSize::getZero();
  Access.Reg = IsLoad ? TII.isLoadFromStackSlot(MI, Access.FI, Access.Bytes)
                      : TII.isStoreToStackSlot(MI, Access.FI, Access.Bytes);
  if (!Access.Reg || !Access.Reg.isPhysical() || !Access.Bytes ||
      Access.Bytes.isScalable() || Access.FI < 0 ||
      !MF.getFrameInfo().isSpillSlotObjectIndex(Access.FI))
    return false;

  const MachineMemOperand &MMO = **MI.memoperands_begin();
  const auto *PSV =
      dyn_cast_or_null<FixedStackPseudoSourceValue>(MMO.getPseudoValue());
  if (!PSV || PSV->getFrameIndex() != Access.FI || MMO.getOffset() != 0 ||
      MMO.getSize() != LocationSize::precise(Access.Bytes))
    return false;

  const TargetRegisterClass *RC = TRI.getMinimalPhysRegClass(Access.Reg);
  if (TRI.getRegSizeInBits(*RC) != Access.Bytes * 8)
    return false;

  // Require a complete value and no hidden definitions. Comparing the other
  // operands below also covers target addressing and predication operands.
  bool FoundData = false;
  for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (MO.isRegMask())
      return false;
    if (!MO.isReg())
      continue;
    if (MO.getSubReg() || MO.isUndef() || MO.isInternalRead())
      return false;
    if (MO.getReg() == Access.Reg) {
      if (FoundData || MO.isDef() != IsLoad || MO.isImplicit())
        return false;
      FoundData = true;
      Access.DataOp = I;
    } else if (MO.isDef()) {
      return false;
    }
  }
  return FoundData;
}

bool SpillStoreElimination::clobbersInputs(const MachineInstr &MI,
                                           const MachineInstr &Store) const {
  for (const MachineOperand &MO : Store.operands()) {
    if (MO.isReg() && MO.getReg() && MI.modifiesRegister(MO.getReg(), &TRI))
      return true;
  }
  return false;
}

MachineInstr *
SpillStoreElimination::findReload(MachineBasicBlock &Pred,
                                  const MachineInstr &Store,
                                  const StackAccess &Access) const {
  unsigned Count = 0;
  for (MachineInstr &MI : llvm::reverse(Pred)) {
    if (++Count > 64 || MI.isBundled() || MI.isCall() || MI.isInlineAsm() ||
        MI.getFlag(MachineInstr::FrameSetup) ||
        MI.getFlag(MachineInstr::FrameDestroy) || MI.mayStore() ||
        MI.hasOrderedMemoryRef())
      return nullptr;
    // Debug uses are not included in physical register liveness.
    if (MI.isDebugInstr() && MI.readsRegister(Access.Reg, &TRI))
      return nullptr;

    StackAccess Load;
    bool IsReload = getAccess(MI, true, Load);
    if (MI.hasUnmodeledSideEffects() && !IsReload)
      return nullptr;
    if (IsReload && Load.Reg == Access.Reg && Load.FI == Access.FI &&
        Load.Bytes == Access.Bytes &&
        MI.getNumOperands() == Store.getNumOperands()) {
      SmallVector<const MachineOperand *, 8> LoadOps, StoreOps;
      for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I) {
        if (I != Load.DataOp)
          LoadOps.push_back(&MI.getOperand(I));
        if (I != Access.DataOp)
          StoreOps.push_back(&Store.getOperand(I));
      }
      if (llvm::all_of(llvm::zip(LoadOps, StoreOps), [](const auto &Ops) {
            return std::get<0>(Ops)->isIdenticalTo(*std::get<1>(Ops));
          }))
        return &MI;
    }
    if (clobbersInputs(MI, Store))
      return nullptr;
  }
  return nullptr;
}

bool SpillStoreElimination::processBlock(MachineBasicBlock &MBB) {
  // Two single-successor predecessors permit partial elimination without edge
  // splitting or duplicating a store. No probability arithmetic is needed,
  // including on targets with divergent control flow.
  if (MBB.pred_size() != 2 || MBB.isEHPad() || MBB.isEHScopeReturnBlock())
    return false;
  for (MachineBasicBlock *Pred : MBB.predecessors()) {
    if (Pred == &MBB || Pred->succ_size() != 1 || Pred->isEHPad())
      return false;
  }

  bool Changed = false;
  for (unsigned Count = 0; Count != 16 && !MBB.empty(); ++Count) {
    MachineInstr &Store = MBB.front();
    StackAccess Access;
    if (!getAccess(Store, false, Access))
      break;

    SmallVector<MachineInstr *, 2> Reloads;
    MachineBasicBlock *NeedsStore = nullptr;
    bool Legal = true;
    for (MachineBasicBlock *Pred : MBB.predecessors()) {
      // Moving a store before a terminator must preserve all its inputs,
      // including execution masks and stack addressing registers.
      for (MachineInstr &MI :
           llvm::make_range(Pred->getFirstTerminator(), Pred->end())) {
        if (!MI.isBranch() || MI.isBundled() || MI.hasUnmodeledSideEffects() ||
            MI.mayLoadOrStore() || clobbersInputs(MI, Store)) {
          Legal = false;
          break;
        }
      }
      if (!Legal)
        break;
      if (MachineInstr *Load = findReload(*Pred, Store, Access))
        Reloads.push_back(Load);
      else
        NeedsStore = Pred;
    }
    if (!Legal || Reloads.empty())
      break;

    if (NeedsStore) {
      MachineInstr *Clone = MF.CloneMachineInstr(&Store);
      Clone->clearKillInfo();
      NeedsStore->insert(NeedsStore->getFirstTerminator(), Clone);
    }
    Store.eraseFromParent();
    ++NumStores;
    Changed = true;
    recomputeLiveIns(MBB);

    for (MachineInstr *Load : Reloads) {
      MachineBasicBlock &Pred = *Load->getParent();
      LiveRegUnits Live(TRI);
      Live.addLiveOuts(Pred);
      for (MachineInstr &MI : llvm::reverse(Pred)) {
        if (&MI == Load)
          break;
        Live.stepBackward(MI);
      }
      // Keep register-based debug locations intact until there is a debug
      // salvage strategy for them.
      if (Live.available(Access.Reg) && !HasDebugInstrs) {
        Load->eraseFromParent();
        ++NumReloads;
      }
    }
    for (MachineBasicBlock *Pred : MBB.predecessors())
      recomputeLivenessFlags(*Pred);
  }
  return Changed;
}

bool SpillStoreElimination::run() {
  if (!MF.getProperties().hasNoVRegs() ||
      !MF.getProperties().hasTracksLiveness() || MF.exposesReturnsTwice())
    return false;
  HasDebugInstrs = llvm::any_of(MF, [](const MachineBasicBlock &MBB) {
    return llvm::any_of(
        MBB, [](const MachineInstr &MI) { return MI.isDebugInstr(); });
  });
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= processBlock(MBB);
  return Changed;
}

PreservedAnalyses
SpillStoreEliminationPass::run(MachineFunction &MF,
                               MachineFunctionAnalysisManager &MFAM) {
  if (!EnableSpillStoreElimination || !SpillStoreElimination(MF).run())
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
