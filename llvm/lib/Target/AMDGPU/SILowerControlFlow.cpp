//===-- SILowerControlFlow.cpp - Use predicates for control flow ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass lowers the pseudo control flow instructions to real
/// machine instructions.
///
/// All control flow is handled using predicated instructions and
/// a predicate stack.  Each Scalar ALU controls the operations of 64 Vector
/// ALUs.  The Scalar ALU can update the predicate for any of the Vector ALUs
/// by writing to the 64-bit EXEC register (each bit corresponds to a
/// single vector ALU).  Typically, for predicates, a vector ALU will write
/// to its bit of the VCC register (like EXEC VCC is 64-bits, one for each
/// Vector ALU) and then the ScalarALU will AND the VCC register with the
/// EXEC to update the predicates.
///
/// For example:
/// %vcc = V_CMP_GT_F32 %vgpr1, %vgpr2
/// %sgpr0 = SI_IF %vcc
///   %vgpr0 = V_ADD_F32 %vgpr0, %vgpr0
/// %sgpr0 = SI_ELSE %sgpr0
///   %vgpr0 = V_SUB_F32 %vgpr0, %vgpr0
/// SI_WAVE_RECONVERGE %sgpr0
///
/// becomes:
///
/// %sgpr0 = S_AND_SAVEEXEC_B64 %vcc  // Save and update the exec mask
/// %sgpr0 = S_XOR_B64 %sgpr0, %exec  // Clear live bits from saved exec mask
/// S_CBRANCH_EXECZ label0            // This instruction is an optional
///                                   // optimization which allows us to
///                                   // branch if all the bits of
///                                   // EXEC are zero.
/// %vgpr0 = V_ADD_F32 %vgpr0, %vgpr0 // Do the IF block of the branch
///
/// label0:
/// %sgpr0 = S_OR_SAVEEXEC_B64 %sgpr0  // Restore the exec mask for the Then
///                                    // block
/// %exec = S_XOR_B64 %sgpr0, %exec    // Update the exec mask
/// S_BRANCH_EXECZ label1              // Use our branch optimization
///                                    // instruction again.
/// %vgpr0 = V_SUB_F32 %vgpr0, %vgpr   // Do the THEN block
/// label1:
/// %exec = S_OR_B64 %exec, %sgpr0     // Re-enable saved exec mask bits
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "si-lower-control-flow"

static cl::opt<bool>
RemoveRedundantEndcf("amdgpu-remove-redundant-endcf",
    cl::init(true), cl::ReallyHidden);

namespace {

class SILowerControlFlow : public MachineFunctionPass {
private:
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  LiveIntervals *LIS = nullptr;
  LiveVariables *LV = nullptr;
  MachineDominatorTree *MDT = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  SetVector<MachineInstr*> LoweredWaveReconverge;
  DenseSet<Register> LoweredIf;
  SmallSet<MachineBasicBlock *, 4> KillBlocks;
  SmallSet<Register, 8> RecomputeRegs;

  const TargetRegisterClass *BoolRC = nullptr;
  uint64_t TestMask;
  unsigned Select;
  unsigned CmovOpc;
  unsigned CmpOpc;
  unsigned AndOpc;
  unsigned AndTermOpc;
  unsigned Andn2Opc;
  unsigned OrOpc;
  unsigned XorOpc;
  unsigned MovTermOpc;
  unsigned Andn2TermOpc;
  unsigned XorTermrOpc;
  unsigned OrTermrOpc;
  unsigned OrSaveExecOpc;
  unsigned Exec;

  bool EnableOptimizeEndCf = false;

  bool hasKill(const MachineBasicBlock *Begin, const MachineBasicBlock *End);

  void emitIf(MachineInstr &MI);
  void emitElse(MachineInstr &MI);
  void emitIfBreak(MachineInstr &MI);
  void emitLoop(MachineInstr &MI);
  void emitWaterfallLoop(MachineInstr &MI);
  void emitWaveDiverge(MachineInstr &MI, Register EnabledLanesMask,
                       Register DisableLanesMask, bool IsIf);

  void emitWaveReconverge(MachineInstr &MI);

  void lowerInitExec(MachineBasicBlock *MBB, MachineInstr &MI);

  void findMaskOperands(MachineInstr &MI, unsigned OpNo,
                        SmallVectorImpl<MachineOperand> &Src) const;

  void combineMasks(MachineInstr &MI);

  bool needMaskWithExec(const MachineInstr &MI) const;

  MachineBasicBlock *process(MachineInstr &MI);

  // Skip to the next instruction, ignoring debug instructions, and trivial
  // block boundaries (blocks that have one (typically fallthrough) successor,
  // and the successor has one predecessor.
  MachineBasicBlock::iterator
  skipIgnoreExecInstsTrivialSucc(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator It) const;

  /// Find the insertion point for a new conditional branch.
  MachineBasicBlock::iterator
  skipToUncondBrOrEnd(MachineBasicBlock &MBB,
                      MachineBasicBlock::iterator I) const {
    assert(I->isTerminator());

    // FIXME: What if we had multiple pre-existing conditional branches?
    MachineBasicBlock::iterator End = MBB.end();
    while (I != End && !I->isUnconditionalBranch())
      ++I;
    return I;
  }

  void optimizeWaveReconverge();

public:
  static char ID;

  SILowerControlFlow() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI Lower control flow pseudo instructions";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addUsedIfAvailable<LiveIntervals>();
    // Should preserve the same set that TwoAddressInstructions does.
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<SlotIndexes>();
    AU.addPreserved<LiveIntervals>();
    AU.addPreservedID(LiveVariablesID);
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char SILowerControlFlow::ID = 0;

INITIALIZE_PASS(SILowerControlFlow, DEBUG_TYPE,
               "SI lower control flow", false, false)

char &llvm::SILowerControlFlowID = SILowerControlFlow::ID;

bool SILowerControlFlow::hasKill(const MachineBasicBlock *Begin,
                                 const MachineBasicBlock *End) {
  DenseSet<const MachineBasicBlock*> Visited;
  SmallVector<MachineBasicBlock *, 4> Worklist(Begin->successors());

  while (!Worklist.empty()) {
    MachineBasicBlock *MBB = Worklist.pop_back_val();

    if (MBB == End || !Visited.insert(MBB).second)
      continue;
    if (KillBlocks.contains(MBB))
      return true;

    Worklist.append(MBB->succ_begin(), MBB->succ_end());
  }

  return false;
}

static bool isSimpleIf(const MachineInstr &MI, const MachineRegisterInfo *MRI) {
  Register SaveExecReg = MI.getOperand(0).getReg();
  auto U = MRI->use_instr_nodbg_begin(SaveExecReg);

  if (U == MRI->use_instr_nodbg_end() ||
      std::next(U) != MRI->use_instr_nodbg_end() ||
      U->getOpcode() != AMDGPU::SI_WAVE_RECONVERGE)
    return false;

  return true;
}

void SILowerControlFlow::emitIf(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();
  MachineBasicBlock::iterator I(&MI);
  Register MaskElse = MI.getOperand(0).getReg();
  MachineOperand &Cond = MI.getOperand(1);
  assert(Cond.getSubReg() == AMDGPU::NoSubRegister);
  Register CondReg = Cond.getReg();
  if (!needMaskWithExec(MI))
    return emitWaveDiverge(MI, CondReg, MaskElse, true);

  Register MaskThen = MRI->createVirtualRegister(BoolRC);
  // Get rid of the garbage bits in the Cond register which might be coming from
  // the bitwise arithmetic when one of the expression operands is coming from
  // the outer scope and hence having extra bits set.
  MachineInstr *CondFiltered = BuildMI(MBB, I, DL, TII->get(AndOpc), MaskThen)
                                   .add(Cond)
                                   .addReg(Exec);
  if (LV)
    LV->replaceKillInstruction(CondReg, MI, *CondFiltered);

  emitWaveDiverge(MI, MaskThen, MaskElse, true);

  if (LIS) {
    LIS->InsertMachineInstrInMaps(*CondFiltered);
    LIS->createAndComputeVirtRegInterval(MaskThen);
  }
}

void SILowerControlFlow::emitElse(MachineInstr &MI) {
  Register InvCondReg = MI.getOperand(0).getReg();
  Register CondReg = MI.getOperand(1).getReg();
  emitWaveDiverge(MI, CondReg, InvCondReg, false);
}

void SILowerControlFlow::emitIfBreak(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();
  auto Dst = MI.getOperand(0).getReg();

  // Skip ANDing with exec if the break condition is already masked by exec
  // because it is a V_CMP in the same basic block. (We know the break
  // condition operand was an i1 in IR, so if it is a VALU instruction it must
  // be one with a carry-out.)
  // AND the break condition operand with exec, then OR that into the "loop
  // exit" mask.
  MachineInstr *And = nullptr, *Or = nullptr;
  Register AndReg;
  if (needMaskWithExec(MI)) {
    AndReg = MRI->createVirtualRegister(BoolRC);
    And = BuildMI(MBB, &MI, DL, TII->get(AndOpc), AndReg)
             .addReg(Exec)
             .add(MI.getOperand(1));
    if (LV)
      LV->replaceKillInstruction(MI.getOperand(1).getReg(), MI, *And);
    Or = BuildMI(MBB, &MI, DL, TII->get(OrOpc), Dst)
             .addReg(AndReg)
             .add(MI.getOperand(2));
  } else {
    Or = BuildMI(MBB, &MI, DL, TII->get(OrOpc), Dst)
             .add(MI.getOperand(1))
             .add(MI.getOperand(2));
    if (LV)
      LV->replaceKillInstruction(MI.getOperand(1).getReg(), MI, *Or);
  }
  if (LV)
    LV->replaceKillInstruction(MI.getOperand(2).getReg(), MI, *Or);

  if (LIS) {
    LIS->ReplaceMachineInstrInMaps(MI, *Or);
    if (And) {
      // Read of original operand 1 is on And now not Or.
      RecomputeRegs.insert(And->getOperand(2).getReg());
      LIS->InsertMachineInstrInMaps(*And);
      LIS->createAndComputeVirtRegInterval(AndReg);
    }
  }

  MI.eraseFromParent();
}

void SILowerControlFlow::emitLoop(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();

  Register Cond = MI.getOperand(0).getReg();
  Register MaskLoop = MRI->createVirtualRegister(BoolRC);
  Register AndZero = MRI->createVirtualRegister(BoolRC);

  MachineInstr *CondLoop = BuildMI(MBB, &MI, DL, TII->get(Andn2Opc), MaskLoop)
                               .addReg(Exec)
                               .addReg(Cond);

  MachineInstr *SetExec= BuildMI(MBB, &MI, DL, TII->get(Select), Exec)
                                     .addReg(MaskLoop)
                                     .addReg(Cond);

  if (LV)
    LV->replaceKillInstruction(MI.getOperand(0).getReg(), MI, *SetExec);

  auto BranchPt = skipToUncondBrOrEnd(MBB, MI.getIterator());
  MachineInstr *Branch =
      BuildMI(MBB, BranchPt, DL, TII->get(AMDGPU::S_CBRANCH_SCC1))
          .add(MI.getOperand(1));

  if (LIS) {
    RecomputeRegs.insert(MI.getOperand(0).getReg());
    LIS->ReplaceMachineInstrInMaps(MI, *SetExec);
    LIS->InsertMachineInstrInMaps(*CondLoop);
    LIS->InsertMachineInstrInMaps(*Branch);
    LIS->createAndComputeVirtRegInterval(MaskLoop);
    LIS->createAndComputeVirtRegInterval(AndZero);
  }

  MI.eraseFromParent();
}

void SILowerControlFlow::emitWaterfallLoop(MachineInstr &MI) {
  Register LoopMask = MI.getOperand(0).getReg();
  Register ExitMask = MI.getOperand(1).getReg();
  MachineBasicBlock *LoopBB = MI.getOperand(2).getMBB();
  MachineBasicBlock *BodyBB = MI.getParent();
  MachineBasicBlock::iterator I = BodyBB->end();
  const DebugLoc DL = MI.getDebugLoc();

  MachineInstr *UpdateExec = BuildMI(*BodyBB, I, DL, TII->get(Select), Exec)
      .addReg(LoopMask)
      .addReg(ExitMask);

  MachineInstr *Branch = BuildMI(*BodyBB, I, DL, TII->get(AMDGPU::S_CBRANCH_SCC1)).addMBB(LoopBB);

  if (LIS) {
    RecomputeRegs.insert(MI.getOperand(0).getReg());
    RecomputeRegs.insert(MI.getOperand(1).getReg());
    LIS->ReplaceMachineInstrInMaps(MI, *UpdateExec);
    LIS->InsertMachineInstrInMaps(*Branch);
  }

  MI.eraseFromParent();
}

void SILowerControlFlow::emitWaveDiverge(MachineInstr &MI,
                                         Register EnabledLanesMask,
                                         Register DisableLanesMask, bool IsIf) {

  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction *MF = MBB.getParent();
  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  bool IsWave32 = ST.isWave32();
  const DebugLoc &DL = MI.getDebugLoc();
  MachineBasicBlock::iterator I(MI);

  bool NeedXor = true;
  if (IsIf) {
    // If there is only one use of save exec register and that use is
    // SI_END_CF, we can optimize SI_IF by returning the full saved exec mask
    // instead of just cleared bits.
    bool SimpleIf = isSimpleIf(MI, MRI);

    if (SimpleIf) {
      // Check for SI_KILL_*_TERMINATOR on path from if to endif.
      // if there is any such terminator simplifications are not safe.
      auto UseMI = MRI->use_instr_nodbg_begin(DisableLanesMask);
      SimpleIf = !hasKill(MI.getParent(), UseMI->getParent());
    }
    NeedXor = !SimpleIf;
  }

  if (NeedXor) {

    MachineInstr *CondInverted =
        BuildMI(MBB, I, DL, TII->get(XorOpc), DisableLanesMask)
            .addReg(EnabledLanesMask)
            .addReg(Exec);

    if (LV) {
      LV->replaceKillInstruction(DisableLanesMask, MI, *CondInverted);
    }

    if (LIS) {
      LIS->InsertMachineInstrInMaps(*CondInverted);
    }
  } else {
    MachineInstr *CopyExec =
        BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), DisableLanesMask)
            .addReg(Exec);
    if (LIS)
      LIS->InsertMachineInstrInMaps(*CopyExec);
    LoweredIf.insert(DisableLanesMask);
  }

  MachineBasicBlock::reverse_iterator J(I);
  while (J != MBB.rend() && !J->definesRegister(AMDGPU::SCC, TRI))
    ++J;
  MachineInstr *EnabledLanesMaskDef = MRI->getUniqueVRegDef(EnabledLanesMask);
  bool SkipSetSCC =
      J != MBB.rend() && EnabledLanesMaskDef &&
      EnabledLanesMaskDef->getParent() == &MBB &&
      EnabledLanesMaskDef->definesRegister(EnabledLanesMask, TRI) &&
      EnabledLanesMaskDef == &*J;
  if (!SkipSetSCC) {
    MachineInstr *IfZeroMask = nullptr;
    bool HasCmpOpc = IsWave32 || AMDGPU::isGFX8Plus(ST);
    if (HasCmpOpc) {
      IfZeroMask = BuildMI(MBB, I, DL, TII->get(CmpOpc))
                       .addReg(EnabledLanesMask)
                       .addImm(0);
      if (LIS)
        LIS->InsertMachineInstrInMaps(*IfZeroMask);
    } else {
      Register TestResultReg = MRI->createVirtualRegister(BoolRC);
      IfZeroMask = BuildMI(MBB, I, DL, TII->get(AndOpc), TestResultReg)
                       .addReg(EnabledLanesMask)
                       .addImm(TestMask);
      if (LIS) {
        LIS->InsertMachineInstrInMaps(*IfZeroMask);
        LIS->createAndComputeVirtRegInterval(TestResultReg);
      }
    }
  }

  MachineInstr *SetExecForSucc =
      BuildMI(MBB, I, DL, TII->get(CmovOpc), Exec).addReg(EnabledLanesMask);

  MachineBasicBlock *FlowBB = MI.getOperand(2).getMBB();
  MachineBasicBlock *TargetBB = nullptr;
  // determine target BBs
  I = skipToUncondBrOrEnd(MBB, I);
  if (I != MBB.end()) {
    // skipToUncondBrOrEnd returns either unconditional branch or end()
    TargetBB = I->getOperand(0).getMBB();
    I->getOperand(0).setMBB(FlowBB);
  } else {
    // assert(MBB.succ_size() == 2);
    for (auto Succ : successors(&MBB)) {
      if (Succ != FlowBB) {
        TargetBB = Succ;
        break;
      }
    }
    I = BuildMI(MBB, I, DL, TII->get(AMDGPU::S_BRANCH)).addMBB(FlowBB);
    if (LIS)
      LIS->InsertMachineInstrInMaps(*I);
  }

  if (TargetBB) {
    MachineInstr *NewBr =
        BuildMI(MBB, I, DL, TII->get(AMDGPU::S_CBRANCH_SCC1)).addMBB(TargetBB);
    if (LIS)
      LIS->InsertMachineInstrInMaps(*NewBr);
  }

  if (!LIS) {
    MI.eraseFromParent();
    return;
  }

  LIS->ReplaceMachineInstrInMaps(MI, *SetExecForSucc);

  RecomputeRegs.insert(MI.getOperand(0).getReg());
  RecomputeRegs.insert(MI.getOperand(1).getReg());

  MI.eraseFromParent();

  LIS->removeAllRegUnitsForPhysReg(Exec);
}

void SILowerControlFlow::emitWaveReconverge(MachineInstr &MI) { 

  MachineBasicBlock &BB = *MI.getParent();
  Register Mask = MI.getOperand(0).getReg();

  MachineInstr *ExecRestore =
      BuildMI(BB, MI, MI.getDebugLoc(), TII->get(OrTermrOpc), Exec)
          .addReg(Exec)
          .addReg(Mask);
  if (LV)
    LV->replaceKillInstruction(Mask, MI, *ExecRestore);

  if (LIS)
    LIS->ReplaceMachineInstrInMaps(MI, *ExecRestore);

  LoweredWaveReconverge.insert(ExecRestore);

  MI.eraseFromParent();
}

// Returns replace operands for a logical operation, either single result
// for exec or two operands if source was another equivalent operation.
void SILowerControlFlow::findMaskOperands(MachineInstr &MI, unsigned OpNo,
       SmallVectorImpl<MachineOperand> &Src) const {
  MachineOperand &Op = MI.getOperand(OpNo);
  if (!Op.isReg() || !Op.getReg().isVirtual()) {
    Src.push_back(Op);
    return;
  }

  MachineInstr *Def = MRI->getUniqueVRegDef(Op.getReg());
  if (!Def || Def->getParent() != MI.getParent() ||
      !(Def->isFullCopy() || (Def->getOpcode() == MI.getOpcode())))
    return;

  // Make sure we do not modify exec between def and use.
  // A copy with implicitly defined exec inserted earlier is an exclusion, it
  // does not really modify exec.
  for (auto I = Def->getIterator(); I != MI.getIterator(); ++I)
    if (I->modifiesRegister(AMDGPU::EXEC, TRI) &&
        !(I->isCopy() && I->getOperand(0).getReg() != Exec))
      return;

  for (const auto &SrcOp : Def->explicit_operands())
    if (SrcOp.isReg() && SrcOp.isUse() &&
        (SrcOp.getReg().isVirtual() || SrcOp.getReg() == Exec))
      Src.push_back(SrcOp);
}

// Search and combine pairs of equivalent instructions, like
// S_AND_B64 x, (S_AND_B64 x, y) => S_AND_B64 x, y
// S_OR_B64  x, (S_OR_B64  x, y) => S_OR_B64  x, y
// One of the operands is exec mask.
void SILowerControlFlow::combineMasks(MachineInstr &MI) {
  assert(MI.getNumExplicitOperands() == 3);
  SmallVector<MachineOperand, 4> Ops;
  unsigned OpToReplace = 1;
  findMaskOperands(MI, 1, Ops);
  if (Ops.size() == 1) OpToReplace = 2; // First operand can be exec or its copy
  findMaskOperands(MI, 2, Ops);
  if (Ops.size() != 3) return;

  unsigned UniqueOpndIdx;
  if (Ops[0].isIdenticalTo(Ops[1])) UniqueOpndIdx = 2;
  else if (Ops[0].isIdenticalTo(Ops[2])) UniqueOpndIdx = 1;
  else if (Ops[1].isIdenticalTo(Ops[2])) UniqueOpndIdx = 1;
  else return;

  Register Reg = MI.getOperand(OpToReplace).getReg();
  MI.removeOperand(OpToReplace);
  MI.addOperand(Ops[UniqueOpndIdx]);
  if (MRI->use_empty(Reg))
    MRI->getUniqueVRegDef(Reg)->eraseFromParent();
}

bool SILowerControlFlow::needMaskWithExec(const MachineInstr &MI) const {
  bool SkipAnding = false;
  if (MI.getOperand(1).isReg()) {
    if (MachineInstr *Def = MRI->getUniqueVRegDef(MI.getOperand(1).getReg())) {
      SkipAnding =
          Def->getParent() == MI.getParent() && SIInstrInfo::isVALU(*Def);
    }
  }
  return !SkipAnding;
}

MachineBasicBlock *SILowerControlFlow::process(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  MachineBasicBlock::iterator I(MI);
  MachineInstr *Prev = (I != MBB.begin()) ? &*(std::prev(I)) : nullptr;

  MachineBasicBlock *SplitBB = &MBB;

  switch (MI.getOpcode()) {
  case AMDGPU::SI_IF:
    emitIf(MI);
    break;

  case AMDGPU::SI_ELSE:
    emitElse(MI);
    break;

  case AMDGPU::SI_IF_BREAK:
    emitIfBreak(MI);
    break;

  case AMDGPU::SI_LOOP:
    emitLoop(MI);
    break;

  case AMDGPU::SI_WATERFALL_LOOP:
    emitWaterfallLoop(MI);
    break;

  case AMDGPU::SI_WAVE_RECONVERGE:
    emitWaveReconverge(MI);
    break;

  default:
    assert(false && "Attempt to process unsupported instruction");
    break;
  }

  MachineBasicBlock::iterator Next;
  for (I = Prev ? Prev->getIterator() : MBB.begin(); I != MBB.end(); I = Next) {
    Next = std::next(I);
    MachineInstr &MaskMI = *I;
    switch (MaskMI.getOpcode()) {
    case AMDGPU::S_AND_B64:
    case AMDGPU::S_OR_B64:
    case AMDGPU::S_AND_B32:
    case AMDGPU::S_OR_B32:
      // Cleanup bit manipulations on exec mask
      combineMasks(MaskMI);
      break;
    default:
      I = MBB.end();
      break;
    }
  }

  return SplitBB;
}

MachineBasicBlock::iterator SILowerControlFlow::skipIgnoreExecInstsTrivialSucc(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator It) const {
  SmallSet<const MachineBasicBlock *, 4> Visited;
  MachineBasicBlock *B = &MBB;
  do {
    if (!Visited.insert(B).second)
      return MBB.end();

    auto E = B->end();
    for (; It != E; ++It) {
      if (TII->mayReadEXEC(*MRI, *It))
        break;
    }

    if (It != E)
      return It;

    if (B->succ_size() != 1)
      return MBB.end();

    // If there is one trivial successor, advance to the next block.
    MachineBasicBlock *Succ = *B->succ_begin();

    It = Succ->begin();
    B = Succ;
  } while (true);
}

void SILowerControlFlow::lowerInitExec(MachineBasicBlock *MBB,
                                       MachineInstr &MI) {
  MachineFunction &MF = *MBB->getParent();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  bool IsWave32 = ST.isWave32();

  if (MI.getOpcode() == AMDGPU::SI_INIT_EXEC) {
    // This should be before all vector instructions.
    MachineInstr *InitMI = BuildMI(*MBB, MBB->begin(), MI.getDebugLoc(),
            TII->get(IsWave32 ? AMDGPU::S_MOV_B32 : AMDGPU::S_MOV_B64), Exec)
        .addImm(MI.getOperand(0).getImm());
    if (LIS) {
      LIS->RemoveMachineInstrFromMaps(MI);
      LIS->InsertMachineInstrInMaps(*InitMI);
    }
    MI.eraseFromParent();
    return;
  }

  // Extract the thread count from an SGPR input and set EXEC accordingly.
  // Since BFM can't shift by 64, handle that case with CMP + CMOV.
  //
  // S_BFE_U32 count, input, {shift, 7}
  // S_BFM_B64 exec, count, 0
  // S_CMP_EQ_U32 count, 64
  // S_CMOV_B64 exec, -1
  Register InputReg = MI.getOperand(0).getReg();
  MachineInstr *FirstMI = &*MBB->begin();
  if (InputReg.isVirtual()) {
    MachineInstr *DefInstr = MRI->getVRegDef(InputReg);
    assert(DefInstr && DefInstr->isCopy());
    if (DefInstr->getParent() == MBB) {
      if (DefInstr != FirstMI) {
        // If the `InputReg` is defined in current block, we also need to
        // move that instruction to the beginning of the block.
        DefInstr->removeFromParent();
        MBB->insert(FirstMI, DefInstr);
        if (LIS)
          LIS->handleMove(*DefInstr);
      } else {
        // If first instruction is definition then move pointer after it.
        FirstMI = &*std::next(FirstMI->getIterator());
      }
    }
  }

  // Insert instruction sequence at block beginning (before vector operations).
  const DebugLoc DL = MI.getDebugLoc();
  const unsigned WavefrontSize = ST.getWavefrontSize();
  const unsigned Mask = (WavefrontSize << 1) - 1;
  Register CountReg = MRI->createVirtualRegister(&AMDGPU::SGPR_32RegClass);
  auto BfeMI = BuildMI(*MBB, FirstMI, DL, TII->get(AMDGPU::S_BFE_U32), CountReg)
                   .addReg(InputReg)
                   .addImm((MI.getOperand(1).getImm() & Mask) | 0x70000);
  if (LV)
    LV->recomputeForSingleDefVirtReg(InputReg);
  auto BfmMI =
      BuildMI(*MBB, FirstMI, DL,
              TII->get(IsWave32 ? AMDGPU::S_BFM_B32 : AMDGPU::S_BFM_B64), Exec)
          .addReg(CountReg)
          .addImm(0);
  auto CmpMI = BuildMI(*MBB, FirstMI, DL, TII->get(AMDGPU::S_CMP_EQ_U32))
                   .addReg(CountReg, RegState::Kill)
                   .addImm(WavefrontSize);
  if (LV)
    LV->getVarInfo(CountReg).Kills.push_back(CmpMI);
  auto CmovMI =
      BuildMI(*MBB, FirstMI, DL,
              TII->get(IsWave32 ? AMDGPU::S_CMOV_B32 : AMDGPU::S_CMOV_B64),
              Exec)
          .addImm(-1);

  if (!LIS) {
    MI.eraseFromParent();
    return;
  }

  LIS->RemoveMachineInstrFromMaps(MI);
  MI.eraseFromParent();

  LIS->InsertMachineInstrInMaps(*BfeMI);
  LIS->InsertMachineInstrInMaps(*BfmMI);
  LIS->InsertMachineInstrInMaps(*CmpMI);
  LIS->InsertMachineInstrInMaps(*CmovMI);

  RecomputeRegs.insert(InputReg);
  LIS->createAndComputeVirtRegInterval(CountReg);
}

void SILowerControlFlow::optimizeWaveReconverge() {
  // If the only instruction immediately following this END_CF is another
  // END_CF in the only successor we can avoid emitting exec mask restore here.
  if (!EnableOptimizeEndCf)
    return;

  for (MachineInstr *MI : reverse(LoweredWaveReconverge)) {
    MachineBasicBlock &MBB = *MI->getParent();
    auto Next =
        skipIgnoreExecInstsTrivialSucc(MBB, std::next(MI->getIterator()));
    if (Next == MBB.end() || !LoweredWaveReconverge.count(&*Next))
      continue;
    // Only skip inner END_CF if outer ENDCF belongs to SI_IF.
    // If that belongs to SI_ELSE then saved mask has an inverted value.
    Register SavedExec =
        TII->getNamedOperand(*Next, AMDGPU::OpName::src1)->getReg();
    assert(SavedExec.isVirtual() && "Expected saved exec to be src1!");

    const MachineInstr *Def = MRI->getUniqueVRegDef(SavedExec);
    if (Def && LoweredIf.count(SavedExec)) {
      LLVM_DEBUG(dbgs() << "Skip redundant "; MI->dump());
      if (LIS)
        LIS->RemoveMachineInstrFromMaps(*MI);
      Register Reg;
      if (LV)
        Reg = TII->getNamedOperand(*MI, AMDGPU::OpName::src1)->getReg();
      MI->eraseFromParent();
      if (LV)
        LV->recomputeForSingleDefVirtReg(Reg);
    }
  }
}

bool SILowerControlFlow::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  EnableOptimizeEndCf = RemoveRedundantEndcf &&
                        MF.getTarget().getOptLevel() > CodeGenOptLevel::None;

  // This doesn't actually need LiveIntervals, but we can preserve them.
  LIS = getAnalysisIfAvailable<LiveIntervals>();
  // This doesn't actually need LiveVariables, but we can preserve them.
  LV = getAnalysisIfAvailable<LiveVariables>();
  auto *MDTWrapper = getAnalysisIfAvailable<MachineDominatorTreeWrapperPass>();
  MDT = MDTWrapper ? &MDTWrapper->getDomTree() : nullptr;
  MRI = &MF.getRegInfo();
  BoolRC = TRI->getBoolRC();

  if (ST.isWave32()) {
    TestMask = 0xffffffff;
    Select = AMDGPU::S_CSELECT_B32_term;
    CmovOpc = AMDGPU::S_CMOV_B32_term;
    CmpOpc = AMDGPU::S_CMP_LG_U32_term;
    AndOpc = AMDGPU::S_AND_B32;
    AndTermOpc = AMDGPU::S_AND_B32_term;
    Andn2Opc = AMDGPU::S_ANDN2_B32;
    OrOpc = AMDGPU::S_OR_B32;
    XorOpc = AMDGPU::S_XOR_B32;
    MovTermOpc = AMDGPU::S_MOV_B32_term;
    Andn2TermOpc = AMDGPU::S_ANDN2_B32_term;
    XorTermrOpc = AMDGPU::S_XOR_B32_term;
    OrTermrOpc = AMDGPU::S_OR_B32_term;
    OrSaveExecOpc = AMDGPU::S_OR_SAVEEXEC_B32;
    Exec = AMDGPU::EXEC_LO;
  } else {
    TestMask = 0xffffffffffffffff;
    Select = AMDGPU::S_CSELECT_B64_term;
    CmovOpc = AMDGPU::S_CMOV_B64_term;
    CmpOpc = AMDGPU::S_CMP_LG_U64_term;
    AndOpc = AMDGPU::S_AND_B64;
    AndTermOpc = AMDGPU::S_AND_B64_term;
    Andn2Opc = AMDGPU::S_ANDN2_B64;
    OrOpc = AMDGPU::S_OR_B64;
    XorOpc = AMDGPU::S_XOR_B64;
    MovTermOpc = AMDGPU::S_MOV_B64_term;
    Andn2TermOpc = AMDGPU::S_ANDN2_B64_term;
    XorTermrOpc = AMDGPU::S_XOR_B64_term;
    OrTermrOpc = AMDGPU::S_OR_B64_term;
    OrSaveExecOpc = AMDGPU::S_OR_SAVEEXEC_B64;
    Exec = AMDGPU::EXEC;
  }

  // Compute set of blocks with kills
  const bool CanDemote =
      MF.getFunction().getCallingConv() == CallingConv::AMDGPU_PS;
  for (auto &MBB : MF) {
    bool IsKillBlock = false;
    for (auto &Term : MBB.terminators()) {
      if (TII->isKillTerminator(Term.getOpcode())) {
        KillBlocks.insert(&MBB);
        IsKillBlock = true;
        break;
      }
    }
    if (CanDemote && !IsKillBlock) {
      for (auto &MI : MBB) {
        if (MI.getOpcode() == AMDGPU::SI_DEMOTE_I1) {
          KillBlocks.insert(&MBB);
          break;
        }
      }
    }
  }

  bool Changed = false;
  MachineFunction::iterator NextBB;
  for (MachineFunction::iterator BI = MF.begin();
       BI != MF.end(); BI = NextBB) {
    NextBB = std::next(BI);
    MachineBasicBlock *MBB = &*BI;

    MachineBasicBlock::iterator I, E, Next;
    E = MBB->end();
    for (I = MBB->begin(); I != E; I = Next) {
      Next = std::next(I);
      MachineInstr &MI = *I;
      MachineBasicBlock *SplitMBB = MBB;

      switch (MI.getOpcode()) {
      case AMDGPU::SI_IF:
      case AMDGPU::SI_ELSE:
      case AMDGPU::SI_IF_BREAK:
      case AMDGPU::SI_WATERFALL_LOOP:
      case AMDGPU::SI_LOOP:
      case AMDGPU::SI_WAVE_RECONVERGE:
        SplitMBB = process(MI);
        Changed = true;
        break;
      }

      if (SplitMBB != MBB) {
        MBB = Next->getParent();
        E = MBB->end();
      }
    }
  }

  optimizeWaveReconverge();

  if (LIS) {
    for (Register Reg : RecomputeRegs) {
      LIS->removeInterval(Reg);
      LIS->createAndComputeVirtRegInterval(Reg);
    }
  }

  RecomputeRegs.clear();
  LoweredIf.clear();
  LoweredWaveReconverge.clear();
  KillBlocks.clear();

  return Changed;
}
