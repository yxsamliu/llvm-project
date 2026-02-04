//===- AMDGPULowerVGPREncoding.cpp - Set MODE & Lower idx Pseudos ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Lower VGPRs above first 256 on gfx1250+.
/// Also lowers dynamic VGPR indexing pseudo instructions to subtarget
/// instructions.
///
/// The pass scans used VGPRs and inserts S_SET_VGPR_MSB instructions on
/// gfx1250 (or S_SET_VGPR_FRAMES on gfx13+) to switch VGPR addressing mode. The
/// mode change is effective until the next change. This instruction provides
/// high bits of a VGPR address for four of the operands: vdst, src0, src1, and
/// src2, or other 4 operands depending on the instruction encoding. If bits are
/// set they are added as MSB to the corresponding operand VGPR number.
///
/// There is no need to replace actual register operands because encoding of the
/// high and low VGPRs is the same. I.e. v0 has the encoding 0x100, so does
/// v256. v1 has the encoding 0x101 and v257 has the same encoding. So high
/// VGPRs will survive until actual encoding and will result in a same actual
/// bit encoding.
///
/// The InstPrinter will take care of the printing a low VGPR instead of a high
/// one. In prinicple this shall be viable to print actual high VGPR numbers,
/// but that would disagree with a disasm printing and create a situation where
/// asm text is not deterministic.
///
/// Another part of the pass is lowering of dynamic VGPR indexing pseudo
/// instructions. V_LOAD/STORE_IDX are lowered to one or several V_MOV_B32,
/// and the index registers they use are encoded in a preceding update to the
/// index select bits in MODE using S_SET_VGPR_FRAMES. Dynamic indexing bundles
/// containing V_LOAD/STORE_IDXs and a CoreMI are lowered by folding
/// V_LOAD/STORE_IDX of CoreMI's operands into CoreMI, and inserting
/// S_SET_VGPR_FRAMES.
///
/// This pass creates a convention where non-fall through basic blocks shall
/// start with all MODE register bits 0. Otherwise a disassembly would not be
/// readable. An optimization here is possible but deemed not desirable because
/// of the readbility concerns.
///
/// Consequentially the ABI is set to expect all 16 MODE bits to be zero on
/// entry. The pass must run very late in the pipeline to make sure no changes
/// to VGPR operands will be made after it.
//
//===----------------------------------------------------------------------===//

#include "AMDGPULowerVGPREncoding.h"
#include "AMDGPU.h"
#include "AMDGPUMachineInstrs.h"
#include "GCNSubtarget.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-lower-vgpr-encoding"

namespace {

class AMDGPULowerVGPREncoding {
  static constexpr unsigned VSrc0 = 0, VDst = 3;
  static constexpr unsigned OpNum = 7;

  static constexpr unsigned VGPRMSBShift =
      llvm::countr_zero_constexpr<unsigned>(AMDGPU::Hwreg::DST_VGPR_MSB);

  enum class EncodeType : unsigned {
    SET_VGPR_MSB = 0,
    SET_VGPR_FRAMES = 1,
    VOPM = 2
  };

  struct OpMode {
    // No MSBs or index register set means they are not required to be
    // of a particular value.
    std::optional<unsigned> MSBits;
    std::optional<MCRegister> IdxReg;

    bool update(const OpMode &New, bool &Rewritten) {
      bool Updated = false;
      if (New.MSBits) {
        if (*New.MSBits != MSBits.value_or(0)) {
          Updated = true;
          Rewritten |= MSBits.has_value();
        }
        MSBits = New.MSBits;
      }
      if (New.IdxReg) {
        if (*New.IdxReg != IdxReg.value_or(AMDGPU::IDX0)) {
          Updated = true;
          Rewritten |= IdxReg.has_value();
        }
        IdxReg = New.IdxReg;
      }
      return Updated;
    }
  };

  struct ModeTy {
    OpMode Ops[OpNum];

    bool update(const ModeTy &New, bool &Rewritten) {
      bool Updated = false;
      for (unsigned I : seq(OpNum))
        Updated |= Ops[I].update(New.Ops[I], Rewritten);
      return Updated;
    }

    bool isSet() const {
      return any_of(Ops, [](const OpMode &Op) {
        return Op.MSBits.value_or(0) != 0 ||
               Op.IdxReg.value_or(AMDGPU::IDX0) != AMDGPU::IDX0;
      });
    }

    unsigned encode(EncodeType type) const {
      switch (type) {
      case EncodeType::SET_VGPR_FRAMES: {
        // GFX13 layout:
        // [src0 idx_sel, src1 idx_sel, src2 idx_sel, dst idx_sel,
        //  src0 msb, src1 msb, src2 msb, dst msb]
        static constexpr unsigned BitsPerField = 2;
        static constexpr unsigned MSBFieldsPos = 8;
        unsigned V = 0;
        for (const auto &[I, Op] : enumerate(Ops)) {
          MCRegister R = Op.IdxReg.value_or(AMDGPU::IDX0);
          assert(AMDGPU::IDX0 <= R && R <= AMDGPU::IDX3);
          V |= (R - AMDGPU::IDX0) << (I * BitsPerField);
          V |= Op.MSBits.value_or(0) << (I * BitsPerField + MSBFieldsPos);
        }
        return V;
      }
      case EncodeType::VOPM: {
        static constexpr unsigned BitsPerField = 4;
        // GFX13 VOPM layout in idxs operand:
        // [dst idx_sel, src0 idx_sel, src1 idx_sel, ... ]
        unsigned V = 0;
        for (const auto &[I, Op] : enumerate(Ops)) {
          MCRegister R = Op.IdxReg.value_or(AMDGPU::IDX0);
          assert(AMDGPU::IDX0 <= R && R <= AMDGPU::IDX3);
          V |= (R - AMDGPU::IDX0) << (I * BitsPerField);
        }
        return V;
      }
      default: {
        assert(type == EncodeType::SET_VGPR_MSB);
        // GFX1250 layout: [src0 msb, src1 msb, src2 msb, dst msb]
        static constexpr unsigned BitsPerField = 2;
        unsigned V = 0;
        for (const auto &[I, Op] : enumerate(Ops))
          V |= Op.MSBits.value_or(0) << (I * BitsPerField);
        return V;
      }
      }
    }
  };

public:
  bool run(MachineFunction &MF);

  struct OpMetaInfo {
    uint32_t Indices = 0;
    uint32_t AccessKinds = 0;

    void Encode(MCRegister Reg, MachineMemOperand &MMO, int OpId) {
      Indices |= ((Reg - AMDGPU::IDX0) & 0x3) << (OpId * 2);
      AccessKinds |= ((MMO.getAddrSpace() == AMDGPUAS::LANE_SHARED)
                          ? AMDGPU::IDX_LANESHARED
                          : AMDGPU::IDX_PRIVATE_ARRAY)
                     << (OpId * 2);
    }

  };

private:
  const GCNSubtarget *ST;
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  SIMachineFunctionInfo *MFI;
  const MCSubtargetInfo *STI;

  // Current basic block.
  MachineBasicBlock *MBB;

  /// Most recent s_set_* instruction.
  MachineInstr *MostRecentModeSet;

  /// Current mode values. The current mode is suitable for all instructions
  /// between the previous mode set, MostRecentModeSet, and the previous
  /// instruction. If it can be updated to include the current instruction we
  /// will do it, and if it can't we will insert a new mode set.
  ModeTy CurrentMode;

  /// Number of current hard clause instructions.
  unsigned ClauseLen;

  /// Number of hard clause instructions remaining.
  unsigned ClauseRemaining;

  /// Clause group breaks.
  unsigned ClauseBreaks;

  /// Last hard clause instruction.
  MachineInstr *Clause;

  /// Insert mode change before \p I. \returns true if mode was changed.
  bool setMode(ModeTy NewMode, MachineBasicBlock::instr_iterator I);

  /// Reset mode to default.
  void resetMode(MachineBasicBlock::instr_iterator I) {
    setMode(ModeTy{{{0, AMDGPU::IDX0},
                    {0, AMDGPU::IDX0},
                    {0, AMDGPU::IDX0},
                    {0, AMDGPU::IDX0}}},
            I);
  }

  /// If \p MO references VGPRs, return the MSBs. Otherwise, return nullopt.
  std::optional<unsigned> getMSBs(const MachineOperand &MO) const;

  /// Handle single \p MI. \return true if changed.
  /// Updates MII to point to the last instruction processed (existing or newly
  /// inserted) for mode update, so that upon increment in runOnMachineFunction
  /// MII is the correct value to process next.
  bool runOnMachineInstr(MachineBasicBlock::instr_iterator &MII);

  /// Compute and set the mode for a single \p MI given \p Ops
  /// operands bit mapping. If MI is BUNDLE, lowers the bundle by replacing the
  /// operands of CoreMI with the dynamic indices from the bundle. Optionally
  /// takes second array \p Ops2 for VOPD. If provided and an operand from \p
  /// Ops is not a VGPR, then \p Ops2 is checked.
  void lowerInstrOrBundle(MachineInstr &MI, MachineInstr *CoreMI,
                          const AMDGPU::OpName Ops[OpNum],
                          const AMDGPU::OpName *Ops2 = nullptr);

  /// Lower V_LOAD/STORE_IDX to one or several V_MOV_B32 instructions and update
  /// the mode. MII is updated to point to the last V_MOV inserted.
  void lowerIDX(MachineBasicBlock::instr_iterator &MII);

  /// Lower bundles which only contain V_LOAD/STORE_IDX, as would be used
  /// to move data, and update the mode. MII is updated to point to the
  /// last V_MOV inserted.
  void lowerMovBundle(MachineInstr &MI, MachineInstr &CoreMI,
                      MachineBasicBlock::instr_iterator &MII);

  /// Check if an instruction \p I is within a clause and returns a suitable
  /// iterator to insert mode change. It may also modify the S_CLAUSE
  /// instruction to extend it or drop the clause if it cannot be adjusted.
  MachineBasicBlock::instr_iterator
  handleClause(MachineBasicBlock::instr_iterator I);

  /// Check if an instruction \p I is immediately after another program state
  /// instruction which it cannot coissue with. If so, insert before that
  /// instruction to encourage more coissuing.
  MachineBasicBlock::instr_iterator
  handleCoissue(MachineBasicBlock::instr_iterator I);

  /// Handle S_SETREG_IMM32_B32 targeting MODE register. On certain hardware,
  /// this instruction clobbers VGPR MSB bits[12:19], so we need to restore
  /// the current mode. \returns true if the instruction was modified or a
  /// new one was inserted.
  bool handleSetregMode(MachineInstr &MI);

  /// Update bits[12:19] of the imm operand in S_SETREG_IMM32_B32 to contain
  /// the VGPR MSB mode value. \returns true if the immediate was changed.
  bool updateSetregModeImm(MachineInstr &MI, int64_t ModeValue);
};

bool AMDGPULowerVGPREncoding::setMode(ModeTy NewMode,
                                      MachineBasicBlock::instr_iterator I) {

  if (I != MBB->instr_end() && AMDGPU::isVOPMPseudo(I->getOpcode())) {

    MachineOperand *IdxsOp = TII->getNamedOperand(*I, AMDGPU::OpName::idxs);
    assert(IdxsOp && IdxsOp->getImm() == 0);
    IdxsOp->setImm(NewMode.encode(EncodeType::VOPM));

    return true;
  }

  EncodeType encodeType = ST->hasVGPRIndexingRegisters()
                              ? EncodeType::SET_VGPR_FRAMES
                              : EncodeType::SET_VGPR_MSB;

  // Record previous mode into high 8 bits of the SET_VGPR_MSB immediate.
  int64_t OldModeBits = (encodeType == EncodeType::SET_VGPR_MSB)
                            ? CurrentMode.encode(encodeType) << 8
                            : 0;

  bool Rewritten = false;
  if (!CurrentMode.update(NewMode, Rewritten))
    return false;

  if (MostRecentModeSet && !Rewritten) {
    // Update MostRecentModeSet with the new mode. It can be either
    // S_SET_VGPR_{MSB|FRAMES} or S_SETREG_IMM32_B32 (with Size <= 12).
    if (MostRecentModeSet->getOpcode() == AMDGPU::S_SET_VGPR_MSB ||
        MostRecentModeSet->getOpcode() == AMDGPU::S_SET_VGPR_FRAMES) {
      MachineOperand &Op = MostRecentModeSet->getOperand(0);

      // Carry old mode bits from the existing instruction.
      OldModeBits = (encodeType == EncodeType::SET_VGPR_MSB)
                        ? OldModeBits = Op.getImm() & 0xff00
                        : 0;

      Op.setImm(CurrentMode.encode(encodeType) | OldModeBits);
    } else {
      assert(MostRecentModeSet->getOpcode() == AMDGPU::S_SETREG_IMM32_B32 &&
             "unexpected MostRecentModeSet opcode");
      updateSetregModeImm(*MostRecentModeSet, CurrentMode.encode(encodeType));
    }
    return true;
  }

  I = handleClause(I);
  I = handleCoissue(I);
  MostRecentModeSet = BuildMI(*MBB, I, {},
                              TII->get(ST->hasVGPRIndexingRegisters()
                                           ? AMDGPU::S_SET_VGPR_FRAMES
                                           : AMDGPU::S_SET_VGPR_MSB))
                          .addImm(NewMode.encode(encodeType) | OldModeBits);

  CurrentMode = NewMode;
  return true;
}

std::optional<unsigned>
AMDGPULowerVGPREncoding::getMSBs(const MachineOperand &MO) const {
  if (!MO.isReg())
    return std::nullopt;

  MCRegister Reg = MO.getReg();
  const TargetRegisterClass *RC = TRI->getPhysRegBaseClass(Reg);
  if (!RC || !TRI->isVGPRClass(RC))
    return std::nullopt;

  unsigned Idx = TRI->getHWRegIndex(Reg);
  return Idx >> 8;
}

static void TransferImplicitVGPROperands(MachineInstr &OldMI,
                                         MachineInstr &NewMI,
                                         const SIRegisterInfo *TRI) {
  for (auto MO : OldMI.implicit_operands()) {
    if (!MO.isReg())
      continue;
    MCRegister Reg = MO.getReg();
    const TargetRegisterClass *RC = TRI->getPhysRegBaseClass(Reg);
    if (!RC || !TRI->isVGPRClass(RC))
      continue;
    auto Existing = std::find_if(NewMI.implicit_operands().begin(),
                                 NewMI.implicit_operands().end(),
                                 [&MO](MachineOperand &ExistingMO) {
                                   return MO.isIdenticalTo(ExistingMO);
                                 });
    if (Existing == NewMI.implicit_operands().end())
      NewMI.addOperand(MO);
  }
}

static void
AddMetadataForOperandIndexing(MachineInstr &MI,
                              AMDGPULowerVGPREncoding::OpMetaInfo Info) {
  LLVMContext &Ctx = MI.getMF()->getFunction().getContext();
  SmallVector<Metadata *, 3> Ops;
  // Use a unique string to identify this metadata.
  Ops.push_back(MDString::get(Ctx, "vgpr_indexing_extra"));
  Type *I32Ty = Type::getInt32Ty(Ctx);
  Ops.push_back(ConstantAsMetadata::get(ConstantInt::get(I32Ty, Info.Indices)));
  Ops.push_back(
      ConstantAsMetadata::get(ConstantInt::get(I32Ty, Info.AccessKinds)));
  MI.addOperand(MachineOperand::CreateMetadata(MDTuple::get(Ctx, Ops)));
}

void AMDGPULowerVGPREncoding::lowerMovBundle(
    MachineInstr &MI, MachineInstr &CoreMI,
    MachineBasicBlock::instr_iterator &MII) {
  auto &StoreMI = cast<AMDGPUMI::VStoreIdxInst>(CoreMI);

  // The RC in MachineInstrDesc for V_LOAD/STORE_IDX can contain many
  // possible register sizes, we need to use the MMO instead to determine size.
  assert(StoreMI.hasOneMemOperand() && "V_LOAD/STORE_IDX must have one MMO");
  MachineMemOperand *MMO = *CoreMI.memoperands_begin();
  auto Size = MMO->getSizeInBits().getValue();
  if (Size % 32 != 0)
    report_fatal_error(
        "TODO-GFX13 Support lowering non-multiple-of-32-bit sizes for "
        "V_LOAD/STORE_IDX");

  auto *LoadMI = cast<AMDGPUMI::VLoadIdxInst>(CoreMI.getPrevNode());
  assert(LoadMI->hasOneMemOperand() && "V_LOAD/STORE_IDX must have one MMO");
  MachineMemOperand *LoadMMO = *LoadMI->memoperands_begin();

#if !defined(NDEBUG)
  // Check if the value loaded by V_LOAD_IDX is the same as stored by
  // V_STORE_IDX
  auto LoadSize = LoadMMO->getSizeInBits().getValue();
  MachineOperand &DataOp = StoreMI.getDataOp();
  const auto *TRI = ST->getRegisterInfo();
  unsigned StoreDataRegNum = TRI->getHWRegIndex(DataOp.getReg());
  MachineOperand &LoadDataOp = LoadMI->getDataOp();
  unsigned LoadDataRegNum = TRI->getHWRegIndex(LoadDataOp.getReg());
  assert(LoadSize == Size && LoadDataRegNum == StoreDataRegNum &&
         "V_LOAD_IDX + V_STORE_IDX Bundle was not created correctly");
#endif

  Register StoreIdxReg = StoreMI.getIdxOp().getReg();
  unsigned StoreOffset = StoreMI.getOffsetOp().getImm();

  Register LoadIdxReg = LoadMI->getIdxOp().getReg();
  unsigned LoadOffset = LoadMI->getOffsetOp().getImm();

  ModeTy NewMode;
  NewMode.Ops[VSrc0] = {0, LoadIdxReg.asMCReg()};
  NewMode.Ops[VDst] = {0, StoreIdxReg.asMCReg()};

  OpMetaInfo OpInfo;
  OpInfo.Encode(StoreIdxReg.asMCReg(), *MMO, 0);
  OpInfo.Encode(LoadIdxReg.asMCReg(), *LoadMMO, 1);

  unsigned MaxVGPR = ST->getAddressableNumVGPRs(MFI->getDynamicVGPRBlockSize()) - 1;
  const MCInstrDesc &OpDesc = TII->get(AMDGPU::V_MOV_B32_e32);
  for (unsigned I = 0; I < Size / 32; ++I) {
    unsigned CurLoadOffset = (LoadOffset + I) & MaxVGPR;
    unsigned CurStoreOffset = (StoreOffset + I) & MaxVGPR;

    auto MIB = BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), OpDesc);
    MIB.addDef(AMDGPU::VGPR0 + CurStoreOffset)
        .addUse(AMDGPU::VGPR0 + CurLoadOffset, RegState::Undef);
    NewMode.Ops[VSrc0].MSBits = CurLoadOffset >> 8;
    NewMode.Ops[VDst].MSBits = CurStoreOffset >> 8;

    AddMetadataForOperandIndexing(*MIB.getInstr(), OpInfo);
    TransferImplicitVGPROperands(CoreMI, *MIB.getInstr(), TRI);
    TransferImplicitVGPROperands(*LoadMI, *MIB.getInstr(), TRI);

    setMode(NewMode, MIB->getIterator());
    MII = MachineBasicBlock::instr_iterator(MIB);
  }
  LoadMI->eraseFromBundle();
  CoreMI.eraseFromBundle();
  MI.eraseFromParent();
}

void AMDGPULowerVGPREncoding::lowerIDX(MachineBasicBlock::instr_iterator &MII) {
  auto &MI = cast<AMDGPUMI::VLoadStoreIdxInst>(*MII);
  bool IsLoad = isa<AMDGPUMI::VLoadIdxInst>(MI);

  // The RC in MachineInstrDesc for V_LOAD/STORE_IDX can contain many
  // possible register sizes, we need to use the MMO instead to determine size.
  assert(MI.hasOneMemOperand() && "V_LOAD/STORE_IDX must have one MMO");
  MachineMemOperand *MMO = *MI.memoperands_begin();
  auto Size = MMO->getSizeInBits().getValue();
  assert((Size % 32) == 0 &&
         "TODO-GFX13 Support lowering non-multiple-of-32-bit sizes for "
         "V_LOAD/STORE_IDX");
  const auto *TRI = ST->getRegisterInfo();
  MachineOperand &DataOp = MI.getDataOp();
  unsigned DataRegNum = TRI->getHWRegIndex(DataOp.getReg());

  Register IdxReg = MI.getIdxOp().getReg();
  unsigned Offset = MI.getOffsetOp().getImm();

  ModeTy NewMode;
  NewMode.Ops[VSrc0] = {0, AMDGPU::IDX0};
  NewMode.Ops[VDst] = {0, AMDGPU::IDX0};
  if (IsLoad)
    NewMode.Ops[VSrc0].IdxReg = IdxReg.asMCReg();
  else
    NewMode.Ops[VDst].IdxReg = IdxReg.asMCReg();

  OpMetaInfo OpInfo;
  OpInfo.Encode(IdxReg.asMCReg(), *MMO, IsLoad ? 1 : 0);

  unsigned MaxVGPR = ST->getAddressableNumVGPRs(MFI->getDynamicVGPRBlockSize()) - 1;
  const MCInstrDesc &OpDesc = TII->get(AMDGPU::V_MOV_B32_e32);
  for (unsigned i = 0; i < Size / 32; ++i) {
    unsigned CurOffset = (Offset + i) & MaxVGPR;
    unsigned CurData = DataRegNum + i;

    auto MIB = BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), OpDesc);
    if (IsLoad) {
      MIB.addDef(AMDGPU::VGPR0 + CurData)
          .addUse(AMDGPU::VGPR0 + CurOffset, RegState::Undef);

      NewMode.Ops[VSrc0].MSBits = CurOffset >> 8;
      NewMode.Ops[VDst].MSBits = CurData >> 8;
    } else {
      MIB.addDef(AMDGPU::VGPR0 + CurOffset)
          .addUse(AMDGPU::VGPR0 + CurData, getUndefRegState(DataOp.isUndef()));
      NewMode.Ops[VSrc0].MSBits = CurData >> 8;
      NewMode.Ops[VDst].MSBits = CurOffset >> 8;
    }
    AddMetadataForOperandIndexing(*MIB.getInstr(), OpInfo);
    TransferImplicitVGPROperands(MI, *MIB.getInstr(), TRI);

    setMode(NewMode, MIB->getIterator());
    MII = MachineBasicBlock::instr_iterator(MIB);
  }

  MI.eraseFromParent();
}

void AMDGPULowerVGPREncoding::lowerInstrOrBundle(
    MachineInstr &MI, MachineInstr *CoreMI, const AMDGPU::OpName Ops[OpNum],
    const AMDGPU::OpName *Ops2) {
  bool IsBundleWithGPRIndexing = CoreMI != nullptr;
  if (!CoreMI)
    CoreMI = &MI;
  ModeTy NewMode;
  OpMetaInfo OpInfo;

  // Record all the v_load/store_idx in the bundle in order to transfer
  // their implicit operands to the CoreMI after the for-loop.
  // Adding Operand in the middle of the loop may invalidate MachineOperand
  // pointers used inside the loop.
  SmallVector<MachineInstr *, 4> AllLoadStoreIdx;
  for (unsigned I = 0; I < OpNum; ++I) {
    MachineOperand *CoreOp = TII->getNamedOperand(*CoreMI, Ops[I]);

    if (CoreOp && IsBundleWithGPRIndexing && CoreOp->isReg()) {
      MachineBasicBlock::instr_iterator II = MI.getIterator();
      MachineBasicBlock::instr_iterator E = MI.getParent()->instr_end();
      while (++II != E && II->isInsideBundle()) {
        if (&*II == CoreMI)
          continue;
        AMDGPUMI::VLoadStoreIdxInst *LSI = nullptr;
        bool IsStore = false;
        if (CoreOp->isDef()) {
          LSI = dyn_cast<AMDGPUMI::VStoreIdxInst>(II);
          IsStore = true;
        } else if (CoreOp->isUse()) {
          if (!CoreOp->isInternalRead())
            continue;
          LSI = dyn_cast<AMDGPUMI::VLoadIdxInst>(II);
        }
        if (!LSI)
          continue;
        Register DataReg = LSI->getDataOp().getReg();
        if (DataReg != CoreOp->getReg())
          continue;

        // Replace CoreOp with a new register of the correct width and offset
        size_t ByteSize = AMDGPU::getRegOperandSize(STI, TII, CoreMI->getDesc(),
                                                    CoreOp->getOperandNo());
        unsigned Offset = LSI->getOffsetOp().getImm();
        assert(Offset < ST->getAddressableNumVGPRs(MFI->getDynamicVGPRBlockSize()) - ByteSize / 4);
        assert(
            !ST->needsAlignedVGPRs() || ByteSize <= 4 ||
            (Offset & 1) == 0 &&
                "Instructions with odd offsets should not have been bundled");
        CoreOp->setReg(
            TRI->getAnyVGPRClassForBitWidth(ByteSize * 8)->getRegister(Offset));
        CoreOp->setIsUndef();
        CoreOp->setIsInternalRead(false);

        NewMode.Ops[I].IdxReg = LSI->getIdxOp().getReg().asMCReg();

        MachineMemOperand *MMO = *(II->memoperands_begin());
        int CoreOpId = AMDGPU::getNamedOperandIdx(CoreMI->getOpcode(), Ops[I]);

        OpInfo.Encode(NewMode.Ops[I].IdxReg.value(), *MMO, CoreOpId);

        bool HasOtherUsers =
            AMDGPU::STAGING_REGRegClass.contains(DataReg) &&
            any_of(CoreMI->explicit_uses(), [&](const auto &Use) {
              return Use.isReg() && Use.getReg() == DataReg;
            });
        // Delete V_LOAD_IDX without other users, and V_STORE_IDX.
        if (IsStore || !HasOtherUsers) {
          --II;
          auto *IdxMI = II->getNextNode();
          AllLoadStoreIdx.push_back(IdxMI);
          IdxMI->removeFromBundle();
          // Insert it before bundle temporarily, erase it later.
          MI.getParent()->insert(MI.getIterator(), IdxMI);
        }
      }
    }

    // VOPM will not read or write the MODE register.
    if (AMDGPU::isVOPMPseudo(CoreMI->getOpcode()))
      continue;

    std::optional<unsigned> MSBits;
    if (CoreOp)
      MSBits = getMSBs(*CoreOp);

#if !defined(NDEBUG)
    if (MSBits.has_value() && Ops2) {
      auto Op2 = TII->getNamedOperand(*CoreMI, Ops2[I]);
      if (Op2) {
        std::optional<unsigned> MSBits2;
        MSBits2 = getMSBs(*Op2);
        if (MSBits2.has_value() && MSBits != MSBits2)
          llvm_unreachable("Invalid VOPD pair was created");
      }
    }
#endif

    if (!MSBits.has_value() && Ops2) {
      CoreOp = TII->getNamedOperand(*CoreMI, Ops2[I]);
      if (CoreOp)
        MSBits = getMSBs(*CoreOp);
    }

    if (!MSBits.has_value())
      continue;

    // Skip tied uses of src2 of VOP2, these will be handled along with defs and
    // only vdst bit affects these operands. We cannot skip tied uses of VOP3,
    // these uses are real even if must match the vdst.
    if (Ops[I] == AMDGPU::OpName::src2 && !CoreOp->isDef() &&
        CoreOp->isTied() &&
        (SIInstrInfo::isVOP2(*CoreMI) ||
         (SIInstrInfo::isVOP3(*CoreMI) &&
          TII->hasVALU32BitEncoding(CoreMI->getOpcode()))))
      continue;

    // Instructions with 10 bit VGPR encodings don't read MSBs.
    if (!SIInstrInfo::isVNBR(*CoreMI))
      NewMode.Ops[I].MSBits = MSBits.value();

    if (ST->hasVGPRIndexingRegisters()) {
      if (!NewMode.Ops[I].IdxReg)
        NewMode.Ops[I].IdxReg = AMDGPU::IDX0;
    }
  }

  if (IsBundleWithGPRIndexing) {
    AddMetadataForOperandIndexing(*CoreMI, OpInfo);
    for (auto IdxMI : AllLoadStoreIdx) {
      TransferImplicitVGPROperands(*IdxMI, *CoreMI, TRI);
      IdxMI->eraseFromParent();
    }

    MachineBasicBlock::instr_iterator Start(MI.getIterator());
    for (MachineBasicBlock::instr_iterator I = ++Start,
                                           E = MI.getParent()->instr_end();
         I != E && I->isBundledWithPred(); ++I) {
      assert(!isa<AMDGPUMI::VLoadStoreIdxInst>(I) &&
             "Failed to lower bundled index instruction");
      I->unbundleFromPred();
    }
    MI.eraseFromParent();
  }
  setMode(NewMode, CoreMI->getIterator());
}

bool AMDGPULowerVGPREncoding::runOnMachineInstr(
    MachineBasicBlock::instr_iterator &MII) {
  MachineInstr &MI = *MII;
  if (MI.isBundle()) {
    for (auto &BundledMI :
         make_range(std::next(MI.getIterator()), MI.getParent()->instr_end())) {
      if (!BundledMI.isBundledWithPred())
        break;
      if (SIInstrInfo::isWMMA(BundledMI) || SIInstrInfo::isSWMMAC(BundledMI) ||
          SIInstrInfo::isConvolve(BundledMI)) {
        MFI->setHasWMMAorConvolve();
        break;
      }
    }
  } else if (SIInstrInfo::isWMMA(MI) || SIInstrInfo::isSWMMAC(MI) ||
             SIInstrInfo::isConvolve(MI)) {
    MFI->setHasWMMAorConvolve();
  }

  if (isa<AMDGPUMI::VLoadStoreIdxInst>(MI)) {
    lowerIDX(MII);
    return true;
  }

  MachineInstr *CoreMI = SIInstrInfo::bundleWithGPRIndexing(MI);
  auto Ops = AMDGPU::getVGPRLoweringOperandTables(CoreMI ? CoreMI->getDesc()
                                                         : MI.getDesc());
  if (Ops.first) {
    if (CoreMI)
      MII = MachineBasicBlock::instr_iterator(CoreMI);
    lowerInstrOrBundle(MI, CoreMI, Ops.first, Ops.second);
    return true;
  }
  if (CoreMI) {
    lowerMovBundle(MI, *CoreMI, MII);
    return true;
  }
  assert(!TII->hasVGPRUses(MI) || MI.isMetaInstruction() || MI.isPseudo());

  return false;
}

MachineBasicBlock::instr_iterator
AMDGPULowerVGPREncoding::handleClause(MachineBasicBlock::instr_iterator I) {
  if (!ClauseRemaining)
    return I;

  // A clause cannot start with a special instruction, place it right before
  // the clause.
  if (ClauseRemaining == ClauseLen) {
    I = Clause->getPrevNode()->getIterator();
    assert(I->isBundle());
    return I;
  }

  // If a clause defines breaks each group cannot start with a mode change.
  // just drop the clause.
  if (ClauseBreaks) {
    Clause->eraseFromBundle();
    ClauseRemaining = 0;
    return I;
  }

  // Otherwise adjust a number of instructions in the clause if it fits.
  // If it does not clause will just become shorter. Since the length
  // recorded in the clause is one less, increment the length after the
  // update. Note that SIMM16[5:0] must be 1-62, not 0 or 63.
  if (ClauseLen < 63)
    Clause->getOperand(0).setImm(ClauseLen | (ClauseBreaks << 8));

  ++ClauseLen;

  return I;
}

MachineBasicBlock::instr_iterator
AMDGPULowerVGPREncoding::handleCoissue(MachineBasicBlock::instr_iterator I) {
  if (I.isEnd())
    return I;

  // "Program State instructions" are instructions which are used to control
  // operation of the GPU rather than performing arithmetic. Such instructions
  // have different coissuing rules w.r.t s_set_vgpr_msb.
  auto isProgramStateInstr = [this](MachineInstr *MI) {
    unsigned Opc = MI->getOpcode();
    return TII->isBarrier(Opc) || TII->isWaitcnt(Opc) ||
           Opc == AMDGPU::S_DELAY_ALU;
  };

  while (!I.isEnd() && I != I->getParent()->begin()) {
    auto Prev = std::prev(I);
    if (!isProgramStateInstr(&*Prev))
      return I;
    I = Prev;
  }

  return I;
}

/// Convert mode value from S_SET_VGPR_MSB format to MODE register format.
/// S_SET_VGPR_MSB uses: (src0[0-1], src1[2-3], src2[4-5], dst[6-7])
/// MODE register uses:  (dst[0-1], src0[2-3], src1[4-5], src2[6-7])
/// This is a left rotation by 2 bits on an 8-bit value.
static int64_t convertModeToSetregFormat(int64_t Mode) {
  assert(isUInt<8>(Mode) && "Mode expected to be 8-bit");
  return llvm::rotl<uint8_t>(static_cast<uint8_t>(Mode), /*R=*/2);
}

bool AMDGPULowerVGPREncoding::updateSetregModeImm(MachineInstr &MI,
                                                  int64_t ModeValue) {
  assert(MI.getOpcode() == AMDGPU::S_SETREG_IMM32_B32);

  // Convert from S_SET_VGPR_MSB format to MODE register format
  int64_t SetregMode = convertModeToSetregFormat(ModeValue);

  MachineOperand *ImmOp = TII->getNamedOperand(MI, AMDGPU::OpName::imm);
  int64_t OldImm = ImmOp->getImm();
  int64_t NewImm =
      (OldImm & ~AMDGPU::Hwreg::VGPR_MSB_MASK) | (SetregMode << VGPRMSBShift);
  ImmOp->setImm(NewImm);
  return NewImm != OldImm;
}

bool AMDGPULowerVGPREncoding::handleSetregMode(MachineInstr &MI) {
  using namespace AMDGPU::Hwreg;

  assert(MI.getOpcode() == AMDGPU::S_SETREG_IMM32_B32 &&
         "only S_SETREG_IMM32_B32 needs to be handled");

  MachineOperand *SIMM16Op = TII->getNamedOperand(MI, AMDGPU::OpName::simm16);
  assert(SIMM16Op && "SIMM16Op must be present");

  auto [HwRegId, Offset, Size] = HwregEncoding::decode(SIMM16Op->getImm());
  (void)Offset;
  if (HwRegId != ID_MODE)
    return false;

  EncodeType encodeType = ST->hasVGPRIndexingRegisters()
                              ? EncodeType::SET_VGPR_FRAMES
                              : EncodeType::SET_VGPR_MSB;

  int64_t ModeValue = CurrentMode.encode(encodeType);

  // Case 1: Size <= 12 - the original instruction uses imm32[0:Size-1], so
  // imm32[12:19] is unused. Safe to set imm32[12:19] to the correct VGPR
  // MSBs.
  if (Size <= VGPRMSBShift) {
    // This instruction now acts as MostRecentModeSet so it can be updated if
    // CurrentMode changes via piggybacking.
    MostRecentModeSet = &MI;
    return updateSetregModeImm(MI, ModeValue);
  }

  // Case 2: Size > 12 - the original instruction uses bits beyond 11, so we
  // cannot arbitrarily modify imm32[12:19]. Check if it already matches VGPR
  // MSBs. Note: imm32[12:19] is in MODE register format, while ModeValue is
  // in S_SET_VGPR_MSB format, so we need to convert before comparing.
  MachineOperand *ImmOp = TII->getNamedOperand(MI, AMDGPU::OpName::imm);
  assert(ImmOp && "ImmOp must be present");
  int64_t ImmBits12To19 = (ImmOp->getImm() & VGPR_MSB_MASK) >> VGPRMSBShift;
  int64_t SetregModeValue = convertModeToSetregFormat(ModeValue);
  if (ImmBits12To19 == SetregModeValue) {
    // Already correct, but we must invalidate MostRecentModeSet because this
    // instruction will overwrite mode[12:19]. We can't update this instruction
    // via piggybacking (bits[12:19] are meaningful), so if CurrentMode changes,
    // a new s_set_vgpr_msb will be inserted after this instruction.
    MostRecentModeSet = nullptr;
    return false;
  }

  // imm32[12:19] doesn't match VGPR MSBs - insert s_set_vgpr_msb after
  // the original instruction to restore the correct value.
  MachineBasicBlock::iterator InsertPt = std::next(MI.getIterator());
  MostRecentModeSet = BuildMI(*MBB, InsertPt, MI.getDebugLoc(),
                              TII->get(AMDGPU::S_SET_VGPR_MSB))
                          .addImm(ModeValue);
  return true;
}

bool AMDGPULowerVGPREncoding::run(MachineFunction &MF) {
  ST = &MF.getSubtarget<GCNSubtarget>();
  if (!ST->has1024AddressableVGPRs())
    return false;

  TII = ST->getInstrInfo();
  TRI = ST->getRegisterInfo();
  MFI = MF.getInfo<SIMachineFunctionInfo>();
  STI = MF.getTarget().getMCSubtargetInfo();

  bool Changed = false;
  ClauseLen = ClauseRemaining = 0;
  CurrentMode = {};
  for (auto &MBB : MF) {
    MostRecentModeSet = nullptr;
    this->MBB = &MBB;

    MachineBasicBlock::instr_iterator I = MBB.instr_begin();
    MachineBasicBlock::instr_iterator E = MBB.instr_end();
    for (; I != E; ++I) {
      if (I->isMetaInstruction())
        continue;

      if (I->isTerminator() || I->isCall()) {
        if (I->getOpcode() == AMDGPU::S_ENDPGM ||
            I->getOpcode() == AMDGPU::S_ENDPGM_SAVED)
          CurrentMode = {};
        else
          resetMode(I->getIterator());
        continue;
      }

      if (I->isInlineAsm()) {
        if (TII->hasVGPRUses(*I))
          resetMode(I->getIterator());
        continue;
      }

      if (I->getOpcode() == AMDGPU::S_CLAUSE) {
        assert(!ClauseRemaining && "Nested clauses are not supported");
        ClauseLen = I->getOperand(0).getImm();
        ClauseBreaks = (ClauseLen >> 8) & 15;
        ClauseLen = ClauseRemaining = (ClauseLen & 63) + 1;
        Clause = &*I;
        continue;
      }

      if (I->getOpcode() == AMDGPU::S_SETREG_IMM32_B32 &&
          ST->hasSetregVGPRMSBFixup()) {
        Changed |= handleSetregMode(*I);
        continue;
      }

      Changed |= runOnMachineInstr(I);

      if (ClauseRemaining)
        --ClauseRemaining;
    }

    resetMode(MBB.instr_end());
  }

  return Changed;
}

class AMDGPULowerVGPREncodingLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPULowerVGPREncodingLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    return AMDGPULowerVGPREncoding().run(MF);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char AMDGPULowerVGPREncodingLegacy::ID = 0;

char &llvm::AMDGPULowerVGPREncodingLegacyID = AMDGPULowerVGPREncodingLegacy::ID;

INITIALIZE_PASS(AMDGPULowerVGPREncodingLegacy, DEBUG_TYPE,
                "AMDGPU Lower VGPR Encoding", false, false)

PreservedAnalyses
AMDGPULowerVGPREncodingPass::run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM) {
  if (!AMDGPULowerVGPREncoding().run(MF))
    return PreservedAnalyses::all();

  return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
}
