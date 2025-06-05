//===- AMDGPUInsertDelayAlu.cpp - Insert s_delay_alu instructions ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Insert s_delay_alu instructions to avoid stalls on GFX11+.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/DenseSet.h" // Changed from SmallPtrSet.h
#include "llvm/Support/raw_ostream.h" // Required for dbgs()
#include <cstdlib>                    // Required for std::getenv

using namespace llvm;

#define DEBUG_TYPE "amdgpu-insert-delay-alu"

namespace {

class AMDGPUInsertDelayAlu {
public:
  const SIInstrInfo *SII;
  const TargetRegisterInfo *TRI;

  const TargetSchedModel *SchedModel;

  // Debugging state per function run
  bool DebugThisFunctionForDelay = false;
  StringRef CurrentFunctionNameForDebug;
  int VDualCndMaskCounter = 0;


  // Return true if MI waits for all outstanding VALU instructions to complete.
  static bool instructionWaitsForVALU(const MachineInstr &MI) {
    // These instruction types wait for VA_VDST==0 before issuing.
    const uint64_t VA_VDST_0 = SIInstrFlags::DS | SIInstrFlags::EXP |
                               SIInstrFlags::FLAT | SIInstrFlags::MIMG |
                               SIInstrFlags::MTBUF | SIInstrFlags::MUBUF;
    if (MI.getDesc().TSFlags & VA_VDST_0)
      return true;
    if (MI.getOpcode() == AMDGPU::S_SENDMSG_RTN_B32 ||
        MI.getOpcode() == AMDGPU::S_SENDMSG_RTN_B64)
      return true;
    if (MI.getOpcode() == AMDGPU::S_WAITCNT_DEPCTR &&
        AMDGPU::DepCtr::decodeFieldVaVdst(MI.getOperand(0).getImm()) == 0)
      return true;
    return false;
  }

  static bool instructionWaitsForSGPRWrites(const MachineInstr &MI) {
    // These instruction types wait for VA_SDST==0 before issuing.
    const uint64_t VA_SDST_0 = SIInstrFlags::SALU | SIInstrFlags::SMRD;

    return MI.getDesc().TSFlags & VA_SDST_0;
  }

  // Types of delay that can be encoded in an s_delay_alu instruction.
  enum DelayType { VALU, TRANS, SALU, OTHER };

  // Get the delay type for an instruction with the specified TSFlags.
  static DelayType getDelayType(uint64_t TSFlags) {
    if (TSFlags & SIInstrFlags::TRANS)
      return TRANS;
    if (TSFlags & SIInstrFlags::VALU)
      return VALU;
    if (TSFlags & SIInstrFlags::SALU)
      return SALU;
    return OTHER;
  }

  // Information about the last instruction(s) that wrote to a particular
  // regunit. In straight-line code there will only be one such instruction, but
  // when control flow converges we merge the delay information from each path
  // to represent the union of the worst-case delays of each type.
  struct DelayInfo {
    // One larger than the maximum number of (non-TRANS) VALU instructions we
    // can encode in an s_delay_alu instruction.
    static constexpr unsigned VALU_MAX = 5;

    // One larger than the maximum number of TRANS instructions we can encode in
    // an s_delay_alu instruction.
    static constexpr unsigned TRANS_MAX = 4;

    // One larger than the maximum number of SALU cycles we can encode in an
    // s_delay_alu instruction.
    static constexpr unsigned SALU_CYCLES_MAX = 4;

    // If it was written by a (non-TRANS) VALU, remember how many clock cycles
    // are left until it completes, and how many other (non-TRANS) VALU we have
    // seen since it was issued.
    uint8_t VALUCycles = 0;
    uint8_t VALUNum = VALU_MAX;

    // If it was written by a TRANS, remember how many clock cycles are left
    // until it completes, and how many other TRANS we have seen since it was
    // issued.
    uint8_t TRANSCycles = 0;
    uint8_t TRANSNum = TRANS_MAX;
    // Also remember how many other (non-TRANS) VALU we have seen since it was
    // issued. When an instruction depends on both a prior TRANS and a prior
    // non-TRANS VALU, this is used to decide whether to encode a wait for just
    // one or both of them.
    uint8_t TRANSNumVALU = VALU_MAX;

    // If it was written by an SALU, remember how many clock cycles are left
    // until it completes.
    uint8_t SALUCycles = 0;

    DelayInfo() = default;

    DelayInfo(DelayType Type, unsigned Cycles) {
      switch (Type) {
      default:
        llvm_unreachable("unexpected type");
      case VALU:
        VALUCycles = Cycles;
        VALUNum = 0;
        break;
      case TRANS:
        TRANSCycles = Cycles;
        TRANSNum = 0;
        TRANSNumVALU = 0;
        break;
      case SALU:
        // Guard against pseudo-instructions like SI_CALL which are marked as
        // SALU but with a very high latency.
        SALUCycles = std::min(Cycles, SALU_CYCLES_MAX);
        break;
      }
    }

    bool operator==(const DelayInfo &RHS) const {
      return VALUCycles == RHS.VALUCycles && VALUNum == RHS.VALUNum &&
             TRANSCycles == RHS.TRANSCycles && TRANSNum == RHS.TRANSNum &&
             TRANSNumVALU == RHS.TRANSNumVALU && SALUCycles == RHS.SALUCycles;
    }

    bool operator!=(const DelayInfo &RHS) const { return !(*this == RHS); }

    // Merge another DelayInfo into this one, to represent the union of the
    // worst-case delays of each type.
    void merge(const DelayInfo &RHS) {
      VALUCycles = std::max(VALUCycles, RHS.VALUCycles);
      VALUNum = std::min(VALUNum, RHS.VALUNum);
      TRANSCycles = std::max(TRANSCycles, RHS.TRANSCycles);
      TRANSNum = std::min(TRANSNum, RHS.TRANSNum);
      TRANSNumVALU = std::min(TRANSNumVALU, RHS.TRANSNumVALU);
      SALUCycles = std::max(SALUCycles, RHS.SALUCycles);
    }

    // Update this DelayInfo after issuing an instruction. IsVALU should be 1
    // when issuing a (non-TRANS) VALU, else 0. IsTRANS should be 1 when issuing
    // a TRANS, else 0. Cycles is the number of cycles it takes to issue the
    // instruction.  Return true if there is no longer any useful delay info.
    bool advance(DelayType Type, unsigned Cycles) {
      bool Erase = true;

      VALUNum += (Type == VALU);
      if (VALUNum >= VALU_MAX || VALUCycles <= Cycles) {
        // Forget about the VALU instruction. It was too far back or has
        // definitely completed by now.
        VALUNum = VALU_MAX;
        VALUCycles = 0;
      } else {
        VALUCycles -= Cycles;
        Erase = false;
      }

      TRANSNum += (Type == TRANS);
      TRANSNumVALU += (Type == VALU);
      if (TRANSNum >= TRANS_MAX || TRANSCycles <= Cycles) {
        // Forget about any TRANS instruction. It was too far back or has
        // definitely completed by now.
        TRANSNum = TRANS_MAX;
        TRANSNumVALU = VALU_MAX;
        TRANSCycles = 0;
      } else {
        TRANSCycles -= Cycles;
        Erase = false;
      }

      if (SALUCycles <= Cycles) {
        // Forget about any SALU instruction. It has definitely completed by
        // now.
        SALUCycles = 0;
      } else {
        SALUCycles -= Cycles;
        Erase = false;
      }

      return Erase;
    }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    void dump() const { // This dump is used by the added debug traces
      bool HasInfo = false;
      if (VALUCycles) {
        dbgs() << " VALUCycles=" << (int)VALUCycles;
        HasInfo = true;
      }
      if (VALUNum < VALU_MAX) {
        dbgs() << " VALUNum=" << (int)VALUNum;
        HasInfo = true;
      }
      if (TRANSCycles) {
        dbgs() << " TRANSCycles=" << (int)TRANSCycles;
        HasInfo = true;
      }
      if (TRANSNum < TRANS_MAX) {
        dbgs() << " TRANSNum=" << (int)TRANSNum;
        HasInfo = true;
      }
      if (TRANSNumVALU < VALU_MAX) {
        dbgs() << " TRANSNumVALU=" << (int)TRANSNumVALU;
        HasInfo = true;
      }
      if (SALUCycles) {
        dbgs() << " SALUCycles=" << (int)SALUCycles;
        HasInfo = true;
      }
      if (!HasInfo) {
        dbgs() << " (empty)";
      }
    }
#endif
  };

  // A map from regunits to the delay info for that regunit.
  struct DelayState : DenseMap<unsigned, DelayInfo> {
    // Merge another DelayState into this one by merging the delay info for each
    // regunit.
    void merge(const DelayState &RHS) {
      for (const auto &KV : RHS) {
        iterator It;
        bool Inserted;
        std::tie(It, Inserted) = insert(KV);
        if (!Inserted)
          It->second.merge(KV.second);
      }
    }

    // Advance the delay info for each regunit, erasing any that are no longer
    // useful.
    void advance(DelayType Type, unsigned Cycles) {
      iterator Next;
      for (auto I = begin(), E = end(); I != E; I = Next) {
        Next = std::next(I);
        if (I->second.advance(Type, Cycles))
          erase(I);
      }
    }

    void advanceByVALUNum(unsigned VALUNum) {
      iterator Next;
      for (auto I = begin(), E = end(); I != E; I = Next) {
        Next = std::next(I);
        if (I->second.VALUNum >= VALUNum && I->second.VALUCycles > 0) {
          erase(I);
        }
      }
    }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    void dump(const TargetRegisterInfo *TRI) const {
      if (empty()) {
        dbgs() << "    empty\n";
        return;
      }

      // Dump DelayInfo for each RegUnit in numerical order.
      SmallVector<const_iterator, 8> Order;
      Order.reserve(size());
      for (const_iterator I = begin(), E = end(); I != E; ++I)
        Order.push_back(I);
      llvm::sort(Order, [](const const_iterator &A, const const_iterator &B) {
        return A->first < B->first;
      });
      for (const_iterator I : Order) {
        dbgs() << "    " << printRegUnit(I->first, TRI);
        I->second.dump();
        dbgs() << "\n";
      }
    }
#endif
  };

  // The saved delay state at the end of each basic block.
  DenseMap<MachineBasicBlock *, DelayState> BlockState;

  // Emit an s_delay_alu instruction if necessary before MI for non-VOPD.
  MachineInstr *emitDelayAlu(MachineInstr &MI, DelayInfo Delay,
                               MachineInstr *LastDelayAlu) {
    unsigned Imm = 0;

    // Wait for a TRANS instruction.
    if (Delay.TRANSNum < DelayInfo::TRANS_MAX)
      Imm |= 4 + Delay.TRANSNum;

    // Wait for a VALU instruction (if it's more recent than any TRANS
    // instruction that we're also waiting for).
    if (Delay.VALUNum < DelayInfo::VALU_MAX &&
        Delay.VALUNum <= Delay.TRANSNumVALU) {
      if (Imm & 0xf)
        Imm |= Delay.VALUNum << 7;
      else
        Imm |= Delay.VALUNum;
    }

    // Wait for an SALU instruction.
    if (Delay.SALUCycles) {
      assert(Delay.SALUCycles < DelayInfo::SALU_CYCLES_MAX);
      if (Imm & 0x780) {
        // We have already encoded a VALU and a TRANS delay. There's no room in
        // the encoding for an SALU delay as well, so just drop it.
      } else if (Imm & 0xf) {
        Imm |= (Delay.SALUCycles + 8) << 7;
      } else {
        Imm |= Delay.SALUCycles + 8;
      }
    }

    // Don't emit the s_delay_alu instruction if there's nothing to wait for.
    if (!Imm)
      return LastDelayAlu;

    // If we only need to wait for one instruction, try encoding it in the last
    // s_delay_alu that we emitted.
    if (!(Imm & 0x780) && LastDelayAlu) {
      unsigned Skip = 0;
      for (auto I = MachineBasicBlock::instr_iterator(LastDelayAlu),
                E = MachineBasicBlock::instr_iterator(MI);
           ++I != E;) {
        if (!I->isBundle() && !I->isMetaInstruction())
          ++Skip;
      }
      if (Skip < 6) {
        MachineOperand &Op = LastDelayAlu->getOperand(0);
        unsigned LastImm = Op.getImm();
        assert((LastImm & ~0xf) == 0 &&
               "Remembered an s_delay_alu with no room for another delay!");
        LastImm |= Imm << 7 | Skip << 4;
        Op.setImm(LastImm);
        return nullptr;
      }
    }

    auto &MBB = *MI.getParent();
    MachineInstr *NewDelayAlu =
        BuildMI(MBB, MI, DebugLoc(), SII->get(AMDGPU::S_DELAY_ALU)).addImm(Imm);
    // Remember the s_delay_alu for next time if there is still room in it to
    // encode another delay.
    return (Imm & 0x780) ? nullptr : NewDelayAlu;
  }

  // Emit an s_delay_alu instruction for VOPD instructions.
  MachineInstr *emitDelayAluForVOPD(MachineInstr &MI,
                                    DelayInfo DelayX, DelayInfo DelayY) {
    unsigned ValuXDep = (DelayX.VALUNum < DelayInfo::VALU_MAX) ? DelayX.VALUNum : 0;
    unsigned ValuYDep = (DelayY.VALUNum < DelayInfo::VALU_MAX) ? DelayY.VALUNum : 0;

    if (ValuXDep == 0 && ValuYDep == 0) {
      return nullptr; // No delay needed
    }

    unsigned FinalImm = 0;
    // ISA: INSTID0 = SIMM16[3:0], INSTSKIP = SIMM16[6:4], INSTID1 = SIMM16[10:7]
    // INSTSKIP_SAME is 0x0 for VOPD (apply INSTID1 to the second op of the same instruction).

    if (ValuXDep != 0 && ValuYDep != 0) {
      // Both X and Y need delay
      FinalImm = (ValuXDep & 0xF);          // instid0 for OpX
      FinalImm |= (0x0 << 4);               // instskip(SAME)
      FinalImm |= ((ValuYDep & 0xF) << 7);  // instid1 for OpY
    } else if (ValuXDep != 0) {
      // Only X needs delay
      FinalImm = (ValuXDep & 0xF);          // instid0 for OpX
                                            // instskip and instid1 remain 0
    } else { // Only Y needs delay (ValuYDep != 0)
      FinalImm = (ValuYDep & 0xF);          // instid0 for OpY (originally for X)
                                            // instskip and instid1 remain 0
    }

    if (FinalImm == 0) return nullptr; // Should be caught by initial check if logic is sound.

    auto &MBB = *MI.getParent();
    MachineInstr *NewDelayAlu =
        BuildMI(MBB, MI, DebugLoc(), SII->get(AMDGPU::S_DELAY_ALU)).addImm(FinalImm);

    return (FinalImm & 0x780) ? nullptr : NewDelayAlu;
  }


  bool runOnMachineBasicBlock(MachineBasicBlock &MBB, bool Emit) {
    DelayState State;
    for (auto *Pred : MBB.predecessors())
      State.merge(BlockState[Pred]);

    LLVM_DEBUG(dbgs() << "  State at start of " << printMBBReference(MBB)
                      << "\n";
               State.dump(TRI););

    bool Changed = false;
    MachineInstr *LastDelayAlu = nullptr;

    MCRegUnit LastSGPRFromVALU = 0;
    // Iterate over the contents of bundles, but don't emit any instructions
    // inside a bundle.
    for (auto &MI : MBB.instrs()) {
      if (MI.isBundle() || MI.isMetaInstruction())
        continue;

      // Ignore some more instructions that do not generate any code.
      switch (MI.getOpcode()) {
      case AMDGPU::SI_RETURN_TO_EPILOG:
        continue;
      }

      bool isTargetVOPD = (MI.getOpcode() == AMDGPU::V_DUAL_CNDMASK_B32_e32_X_CNDMASK_B32_e32_gfx11);
      bool isTrackedInstruction = DebugThisFunctionForDelay && isTargetVOPD;

      if (isTrackedInstruction) {
          VDualCndMaskCounter++;
          dbgs() << "DB_DELAY: Processing V_DUAL_CNDMASK_B32 #" << VDualCndMaskCounter
                 << " in function " << CurrentFunctionNameForDebug << "\n";
          dbgs() << "DB_DELAY: MI: "; MI.dump();
          dbgs() << "DB_DELAY: State BEFORE processing this MI uses/defs:\n";
          State.dump(TRI);
      }

      DelayType Type = getDelayType(MI.getDesc().TSFlags); // For VOPD, this is VALU

      if (instructionWaitsForSGPRWrites(MI)) {
        auto It = State.find(LastSGPRFromVALU);
        if (It != State.end()) {
          DelayInfo Info = It->getSecond();
          State.advanceByVALUNum(Info.VALUNum);
          LastSGPRFromVALU = 0;
        }
      }

      DenseSet<unsigned> ConsumedUnitsForMI; // Changed from SmallPtrSet

      if (instructionWaitsForVALU(MI)) { // True for S_WAITCNT VA_VDST=0, etc. Not for VOPD itself.
        if (isTrackedInstruction) { // Should not happen for the VOPD itself
            dbgs() << "DB_DELAY: Instruction waits for all VALU. Clearing DelayState.\n";
        }
        State = DelayState();
      } else if (isTargetVOPD) {
        DelayInfo DelayForOpX, DelayForOpY;

        // Operands for V_DUAL_CNDMASK_B32_e32_X_CNDMASK_B32_e32_gfx11:
        // 0: dstX, 1: dstY, 2: src0X (tied to dstX), 3: src1X, 4: src0Y (tied to dstY), 5: src1Y
        // Implicit uses: VCC (e.g., VCC_LO), EXEC

        // Process OpX sources: MI.getOperand(2), MI.getOperand(3)
        for (MCRegUnit Unit : TRI->regunits(MI.getOperand(2).getReg())) { // src0X (original value of dstX)
            auto It = State.find(Unit);
            if (It != State.end()) { DelayForOpX.merge(It->second); ConsumedUnitsForMI.insert(Unit); }
        }
        for (MCRegUnit Unit : TRI->regunits(MI.getOperand(3).getReg())) { // src1X
            auto It = State.find(Unit);
            if (It != State.end()) { DelayForOpX.merge(It->second); ConsumedUnitsForMI.insert(Unit); }
        }

        // Process OpY sources: MI.getOperand(4), MI.getOperand(5)
        for (MCRegUnit Unit : TRI->regunits(MI.getOperand(4).getReg())) { // src0Y (original value of dstY)
            auto It = State.find(Unit);
            if (It != State.end()) { DelayForOpY.merge(It->second); ConsumedUnitsForMI.insert(Unit); }
        }
        for (MCRegUnit Unit : TRI->regunits(MI.getOperand(5).getReg())) { // src1Y
            auto It = State.find(Unit);
            if (It != State.end()) { DelayForOpY.merge(It->second); ConsumedUnitsForMI.insert(Unit); }
        }

        // Process implicit uses (e.g., VCC) - ONLY FOR VOPD
        for (const MachineOperand &ImpOp : MI.implicit_operands()) {
            if (ImpOp.isReg() && ImpOp.isUse()) {
                // Assuming VCC is the primary relevant implicit VALU-related dependency
                if (TRI->isSubRegisterEq(AMDGPU::VCC, ImpOp.getReg()) || ImpOp.getReg() == AMDGPU::VCC) {
                     for (MCRegUnit Unit : TRI->regunits(ImpOp.getReg())) {
                        auto It = State.find(Unit);
                        if (It != State.end()) {
                            DelayForOpX.merge(It->second); // VCC used by both X and Y
                            DelayForOpY.merge(It->second);
                            ConsumedUnitsForMI.insert(Unit);
                            if(isTrackedInstruction) {
                                dbgs() << "DB_DELAY: Merging DelayInfo for implicit VCC Unit " << printRegUnit(Unit, TRI) << ": ";
                                It->second.dump(); dbgs() << "\n";
                            }
                        }
                    }
                }
            }
        }
        if (isTrackedInstruction) {
            dbgs() << "DB_DELAY: DelayInfo for OpX: "; DelayForOpX.dump(); dbgs() << "\n";
            dbgs() << "DB_DELAY: DelayInfo for OpY: "; DelayForOpY.dump(); dbgs() << "\n";
        }

        if (Emit && !MI.isBundledWithPred()) {
           MachineInstr* NewDelay = emitDelayAluForVOPD(MI, DelayForOpX, DelayForOpY);
           if (NewDelay) LastDelayAlu = (NewDelay->getOperand(0).getImm() & 0x780) ? nullptr : NewDelay;
        }

      } else if (Type != OTHER) { // Non-VOPD VALU/TRANS/SALU
        DelayInfo DelayForMI;
        // TODO: Scan implicit uses too? (Original comment from LLVM)
        for (const auto &Op : MI.explicit_uses()) {
          if (Op.isReg()) {
            if (MI.getOpcode() == AMDGPU::V_WRITELANE_B32 && Op.isTied()) continue;
            for (MCRegUnit Unit : TRI->regunits(Op.getReg())) {
              auto It = State.find(Unit);
              if (It != State.end()) {
                DelayForMI.merge(It->second);
                ConsumedUnitsForMI.insert(Unit);
              }
            }
          }
        }
        // Implicit operand handling for non-VOPD is NOT done here.

        if (Emit && !MI.isBundledWithPred()) {
          LastDelayAlu = emitDelayAlu(MI, DelayForMI, LastDelayAlu);
        }
      }

      // Erase consumed units from State *after* all uses by MI (or its parts) have been processed
      for (unsigned Unit : ConsumedUnitsForMI) {
        State.erase(Unit);
      }

      // Process defs of the current instruction MI
      if (Type != OTHER) { // VALU, TRANS, SALU (VOPD is VALU type)
        if (isTargetVOPD) {
            // OpX defines MI.getOperand(0), OpY defines MI.getOperand(1)
            unsigned LatencyX = SchedModel->computeOperandLatency(&MI, MI.getOperand(0).getOperandNo(), nullptr, 0);
            for (MCRegUnit Unit : TRI->regunits(MI.getOperand(0).getReg()))
                State[Unit] = DelayInfo(Type, LatencyX);

            unsigned LatencyY = SchedModel->computeOperandLatency(&MI, MI.getOperand(1).getOperandNo(), nullptr, 0);
            for (MCRegUnit Unit : TRI->regunits(MI.getOperand(1).getReg()))
                State[Unit] = DelayInfo(Type, LatencyY);
        } else { // Non-VOPD VALU/TRANS/SALU
            for (const auto &Op : MI.defs()) {
              if (Op.isReg()) { // Ensure it's a register definition
                unsigned Latency = SchedModel->computeOperandLatency(
                    &MI, Op.getOperandNo(), nullptr, 0);
                for (MCRegUnit Unit : TRI->regunits(Op.getReg()))
                  State[Unit] = DelayInfo(Type, Latency);
              }
            }
        }
        if (SII->isVALU(MI.getOpcode())) { // Including VOPD
          for (const auto &Op : MI.defs()) {
            if (Op.isReg()) {
                Register Reg = Op.getReg();
                if (AMDGPU::isSGPR(Reg, TRI)) {
                  LastSGPRFromVALU = *TRI->regunits(Reg).begin();
                  break;
                }
            }
          }
        }
      }

      // Advance by the number of cycles it takes to issue this instruction.
      unsigned Cycles = SIInstrInfo::getNumWaitStates(MI);
      State.advance(Type, Cycles); // Type is for MI itself

      if (isTrackedInstruction) {
          dbgs() << "DB_DELAY: State AFTER processing MI defs and advancing state by " << Cycles << " cycles (instr type " << (int)Type << "):\n";
          State.dump(TRI);
          dbgs() << "-----\n";
      }
      LLVM_DEBUG(dbgs() << "  State after " << MI; State.dump(TRI););
    } // end loop over MI

    if (Emit) {
      assert(State == BlockState[&MBB] &&
             "Basic block state should not have changed on final pass!");
    } else if (DelayState &BS = BlockState[&MBB]; State != BS) {
      BS = std::move(State);
      Changed = true;
    }
    return Changed;
  }

  bool run(MachineFunction &MF) {
    LLVM_DEBUG(dbgs() << "AMDGPUInsertDelayAlu running on " << MF.getName()
                      << "\n");

    const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
    if (!ST.hasDelayAlu())
      return false;

    // Initialize debugging state for this function run
    CurrentFunctionNameForDebug = MF.getName();
    DebugThisFunctionForDelay = false;
    VDualCndMaskCounter = 0; // Reset counter for each function
    if (const char *EnvVar = std::getenv("DB_DELAY")) {
      if (CurrentFunctionNameForDebug == EnvVar) {
        DebugThisFunctionForDelay = true;
        dbgs() << "DB_DELAY: Debugging enabled for function: " << EnvVar << "\n";
      }
    }


    SII = ST.getInstrInfo();
    TRI = ST.getRegisterInfo();
    SchedModel = &SII->getSchedModel();

    // Calculate the delay state for each basic block, iterating until we reach
    // a fixed point.
    SetVector<MachineBasicBlock *> WorkList;
    for (auto &MBB : reverse(MF))
      WorkList.insert(&MBB);
    while (!WorkList.empty()) {
      auto &MBB = *WorkList.pop_back_val();
      bool ChangedFromNonEmit = runOnMachineBasicBlock(MBB, false);
      if (ChangedFromNonEmit) // Use a different variable name to avoid confusion
        WorkList.insert_range(MBB.successors());
    }

    LLVM_DEBUG(dbgs() << "Final pass over all BBs\n");

    // Make one last pass over all basic blocks to emit s_delay_alu
    // instructions.
    bool ChangedOverall = false; // Use a different variable name
    for (auto &MBB : MF)
      ChangedOverall |= runOnMachineBasicBlock(MBB, true);
    return ChangedOverall;
  }
};

class AMDGPUInsertDelayAluLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUInsertDelayAluLegacy() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;
    AMDGPUInsertDelayAlu Impl; // Impl is created per-function call
    return Impl.run(MF);
  }
};
} // namespace

PreservedAnalyses
AMDGPUInsertDelayAluPass::run(MachineFunction &MF,
                               MachineFunctionAnalysisManager &MFAM) {
  AMDGPUInsertDelayAlu Impl; // Impl is created per-function call
  if (!Impl.run(MF))
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
} // end namespace llvm

char AMDGPUInsertDelayAluLegacy::ID = 0;

char &llvm::AMDGPUInsertDelayAluID = AMDGPUInsertDelayAluLegacy::ID;

INITIALIZE_PASS(AMDGPUInsertDelayAluLegacy, DEBUG_TYPE,
                "AMDGPU Insert Delay ALU", false, false)
