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
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

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
      if (Imm & 0xf) // If Imm already has a TRANS part (lower 4 bits)
        Imm |= Delay.VALUNum << 7; // Encode VALUNum in higher bits (INSTID1)
      else
        Imm |= Delay.VALUNum;     // Encode VALUNum in lower bits (INSTID0)
    }

    // Wait for an SALU instruction.
    if (Delay.SALUCycles) {
      assert(Delay.SALUCycles < DelayInfo::SALU_CYCLES_MAX);
      if (Imm & 0x780) { // If VALU and (potentially) TRANS already encoded in high bits
        // No room for SALU if both VALU and TRANS (as INSTID1 and INSTID0 resp.) are set.
        // Or if VALU is in INSTID1, and INSTID0 is occupied.
      } else if (Imm & 0xf) { // If INSTID0 is already set (e.g. by TRANS or simple VALU)
        Imm |= (Delay.SALUCycles + 8) << 7; // Encode SALU in higher bits (INSTID1)
      } else { // INSTID0 and INSTID1 are free
        Imm |= Delay.SALUCycles + 8; // Encode SALU in lower bits (INSTID0)
      }
    }

    // Don't emit the s_delay_alu instruction if there's nothing to wait for.
    if (!Imm)
      return LastDelayAlu;

    // If we only need to wait for one instruction (Imm has only INSTID0 set),
    // try encoding it in the last s_delay_alu that we emitted.
    if (!(Imm & 0x780) && LastDelayAlu) { // If only lower 4 bits (INSTID0) are set
      unsigned Skip = 0;
      for (auto I = MachineBasicBlock::instr_iterator(LastDelayAlu),
                E = MachineBasicBlock::instr_iterator(MI);
           ++I != E;) {
        if (!I->isBundle() && !I->isMetaInstruction())
          ++Skip;
      }

      if (Skip < 6) { // Max skip is 5 (encoded as 0-5 for skip 0-5)
        MachineOperand &Op = LastDelayAlu->getOperand(0);
        unsigned OldImm = Op.getImm();
        // Ensure LastDelayAlu also only had INSTID0 set and no INSTID1/SKIP
        if ((OldImm & ~0xf) == 0) {
            unsigned CombinedImm = OldImm;
            CombinedImm |= (Imm << 7);   // New INSTID0 becomes INSTID1
            CombinedImm |= (Skip << 4); // Set skip
            Op.setImm(CombinedImm);
            return nullptr; // Modified existing, don't return the potential NewDelayAlu
        }
      }
    }

    auto &MBB = *MI.getParent();
    MachineInstr *NewDelayAlu =
        BuildMI(MBB, MI, DebugLoc(), SII->get(AMDGPU::S_DELAY_ALU)).addImm(Imm);

    // Remember the s_delay_alu for next time if there is still room in it to
    // encode another delay (i.e., only INSTID0 was set).
    return (Imm & 0x780) ? nullptr : NewDelayAlu;
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

    for (auto &MI : MBB.instrs()) {
      if (MI.isBundle() || MI.isMetaInstruction())
        continue;

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

      if (instructionWaitsForSGPRWrites(MI)) {
        auto It = State.find(LastSGPRFromVALU);
        if (It != State.end()) {
          DelayInfo Info = It->getSecond();
          State.advanceByVALUNum(Info.VALUNum);
          LastSGPRFromVALU = 0;
        }
      }

      if (instructionWaitsForVALU(MI)) {
        if (isTrackedInstruction && isTargetVOPD) {
            dbgs() << "DB_DELAY: VOPD instruction implies wait for all VALU. Clearing DelayState.\n";
        }
        State = DelayState();
      } else if (isTargetVOPD) {
        DelayInfo DelayForOpX_Emission;
        DenseSet<unsigned> ConsumedUnitsOpX_ForStateUpdate;

        for (MCRegUnit Unit : TRI->regunits(MI.getOperand(2).getReg())) {
            auto It = State.find(Unit);
            if (It != State.end()) { DelayForOpX_Emission.merge(It->second); ConsumedUnitsOpX_ForStateUpdate.insert(Unit); }
        }
        for (MCRegUnit Unit : TRI->regunits(MI.getOperand(3).getReg())) {
            auto It = State.find(Unit);
            if (It != State.end()) { DelayForOpX_Emission.merge(It->second); ConsumedUnitsOpX_ForStateUpdate.insert(Unit); }
        }

        DelayState State_For_OpY_EmissionCalc = State;
        for (unsigned Unit : ConsumedUnitsOpX_ForStateUpdate) { State_For_OpY_EmissionCalc.erase(Unit); }
        unsigned LatencyX_Virtual = SchedModel->computeOperandLatency(&MI, MI.getOperand(0).getOperandNo(), nullptr, 0);
        for (MCRegUnit Unit : TRI->regunits(MI.getOperand(0).getReg())) {
            State_For_OpY_EmissionCalc[Unit] = DelayInfo(VALU, LatencyX_Virtual);
        }
        State_For_OpY_EmissionCalc.advance(VALU, 1);

        DelayInfo DelayForOpY_Emission;
        for (MCRegUnit Unit : TRI->regunits(MI.getOperand(4).getReg())) {
            auto It = State_For_OpY_EmissionCalc.find(Unit);
            if (It != State_For_OpY_EmissionCalc.end()) { DelayForOpY_Emission.merge(It->second); }
        }
        for (MCRegUnit Unit : TRI->regunits(MI.getOperand(5).getReg())) {
            auto It = State_For_OpY_EmissionCalc.find(Unit);
            if (It != State_For_OpY_EmissionCalc.end()) { DelayForOpY_Emission.merge(It->second); }
        }

        if (Emit && !MI.isBundledWithPred()) {
          auto getSimpleImm = [](const DelayInfo& Delay) -> unsigned {
            unsigned Imm = 0;
            // This simplified getter only handles VALU dependencies, as that is
            // what v_dual_cndmask generates.
            if (Delay.VALUNum < DelayInfo::VALU_MAX) {
              Imm = Delay.VALUNum;
            }
            return Imm;
          };

          unsigned ImmX = getSimpleImm(DelayForOpX_Emission);
          unsigned ImmY = getSimpleImm(DelayForOpY_Emission);

          if (ImmX > 0 || ImmY > 0) {
            // If one of the dual operations has a dependency and the other
            // doesn't, we still need to emit a combined s_delay_alu. An
            // instid of 0 for the one with no dependency will be treated as
            // VALU_DEP_0, which is effectively a no-op if there's no real
            // dependency on the immediately preceding instruction.
            unsigned FinalImmX = ImmX;
            unsigned FinalImmY = ImmY;

            // The two ops in a dual instruction are effectively back-to-back,
            // so instskip should be NEXT (0).
            unsigned Skip = 0;
            unsigned CombinedImm = FinalImmX | (FinalImmY << 7) | (Skip << 4);

            BuildMI(MBB, MI, DebugLoc(), SII->get(AMDGPU::S_DELAY_ALU)).addImm(CombinedImm);
            Changed = true;
          }
        }

        // --- Main State update proceeds sequentially ---
        for (unsigned Unit : ConsumedUnitsOpX_ForStateUpdate) { State.erase(Unit); }
        unsigned LatencyX_Actual = SchedModel->computeOperandLatency(&MI, MI.getOperand(0).getOperandNo(), nullptr, 0);
        for (MCRegUnit Unit : TRI->regunits(MI.getOperand(0).getReg())) {
            State[Unit] = DelayInfo(VALU, LatencyX_Actual);
        }
        if (AMDGPU::isSGPR(MI.getOperand(0).getReg(), TRI)) {
             LastSGPRFromVALU = *TRI->regunits(MI.getOperand(0).getReg()).begin();
        }
        State.advance(VALU, 1);
        if(isTrackedInstruction) {dbgs() << "DB_DELAY: Main State AFTER actual OpX exec:\n"; State.dump(TRI);}

        // After OpX, we now clear the uses for OpY from the state.
        for (MCRegUnit Unit : TRI->regunits(MI.getOperand(4).getReg())) {
            State.erase(Unit);
        }
        for (MCRegUnit Unit : TRI->regunits(MI.getOperand(5).getReg())) {
            State.erase(Unit);
        }
        unsigned LatencyY_Actual = SchedModel->computeOperandLatency(&MI, MI.getOperand(1).getOperandNo(), nullptr, 0);
        for (MCRegUnit Unit : TRI->regunits(MI.getOperand(1).getReg())) {
            State[Unit] = DelayInfo(VALU, LatencyY_Actual);
        }
        if (AMDGPU::isSGPR(MI.getOperand(1).getReg(), TRI)) {
             LastSGPRFromVALU = *TRI->regunits(MI.getOperand(1).getReg()).begin();
        }
        State.advance(VALU, 1);

        if (isTrackedInstruction) {
            dbgs() << "DB_DELAY: Main State AFTER VOPD (actual OpY exec):\n";
            State.dump(TRI);
            dbgs() << "-----\n";
        }
        LLVM_DEBUG(dbgs() << "  State after VOPD (simulated as 2 ops) " << MI; State.dump(TRI););
        continue;
      }

      // --- Generic processing for non-VOPD or post-waitcnt instructions ---
      DenseSet<unsigned> ConsumedUnitsForThisMI;
      DelayType CurrentMIType = getDelayType(MI.getDesc().TSFlags);

      if (CurrentMIType != OTHER) {
        DelayInfo DelayForCurrentMI;
        for (const auto &Op : MI.explicit_uses()) {
          if (Op.isReg()) {
            if (MI.getOpcode() == AMDGPU::V_WRITELANE_B32 && Op.isTied()) continue;
            for (MCRegUnit Unit : TRI->regunits(Op.getReg())) {
              auto It = State.find(Unit);
              if (It != State.end()) {
                DelayForCurrentMI.merge(It->second);
                ConsumedUnitsForThisMI.insert(Unit);
              }
            }
          }
        }
        if (Emit && !MI.isBundledWithPred()) {
          LastDelayAlu = emitDelayAlu(MI, DelayForCurrentMI, LastDelayAlu);
        }
      }

      for (unsigned Unit : ConsumedUnitsForThisMI) {
        State.erase(Unit);
      }

      // Generic def processing
      if (CurrentMIType != OTHER) {
        for (const auto &Op : MI.defs()) {
            if (Op.isReg()){
                unsigned Latency = SchedModel->computeOperandLatency(
                    &MI, Op.getOperandNo(), nullptr, 0);
                for (MCRegUnit Unit : TRI->regunits(Op.getReg()))
                  State[Unit] = DelayInfo(CurrentMIType, Latency);
            }
        }
        if (SII->isVALU(MI.getOpcode())) {
          for (const auto &Op : MI.defs()) {
            if (Op.isReg()){
                Register Reg = Op.getReg();
                if (AMDGPU::isSGPR(Reg, TRI)) {
                  LastSGPRFromVALU = *TRI->regunits(Reg).begin();
                  break;
                }
            }
          }
        }
      }

      unsigned Cycles = SIInstrInfo::getNumWaitStates(MI);
      State.advance(CurrentMIType, Cycles);

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

    CurrentFunctionNameForDebug = MF.getName();
    DebugThisFunctionForDelay = false;
    VDualCndMaskCounter = 0;
    if (const char *EnvVar = std::getenv("DB_DELAY")) {
      if (CurrentFunctionNameForDebug == EnvVar) {
        DebugThisFunctionForDelay = true;
        dbgs() << "DB_DELAY: Debugging enabled for function: " << EnvVar << "\n";
      }
    }

    SII = ST.getInstrInfo();
    TRI = ST.getRegisterInfo();
    SchedModel = &SII->getSchedModel();

    SetVector<MachineBasicBlock *> WorkList;
    for (auto &MBB : reverse(MF))
      WorkList.insert(&MBB);
    while (!WorkList.empty()) {
      auto &MBB = *WorkList.pop_back_val();
      bool ChangedFromNonEmit = runOnMachineBasicBlock(MBB, false);
      if (ChangedFromNonEmit)
        WorkList.insert_range(MBB.successors());
    }

    LLVM_DEBUG(dbgs() << "Final pass over all BBs\n");

    bool ChangedOverall = false;
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
    AMDGPUInsertDelayAlu Impl;
    return Impl.run(MF);
  }
};
} // namespace

PreservedAnalyses
AMDGPUInsertDelayAluPass::run(MachineFunction &MF,
                               MachineFunctionAnalysisManager &MFAM) {
  AMDGPUInsertDelayAlu Impl;
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
