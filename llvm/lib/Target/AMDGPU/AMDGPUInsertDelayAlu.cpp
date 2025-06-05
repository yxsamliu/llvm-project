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

  // Emit an s_delay_alu instruction if necessary before MI.
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

      bool isTrackedInstruction = DebugThisFunctionForDelay && (MI.getOpcode() == AMDGPU::V_DUAL_CNDMASK_B32_e32_X_CNDMASK_B32_e32_gfx11);
      if (isTrackedInstruction) {
          VDualCndMaskCounter++;
          dbgs() << "DB_DELAY: Processing V_DUAL_CNDMASK_B32 #" << VDualCndMaskCounter
                 << " in function " << CurrentFunctionNameForDebug << "\n";
          dbgs() << "DB_DELAY: MI: "; MI.dump();
          dbgs() << "DB_DELAY: State BEFORE processing this MI uses/defs:\n";
          State.dump(TRI);
      }

      DelayType Type = getDelayType(MI.getDesc().TSFlags);

      if (instructionWaitsForSGPRWrites(MI)) {
        auto It = State.find(LastSGPRFromVALU);
        if (It != State.end()) {
          DelayInfo Info = It->getSecond();
          State.advanceByVALUNum(Info.VALUNum);
          LastSGPRFromVALU = 0;
        }
      }

      DelayInfo DelayForMI; // Stores the combined delay requirements for MI's uses

      if (instructionWaitsForVALU(MI)) {
        if (isTrackedInstruction) {
            dbgs() << "DB_DELAY: Instruction waits for all VALU. Clearing DelayState.\n";
        }
        // Forget about all outstanding VALU delays.
        // TODO: This is overkill since it also forgets about SALU delays.
        State = DelayState();
      } else if (Type != OTHER) { // This instruction itself is a VALU/TRANS/SALU, check its uses
        // TODO: Scan implicit uses too?
        for (const auto &Op : MI.explicit_uses()) {
          if (Op.isReg()) {
            // One of the operands of the writelane is also the output operand.
            // This creates the insertion of redundant delays. Hence, we have to
            // ignore this operand.
            if (MI.getOpcode() == AMDGPU::V_WRITELANE_B32 && Op.isTied())
              continue;
            for (MCRegUnit Unit : TRI->regunits(Op.getReg())) {
              auto It = State.find(Unit);
              if (It != State.end()) {
                if (isTrackedInstruction) {
                    dbgs() << "DB_DELAY: Merging DelayInfo for used RegUnit " << printRegUnit(Unit, TRI) << ": ";
                    It->second.dump(); dbgs() << "\n";
                }
                DelayForMI.merge(It->second);
                State.erase(Unit); // Consumed by this instruction
              }
            }
          }
        }

        if (SII->isVALU(MI.getOpcode())) {
          for (const auto &Op : MI.defs()) {
            Register Reg = Op.getReg();
            if (AMDGPU::isSGPR(Reg, TRI)) {
              LastSGPRFromVALU = *TRI->regunits(Reg).begin();
              break;
            }
          }
        }

        if (Emit && !MI.isBundledWithPred()) {
          if (isTrackedInstruction) {
              dbgs() << "DB_DELAY: Considering S_DELAY_ALU for current V_DUAL_CNDMASK_B32.\n";
              dbgs() << "DB_DELAY: DelayInfo calculated for its uses: "; DelayForMI.dump(); dbgs() << "\n";
              dbgs() << "DB_DELAY: LastDelayAlu before emitDelayAlu call: ";
              if (LastDelayAlu) LastDelayAlu->dump(); else dbgs() << "nullptr\n";
          }

          MachineInstr* ArgLastDelayAlu = LastDelayAlu;
          unsigned ArgLastDelayAluImm = 0;
          if (ArgLastDelayAlu) ArgLastDelayAluImm = ArgLastDelayAlu->getOperand(0).getImm();

          MachineInstr* ResultFromEmit = emitDelayAlu(MI, DelayForMI, ArgLastDelayAlu);

          if (isTrackedInstruction) {
            bool ActionTaken = false;
            // Check if ArgLastDelayAlu (the one passed in) was modified
            if (ArgLastDelayAlu && ArgLastDelayAlu->getOperand(0).getImm() != ArgLastDelayAluImm) {
                dbgs() << "DB_DELAY: Modified existing S_DELAY_ALU (passed as LastDelayAlu):\n";
                dbgs() << "DB_DELAY: Old Imm: 0x" << Twine::utohexstr(ArgLastDelayAluImm)
                       << ", New Imm: 0x" << Twine::utohexstr(ArgLastDelayAlu->getOperand(0).getImm()) << "\n";
                ArgLastDelayAlu->dump();
                ActionTaken = true;
            // Check if a new instruction was inserted before MI
            } else if (MI.getPrevNode() && MI.getPrevNode()->getOpcode() == AMDGPU::S_DELAY_ALU &&
                       MI.getPrevNode() != ArgLastDelayAlu) {
                dbgs() << "DB_DELAY: Inserted new S_DELAY_ALU before V_DUAL_CNDMASK_B32:\n";
                MI.getPrevNode()->dump();
                ActionTaken = true;
            }

            if (!ActionTaken) {
                dbgs() << "DB_DELAY: No S_DELAY_ALU inserted or modified for V_DUAL_CNDMASK_B32 by this call.\n";
            }
            dbgs() << "DB_DELAY: emitDelayAlu call returned: ";
            if (ResultFromEmit) ResultFromEmit->dump(); else dbgs() << "nullptr\n";
          }
          LastDelayAlu = ResultFromEmit; // Update LastDelayAlu with the result
        }
      } // end if (Type != OTHER) for processing uses and emitting s_delay_alu

      // Process defs of the current instruction MI
      if (Type != OTHER) {
        // TODO: Scan implicit defs too?
        for (const auto &Op : MI.defs()) {
          unsigned Latency = SchedModel->computeOperandLatency(
              &MI, Op.getOperandNo(), nullptr, 0);
          if (isTrackedInstruction) {
              dbgs() << "DB_DELAY: Def operand " << Op.getOperandNo() << " (Reg " << printReg(Op.getReg(), TRI)
                     << ") has Latency " << Latency << ". Updating State for its RegUnits.\n";
          }
          for (MCRegUnit Unit : TRI->regunits(Op.getReg())) {
            State[Unit] = DelayInfo(Type, Latency);
            if (isTrackedInstruction) {
                 dbgs() << "DB_DELAY: State for RegUnit " << printRegUnit(Unit, TRI) << " set to: ";
                 State[Unit].dump(); dbgs() << "\n";
            }
          }
        }
      }

      // Advance by the number of cycles it takes to issue this instruction.
      unsigned Cycles = SIInstrInfo::getNumWaitStates(MI);
      State.advance(Type, Cycles);

      if (isTrackedInstruction) {
          dbgs() << "DB_DELAY: State AFTER processing MI defs and advancing state by " << Cycles << " cycles (instr type " << Type << "):\n";
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
      bool Changed = runOnMachineBasicBlock(MBB, false);
      if (Changed)
        WorkList.insert_range(MBB.successors());
    }

    LLVM_DEBUG(dbgs() << "Final pass over all BBs\n");

    // Make one last pass over all basic blocks to emit s_delay_alu
    // instructions.
    bool Changed = false;
    for (auto &MBB : MF)
      Changed |= runOnMachineBasicBlock(MBB, true);
    return Changed;
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
