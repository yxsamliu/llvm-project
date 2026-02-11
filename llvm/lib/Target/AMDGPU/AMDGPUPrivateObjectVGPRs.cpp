//===----------- AMDGPUPrivateObjectVGPRs.cpp - Private object VGPRs ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Mark physical VGPRs allocated to promoted private objects as used to prevent
/// the register allocator from using these VGPRs where the objects are live:
///
///  * Add implicit use/def operands to VGPR_LIFETIME_{START,END} pseudos
///  * Add the VGPRs to basic block live-ins
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUMachineInstrs.h"
#include "AMDGPUMemoryUtils.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-private-object-vgprs"

static cl::opt<unsigned>
    RegChunkSizeInDWords("private-object-reg-chunk-size", cl::Hidden,
                         cl::desc("Number of 32-bit VGPRs per register chunk "
                                  "for promoted private objects"),
                         cl::init(1));

using ObjectRegs = SmallVector<MCPhysReg, 50>;

namespace {

class AMDGPUPrivateObjectVGPRs : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUPrivateObjectVGPRs() : MachineFunctionPass(ID) {
    initializeAMDGPUPrivateObjectVGPRsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Def/use private object VGPRs";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    // LiveVariables only tracks virtual registers and we only touch physical
    // registers.
    AU.addPreserved<LiveVariablesWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreservedID(MachineLoopInfoID);
    AU.addPreservedID(MachineDominatorsID);
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  struct AllocaBBInfo {
    const AllocaInst *Alloca = nullptr;
    bool LiveIn = false;
    bool Starts = false;
    bool Ends = false;
  };

  // Private/shared indexing analysis
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;
  LiveIntervals *LIS = nullptr;

  DenseMap<const AllocaInst *, std::pair<ObjectRegs, MachineMemOperand *>>
      AllocaObjectRegs;

  ObjectRegs computeObjectRegs(const AllocaInst &Alloca) const;
};

} // End anonymous namespace.

INITIALIZE_PASS(AMDGPUPrivateObjectVGPRs, DEBUG_TYPE,
                "AMDGPU Add defs/uses for private object VGPRs", false, false)

char AMDGPUPrivateObjectVGPRs::ID = 0;

char &llvm::AMDGPUPrivateObjectVGPRsID = AMDGPUPrivateObjectVGPRs::ID;

FunctionPass *llvm::createAMDGPUPrivateObjectVGPRsPass() {
  return new AMDGPUPrivateObjectVGPRs();
}

ObjectRegs
AMDGPUPrivateObjectVGPRs::computeObjectRegs(const AllocaInst &Alloca) const {
  ObjectRegs Regs;
  auto &MD = AMDGPU::AllocatedVGPRsMetadata::get(Alloca);
  unsigned Offset = MD.getAddress();
  unsigned Size = MD.getSize();
  assert(Offset % 4 == 0 && Size % 4 == 0);
  unsigned RegWidth = RegChunkSizeInDWords * 32;
  const TargetRegisterClass *BaseRegRC =
      TRI->getAnyVGPRClassForBitWidth(RegWidth);
  if (!BaseRegRC)
    report_fatal_error("Invalid VGPR width " + Twine(RegWidth));
  unsigned BaseRegIdx = Offset / 4;
  MCPhysReg BaseReg = BaseRegRC->getRegister(BaseRegIdx);
  unsigned NumRegs = Size / (RegChunkSizeInDWords * 4);
  for (unsigned I : seq(NumRegs)) {
    MCPhysReg Reg = BaseReg + I * RegChunkSizeInDWords;
    Regs.push_back(Reg);
  }

  if (unsigned LastChunkSize = Size % (RegChunkSizeInDWords * 4)) {
    unsigned LastRegWidth = LastChunkSize * 8;
    const TargetRegisterClass *LastRegRC =
        TRI->getAnyVGPRClassForBitWidth(LastRegWidth);
    if (!LastRegRC)
      report_fatal_error("Invalid VGPR width " + Twine(LastRegWidth));
    unsigned LastRegIdx = BaseRegIdx + RegChunkSizeInDWords * NumRegs;
    MCPhysReg Reg = LastRegRC->getRegister(LastRegIdx);
    Regs.push_back(Reg);
  }

  return Regs;
}

bool AMDGPUPrivateObjectVGPRs::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasVGPRIndexingRegisters())
    return false;
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  LIS = nullptr;
  if (auto *LISWrapper = getAnalysisIfAvailable<LiveIntervalsWrapperPass>())
    LIS = &LISWrapper->getLIS();

  // Sort basic blocks in reverse post-order for the live-out/live-in
  // propagation.
  DenseMap<MachineBasicBlock *, unsigned> BlockToIndex;
  SmallVector<MachineBasicBlock *> IndexToBlock;
  ReversePostOrderTraversal<MachineBasicBlock *> RPOT(&*MF.begin());
  for (MachineBasicBlock *MBB : RPOT) {
    BlockToIndex[MBB] = IndexToBlock.size();
    IndexToBlock.push_back(MBB);
  }

  // Fixed-point iteration to determine basic block live-ins.
  //
  // The first pass of the fixed-point iteration also scans instructions.
  SmallVector<SmallVector<AllocaBBInfo>> BBInfos(IndexToBlock.size());
  SmallBitVector Worklist(IndexToBlock.size());
  bool Changed = false;

  for (bool Dirty = true, FirstPass = true; Dirty; FirstPass = false) {
    Dirty = false;

    for (auto [MBBI, MBB] : enumerate(IndexToBlock)) {
      auto &BBI = BBInfos[MBBI];

      // During the first outer iteration, augment VGPR_LIFETIME_{START,END}
      // with implicit operands and record the initial per-basic block
      // information to compute live-ins.
      if (FirstPass) {
        for (MachineInstr &MI : *MBB) {
          if (MI.getOpcode() == AMDGPU::VGPR_LIFETIME_START ||
              MI.getOpcode() == AMDGPU::VGPR_LIFETIME_END) {
            bool IsStart = MI.getOpcode() == AMDGPU::VGPR_LIFETIME_START;
            MachineMemOperand *MMO = *MI.memoperands_begin();
            auto *Alloca = cast<AllocaInst>(MMO->getValue());

            // Add the implicit operand(s).
            auto ObjRegsIt = AllocaObjectRegs.find(Alloca);
            if (ObjRegsIt == AllocaObjectRegs.end()) {
              ObjRegsIt =
                  AllocaObjectRegs
                      .try_emplace(Alloca, computeObjectRegs(*Alloca), MMO)
                      .first;
            }

            for (MCPhysReg Reg : ObjRegsIt->second.first) {
              MachineOperand Op = MachineOperand::CreateReg(
                  Reg, /*isDef=*/IsStart, /*isImp=*/true, /*isKill=*/!IsStart);
              MI.addOperand(Op);
            }

            // Record the basic block behavior.
            auto It = find_if(BBI, [&](const AllocaBBInfo &Info) {
              return Info.Alloca == Alloca;
            });
            if (It == BBI.end()) {
              BBI.push_back({Alloca, false, false, false});
              It = std::prev(BBI.end());
            }
            It->Starts = IsStart;
            It->Ends = !IsStart;

            Changed = true;
          }
        }
      } else {
        if (!Worklist[MBBI])
          continue;
        Worklist[MBBI] = false;
      }

      // Propagate live-outs into successors.
      for (const auto &ABBI : BBI) {
        if ((ABBI.LiveIn && !ABBI.Ends) || ABBI.Starts) {
          auto &ObjectRegs = AllocaObjectRegs.at(ABBI.Alloca).first;
          for (MachineBasicBlock *Succ : MBB->successors()) {
            unsigned SuccI = BlockToIndex.at(Succ);
            auto &SuccBBI = BBInfos[SuccI];
            auto It = find_if(BBInfos[SuccI], [&](const AllocaBBInfo &Info) {
              return Info.Alloca == ABBI.Alloca;
            });
            bool Update = false;
            if (It == SuccBBI.end()) {
              SuccBBI.push_back({ABBI.Alloca, true, false, false});
              It = std::prev(SuccBBI.end());
              Update = true;
            } else if (!It->LiveIn) {
              It->LiveIn = true;
              Update = true;
            }

            if (Update) {
              if (!It->Starts && !It->Ends) {
                // We are live-out from the successor because of the newly found
                // live-in. If the successor is earlier in RPOT, we will have
                // to re-evaluate it on the next outer iteration.
                if (SuccI < MBBI) {
                  Worklist[SuccI] = true;
                  Dirty = true;
                }
              }

              for (MCPhysReg Reg : ObjectRegs)
                Succ->addLiveIn(Reg);
            }
          }
        }
      }
    }
  }

  // It is legal for the pre-isel LLVM IR to have a lifetime.start without a
  // lifetime.end. Liveness analysis is strong enough to mark physical registers
  // as unused immediately after VGPR_LIFETIME_START in this case.
  //
  // Add VGPR_LIFETIME_END instructions at the end of basic blocks that end the
  // function.
  for (unsigned BBIdx = 0; BBIdx != IndexToBlock.size(); ++BBIdx) {
    MachineBasicBlock *MBB = IndexToBlock[BBIdx];
    if (!MBB->succ_empty())
      continue;

    auto &BBI = BBInfos[BBIdx];
    for (const auto &ABBI : BBI) {
      if ((ABBI.LiveIn || ABBI.Starts) && !ABBI.Ends) {
        // There may be a COPY to a conflicting physical VGPR before a function
        // return. We just iterate backwards over instructions that don't touch
        // memory.
        MachineBasicBlock::iterator IP = MBB->getFirstTerminator();
        while (IP != MBB->begin()) {
          --IP;
          if (IP->mayStore() || IP->mayLoad()) {
            ++IP;
            break;
          }
        }

        const auto &[ObjRegs, MMO] = AllocaObjectRegs.at(ABBI.Alloca);
        MachineInstr *MI =
            BuildMI(*MBB, IP, {}, TII->get(AMDGPU::VGPR_LIFETIME_END))
                .addMemOperand(MMO);

        for (MCPhysReg Reg : ObjRegs) {
          MachineOperand Op = MachineOperand::CreateReg(
              Reg, /*isDef=*/false, /*isImp=*/true, /*isKill=*/true);
          MI->addOperand(Op);
        }

        Changed = true;
      }
    }
  }

  // Remove live ranges from LiveIntervals. They will be recalculated lazily.
  if (LIS) {
    for (const auto &Pair : AllocaObjectRegs) {
      const ObjectRegs &Regs = Pair.second.first;
      for (MCPhysReg Reg : Regs)
        LIS->removeAllRegUnitsForPhysReg(Reg);
    }
  }

  return Changed;
}
