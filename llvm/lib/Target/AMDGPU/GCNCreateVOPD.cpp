//===- GCNCreateVOPD.cpp - Create VOPD Instructions ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Combine VALU pairs into VOPD instructions
/// Only works on wave32
/// Has register requirements, we reject creating VOPD if the requirements are
/// not met.
/// shouldCombineVOPD mutator in postRA machine scheduler puts candidate
/// instructions for VOPD back-to-back
///
//
//===----------------------------------------------------------------------===//
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "GCNVOPDUtils.h"
#include "SIInstrInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gcn-create-vopd"
STATISTIC(NumVOPDCreated, "Number of VOPD Insts Created.");

using namespace llvm;

namespace {

class GCNCreateVOPD {
private:
  class VOPDCombineInfo {
  public:
    VOPDCombineInfo() = default;
    VOPDCombineInfo(MachineInstr *First, MachineInstr *Second)
        : FirstMI(First), SecondMI(Second) {}

    MachineInstr *FirstMI;
    MachineInstr *SecondMI;
  };
  bool TraceEnabled = false;
  int TraceCount = 0;
  llvm::StringRef TraceName;
  int Limit = -1;
  int CombineCount = 0;

public:
  const GCNSubtarget *ST = nullptr;

  bool doReplace(const SIInstrInfo *SII, VOPDCombineInfo &CI) {
    // enforce the limit (only if TEST_VAR was set)
    if (Limit >= 0 && CombineCount >= Limit)
      return false;
    ++CombineCount;

    auto *FirstMI = CI.FirstMI;
    auto *SecondMI = CI.SecondMI;
    unsigned Opc1 = FirstMI->getOpcode();
    unsigned Opc2 = SecondMI->getOpcode();
    unsigned EncodingFamily =
        AMDGPU::getVOPDEncodingFamily(SII->getSubtarget());
    int NewOpcode =
        AMDGPU::getVOPDFull(AMDGPU::getVOPDOpcode(Opc1),
                            AMDGPU::getVOPDOpcode(Opc2), EncodingFamily);
    assert(NewOpcode != -1 &&
           "Should have previously determined this as a possible VOPD\n");

    auto VOPDInst = BuildMI(*FirstMI->getParent(), FirstMI,
                            FirstMI->getDebugLoc(), SII->get(NewOpcode))
                        .setMIFlags(FirstMI->getFlags() | SecondMI->getFlags());

    namespace VOPD = AMDGPU::VOPD;
    MachineInstr *MI[] = {FirstMI, SecondMI};
    auto InstInfo =
        AMDGPU::getVOPDInstInfo(FirstMI->getDesc(), SecondMI->getDesc());

    for (auto CompIdx : VOPD::COMPONENTS) {
      auto MCOprIdx = InstInfo[CompIdx].getIndexOfDstInMCOperands();
      VOPDInst.add(MI[CompIdx]->getOperand(MCOprIdx));
    }

    for (auto CompIdx : VOPD::COMPONENTS) {
      auto CompSrcOprNum = InstInfo[CompIdx].getCompSrcOperandsNum();
      for (unsigned CompSrcIdx = 0; CompSrcIdx < CompSrcOprNum; ++CompSrcIdx) {
        auto MCOprIdx = InstInfo[CompIdx].getIndexOfSrcInMCOperands(CompSrcIdx);
        VOPDInst.add(MI[CompIdx]->getOperand(MCOprIdx));
      }
    }

    SII->fixImplicitOperands(*VOPDInst);
    for (auto CompIdx : VOPD::COMPONENTS)
      VOPDInst.copyImplicitOps(*MI[CompIdx]);

    if (TraceEnabled) {
      ++TraceCount;
      errs() << "VOPD Combine #" << TraceCount << " in function '" << TraceName
             << "':\n";
      errs() << "  Original1: " << *FirstMI;
      errs() << "  Original2: " << *SecondMI;
      errs() << "  Combined:  " << *VOPDInst.getInstr();
    }

    LLVM_DEBUG(dbgs() << "VOPD Fused: " << *VOPDInst << " from\tX: "
                      << *CI.FirstMI << "\tY: " << *CI.SecondMI << "\n");

    for (auto CompIdx : VOPD::COMPONENTS)
      MI[CompIdx]->eraseFromParent();

    ++NumVOPDCreated;
    return true;
  }

  bool run(MachineFunction &MF) {
    ST = &MF.getSubtarget<GCNSubtarget>();
    if (!AMDGPU::hasVOPD(*ST) || !ST->isWave32())
      return false;
    LLVM_DEBUG(dbgs() << "CreateVOPD Pass:\n");

    const SIInstrInfo *SII = ST->getInstrInfo();
    bool Changed = false;
    if (const char *Env = getenv("DB_VOPD_FUN")) {
      TraceName = MF.getName();
      if (TraceName == Env) {
        TraceEnabled = true;
        TraceCount = 0;
      }
    }
    // read TEST_VAR limit
    if (const char *LE = getenv("TEST_VAR"))
      Limit = std::atoi(LE);

    SmallVector<VOPDCombineInfo> ReplaceCandidates;

    for (auto &MBB : MF) {
      auto MII = MBB.begin(), E = MBB.end();
      while (MII != E) {
        auto *FirstMI = &*MII;
        MII = next_nodbg(MII, MBB.end());
        if (MII == MBB.end())
          break;
        if (FirstMI->isDebugInstr())
          continue;
        auto *SecondMI = &*MII;
        unsigned Opc = FirstMI->getOpcode();
        unsigned Opc2 = SecondMI->getOpcode();
        llvm::AMDGPU::CanBeVOPD FirstCanBeVOPD = AMDGPU::getCanBeVOPD(Opc);
        llvm::AMDGPU::CanBeVOPD SecondCanBeVOPD = AMDGPU::getCanBeVOPD(Opc2);
        VOPDCombineInfo CI;

        if (FirstCanBeVOPD.X && SecondCanBeVOPD.Y)
          CI = VOPDCombineInfo(FirstMI, SecondMI);
        else if (FirstCanBeVOPD.Y && SecondCanBeVOPD.X)
          CI = VOPDCombineInfo(SecondMI, FirstMI);
        else
          continue;
        // checkVOPDRegConstraints cares about program order, but doReplace
        // cares about X-Y order in the constituted VOPD
        if (llvm::checkVOPDRegConstraints(*SII, *FirstMI, *SecondMI)) {
          ReplaceCandidates.push_back(CI);
          ++MII;
        }
      }
    }
    for (auto &CI : ReplaceCandidates) {
      Changed |= doReplace(SII, CI);
    }

    return Changed;
  }
};

class GCNCreateVOPDLegacy : public MachineFunctionPass {
public:
  static char ID;
  GCNCreateVOPDLegacy() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return "GCN Create VOPD Instructions";
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;

    return GCNCreateVOPD().run(MF);
  }
};

} // namespace

PreservedAnalyses
llvm::GCNCreateVOPDPass::run(MachineFunction &MF,
                             MachineFunctionAnalysisManager &AM) {
  if (!GCNCreateVOPD().run(MF))
    return PreservedAnalyses::all();
  return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
}

char GCNCreateVOPDLegacy::ID = 0;

char &llvm::GCNCreateVOPDID = GCNCreateVOPDLegacy::ID;

INITIALIZE_PASS(GCNCreateVOPDLegacy, DEBUG_TYPE, "GCN Create VOPD Instructions",
                false, false)
