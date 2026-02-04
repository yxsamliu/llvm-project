//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower LDS global variables with target extension type "amdgpu.named.barrier"
// that require specialized address assignment. It assigns a unique
// barrier identifier to each named-barrier LDS variable and encodes
// this identifier within the !absolute_symbol metadata of that global.
// This encoding ensures that subsequent LDS lowering passes can process these
// barriers correctly without conflicts.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUMemoryUtils.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

#include <algorithm>

#define DEBUG_TYPE "amdgpu-lower-exec-sync"

using namespace llvm;
using namespace AMDGPU;

namespace {

// If GV is also used directly by other kernels, create a new GV
// used only by this kernel and its function.
static GlobalVariable *uniquifyGVPerKernel(Module &M, GlobalVariable *GV,
                                           Function *KF) {
  bool NeedsReplacement = false;
  for (Use &U : GV->uses()) {
    if (auto *I = dyn_cast<Instruction>(U.getUser())) {
      Function *F = I->getFunction();
      if (isKernel(*F) && !getWavegroupRankFunction(*F) && F != KF) {
        NeedsReplacement = true;
        break;
      }
    }
  }
  if (!NeedsReplacement)
    return GV;
  // Create a new GV used only by this kernel and its function
  GlobalVariable *NewGV = new GlobalVariable(
      M, GV->getValueType(), GV->isConstant(), GV->getLinkage(),
      GV->getInitializer(), GV->getName() + "." + KF->getName(), nullptr,
      GV->getThreadLocalMode(), GV->getType()->getAddressSpace());
  NewGV->copyAttributesFrom(GV);
  for (Use &U : make_early_inc_range(GV->uses())) {
    if (auto *I = dyn_cast<Instruction>(U.getUser())) {
      Function *F = I->getFunction();
      if (!isKernel(*F) || getWavegroupRankFunction(*F) || F == KF) {
        U.getUser()->replaceUsesOfWith(GV, NewGV);
      }
    }
  }
  return NewGV;
}

// Write the specified address into metadata where it can be retrieved by
// the assembler. Format is a half open range, [Address Address+1)
static void recordLDSAbsoluteAddress(Module *M, GlobalVariable *GV,
                                     uint32_t Address) {
  LLVMContext &Ctx = M->getContext();
  auto *IntTy = M->getDataLayout().getIntPtrType(Ctx, AMDGPUAS::LOCAL_ADDRESS);
  auto *MinC = ConstantAsMetadata::get(ConstantInt::get(IntTy, Address));
  auto *MaxC = ConstantAsMetadata::get(ConstantInt::get(IntTy, Address + 1));
  GV->setMetadata(LLVMContext::MD_absolute_symbol,
                  MDNode::get(Ctx, {MinC, MaxC}));
}

template <typename T> SmallVector<T> sortByName(SmallVector<T> &&V) {
  sort(V, [](const auto *L, const auto *R) {
    return L->getName() < R->getName();
  });
  return {std::move(V)};
}

// Main utility function for special LDS variables lowering.
static bool lowerExecSyncGlobalVariables(
    Module &M, LDSUsesInfoTy &LDSUsesInfo,
    VariableFunctionMap &LDSToKernelsThatNeedToAccessItIndirectly) {
  bool Changed = false;
  const DataLayout &DL = M.getDataLayout();

  unsigned NumSemAbsolutes[MAX_WAVES_PER_WAVEGROUP] = {0};
  constexpr unsigned NumBarScopes =
      static_cast<unsigned>(Barrier::Scope::NUM_SCOPES);
  unsigned NumBarAbsolutes[NumBarScopes] = {0};

  // The 1st round: give module-absolute assignments
  SmallVector<GlobalVariable *> OrderedGVs;
  for (auto &K : LDSToKernelsThatNeedToAccessItIndirectly) {
    GlobalVariable *GV = K.first;
    if (!(isNamedBarrier(*GV) || isLDSSemaphore(*GV)))
      continue;

    // Give a module-absolute assignment if it is indirectly accessed by
    // multiple kernels. This is not precise, but we don't want to duplicate
    // a function when it is called by multiple kernels.
    if (LDSToKernelsThatNeedToAccessItIndirectly[GV].size() > 1) {
      OrderedGVs.push_back(GV);
    } else {
      // Leave it to the 2nd round, which will give a kernel-relative
      // assignment if it is only indirectly accessed by one kernel
      LDSUsesInfo.direct_access[*K.second.begin()].insert(GV);
    }
    LDSToKernelsThatNeedToAccessItIndirectly.erase(GV);
  }
  OrderedGVs = sortByName(std::move(OrderedGVs));
  for (GlobalVariable *GV : OrderedGVs) {
    unsigned Offset;
    if (TargetExtType *ExtTy = isNamedBarrier(*GV)) {
      unsigned BarrierScope = ExtTy->getIntParameter(0);
      unsigned BarId = NumBarAbsolutes[BarrierScope] + 1;
      unsigned BarCnt = GV->getGlobalSize(DL) / 16;
      NumBarAbsolutes[BarrierScope] += BarCnt;

      // 4 bits for alignment, 5 bits for the barrier num,
      // 3 bits for the barrier scope
      Offset = 0x802000u | BarrierScope << 9 | BarId << 4;

    } else if (TargetExtType *ExtTy = isLDSSemaphore(*GV)) {
      unsigned OwningRank = ExtTy->getIntParameter(0);
      assert(OwningRank < MAX_WAVES_PER_WAVEGROUP); 
      unsigned Num = ++NumSemAbsolutes[OwningRank];

      // 4 bits for alignment, 4 bits for the semaphore num,
      // 4 bits for the owning rank
      Offset = 0x801000u | OwningRank << 8 | Num << 4;

    } else
      llvm_unreachable("Unhandled special variable type.");

    recordLDSAbsoluteAddress(&M, GV, Offset);
  }
  OrderedGVs.clear();

  // The 2nd round: give a kernel-relative assignment for GV that
  // either only indirectly accessed by single kernel or only directly
  // accessed by multiple kernels.
  SmallVector<Function *> OrderedKernels;
  for (auto &K : LDSUsesInfo.direct_access) {
    Function *F = K.first;
    assert(isKernel(*F));
    OrderedKernels.push_back(F);
  }
  OrderedKernels = sortByName(std::move(OrderedKernels));

  DenseMap<Function *, unsigned> Kernel2BarId[NumBarScopes];
  DenseMap<Function *, unsigned> Kernel2SemRelative[MAX_WAVES_PER_WAVEGROUP];
  for (Function *F : OrderedKernels) {

    // Collect all globals for each kernel.
    for (GlobalVariable *GV : LDSUsesInfo.direct_access[F]) {
      if (!(isNamedBarrier(*GV) || isLDSSemaphore(*GV)))
        continue;

      LDSUsesInfo.direct_access[F].erase(GV);
      if (GV->isAbsoluteSymbolRef()) {
        // Already assigned
        continue;
      }
      OrderedGVs.push_back(GV);
    }

    OrderedGVs = sortByName(std::move(OrderedGVs));
    for (GlobalVariable *GV : OrderedGVs) {
      // GV could also be used directly by other kernels. If so, we need to
      // create a new GV used only by this kernel and its function.
      auto NewGV = uniquifyGVPerKernel(M, GV, F);
      Changed |= (NewGV != GV);
      unsigned Offset;
      if (TargetExtType *ExtTy = isNamedBarrier(*GV)) {
        // Place each barrier in the next open slot above the module-relative
        // and already assigned kernel-relative barriers.
        unsigned BarrierScope = ExtTy->getIntParameter(0);
        unsigned BarId = Kernel2BarId[BarrierScope][F];
        BarId += NumBarAbsolutes[BarrierScope] + 1;
        unsigned BarCnt = GV->getGlobalSize(DL) / 16;
        Kernel2BarId[BarrierScope][F] += BarCnt;
        Offset = 0x802000u | BarrierScope << 9 | BarId << 4;

      } else if (TargetExtType *ExtTy = isLDSSemaphore(*GV)) {
        // Determine which semaphore GVs were already assigned, and for the
        // remaining ones assign the semaphore nums above.
        unsigned OwningRank =
            ExtTy->getIntParameter(0) % MAX_WAVES_PER_WAVEGROUP;
        unsigned Num = NumSemAbsolutes[OwningRank];
        Kernel2SemRelative[OwningRank][F]++;
        Num += Kernel2SemRelative[OwningRank][F];
        Offset = 0x801000u | OwningRank << 8 | Num << 4;

      } else
        llvm_unreachable("Unhandled special variable type.");
      recordLDSAbsoluteAddress(&M, NewGV, Offset);
    }
    OrderedGVs.clear();
  }
  // Also erase those special LDS variables from indirect_access.
  for (auto &K : LDSUsesInfo.indirect_access) {
    assert(isKernel(*K.first));
    for (GlobalVariable *GV : K.second) {
      if (isNamedBarrier(*GV) || isLDSSemaphore(*GV))
        K.second.erase(GV);
    }
  }
  return Changed;
}

static bool runLowerExecSyncGlobals(Module &M) {
  CallGraph CG = CallGraph(M);
  bool Changed = false;
  Changed |= eliminateConstantExprUsesOfLDSFromAllInstructions(M);

  // For each kernel, what variables does it access directly or through
  // callees
  LDSUsesInfoTy LDSUsesInfo = getTransitiveUsesOfLDS(CG, M);

  // For each variable accessed through callees, which kernels access it
  VariableFunctionMap LDSToKernelsThatNeedToAccessItIndirectly;
  for (auto &K : LDSUsesInfo.indirect_access) {
    Function *F = K.first;
    assert(isKernel(*F));
    for (GlobalVariable *GV : K.second) {
      LDSToKernelsThatNeedToAccessItIndirectly[GV].insert(F);
    }
  }

  if (LDSUsesInfo.HasSpecialGVs) {
    // Special LDS variables need special address assignment
    Changed |= lowerExecSyncGlobalVariables(
        M, LDSUsesInfo, LDSToKernelsThatNeedToAccessItIndirectly);
  }
  return Changed;
}

class AMDGPULowerExecSyncLegacy : public ModulePass {
public:
  static char ID;
  AMDGPULowerExecSyncLegacy() : ModulePass(ID) {}
  bool runOnModule(Module &M) override;
};

} // namespace

char AMDGPULowerExecSyncLegacy::ID = 0;
char &llvm::AMDGPULowerExecSyncLegacyPassID = AMDGPULowerExecSyncLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPULowerExecSyncLegacy, DEBUG_TYPE,
                      "AMDGPU lowering of execution synchronization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AMDGPULowerExecSyncLegacy, DEBUG_TYPE,
                    "AMDGPU lowering of execution synchronization", false,
                    false)

bool AMDGPULowerExecSyncLegacy::runOnModule(Module &M) {
  return runLowerExecSyncGlobals(M);
}

ModulePass *llvm::createAMDGPULowerExecSyncLegacyPass() {
  return new AMDGPULowerExecSyncLegacy();
}

PreservedAnalyses AMDGPULowerExecSyncPass::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  return runLowerExecSyncGlobals(M) ? PreservedAnalyses::none()
                                    : PreservedAnalyses::all();
}
