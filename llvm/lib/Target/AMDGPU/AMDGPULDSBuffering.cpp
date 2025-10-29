//===-- AMDGPULDSBuffering.cpp - Per-thread LDS buffering -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass identifies profitable per-thread global memory aggregates and
// buffers them through LDS (addrspace(3)) to improve performance in
// memory-bound kernels. It is intended to run late, after SROA and
// AMDGPUPromoteAlloca, and uses leftover LDS budget.
//
// The initial implementation provides the pass skeleton and shared
// utilities for LDS allocation and thread id computation. Transformation
// logic will be added incrementally under a feature flag.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Instructions.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "amdgpu-lds-buffering"

using namespace llvm;

namespace {

static cl::opt<bool>
    EnableLDSBuffering("amdgpu-lds-buffering-enable",
                       cl::desc("Enable per-thread LDS buffering of globals"),
                       cl::init(true));

static cl::opt<unsigned> LDSBufferingMaxBytes(
    "amdgpu-lds-buffering-max-bytes",
    cl::desc("Max byte size for LDS buffering candidates"), cl::init(64));

class AMDGPULDSBufferingImpl {
  const TargetMachine &TM;
  Module *Mod = nullptr;
  const DataLayout *DL = nullptr;
  bool IsAMDGCN = false;
  bool IsAMDHSA = false;

public:
  AMDGPULDSBufferingImpl(const TargetMachine &TM) : TM(TM) {}

  bool run(Function &F) {
    LLVM_DEBUG(dbgs() << "[LDSBuffer] Visit function: " << F.getName()
                      << '\n');
    if (!EnableLDSBuffering)
      return false;

    // Only consider AMDGPU targets.
    const Triple &TT = TM.getTargetTriple();
    if (!TT.isAMDGCN())
      return false;
    IsAMDGCN = true;
    IsAMDHSA = TT.getOS() == Triple::AMDHSA;

    // Kernels only for now.
    if (!AMDGPU::isEntryFunctionCC(F.getCallingConv()))
      return false;

    Mod = F.getParent();
    DL = &Mod->getDataLayout();

    bool Changed = false;
    unsigned NumTransformed = 0;

    // Minimal pattern: a load from AS(1) whose only use is a store back to the
    // exact same pointer later. Replace with global<->LDS memcpy pair to
    // shorten the live range and free VGPRs.
    SmallVector<Instruction *> ToErase;
    for (BasicBlock &BB : F) {
      for (Instruction &I : llvm::make_early_inc_range(BB)) {
        auto *LI = dyn_cast<LoadInst>(&I);
        if (!LI || LI->isVolatile())
          continue;

        Type *ValTy = LI->getType();
        // Only handle fixed-size first-class types (e.g. vectors, i64/i128).
        if (!ValTy->isFirstClassType())
          continue;

        // Only from global addrspace(1) for now.
        Value *Ptr = LI->getPointerOperand();
        auto *PtrTy = dyn_cast<PointerType>(Ptr->getType());
        if (!PtrTy || PtrTy->getAddressSpace() != AMDGPUAS::GLOBAL_ADDRESS)
          continue;

        // Ensure single user which is a store of the loaded value.
        if (!LI->hasOneUse())
          continue;
        auto *SI = dyn_cast<StoreInst>(LI->user_back());
        if (!SI || SI->isVolatile())
          continue;
        if (SI->getValueOperand() != LI)
          continue;

        // Store must target the same pointer SSA value (bitcasts allowed).
        Value *SPtr = SI->getPointerOperand();
        if (SPtr->stripPointerCasts() != Ptr->stripPointerCasts())
          continue;

        // Size/align guards.
    TypeSize TS = DL->getTypeStoreSize(ValTy);
    if (TS.isScalable())
          continue;
    uint64_t Size = TS.getFixedValue();
    if (Size == 0 || Size > LDSBufferingMaxBytes)
          continue;
        Align LoadAlign = LI->getAlign();
        Align MinAlign = Align(16);
        if (LoadAlign < MinAlign)
          continue;

        // Create LDS slot near the load and emit memcpy global->LDS.
        LLVM_DEBUG({
          dbgs() << "[LDSBuffer] Candidate found: load->store same ptr in "
                  << F.getName() << '\n';
          dbgs() << "            size=" << Size << "B, align="
                  << LoadAlign.value() << ", ptr AS="
                  << PtrTy->getAddressSpace() << "\n";
        });
        IRBuilder<> BLoad(LI);
        Align Alignment = LoadAlign;
        auto [GV, SlotPtr] =
            createLDSGlobalAndThreadSlot(F, ValTy, Alignment, "ldsbuf", BLoad);
        // memcpy p3 <- p1
        LLVM_DEBUG(dbgs() << "[LDSBuffer] Insert memcpy global->LDS: "
                          << GV->getName() << ", bytes=" << Size
                          << ", align=" << Alignment.value() << '\n');
        BLoad.CreateMemCpy(SlotPtr, Alignment, Ptr, Alignment, TS);

        // Replace the final store with memcpy LDS->global.
        IRBuilder<> BStore(SI);
        LLVM_DEBUG(dbgs() << "[LDSBuffer] Insert memcpy LDS->global: "
                          << GV->getName() << ", bytes=" << Size
                          << ", align=" << Alignment.value() << '\n');
        BStore.CreateMemCpy(SPtr, Alignment, SlotPtr, Alignment, TS);

        ToErase.push_back(SI);
        ToErase.push_back(LI);
        LLVM_DEBUG(dbgs() << "[LDSBuffer] Erase original load/store pair\n");
        Changed = true;
        ++NumTransformed;
      }
    }

    for (Instruction *E : ToErase)
      E->eraseFromParent();

    LLVM_DEBUG(dbgs() << "[LDSBuffer] Transformations applied: "
                      << NumTransformed << "\n");

    return Changed;
  }

private:
  // Get local size Y and Z from the dispatch packet on HSA.
  std::pair<Value *, Value *> getLocalSizeYZ(IRBuilder<> &Builder) {
    Function &F = *Builder.GetInsertBlock()->getParent();
    const AMDGPUSubtarget &ST = AMDGPUSubtarget::get(TM, F);

    // Read from dispatch packet.
    CallInst *DispatchPtr =
        Builder.CreateIntrinsic(Intrinsic::amdgcn_dispatch_ptr, {}, {});
    DispatchPtr->addRetAttr(Attribute::NoAlias);
    DispatchPtr->addRetAttr(Attribute::NonNull);
    F.removeFnAttr("amdgpu-no-dispatch-ptr");
    DispatchPtr->addDereferenceableRetAttr(64);

    Type *I32Ty = Type::getInt32Ty(Mod->getContext());
    Value *GEPXY = Builder.CreateConstInBoundsGEP1_64(I32Ty, DispatchPtr, 1);
    LoadInst *LoadXY = Builder.CreateAlignedLoad(I32Ty, GEPXY, Align(4));
    Value *GEPZU = Builder.CreateConstInBoundsGEP1_64(I32Ty, DispatchPtr, 2);
    LoadInst *LoadZU = Builder.CreateAlignedLoad(I32Ty, GEPZU, Align(4));
    MDNode *MD = MDNode::get(Mod->getContext(), {});
    LoadXY->setMetadata(LLVMContext::MD_invariant_load, MD);
    LoadZU->setMetadata(LLVMContext::MD_invariant_load, MD);
    ST.makeLIDRangeMetadata(LoadZU);
    Value *Y = Builder.CreateLShr(LoadXY, 16);
    return std::pair(Y, LoadZU);
  }

  // Get workitem id for dimension N (0,1,2).
  Value *getWorkitemID(IRBuilder<> &Builder, unsigned N) {
    Function *F = Builder.GetInsertBlock()->getParent();
    const AMDGPUSubtarget &ST = AMDGPUSubtarget::get(TM, *F);
    Intrinsic::ID IntrID = Intrinsic::not_intrinsic;
    StringRef AttrName;
    switch (N) {
    case 0:
      IntrID = Intrinsic::amdgcn_workitem_id_x;
      AttrName = "amdgpu-no-workitem-id-x";
      break;
    case 1:
      IntrID = Intrinsic::amdgcn_workitem_id_y;
      AttrName = "amdgpu-no-workitem-id-y";
      break;
    case 2:
      IntrID = Intrinsic::amdgcn_workitem_id_z;
      AttrName = "amdgpu-no-workitem-id-z";
      break;
    default:
      llvm_unreachable("invalid dimension");
    }
    Function *WorkitemIdFn = Intrinsic::getOrInsertDeclaration(Mod, IntrID);
    CallInst *CI = Builder.CreateCall(WorkitemIdFn);
    ST.makeLIDRangeMetadata(CI);
    F->removeFnAttr(AttrName);
    return CI;
  }

  // Compute linear thread id within a workgroup.
  Value *buildLinearThreadId(IRBuilder<> &Builder) {
    Value *TCntY, *TCntZ;
    std::tie(TCntY, TCntZ) = getLocalSizeYZ(Builder);
    Value *TIdX = getWorkitemID(Builder, 0);
    Value *TIdY = getWorkitemID(Builder, 1);
    Value *TIdZ = getWorkitemID(Builder, 2);
    Value *Tmp0 = Builder.CreateMul(TCntY, TCntZ, "", true, true);
    Tmp0 = Builder.CreateMul(Tmp0, TIdX);
    Value *Tmp1 = Builder.CreateMul(TIdY, TCntZ, "", true, true);
    Value *TID = Builder.CreateAdd(Tmp0, Tmp1);
    TID = Builder.CreateAdd(TID, TIdZ);
    return TID;
  }

  // Create an LDS array [WGSize x ElemTy] and return pointer to per-thread slot.
  std::pair<GlobalVariable *, Value *>
  createLDSGlobalAndThreadSlot(Function &F, Type *ElemTy, Align Alignment,
                               StringRef BaseName, IRBuilder<> &Builder) {
    const AMDGPUSubtarget &ST = AMDGPUSubtarget::get(TM, F);
    unsigned WorkGroupSize = ST.getFlatWorkGroupSizes(F).second;
    Type *ArrTy = ArrayType::get(ElemTy, WorkGroupSize);
    GlobalVariable *GV = new GlobalVariable(
        *Mod, ArrTy, /*isConstant=*/false, GlobalValue::InternalLinkage,
        PoisonValue::get(ArrTy), (F.getName() + "." + BaseName).str(),
        nullptr, GlobalVariable::NotThreadLocal, AMDGPUAS::LOCAL_ADDRESS);
    GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    GV->setAlignment(Alignment);

    LLVM_DEBUG({
      dbgs() << "[LDSBuffer] Create LDS global: name=" << GV->getName()
              << ", elemTy=" << *ElemTy << ", WGSize=" << WorkGroupSize
              << ", align=" << Alignment.value() << '\n';
    });

    Value *LinearTID = buildLinearThreadId(Builder);
    LLVMContext &Ctx = Mod->getContext();
    Value *Indices[] = {Constant::getNullValue(Type::getInt32Ty(Ctx)),
                        LinearTID};
    Value *SlotPtr = Builder.CreateInBoundsGEP(ArrTy, GV, Indices);
    return {GV, SlotPtr};
  }
};

} // end anonymous namespace

PreservedAnalyses
AMDGPULDSBufferingPass::run(Function &F, FunctionAnalysisManager &AM) {
  bool Changed = AMDGPULDSBufferingImpl(TM).run(F);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

//===----------------------------------------------------------------------===//
// Legacy PM wrapper
//===----------------------------------------------------------------------===//

namespace {

class AMDGPULDSBufferingLegacy : public FunctionPass {
public:
  static char ID;
  AMDGPULDSBufferingLegacy() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "AMDGPU LDS Buffering"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    FunctionPass::getAnalysisUsage(AU);
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    if (auto *TPC = getAnalysisIfAvailable<TargetPassConfig>())
      return AMDGPULDSBufferingImpl(TPC->getTM<TargetMachine>()).run(F);
    return false;
  }
};

} // end anonymous namespace

char AMDGPULDSBufferingLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPULDSBufferingLegacy, DEBUG_TYPE,
                      "AMDGPU per-thread LDS buffering", false, false)
INITIALIZE_PASS_END(AMDGPULDSBufferingLegacy, DEBUG_TYPE,
                    "AMDGPU per-thread LDS buffering", false, false)

FunctionPass *llvm::createAMDGPULDSBufferingLegacyPass() {
  return new AMDGPULDSBufferingLegacy();
}
