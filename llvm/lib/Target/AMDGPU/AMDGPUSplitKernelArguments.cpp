#include "AMDGPU.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Cloning.h"

#define DEBUG_TYPE "amdgpu-split-kernel-arguments"

using namespace llvm;

namespace {

class AMDGPUSplitKernelArguments : public FunctionPass {
public:
  static char ID;

  AMDGPUSplitKernelArguments() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};

} // end anonymous namespace

bool AMDGPUSplitKernelArguments::runOnFunction(Function &F) {
  LLVM_DEBUG(dbgs() << "Entering AMDGPUSplitKernelArguments::runOnFunction " 
                    << F.getName() << '\n');
  if (skipFunction(F)) {
    LLVM_DEBUG(dbgs() << "skipFunction\n");
    return false;
  }

  CallingConv::ID CC = F.getCallingConv();
  if (CC != CallingConv::AMDGPU_KERNEL || F.arg_empty()) {
    LLVM_DEBUG(dbgs() << "non-kernel or arg_empty\n");
    return false;
  }

  SmallVector<Argument *, 8> StructArgs;
  SmallVector<Type *, 8> NewArgTypes;
  SmallVector<Value *, 8> NewArgs;

  // Collect struct arguments and new argument types
  for (Argument &Arg : F.args()) {
    LLVM_DEBUG(dbgs() << "Processing argument: " << Arg << "\n");
    if (Arg.use_empty()) {
      NewArgTypes.push_back(Arg.getType());
      LLVM_DEBUG(dbgs() << "use empty\n");
      continue;
    }

    PointerType *PT = dyn_cast<PointerType>(Arg.getType());
    if (!PT) {
      NewArgTypes.push_back(Arg.getType());
      LLVM_DEBUG(dbgs() << "not a pointer\n");
      continue;
    }
    
    const bool IsByRef = Arg.hasByRefAttr();
    if (!IsByRef) {
      NewArgTypes.push_back(Arg.getType());
      LLVM_DEBUG(dbgs() << "not byref\n");
      continue;
    }
    
    Type *ArgTy = Arg.getParamByRefType();
    StructType *ST = dyn_cast<StructType>(ArgTy);
    if (!ST) {
      NewArgTypes.push_back(Arg.getType());
      LLVM_DEBUG(dbgs() << "not a struct\n");
      continue;
    }

    bool AllLoadsOrGEPs = true;
    for (User *U : Arg.users()) {
      LLVM_DEBUG(dbgs() << "  User: " << *U << "\n");
      if (auto *LI = dyn_cast<LoadInst>(U)) {
        NewArgTypes.push_back(LI->getType());
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        for (User *GEPUser : GEP->users()) {
        LLVM_DEBUG(dbgs() << "    GEP User: " << *GEPUser << "\n");
          if (auto *GEPLoad = dyn_cast<LoadInst>(GEPUser)) {
            NewArgTypes.push_back(GEPLoad->getType());
          } else {
            AllLoadsOrGEPs = false;
            break;
          }
        }
      } else {
        AllLoadsOrGEPs = false;
        break;
      }
    }
    LLVM_DEBUG(dbgs() << "  AllLoadsOrGEPs: " << (AllLoadsOrGEPs ? "true" : "false") << "\n");

    if (AllLoadsOrGEPs) {
      StructArgs.push_back(&Arg);
    } else {
      NewArgTypes.push_back(Arg.getType());
    }
  }

  if (StructArgs.empty())
    return false;

  // Create new function type
  FunctionType *NewFT = FunctionType::get(F.getReturnType(), NewArgTypes, F.isVarArg());
  Function *NewF = Function::Create(NewFT, F.getLinkage(), F.getName(), F.getParent());

  // Transfer function attributes
  NewF->copyAttributesFrom(&F);

  // Set calling convention
  NewF->setCallingConv(F.getCallingConv());

  // Map old arguments to new arguments
  ValueToValueMapTy VMap;
  auto NewArgIt = NewF->arg_begin();
  for (Argument &Arg : F.args()) {
    if (std::find(StructArgs.begin(), StructArgs.end(), &Arg) != StructArgs.end()) {
      for (User *U : Arg.users()) {
        if (auto *LI = dyn_cast<LoadInst>(U)) {
          VMap[LI] = &*NewArgIt++;
        } else if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
          for (User *GEPUser : GEP->users()) {
            if (auto *GEPLoad = dyn_cast<LoadInst>(GEPUser)) {
              VMap[GEPLoad] = &*NewArgIt++;
            }
          }
        }
      }
    } else {
      VMap[&Arg] = &*NewArgIt++;
    }
  }

  // Clone the function body
  SmallVector<ReturnInst *, 8> Returns;
  CloneFunctionInto(NewF, &F, VMap, CloneFunctionChangeType::LocalChangesOnly, Returns);

  // Replace old function with new function
  F.replaceAllUsesWith(NewF);
  F.eraseFromParent();

  return true;
}

INITIALIZE_PASS_BEGIN(AMDGPUSplitKernelArguments, DEBUG_TYPE,
                      "AMDGPU Split Kernel Arguments", false, false)
INITIALIZE_PASS_END(AMDGPUSplitKernelArguments, DEBUG_TYPE,
                    "AMDGPU Split Kernel Arguments", false, false)

char AMDGPUSplitKernelArguments::ID = 0;

FunctionPass *llvm::createAMDGPUSplitKernelArgumentsPass() {
  return new AMDGPUSplitKernelArguments();
}

PreservedAnalyses
AMDGPUSplitKernelArgumentsPass::run(Function &F,
                                    FunctionAnalysisManager &AM) {
  if (AMDGPUSplitKernelArguments().runOnFunction(F)) {
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    return PA;
  }
  return PreservedAnalyses::all();
}