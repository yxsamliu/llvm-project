#include "AMDGPU.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Cloning.h"

#define DEBUG_TYPE "amdgpu-split-kernel-arguments"

using namespace llvm;

namespace {
static llvm::cl::opt<bool> DisableSplitKernelArgs(
    "disable-amdgpu-split-kernel-args",
    llvm::cl::desc("Disable splitting of AMDGPU kernel arguments"),
    llvm::cl::init(false));

class AMDGPUSplitKernelArguments : public ModulePass {
public:
  static char ID;

  AMDGPUSplitKernelArguments() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

private:
  bool processFunction(Function &F);
};
} // end anonymous namespace

bool AMDGPUSplitKernelArguments::processFunction(Function &F) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  LLVM_DEBUG(dbgs() << "Entering AMDGPUSplitKernelArguments::processFunction "
                    << F.getName() << '\n');
  if (F.isDeclaration()) {
    LLVM_DEBUG(dbgs() << "Function is a declaration, skipping\n");
    return false;
  }

  CallingConv::ID CC = F.getCallingConv();
  if (CC != CallingConv::AMDGPU_KERNEL || F.arg_empty()) {
    LLVM_DEBUG(dbgs() << "non-kernel or arg_empty\n");
    return false;
  }

  // Define a struct to hold load information for sorting
  struct LoadInfo {
    LoadInst *Load;
    Argument *Arg;
    uint64_t Offset;
    unsigned OriginalArgIndex;
  };

  SmallVector<LoadInfo, 16> AllLoads;
  SmallVector<Type *, 8> NewArgTypes;
  SmallVector<std::tuple<unsigned, unsigned, uint64_t>, 8> NewArgMappings;

  unsigned OriginalArgIndex = 0;
  unsigned NewArgIndex = 0;
  for (Argument &Arg : F.args()) {
    LLVM_DEBUG(dbgs() << "Processing argument: " << Arg << "\n");
    if (Arg.use_empty()) {
      NewArgTypes.push_back(Arg.getType());
      NewArgMappings.emplace_back(NewArgIndex, OriginalArgIndex, 0);
      ++NewArgIndex;
      ++OriginalArgIndex;
      LLVM_DEBUG(dbgs() << "use empty\n");
      continue;
    }

    PointerType *PT = dyn_cast<PointerType>(Arg.getType());
    if (!PT) {
      NewArgTypes.push_back(Arg.getType());
      LLVM_DEBUG(dbgs() << "not a pointer\n");
      if (NewArgIndex != OriginalArgIndex)
        NewArgMappings.emplace_back(NewArgIndex, OriginalArgIndex, 0);
      ++NewArgIndex;
      ++OriginalArgIndex;
      continue;
    }

    const bool IsByRef = Arg.hasByRefAttr();
    if (!IsByRef) {
      NewArgTypes.push_back(Arg.getType());
      LLVM_DEBUG(dbgs() << "not byref\n");
      if (NewArgIndex != OriginalArgIndex)
        NewArgMappings.emplace_back(NewArgIndex, OriginalArgIndex, 0);
      ++NewArgIndex;
      ++OriginalArgIndex;
      continue;
    }

    Type *ArgTy = Arg.getParamByRefType();
    StructType *ST = dyn_cast<StructType>(ArgTy);
    if (!ST) {
      NewArgTypes.push_back(Arg.getType());
      LLVM_DEBUG(dbgs() << "not a struct\n");
      if (NewArgIndex != OriginalArgIndex)
        NewArgMappings.emplace_back(NewArgIndex, OriginalArgIndex, 0);
      ++NewArgIndex;
      ++OriginalArgIndex;
      continue;
    }

    bool AllLoadsOrGEPs = true;
    SmallVector<LoadInst *, 8> Loads;
    SmallVector<GetElementPtrInst *, 8> GEPs;
    for (User *U : Arg.users()) {
      LLVM_DEBUG(dbgs() << "  User: " << *U << "\n");
      if (auto *LI = dyn_cast<LoadInst>(U)) {
        Loads.push_back(LI);
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        GEPs.push_back(GEP);
        for (User *GEPUser : GEP->users()) {
          LLVM_DEBUG(dbgs() << "    GEP User: " << *GEPUser << "\n");
          if (auto *GEPLoad = dyn_cast<LoadInst>(GEPUser)) {
            Loads.push_back(GEPLoad);
          } else {
            AllLoadsOrGEPs = false;
            break;
          }
        }
      } else {
        AllLoadsOrGEPs = false;
        break;
      }
      if (!AllLoadsOrGEPs)
        break;
    }
    LLVM_DEBUG(dbgs() << "  AllLoadsOrGEPs: "
                      << (AllLoadsOrGEPs ? "true" : "false") << "\n");

    if (AllLoadsOrGEPs) {
      for (LoadInst *LI : Loads) {
        // Compute offset
        uint64_t Offset = 0;
        if (auto *GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand())) {
          APInt OffsetAPInt(DL.getPointerSizeInBits(), 0);
          if (GEP->accumulateConstantOffset(DL, OffsetAPInt))
            Offset = OffsetAPInt.getZExtValue();
        }

        AllLoads.emplace_back(LoadInfo{LI, &Arg, Offset, OriginalArgIndex});
      }
    } else {
      // Argument is not eligible for splitting; keep it as is
      NewArgTypes.push_back(Arg.getType());
      if (NewArgIndex != OriginalArgIndex)
        NewArgMappings.emplace_back(NewArgIndex, OriginalArgIndex, 0);
      ++NewArgIndex;
    }
    ++OriginalArgIndex;
  }

  if (AllLoads.empty())
    return false;

  // Sort the loads by OriginalArgIndex and Offset
  std::sort(AllLoads.begin(), AllLoads.end(), [](const LoadInfo &A, const LoadInfo &B) {
    if (A.OriginalArgIndex != B.OriginalArgIndex)
      return A.OriginalArgIndex < B.OriginalArgIndex;
    return A.Offset < B.Offset;
  });

  // Rebuild NewArgTypes and NewArgMappings
  // First, process non-split arguments
  // Then, append sorted scalar arguments from split struct arguments
  SmallVector<Type *, 8> FinalNewArgTypes;
  SmallVector<std::tuple<unsigned, unsigned, uint64_t>, 8> FinalNewArgMappings;
  unsigned NewArgIndexFinal = 0;
  unsigned OriginalArgIndexFinal = 0;
  for (Argument &Arg : F.args()) {
    bool IsSplit = false;
    // Check if this argument was split
    for (const LoadInfo &LI : AllLoads) {
      if (LI.Arg == &Arg) {
        IsSplit = true;
        break;
      }
    }

    if (!IsSplit) {
      // Argument is not split; add its type
      FinalNewArgTypes.push_back(Arg.getType());
      if (NewArgIndexFinal != OriginalArgIndexFinal)
        FinalNewArgMappings.emplace_back(NewArgIndexFinal, OriginalArgIndexFinal, 0);
      ++NewArgIndexFinal;
    }
    // If split, scalar arguments will be added next
    ++OriginalArgIndexFinal;
  }

  // Now, add the sorted loads
  for (const LoadInfo &LI : AllLoads) {
    FinalNewArgTypes.push_back(LI.Load->getType());
    FinalNewArgMappings.emplace_back(NewArgIndexFinal, LI.OriginalArgIndex, LI.Offset);
    ++NewArgIndexFinal;
  }

  // Collect original function attributes
  AttributeList OldAttrs = F.getAttributes();
  AttributeSet FnAttrs = OldAttrs.getFnAttrs();
  AttributeSet RetAttrs = OldAttrs.getRetAttrs();

  // Create new function type
  FunctionType *NewFT =
      FunctionType::get(F.getReturnType(), FinalNewArgTypes, F.isVarArg());
  Function *NewF =
      Function::Create(NewFT, F.getLinkage(), F.getAddressSpace(), F.getName());
  F.getParent()->getFunctionList().insert(F.getIterator(), NewF);
  NewF->takeName(&F);

  // Set calling convention
  NewF->setCallingConv(F.getCallingConv());

  // Build new parameter attributes
  SmallVector<AttributeSet, 8> NewArgAttrSets;
  for (Argument &Arg : F.args()) {
    bool IsSplit = false;
    for (const LoadInfo &LI : AllLoads) {
      if (LI.Arg == &Arg) {
        IsSplit = true;
        break;
      }
    }

    if (!IsSplit) {
      // Copy existing attributes for this argument
      AttributeSet ArgAttrs = OldAttrs.getParamAttrs(Arg.getArgNo());
      NewArgAttrSets.push_back(ArgAttrs);
      ++NewArgIndex;
    }
    // Split arguments' attributes are not copied to scalar arguments
    // New scalar arguments will receive default (empty) attributes
  }

  // Add default attributes for the new scalar arguments
  for (size_t i = 0; i < AllLoads.size(); ++i) {
    NewArgAttrSets.emplace_back(AttributeSet());
  }

  // Build the new AttributeList
  AttributeList NewAttrList = AttributeList::get(
      F.getContext(), FnAttrs, RetAttrs, NewArgAttrSets);

  // Set the attributes on the new function
  NewF->setAttributes(NewAttrList);

  // Add the mapping information as a function attribute
  // Format: "NewArgIndex:OriginalArgIndex:Offset;..."
  std::string MappingStr;
  for (const auto &Info : FinalNewArgMappings) {
    unsigned NewArgIdx, OrigArgIdx;
    uint64_t Offset;
    std::tie(NewArgIdx, OrigArgIdx, Offset) = Info;

    if (!MappingStr.empty())
      MappingStr += ";";
    MappingStr += std::to_string(NewArgIdx) + ":" + std::to_string(OrigArgIdx) +
                  ":" + std::to_string(Offset);
  }

  // Add the function attribute to the new function
  NewF->addFnAttr("amdgpu-argument-mapping", MappingStr);

  LLVM_DEBUG(dbgs() << "New empty function:\n" << *NewF << '\n');

  // Move the body of the old function to the new function
  NewF->splice(NewF->begin(), &F);

  // Map old arguments and loads to new arguments
  DenseMap<Value *, Value *> VMap;
  DenseMap<LoadInst *, Argument *> LoadToNewArgMap;

  // Iterate over new arguments and map them
  auto NewArgIt = NewF->arg_begin();
  unsigned ArgIdx = 0;
  for (Argument &Arg : F.args()) {
    bool IsSplit = false;
    for (const LoadInfo &LI : AllLoads) {
      if (LI.Arg == &Arg) {
        IsSplit = true;
        break;
      }
    }

    if (!IsSplit) {
      NewArgIt->setName(Arg.getName());
      VMap[&Arg] = &*NewArgIt;
      ++NewArgIt;
    }
    ++ArgIdx;
  }

  // Then, map split load arguments
  for (const LoadInfo &LI : AllLoads) {
    NewArgIt->setName(LI.Load->getName());
    LoadToNewArgMap[LI.Load] = &*NewArgIt;
    ++NewArgIt;
  }

  // Replace original arguments with new arguments
  for (Argument &Arg : F.args()) {
    bool IsSplit = false;
    for (const LoadInfo &LI : AllLoads) {
      if (LI.Arg == &Arg) {
        IsSplit = true;
        break;
      }
    }

    if (!IsSplit) {
      Argument *NewArg = dyn_cast<Argument>(VMap[&Arg]);
      if (NewArg) {
        Arg.replaceAllUsesWith(NewArg);
      }
    } else {
      // Replace uses of the original struct argument with undef
      UndefValue *UndefArg = UndefValue::get(Arg.getType());
      Arg.replaceAllUsesWith(UndefArg);
    }
  }

  // Replace LoadInsts with new scalar arguments
  for (const LoadInfo &LI : AllLoads) {
    Value *PtrVal = LI.Load->getPointerOperand();
    // Check if still a GEP
    if (auto *GEP = dyn_cast<GetElementPtrInst>(PtrVal)) {
      if (GEP->use_empty()) {
        GEP->eraseFromParent();
      } else {
        GEP->replaceAllUsesWith(UndefValue::get(GEP->getType()));
        GEP->eraseFromParent();
      }
    }
  }


  LLVM_DEBUG(dbgs() << "New function after transformation:\n" << *NewF << '\n');

  // Replace old function with new function
  F.replaceAllUsesWith(NewF);
  F.eraseFromParent();

  return true;
}

bool AMDGPUSplitKernelArguments::runOnModule(Module &M) {
  if (DisableSplitKernelArgs)
    return false;
  bool Changed = false;
  SmallVector<Function *, 16> FunctionsToProcess;

  // Collect functions to process
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    FunctionsToProcess.push_back(&F);
  }

  // Now process the functions
  for (Function *F : FunctionsToProcess) {
    if (F->isDeclaration())
      continue;
    Changed |= processFunction(*F);
  }
  LLVM_DEBUG(dbgs() << "Module after transformation:\n" << M << '\n');

  return Changed;
}

INITIALIZE_PASS_BEGIN(AMDGPUSplitKernelArguments, DEBUG_TYPE,
                      "AMDGPU Split Kernel Arguments", false, false)
INITIALIZE_PASS_END(AMDGPUSplitKernelArguments, DEBUG_TYPE,
                    "AMDGPU Split Kernel Arguments", false, false)

char AMDGPUSplitKernelArguments::ID = 0;

ModulePass *llvm::createAMDGPUSplitKernelArgumentsPass() {
  return new AMDGPUSplitKernelArguments();
}

PreservedAnalyses AMDGPUSplitKernelArgumentsPass::run(Module &M, ModuleAnalysisManager &AM) {
  AMDGPUSplitKernelArguments Splitter;
  bool Changed = Splitter.runOnModule(M);

  if (!Changed)
    return PreservedAnalyses::all();

  // Since we modified the module, we need to report that analyses are invalidated
  return PreservedAnalyses::none();
}
