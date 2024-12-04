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

  // Struct to hold information about loads from split arguments
  struct LoadInfo {
    LoadInst *Load;
    Argument *Arg;
    uint64_t Offset;
    unsigned OriginalArgIndex;
  };

  // Struct to represent final arguments (both non-split and split)
  struct FinalArgInfo {
    Type *ArgType;
    unsigned OrigArgIndex;
    uint64_t Offset;
    Argument *OrigArg; // For non-split arguments only (used for attributes)
    bool IsSplit;
  };

  SmallVector<LoadInfo, 16> AllLoads;
  SmallVector<FinalArgInfo, 16> AllFinalArgs;

  // Collect arguments
  unsigned OriginalArgIndex = 0;
  for (Argument &Arg : F.args()) {
    LLVM_DEBUG(dbgs() << "Processing argument: " << Arg << "\n");
    if (Arg.use_empty()) {
      // This argument is unused, just keep it as is
      AllFinalArgs.push_back({Arg.getType(), OriginalArgIndex, 0, &Arg, false});
      ++OriginalArgIndex;
      LLVM_DEBUG(dbgs() << "use empty\n");
      continue;
    }

    PointerType *PT = dyn_cast<PointerType>(Arg.getType());
    if (!PT) {
      // Non-pointer argument, keep it as is
      AllFinalArgs.push_back({Arg.getType(), OriginalArgIndex, 0, &Arg, false});
      ++OriginalArgIndex;
      continue;
    }

    const bool IsByRef = Arg.hasByRefAttr();
    if (!IsByRef) {
      // Pointer but not byref, just keep it
      AllFinalArgs.push_back({Arg.getType(), OriginalArgIndex, 0, &Arg, false});
      ++OriginalArgIndex;
      continue;
    }

    // By-ref pointer argument. Check if it's a struct we can split
    Type *ArgTy = Arg.getParamByRefType();
    StructType *ST = dyn_cast<StructType>(ArgTy);
    if (!ST) {
      // It's a byref pointer to a non-struct type, keep it as is
      AllFinalArgs.push_back({Arg.getType(), OriginalArgIndex, 0, &Arg, false});
      ++OriginalArgIndex;
      continue;
    }

    // It's a struct argument. Check all uses for loads and GEPs
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
      }
      if (!AllLoadsOrGEPs)
        break;
    }
    LLVM_DEBUG(dbgs() << "  AllLoadsOrGEPs: "
                      << (AllLoadsOrGEPs ? "true" : "false") << "\n");

    if (AllLoadsOrGEPs) {
      // We'll not add the original struct argument itself.
      // Instead, we will add each load as a separate argument later.
      for (LoadInst *LI : Loads) {
        // Compute offset for each load
        uint64_t Offset = 0;
        if (auto *GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand())) {
          APInt OffsetAPInt(DL.getPointerSizeInBits(), 0);
          if (GEP->accumulateConstantOffset(DL, OffsetAPInt))
            Offset = OffsetAPInt.getZExtValue();
        }

        AllLoads.push_back({LI, &Arg, Offset, OriginalArgIndex});
      }
    } else {
      // Argument is not eligible for splitting; keep it as is
      AllFinalArgs.push_back({Arg.getType(), OriginalArgIndex, 0, &Arg, false});
    }

    ++OriginalArgIndex;
  }

  // If we didn't find any argument to split, no changes are needed
  if (AllLoads.empty())
    return false;

  // Sort the loads by OriginalArgIndex and Offset so they appear in proper order
  std::sort(AllLoads.begin(), AllLoads.end(), [](const LoadInfo &A, const LoadInfo &B) {
    if (A.OriginalArgIndex != B.OriginalArgIndex)
      return A.OriginalArgIndex < B.OriginalArgIndex;
    return A.Offset < B.Offset;
  });

  // Add the split load arguments to AllFinalArgs
  for (const LoadInfo &LI : AllLoads) {
    AllFinalArgs.push_back({LI.Load->getType(), LI.OriginalArgIndex, LI.Offset, LI.Arg, true});
  }

  // Now sort AllFinalArgs by OrigArgIndex, then by Offset
  std::sort(AllFinalArgs.begin(), AllFinalArgs.end(), [](const FinalArgInfo &A, const FinalArgInfo &B) {
    if (A.OrigArgIndex != B.OrigArgIndex)
      return A.OrigArgIndex < B.OrigArgIndex;
    return A.Offset < B.Offset;
  });

  // Build final argument arrays
  SmallVector<Type *, 8> FinalNewArgTypes;
  SmallVector<std::tuple<unsigned, unsigned, uint64_t>, 8> FinalNewArgMappings;
  unsigned NewArgIndexFinal = 0;
  for (auto &FA : AllFinalArgs) {
    FinalNewArgTypes.push_back(FA.ArgType);
    FinalNewArgMappings.push_back(std::make_tuple(NewArgIndexFinal, FA.OrigArgIndex, FA.Offset));
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
  // For non-split arguments, copy existing attributes.
  // For split arguments, give them default (empty) attributes.
  SmallVector<AttributeSet, 8> NewArgAttrSets;
  {
    // Map from OrigArgIndex to the original Arg for attribute lookup
    // Non-split arguments have a direct Arg pointer, split ones have IsSplit=true
    DenseMap<unsigned, Argument*> OrigIndexToArg;
    for (auto &FA : AllFinalArgs) {
      if (!FA.IsSplit && FA.OrigArg)
        OrigIndexToArg[FA.OrigArgIndex] = FA.OrigArg;
    }

    for (unsigned i = 0; i < AllFinalArgs.size(); ++i) {
      auto &FA = AllFinalArgs[i];
      if (FA.IsSplit) {
        // Split arguments get default attributes
        NewArgAttrSets.push_back(AttributeSet());
      } else {
        // Non-split argument: copy attributes from the original argument
        Argument *OrigArg = OrigIndexToArg[FA.OrigArgIndex];
        if (OrigArg) {
          AttributeSet ArgAttrs = OldAttrs.getParamAttrs(OrigArg->getArgNo());
          NewArgAttrSets.push_back(ArgAttrs);
        } else {
          // If we can't find an original Arg (shouldn't happen), just default
          NewArgAttrSets.push_back(AttributeSet());
        }
      }
    }
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

  // Replace old arguments and loads
  DenseMap<Value *, Value *> VMap;
  DenseMap<LoadInst *, Argument *> LoadToNewArgMap;

  // Map new arguments back to old arguments/loads
  // After sorting, we know NewF->arg_begin() matches the order in AllFinalArgs
  auto NewArgIt = NewF->arg_begin();
  for (unsigned i = 0; i < AllFinalArgs.size(); ++i, ++NewArgIt) {
    auto &FA = AllFinalArgs[i];
    NewArgIt->setName("arg" + std::to_string(i));

    if (FA.IsSplit) {
      // This corresponds to a load from a split argument
      // Find the corresponding LoadInfo
      // We'll match by OrigArgIndex and Offset
      LoadInst *MatchedLoad = nullptr;
      for (auto &LI : AllLoads) {
        if (LI.OriginalArgIndex == FA.OrigArgIndex && LI.Offset == FA.Offset) {
          MatchedLoad = LI.Load;
          break;
        }
      }

      if (MatchedLoad) {
        LoadToNewArgMap[MatchedLoad] = &*NewArgIt;
      }
    } else {
      // Non-split argument: replace its uses with this new argument
      // Find the original argument by OrigArgIndex
      Argument *OrigArg = FA.OrigArg;
      if (OrigArg)
        VMap[OrigArg] = &*NewArgIt;
    }
  }

  // Replace non-split arguments
  for (Argument &Arg : F.args()) {
    unsigned ArgNo = Arg.getArgNo();
    // Check if this argument was split
    bool IsSplit = false;
    for (auto &LI : AllLoads) {
      if (LI.Arg == &Arg) {
        IsSplit = true;
        break;
      }
    }

    if (!IsSplit) {
      // Replace uses with the new argument
      if (VMap.find(&Arg) != VMap.end()) {
        Arg.replaceAllUsesWith(VMap[&Arg]);
      }
    } else {
      // For split arguments, replace original argument uses with undef
      UndefValue *UndefArg = UndefValue::get(Arg.getType());
      Arg.replaceAllUsesWith(UndefArg);
    }
  }

  // Erase GEPs associated with split arguments if they exist
  // If the pointer operand is no longer a GEP, just skip
  for (auto &LI : AllLoads) {
    Value *PtrVal = LI.Load->getPointerOperand();
    if (auto *GEP = dyn_cast_or_null<GetElementPtrInst>(PtrVal)) {
      if (GEP->use_empty()) {
        GEP->eraseFromParent();
      } else {
        GEP->replaceAllUsesWith(UndefValue::get(GEP->getType()));
        GEP->eraseFromParent();
      }
    }
  }

  // Replace LoadInst uses with the corresponding new arguments
  for (auto &Entry : LoadToNewArgMap) {
    LoadInst *LI = Entry.first;
    Argument *NewArg = Entry.second;
    LI->replaceAllUsesWith(NewArg);
    LI->eraseFromParent();
  }

  LLVM_DEBUG(dbgs() << "New function after transformation:\n" << *NewF << '\n');

  // Replace old function references with the new one
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
