#include "AMDGPU.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
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

  std::string TargetFunction;

  AMDGPUSplitKernelArguments() : ModulePass(ID) {
    if (const char *EnvVar = std::getenv("DBG_SPLIT_ARG")) {
      TargetFunction = EnvVar;
    }
  }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

private:
  bool processFunction(Function &F);
  void dumpModuleToFile(Module &M, StringRef Suffix);
};

} // end anonymous namespace
void AMDGPUSplitKernelArguments::dumpModuleToFile(Module &M, StringRef Suffix) {
  std::error_code EC;
  std::string Filename =
      (Twine(M.getModuleIdentifier()) + Twine(Suffix) + Twine(".ll")).str();
  raw_fd_ostream OS(Filename, EC, sys::fs::OF_None);

  if (EC) {
    errs() << "Error opening file " << Filename << ": " << EC.message() << "\n";
    return;
  }

  M.print(OS, nullptr);
  OS.close();

  if (EC) {
    errs() << "Error writing to file " << Filename << ": " << EC.message()
           << "\n";
    return;
  }
}

bool AMDGPUSplitKernelArguments::processFunction(Function &F) {
  // Check if we should process this function based on env var
  if (!TargetFunction.empty() && F.getName() != TargetFunction) {
    LLVM_DEBUG(dbgs() << "Skipping function " << F.getName()
                      << " as it doesn't match target " << TargetFunction
                      << "\n");
    return false;
  }
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

  SmallVector<std::tuple<unsigned, unsigned, uint64_t>, 8> NewArgMappings;
  DenseMap<Argument *, SmallVector<LoadInst *, 8>> ArgToLoadsMap;
  DenseMap<Argument *, SmallVector<GetElementPtrInst *, 8>> ArgToGEPsMap;
  SmallVector<Argument *, 8> StructArgs;
  SmallVector<Type *, 8> NewArgTypes;

  // Helper function to convert address space if needed
  auto convertAddressSpace = [](Type *Ty) -> Type * {
    if (auto *PtrTy = dyn_cast<PointerType>(Ty)) {
      if (PtrTy->getAddressSpace() == AMDGPUAS::FLAT_ADDRESS) {
        // Create a new pointer type in the global address space
        return PointerType::get(PtrTy->getContext(), AMDGPUAS::GLOBAL_ADDRESS);
      }
    }
    return Ty;
  };

  // Collect struct arguments and new argument types
  unsigned OriginalArgIndex = 0;
  unsigned NewArgIndex = 0;
  for (Argument &Arg : F.args()) {
    LLVM_DEBUG(dbgs() << "Processing argument: " << Arg << "\n");
    if (Arg.use_empty()) {
      NewArgTypes.push_back(convertAddressSpace(Arg.getType()));
      NewArgMappings.push_back(
          std::make_tuple(NewArgIndex, OriginalArgIndex, 0));
      ++NewArgIndex;
      ++OriginalArgIndex;
      LLVM_DEBUG(dbgs() << "use empty\n");
      continue;
    }

    PointerType *PT = dyn_cast<PointerType>(Arg.getType());
    if (!PT) {
      NewArgTypes.push_back(Arg.getType());
      LLVM_DEBUG(dbgs() << "not a pointer\n");
      // Include mapping if indices have changed
      if (NewArgIndex != OriginalArgIndex)
        NewArgMappings.push_back(
            std::make_tuple(NewArgIndex, OriginalArgIndex, 0));
      ++NewArgIndex;
      ++OriginalArgIndex;
      continue;
    }

    const bool IsByRef = Arg.hasByRefAttr();
    if (!IsByRef) {
      NewArgTypes.push_back(Arg.getType());
      LLVM_DEBUG(dbgs() << "not byref\n");
      // Include mapping if indices have changed
      if (NewArgIndex != OriginalArgIndex)
        NewArgMappings.push_back(
            std::make_tuple(NewArgIndex, OriginalArgIndex, 0));
      ++NewArgIndex;
      ++OriginalArgIndex;
      continue;
    }

    Type *ArgTy = Arg.getParamByRefType();
    StructType *ST = dyn_cast<StructType>(ArgTy);
    if (!ST) {
      NewArgTypes.push_back(Arg.getType());
      LLVM_DEBUG(dbgs() << "not a struct\n");
      // Include mapping if indices have changed
      if (NewArgIndex != OriginalArgIndex)
        NewArgMappings.push_back(
            std::make_tuple(NewArgIndex, OriginalArgIndex, 0));
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
      StructArgs.push_back(&Arg);
      ArgToLoadsMap[&Arg] = Loads;
      ArgToGEPsMap[&Arg] = GEPs;
      for (LoadInst *LI : Loads) {
        Type *NewType = convertAddressSpace(LI->getType());
        NewArgTypes.push_back(NewType);

        // Compute offset
        uint64_t Offset = 0;
        if (auto *GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand())) {
          // Compute the offset using DataLayout
          APInt OffsetAPInt(DL.getPointerSizeInBits(), 0);
          if (GEP->accumulateConstantOffset(DL, OffsetAPInt))
            Offset = OffsetAPInt.getZExtValue();
        }

        // Map each new argument to the original argument index and offset
        NewArgMappings.push_back(
            std::make_tuple(NewArgIndex, OriginalArgIndex, Offset));
        ++NewArgIndex;
      }
    } else {
      NewArgTypes.push_back(convertAddressSpace(Arg.getType()));
      // Include mapping if indices have changed
      if (NewArgIndex != OriginalArgIndex)
        NewArgMappings.push_back(
            std::make_tuple(NewArgIndex, OriginalArgIndex, 0));
      ++NewArgIndex;
    }
    ++OriginalArgIndex;
  }

  if (StructArgs.empty())
    return false;

  // Collect function and return attributes
  AttributeList OldAttrs = F.getAttributes();
  AttributeSet FnAttrs = OldAttrs.getFnAttrs();
  AttributeSet RetAttrs = OldAttrs.getRetAttrs();

  // Create new function type
  FunctionType *NewFT =
      FunctionType::get(F.getReturnType(), NewArgTypes, F.isVarArg());
  Function *NewF =
      Function::Create(NewFT, F.getLinkage(), F.getAddressSpace(), F.getName());
  F.getParent()->getFunctionList().insert(F.getIterator(), NewF);
  NewF->takeName(&F);
  NewF->setVisibility(F.getVisibility());
  if (F.hasComdat())
    NewF->setComdat(F.getComdat());
  NewF->setDSOLocal(F.isDSOLocal());
  NewF->setUnnamedAddr(F.getUnnamedAddr());
  NewF->setCallingConv(F.getCallingConv());

  // Build new parameter attributes
  SmallVector<AttributeSet, 8> NewArgAttrSets;
  NewArgIndex = 0;
  for (Argument &Arg : F.args()) {
    if (ArgToLoadsMap.count(&Arg)) {
      for (LoadInst *LI : ArgToLoadsMap[&Arg]) {
        // No attributes for the new scalar arguments
        NewArgAttrSets.push_back(AttributeSet());
        ++NewArgIndex;
      }
    } else {
      // Copy existing attributes for this argument
      AttributeSet ArgAttrs = OldAttrs.getParamAttrs(Arg.getArgNo());
      NewArgAttrSets.push_back(ArgAttrs);
      ++NewArgIndex;
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
  for (const auto &Info : NewArgMappings) {
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
  auto NewArgIt = NewF->arg_begin();
  for (Argument &Arg : F.args()) {
    if (ArgToLoadsMap.count(&Arg)) {
      for (LoadInst *LI : ArgToLoadsMap[&Arg]) {
        std::string OldName = LI->getName().str();
        LI->setName(OldName + ".old");
        NewArgIt->setName(OldName);
        Value *NewArg = &*NewArgIt++;
        // Only insert cast if we're dealing with pointers
        if (isa<PointerType>(NewArg->getType()) &&
            isa<PointerType>(LI->getType())) {
          IRBuilder<> Builder(LI);
          Value *CastedArg = Builder.CreatePointerBitCastOrAddrSpaceCast(
              NewArg, LI->getType());
          VMap[LI] = CastedArg;
        } else {
          VMap[LI] = NewArg;
        }
      }
      UndefValue *UndefArg = UndefValue::get(Arg.getType());
      Arg.replaceAllUsesWith(UndefArg);
    } else {
      std::string OldName = Arg.getName().str();
      Arg.setName(OldName + ".old");
      NewArgIt->setName(OldName);
      Value *NewArg = &*NewArgIt;
      // Only insert cast if we're dealing with pointers
      if (isa<PointerType>(NewArg->getType()) &&
          isa<PointerType>(Arg.getType())) {
        IRBuilder<> Builder(&*NewF->begin()->begin());
        Value *CastedArg =
            Builder.CreatePointerBitCastOrAddrSpaceCast(NewArg, Arg.getType());
        Arg.replaceAllUsesWith(CastedArg);
      } else {
        Arg.replaceAllUsesWith(NewArg);
      }
      ++NewArgIt;
    }
  }

  // Replace LoadInsts with new arguments
  for (auto &Entry : ArgToLoadsMap) {
    for (LoadInst *LI : Entry.second) {
      // Replace uses of LoadInst with the corresponding new argument
      Value *NewArg = VMap[LI];
      LI->replaceAllUsesWith(NewArg);
      // Now LI has no uses and can be safely erased
      LI->eraseFromParent();
    }
  }

  // Erase GEPs
  for (auto &Entry : ArgToGEPsMap) {
    for (GetElementPtrInst *GEP : Entry.second) {
      // GEP might have been used by the LoadInsts which are now erased
      // So GEP should have no uses and can be safely erased
      if (GEP->use_empty()) {
        GEP->eraseFromParent();
      } else {
        // If GEP still has uses, we need to replace them with Undef
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
  // Dump initial module state
  dumpModuleToFile(M, ".pre_split");

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
  // Dump final module state if any changes were made
  if (Changed) {
    dumpModuleToFile(M, ".post_split");
  }
  // LLVM_DEBUG(dbgs() << "Module after transformation:\n" << M << '\n');

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

