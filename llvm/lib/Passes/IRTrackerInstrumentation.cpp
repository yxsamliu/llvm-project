//===- IRTrackerInstrumentation.cpp - IR tracker recorder -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/IRTrackerInstrumentation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StableHashing.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PrintPasses.h"
#include "llvm/IR/StructuralHash.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// CLI option (the recorder owns its own flag so this TU is self-contained).
//===----------------------------------------------------------------------===//

static cl::opt<std::string> IRTrackerJSONOutput(
    "ir-tracker-json-output",
    cl::desc(
        "IR tracker: per-pass IR snapshot output path (TSV row format; CLI "
        "name kept for compatibility)"),
    cl::value_desc("file"), cl::init(""), cl::Hidden);

namespace {

//===----------------------------------------------------------------------===//
// Local copies of small helpers shared with StandardInstrumentations.cpp.
// Duplicated here so this TU is self-contained.
//===----------------------------------------------------------------------===//

template <typename IRUnitT> static const IRUnitT *unwrapIR(Any IR) {
  const IRUnitT **IRPtr = llvm::any_cast<const IRUnitT *>(&IR);
  return IRPtr ? *IRPtr : nullptr;
}

static std::string getIRName(Any IR) {
  if (unwrapIR<Module>(IR))
    return "[module]";
  if (const auto *F = unwrapIR<Function>(IR))
    return F->getName().str();
  if (const auto *C = unwrapIR<LazyCallGraph::SCC>(IR))
    return C->getName();
  if (const auto *L = unwrapIR<Loop>(IR))
    return "loop %" + L->getName().str() + " in function " +
           L->getHeader()->getParent()->getName().str();
  llvm_unreachable("Unknown wrapped IR type");
}

static bool moduleContainsFilterPrintFunc(const Module &M) {
  return any_of(M.functions(),
                [](const Function &F) {
                  return isFunctionInPrintList(F.getName());
                }) ||
         isFunctionInPrintList("*");
}

static bool sccContainsFilterPrintFunc(const LazyCallGraph::SCC &C) {
  return any_of(C,
                [](const LazyCallGraph::Node &N) {
                  return isFunctionInPrintList(N.getName());
                }) ||
         isFunctionInPrintList("*");
}

static bool shouldPrintIR(Any IR) {
  if (const auto *M = unwrapIR<Module>(IR))
    return moduleContainsFilterPrintFunc(*M);
  if (const auto *F = unwrapIR<Function>(IR))
    return isFunctionInPrintList(F->getName());
  if (const auto *C = unwrapIR<LazyCallGraph::SCC>(IR))
    return sccContainsFilterPrintFunc(*C);
  if (const auto *L = unwrapIR<Loop>(IR))
    return isFunctionInPrintList(L->getHeader()->getParent()->getName());
  return false;
}

static bool isIgnored(StringRef PassID) {
  return isSpecialPass(PassID,
                       {"PassManager", "PassAdaptor", "AnalysisManagerProxy",
                        "DevirtSCCRepeatedPass", "ModuleInlinerWrapperPass",
                        "VerifierPass", "PrintModulePass", "PrintMIRPass",
                        "PrintMIRPreparePass"});
}

//===----------------------------------------------------------------------===//
// Recorder implementation (cost-improvement port: hash-skip C1..C5, loc
// identity, lightweight printer, see scripts_shared .cursor/skills/
// ir-tracker SKILL.md for the design + measurement story).
//===----------------------------------------------------------------------===//

static std::string getIRTrackerFilePath(const DILocation *Loc) {
  if (!Loc)
    return {};

  StringRef Dir = Loc->getDirectory();
  StringRef File = Loc->getFilename();
  if (File.empty())
    return {};
  if (Dir.empty())
    return File.str();

  SmallString<256> Path(Dir);
  sys::path::append(Path, File);
  return std::string(Path);
}

static std::string getIRTrackerInstructionText(const Instruction &I,
                                               ModuleSlotTracker &MST) {
  std::string Text;
  raw_string_ostream OS(Text);
  I.print(OS, MST);
  OS.flush();
  return Text;
}

static stable_hash hashInstruction(const Instruction &I) {
  stable_hash H = stable_hash_combine(I.getOpcode(), I.getType()->getTypeID(),
                                      I.getNumOperands());
  for (const Use &U : I.operands()) {
    Value *V = U.get();
    H = stable_hash_combine(H, V->getType()->getTypeID(),
                            isa<Constant>(V) ? 1 : 0, isa<Argument>(V) ? 1 : 0);
    if (auto *C = dyn_cast<ConstantInt>(V))
      H = stable_hash_combine(H, static_cast<stable_hash>(hash_value(C->getValue())));
  }
  if (I.isCommutative())
    H = stable_hash_combine(H, 1);
  if (auto *CI = dyn_cast<CmpInst>(&I))
    H = stable_hash_combine(H, CI->getPredicate());
  return H;
}

static stable_hash hashBlockFingerprint(const BasicBlock &BB) {
  if (BB.empty())
    return 0;
  const Instruction &First = BB.front();
  const Instruction &Last = BB.back();
  return stable_hash_combine(static_cast<stable_hash>(BB.size()),
                             hashInstruction(First), hashInstruction(Last));
}

static stable_hash hashTrackerIdentity(const DILocation *Loc) {
  if (!Loc)
    return 0;
  unsigned ScopeLine = 0;
  if (DISubprogram *SP = Loc->getScope()->getSubprogram())
    ScopeLine = SP->getLine();
  return stable_hash_combine(Loc->getLine(), Loc->getColumn(), ScopeLine);
}

static void printAPIntValue(raw_ostream &OS, const APInt &V) {
  SmallString<32> Tmp;
  if (V.isNegative())
    V.toStringSigned(Tmp, 10);
  else
    V.toStringUnsigned(Tmp, 10);
  OS << Tmp;
}

class IRTrackerJSONState {
  std::unique_ptr<raw_fd_ostream> OS;
  unsigned NextSeq = 1;
  bool InitialCaptured = false;
  unsigned NextTrackerID = 1;
  // Cached pointer to the last module observed by afterPass, used by the
  // destructor to emit a final full-instruction snapshot so downstream
  // tooling can reconstruct the post-optimization IR from the DB alone.
  const Module *LastModule = nullptr;
  DenseMap<const Function *, stable_hash> FunctionHashes;
  DenseMap<const Function *, SmallVector<stable_hash>> BlockFingerprints;
  DenseMap<const Function *, SmallVector<stable_hash>> BlockHashes;
  DenseMap<const Function *, SmallVector<bool>> BlockHasZeroIDs;
  DenseMap<const Function *, SmallVector<SmallVector<stable_hash>>>
      BlockInstHashes;
  DenseMap<const Function *, SmallVector<SmallVector<unsigned>>> BlockTempIDs;
  DenseMap<stable_hash, unsigned> LocKeyToTrackerID;
  DenseMap<unsigned, stable_hash> TrackerIDToPrevHash;
  DenseSet<unsigned> EmittedTrackerMetadata;

  void writePassRecord(unsigned Seq, StringRef Phase, StringRef PassName,
                       StringRef IRUnit) {
    *OS << "P\t" << Seq << '\t' << Phase << '\t' << PassName << '\t' << IRUnit
        << '\n';
  }

  void writeTrackerRecord(unsigned ID, const DILocation *Loc) {
    if (ID == 0 || !EmittedTrackerMetadata.insert(ID).second)
      return;
    // Always emit a T record for every tracker ID, including synthesized
    // temp IDs for zero-location instructions (phi, many LCSSA-inserted
    // ops, etc.). Placeholder ``<synthetic>`` / line 0 / col 0 keeps the
    // DB row count aligned with the I rows so downstream tooling does not
    // silently drop those instructions.
    std::string FilePath = Loc ? getIRTrackerFilePath(Loc) : "<synthetic>";
    unsigned LineN = Loc ? Loc->getLine() : 0;
    unsigned ColN = Loc ? Loc->getColumn() : 0;
    *OS << "T\t" << ID << '\t' << FilePath << '\t' << LineN << '\t'
        << ColN << '\n';
  }

  unsigned getOrCreateTrackerID(const DILocation *Loc) {
    stable_hash Key = hashTrackerIdentity(Loc);
    if (Key == 0)
      return 0;
    auto It = LocKeyToTrackerID.find(Key);
    if (It != LocKeyToTrackerID.end())
      return It->second;
    unsigned ID = NextTrackerID++;
    LocKeyToTrackerID[Key] = ID;
    return ID;
  }

  void writeInstructionsInFunction(const Function &F, bool SkipUnchanged) {
    if (F.isDeclaration() || !isFunctionInPrintList(F.getName()))
      return;

    auto &PrevBlkH = BlockHashes[&F];
    auto &PrevBlkFP = BlockFingerprints[&F];
    auto &PrevBlkHasZeroIDs = BlockHasZeroIDs[&F];
    auto &PrevInstH = BlockInstHashes[&F];
    auto &PrevTempIDs = BlockTempIDs[&F];
    SmallVector<stable_hash> NewBlkH;
    SmallVector<stable_hash> NewBlkFP;
    SmallVector<bool> NewBlkHasZeroIDs;
    SmallVector<unsigned> ChangedBlocks;
    stable_hash FuncH = 0;

    unsigned BlkIdx = 0;
    for (const BasicBlock &BB : F) {
      stable_hash BlkFP = hashBlockFingerprint(BB);
      NewBlkFP.push_back(BlkFP);
      stable_hash BlkH = 0;
      bool HasZeroID = false;
      bool FingerprintUnchanged = SkipUnchanged && BlkIdx < PrevBlkFP.size() &&
                                  PrevBlkFP[BlkIdx] == BlkFP;
      if (FingerprintUnchanged && BlkIdx < PrevBlkH.size()) {
        BlkH = PrevBlkH[BlkIdx];
        if (BlkIdx < PrevBlkHasZeroIDs.size())
          HasZeroID = PrevBlkHasZeroIDs[BlkIdx];
      } else {
        for (const Instruction &I : BB) {
          stable_hash H = hashInstruction(I);
          BlkH = stable_hash_combine(BlkH, H);
          if (!I.getDebugLoc())
            HasZeroID = true;
        }
      }
      NewBlkHasZeroIDs.push_back(HasZeroID);
      NewBlkH.push_back(BlkH);
      FuncH = stable_hash_combine(FuncH, BlkH);
      if (!SkipUnchanged || BlkIdx >= PrevBlkH.size() ||
          PrevBlkH[BlkIdx] != BlkH)
        ChangedBlocks.push_back(BlkIdx);
      ++BlkIdx;
    }

    if (SkipUnchanged) {
      auto It = FunctionHashes.find(&F);
      if (It != FunctionHashes.end() && It->second == FuncH) {
        return;
      }
      FunctionHashes[&F] = FuncH;
    } else {
      FunctionHashes[&F] = FuncH;
    }

    if (ChangedBlocks.empty()) {
      PrevBlkFP = std::move(NewBlkFP);
      PrevBlkHasZeroIDs = std::move(NewBlkHasZeroIDs);
      PrevBlkH = std::move(NewBlkH);
      return;
    }

    StringRef FunctionName = F.getName();
    SmallString<256> InstBuf;
    DenseMap<const Value *, unsigned> LocalValueNames;
    unsigned NextLocalValueName = 0;

    auto getValueName = [&](const Value *V) -> std::string {
      if (auto *GV = dyn_cast<GlobalValue>(V)) {
        if (GV->hasName())
          return (Twine("@") + GV->getName()).str();
      }
      if (auto *BB = dyn_cast<BasicBlock>(V)) {
        if (BB->hasName())
          return (Twine("%") + BB->getName()).str();
      }
      if (V->hasName())
        return (Twine("%") + V->getName()).str();
      if (auto *I = dyn_cast<Instruction>(V)) {
        if (const DILocation *Loc =
                I->getDebugLoc() ? I->getDebugLoc().get() : nullptr) {
          unsigned ID = getOrCreateTrackerID(Loc);
          if (ID != 0)
            return (Twine("%t") + Twine(ID)).str();
        }
      }
      if (auto It = LocalValueNames.find(V); It != LocalValueNames.end())
        return (Twine("%u") + Twine(It->second)).str();
      unsigned ID = NextLocalValueName++;
      LocalValueNames[V] = ID;
      return (Twine("%u") + Twine(ID)).str();
    };

    std::function<void(raw_ostream &, const Value *)> writeValueRef =
        [&](raw_ostream &OS, const Value *V) {
          if (auto *CI = dyn_cast<ConstantInt>(V)) {
            printAPIntValue(OS, CI->getValue());
            return;
          }
          if (auto *CF = dyn_cast<ConstantFP>(V)) {
            SmallString<32> Tmp;
            CF->getValueAPF().toString(Tmp);
            OS << Tmp;
            return;
          }
          if (isa<ConstantPointerNull>(V)) {
            OS << "null";
            return;
          }
          if (isa<UndefValue>(V)) {
            OS << "undef";
            return;
          }
          if (isa<PoisonValue>(V)) {
            OS << "poison";
            return;
          }
          if (isa<ConstantAggregateZero>(V)) {
            OS << "zeroinitializer";
            return;
          }
          if (auto *CA = dyn_cast<ConstantDataArray>(V)) {
            if (CA->isString()) {
              OS << "c\"";
              printEscapedString(CA->getAsString(), OS);
              OS << "\"";
              return;
            }
          }
          OS << getValueName(V);
        };

    auto printInstructionText = [&](raw_ostream &OS, const Instruction &I,
                                    unsigned CurID) {
      if (!I.getType()->isVoidTy()) {
        if (I.hasName())
          OS << "%" << I.getName();
        else if (CurID != 0)
          OS << "%t" << CurID;
        else
          OS << getValueName(&I);
        OS << " = ";
      }

      OS << I.getOpcodeName();
      if (const auto *CI = dyn_cast<CmpInst>(&I))
        OS << ' ' << CI->getPredicate();

      if (const auto *RI = dyn_cast<ReturnInst>(&I)) {
        if (RI->getNumOperands() == 0) {
          OS << " void";
          return;
        }
        OS << ' ';
        RI->getReturnValue()->getType()->print(OS);
        OS << ' ';
        writeValueRef(OS, RI->getReturnValue());
        return;
      }

      if (const auto *BI = dyn_cast<BranchInst>(&I)) {
        if (BI->isUnconditional()) {
          OS << ' ';
          writeValueRef(OS, BI->getSuccessor(0));
        } else {
          OS << ' ';
          writeValueRef(OS, BI->getCondition());
          OS << ", ";
          writeValueRef(OS, BI->getSuccessor(0));
          OS << ", ";
          writeValueRef(OS, BI->getSuccessor(1));
        }
        return;
      }

      if (const auto *PN = dyn_cast<PHINode>(&I)) {
        OS << ' ';
        I.getType()->print(OS);
        bool First = true;
        for (unsigned Idx = 0; Idx < PN->getNumIncomingValues(); ++Idx) {
          OS << (First ? ' ' : ',');
          if (!First)
            OS << ' ';
          First = false;
          OS << "[ ";
          writeValueRef(OS, PN->getIncomingValue(Idx));
          OS << ", ";
          writeValueRef(OS, PN->getIncomingBlock(Idx));
          OS << " ]";
        }
        return;
      }

      if (const auto *CB = dyn_cast<CallBase>(&I)) {
        if (!CB->getType()->isVoidTy()) {
          OS << ' ';
          CB->getType()->print(OS);
        }
        OS << ' ';
        writeValueRef(OS, CB->getCalledOperand());
        OS << '(';
        for (unsigned Idx = 0; Idx < CB->arg_size(); ++Idx) {
          if (Idx)
            OS << ", ";
          writeValueRef(OS, CB->getArgOperand(Idx));
        }
        OS << ')';
        return;
      }

      if (I.getNumOperands()) {
        if (!I.getType()->isVoidTy()) {
          OS << ' ';
          I.getType()->print(OS);
        }
        OS << ' ';
        for (unsigned Idx = 0; Idx < I.getNumOperands(); ++Idx) {
          if (Idx)
            OS << ", ";
          writeValueRef(OS, I.getOperand(Idx));
        }
      }
    };
    BlkIdx = 0;
    unsigned ChangedBlockPos = 0;

    for (const BasicBlock &BB : F) {
      bool BlockChanged = ChangedBlockPos < ChangedBlocks.size() &&
                          ChangedBlocks[ChangedBlockPos] == BlkIdx;

      if (BlockChanged) {
        ++ChangedBlockPos;
        StringRef BBLabel =
            BB.hasName() ? BB.getName() : StringRef("<unnamed>");
        bool NeedFallback = NewBlkHasZeroIDs[BlkIdx];
        SmallVector<stable_hash> CurInstH;
        SmallVector<unsigned> CurTempIDs;
        auto *OldInstH = (SkipUnchanged && BlkIdx < PrevInstH.size())
                             ? &PrevInstH[BlkIdx]
                             : nullptr;
        auto *OldTempIDs = (SkipUnchanged && BlkIdx < PrevTempIDs.size())
                               ? &PrevTempIDs[BlkIdx]
                               : nullptr;
        SmallVector<bool> UsedOldTempIDs;
        if (OldTempIDs)
          UsedOldTempIDs.assign(OldTempIDs->size(), false);
        unsigned InstSeq = 0;
        unsigned InstIdx = 0;
        for (Instruction &I : const_cast<BasicBlock &>(BB)) {
          stable_hash CurH = hashInstruction(I);
          if (NeedFallback)
            CurInstH.push_back(CurH);
          const DILocation *Loc =
              I.getDebugLoc() ? I.getDebugLoc().get() : nullptr;
          unsigned CurID = getOrCreateTrackerID(Loc);
          if (CurID == 0) {
            int MatchedIdx = -1;
            if (OldTempIDs && OldInstH) {
              if (InstIdx < OldTempIDs->size() && InstIdx < OldInstH->size() &&
                  (*OldTempIDs)[InstIdx] != 0 && !UsedOldTempIDs[InstIdx] &&
                  (*OldInstH)[InstIdx] == CurH) {
                MatchedIdx = InstIdx;
              } else {
                int BestIdx = -1;
                int BestDist = std::numeric_limits<int>::max();
                bool AmbiguousBest = false;
                for (size_t J = 0,
                            E = std::min(OldTempIDs->size(), OldInstH->size());
                     J != E; ++J) {
                  if ((*OldTempIDs)[J] == 0 || UsedOldTempIDs[J] ||
                      (*OldInstH)[J] != CurH)
                    continue;
                  int Dist =
                      std::abs(static_cast<int>(J) - static_cast<int>(InstIdx));
                  if (Dist < BestDist) {
                    BestDist = Dist;
                    BestIdx = static_cast<int>(J);
                    AmbiguousBest = false;
                  } else if (Dist == BestDist) {
                    AmbiguousBest = true;
                  }
                }
                if (BestIdx >= 0 && !AmbiguousBest)
                  MatchedIdx = BestIdx;
              }
            }
            if (MatchedIdx >= 0) {
              CurID = (*OldTempIDs)[MatchedIdx];
              UsedOldTempIDs[MatchedIdx] = true;
            } else {
              CurID = NextTrackerID++;
            }
            CurTempIDs.push_back(CurID);
          } else if (NeedFallback) {
            CurTempIDs.push_back(0);
          }
          bool InstChanged = true;
          if (CurID != 0) {
            auto It = TrackerIDToPrevHash.find(CurID);
            InstChanged = It == TrackerIDToPrevHash.end() || It->second != CurH;
          } else {
            InstChanged = !OldInstH || InstIdx >= OldInstH->size() ||
                          (*OldInstH)[InstIdx] != CurH;
          }

          if (InstChanged) {
            InstBuf.clear();
            raw_svector_ostream IOS(InstBuf);
            printInstructionText(IOS, I, CurID);

            if (CurID != 0)
              writeTrackerRecord(CurID, Loc);

            *OS << "I\t" << FunctionName << '\t' << BBLabel << '\t' << InstSeq
                << '\t' << I.getOpcodeName() << '\t' << CurID << '\t' << InstBuf
                << '\n';
          }
          if (CurID != 0)
            TrackerIDToPrevHash[CurID] = CurH;
          ++InstSeq;
          ++InstIdx;
        }
        if (NeedFallback) {
          if (BlkIdx >= PrevInstH.size())
            PrevInstH.resize(BlkIdx + 1);
          PrevInstH[BlkIdx] = std::move(CurInstH);
          if (BlkIdx >= PrevTempIDs.size())
            PrevTempIDs.resize(BlkIdx + 1);
          PrevTempIDs[BlkIdx] = std::move(CurTempIDs);
        }
      }
      ++BlkIdx;
    }
    PrevBlkFP = std::move(NewBlkFP);
    PrevBlkHasZeroIDs = std::move(NewBlkHasZeroIDs);
    PrevBlkH = std::move(NewBlkH);
  }

  void writeIR(Any IR, unsigned Seq, StringRef Phase, StringRef PassName,
               StringRef IRUnit, bool SkipUnchanged) {
    writePassRecord(Seq, Phase, PassName, IRUnit);
    if (const auto *M = unwrapIR<Module>(IR)) {
      for (const Function &F : *M)
        writeInstructionsInFunction(F, SkipUnchanged);
      return;
    }
    if (const auto *F = unwrapIR<Function>(IR)) {
      writeInstructionsInFunction(*F, SkipUnchanged);
      return;
    }
    if (const auto *C = unwrapIR<LazyCallGraph::SCC>(IR)) {
      for (const LazyCallGraph::Node &N : *C)
        writeInstructionsInFunction(N.getFunction(), SkipUnchanged);
      return;
    }
    if (const auto *L = unwrapIR<Loop>(IR))
      writeInstructionsInFunction(*L->getHeader()->getParent(), SkipUnchanged);
  }

  bool allFunctionsKnown(Any IR) {
    if (const auto *M = unwrapIR<Module>(IR)) {
      for (const Function &F : *M)
        if (!F.isDeclaration() && !FunctionHashes.count(&F))
          return false;
      return true;
    }
    if (const auto *F = unwrapIR<Function>(IR))
      return F->isDeclaration() || FunctionHashes.count(F);
    if (const auto *C = unwrapIR<LazyCallGraph::SCC>(IR)) {
      for (const LazyCallGraph::Node &N : *C)
        if (!FunctionHashes.count(&N.getFunction()))
          return false;
      return true;
    }
    if (const auto *L = unwrapIR<Loop>(IR))
      return FunctionHashes.count(L->getHeader()->getParent());
    return false;
  }

public:
  explicit IRTrackerJSONState(StringRef Path) {
    std::error_code EC;
    OS = std::make_unique<raw_fd_ostream>(Path, EC, sys::fs::OF_Text);
    if (EC)
      report_fatal_error(Twine("ir-tracker json output open: ") + EC.message());
  }

  // Emit one synthetic "final" pass record on teardown covering every
  // function of the last-seen module with ``SkipUnchanged=false`` so the
  // post-optimization IR is fully materialized in the DB. The pass
  // manager's ``PassInstrumentationCallbacks`` (which owns the shared_ptr
  // holding this object) is destroyed before the module in the standard
  // opt/clang flow, so ``LastModule`` is still live at this point. The
  // cache reset is required because writeInstructionsInFunction also
  // short-circuits per-instruction on matching ``TrackerIDToPrevHash``
  // even when ``SkipUnchanged`` is false; without the reset the final
  // record would only contain whatever changed since the last pass.
  ~IRTrackerJSONState() {
    if (!LastModule)
      return;
    FunctionHashes.clear();
    BlockHashes.clear();
    BlockFingerprints.clear();
    BlockHasZeroIDs.clear();
    BlockInstHashes.clear();
    BlockTempIDs.clear();
    TrackerIDToPrevHash.clear();
    writePassRecord(NextSeq++, "final", "<final>", "[module]");
    for (const Function &F : *LastModule)
      writeInstructionsInFunction(F, /*SkipUnchanged=*/false);
  }

  void beforePass(StringRef PassID, Any IR) {
    if (InitialCaptured || isIgnored(PassID) || !shouldPrintIR(IR))
      return;
    InitialCaptured = true;
    writeIR(IR, 0, "initial", "<initial>", getIRName(IR),
            /*SkipUnchanged=*/false);
  }

  void afterPass(StringRef PassID, Any IR, PassInstrumentationCallbacks &PIC,
                 const PreservedAnalyses &PA) {
    if (isIgnored(PassID) || !shouldPrintIR(IR))
      return;

    // Track the enclosing module so the destructor can dump a final
    // full-instruction snapshot regardless of the IR unit type this pass
    // saw.
    if (const auto *M = unwrapIR<Module>(IR))
      LastModule = M;
    else if (const auto *F = unwrapIR<Function>(IR))
      LastModule = F->getParent();
    else if (const auto *C = unwrapIR<LazyCallGraph::SCC>(IR)) {
      if (C->begin() != C->end())
        LastModule = C->begin()->getFunction().getParent();
    } else if (const auto *L = unwrapIR<Loop>(IR))
      LastModule = L->getHeader()->getParent()->getParent();

    StringRef PassName = PIC.getPassNameForClassName(PassID);
    if (PassName.empty())
      PassName = PassID;

    if (PA.areAllPreserved() && allFunctionsKnown(IR)) {
      writePassRecord(NextSeq++, "after", PassName, getIRName(IR));
      return;
    }

    writeIR(IR, NextSeq++, "after", PassName, getIRName(IR),
            /*SkipUnchanged=*/true);
  }
};

} // namespace

void IRTrackerInstrumentation::registerCallbacks(
    PassInstrumentationCallbacks &PIC) {
  StringRef Path = IRTrackerJSONOutput;
  if (Path.empty())
    return;

  auto State = std::make_shared<IRTrackerJSONState>(Path);
  PIC.registerBeforeNonSkippedPassCallback(
      [State](StringRef PassID, Any IR) { State->beforePass(PassID, IR); });
  PIC.registerAfterPassCallback(
      [State, &PIC](StringRef PassID, Any IR, const PreservedAnalyses &PA) {
        State->afterPass(PassID, IR, PIC, PA);
      });
}
