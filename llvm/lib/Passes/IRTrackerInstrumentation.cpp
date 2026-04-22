//===- IRTrackerInstrumentation.cpp - IR tracker recorder -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/IRTrackerInstrumentation.h"

#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PrintPasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// CLI option
//
// The flag and its accessor live here (rather than in ``lib/IR/PrintPasses``)
// so the IR tracker is self-contained: anyone removing the recorder removes
// exactly one TU.
//===----------------------------------------------------------------------===//

static cl::opt<std::string> IRTrackerJSONOutput(
    "ir-tracker-json-output",
    cl::desc(
        "IR tracker: JSON Lines output path; record instructions with debug "
        "locations after each pass"),
    cl::value_desc("file"), cl::init(""), cl::Hidden);

namespace {

//===----------------------------------------------------------------------===//
// Local copies of small helpers shared with ``StandardInstrumentations.cpp``.
//
// Duplicated here so this TU is self-contained. If we later add more
// instrumentations factored out into their own files, hoist these into a
// shared internal header.
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
// Recorder implementation.
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

class IRTrackerJSONState {
  std::unique_ptr<raw_fd_ostream> OS;
  unsigned NextSeq = 1;
  bool InitialCaptured = false;

  void writePassRecord(unsigned Seq, StringRef Phase, StringRef PassName,
                       StringRef IRUnit) {
    json::Object Obj;
    Obj["kind"] = "pass";
    Obj["seq"] = Seq;
    Obj["phase"] = Phase.str();
    Obj["pass"] = PassName.str();
    Obj["ir_unit"] = IRUnit.str();
    *OS << formatv("{0}\n", json::Value(std::move(Obj)));
  }

  void writeInstructionsInFunction(const Function &F) {
    if (F.isDeclaration() || !isFunctionInPrintList(F.getName()))
      return;

    std::string FunctionName = F.getName().str();
    ModuleSlotTracker MST(F.getParent());
    MST.incorporateFunction(F);
    for (const BasicBlock &BB : F) {
      std::string BBLabel =
          BB.hasName() ? BB.getName().str() : std::string("<unnamed>");
      unsigned InstSeq = 0;
      for (const Instruction &I : BB) {
        const DebugLoc &DL = I.getDebugLoc();
        if (!DL)
          continue;
        const DILocation *Loc = DL.get();
        if (!Loc || Loc->getLine() == 0)
          continue;

        std::string FilePath = getIRTrackerFilePath(Loc);
        if (FilePath.empty())
          continue;

        json::Object Obj;
        Obj["kind"] = "inst";
        Obj["file"] = FilePath;
        Obj["line"] = Loc->getLine();
        Obj["col"] = Loc->getColumn();
        Obj["function"] = FunctionName;
        Obj["block"] = BBLabel;
        Obj["inst_seq"] = InstSeq++;
        Obj["opcode"] = std::string(I.getOpcodeName());
        Obj["text"] = getIRTrackerInstructionText(I, MST);
        *OS << formatv("{0}\n", json::Value(std::move(Obj)));
      }
    }
  }

  void writeIR(Any IR, unsigned Seq, StringRef Phase, StringRef PassName,
               StringRef IRUnit) {
    writePassRecord(Seq, Phase, PassName, IRUnit);
    if (const auto *M = unwrapIR<Module>(IR)) {
      for (const Function &F : *M)
        writeInstructionsInFunction(F);
      return;
    }
    if (const auto *F = unwrapIR<Function>(IR)) {
      writeInstructionsInFunction(*F);
      return;
    }
    if (const auto *C = unwrapIR<LazyCallGraph::SCC>(IR)) {
      for (const LazyCallGraph::Node &N : *C)
        writeInstructionsInFunction(N.getFunction());
      return;
    }
    if (const auto *L = unwrapIR<Loop>(IR))
      writeInstructionsInFunction(*L->getHeader()->getParent());
  }

public:
  explicit IRTrackerJSONState(StringRef Path) {
    std::error_code EC;
    OS = std::make_unique<raw_fd_ostream>(Path, EC, sys::fs::OF_Text);
    if (EC)
      report_fatal_error(Twine("ir-tracker json output open: ") + EC.message());
  }

  void beforePass(StringRef PassID, Any IR) {
    if (InitialCaptured || isIgnored(PassID) || !shouldPrintIR(IR))
      return;
    InitialCaptured = true;
    writeIR(IR, 0, "initial", "<initial>", getIRName(IR));
  }

  void afterPass(StringRef PassID, Any IR, PassInstrumentationCallbacks &PIC) {
    if (isIgnored(PassID) || !shouldPrintIR(IR))
      return;

    StringRef PassName = PIC.getPassNameForClassName(PassID);
    if (PassName.empty())
      PassName = PassID;
    writeIR(IR, NextSeq++, "after", PassName, getIRName(IR));
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
        (void)PA;
        State->afterPass(PassID, IR, PIC);
      });
}
