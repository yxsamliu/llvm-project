//===- LLSourceDebugLoc.cpp - Attach .ll file lines as !dbg metadata -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLSourceDebugLoc.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

static bool isBitcodeBuffer(StringRef Buffer) {
  return Buffer.starts_with("BC\xc0\xde");
}

/// Return true if a trimmed line looks like the start of a function definition
/// or declaration (starts with "define " or "declare ").
static bool isFunctionDefOrDecl(StringRef S) {
  return S.starts_with("define ") || S.starts_with("declare ");
}

/// Return true if a trimmed line looks like a basic-block label.
///
/// In LLVM assembly, a BB label is a line whose trimmed form ends with ':'
/// and contains no '=' sign before the colon (to avoid matching things like
/// "ret i32 0 ; [label]:"). We also require no whitespace before the colon
/// (BB labels are at the start of the trimmed line).
static bool isBBLabel(StringRef S) {
  // Must end with ':'
  if (!S.ends_with(":"))
    return false;
  StringRef Body = S.drop_back(); // drop the trailing ':'
  // Must not contain '=' (instruction result assignments contain '=')
  if (Body.contains('='))
    return false;
  // Must not contain spaces (a label is a single word)
  if (Body.contains(' ') || Body.contains('\t'))
    return false;
  return true;
}

/// Strip a line comment (; ...) from \p Line, respecting quoted strings.
static StringRef stripComment(StringRef Line) {
  bool InQuote = false;
  for (unsigned I = 0, E = Line.size(); I < E; ++I) {
    char C = Line[I];
    if (C == '"') {
      InQuote = !InQuote;
    } else if (C == ';' && !InQuote) {
      return Line.take_front(I);
    }
  }
  return Line;
}

struct FunctionLineMap {
  unsigned DefineLine = 1;
  SmallVector<SmallVector<unsigned>> BBInstLines;
};

using LineMap = SmallVector<FunctionLineMap>;

/// Scan \p Text (LLVM assembly) and build a 3-D positional map:
///   LineMap[func_idx][bb_idx][inst_idx] = 1-based line number in the file.
///
/// Indices parallel the order functions/BBs/instructions appear in text.
/// Returns the number of functions found.
static LineMap buildLineMap(StringRef Text) {
  LineMap LM;

  bool InFunction = false;
  unsigned FuncIdx = 0;
  unsigned BBIdx = 0;
  unsigned LineNo = 0;

  SmallVector<StringRef> Lines;
  Text.split(Lines, '\n');

  for (StringRef RawLine : Lines) {
    ++LineNo;
    StringRef Trimmed = stripComment(RawLine).trim();

    if (Trimmed.empty())
      continue;

    if (isFunctionDefOrDecl(Trimmed)) {
      // Only definitions contribute instruction line information. Declarations
      // have no body and must not consume a function slot, or the positional
      // mapping will become misaligned.
      if (Trimmed.starts_with("define ")) {
        InFunction = true;
        LM.emplace_back(); // new function slot
        LM.back().DefineLine = LineNo;
        FuncIdx = LM.size() - 1;
        BBIdx = 0;
        // If the opening brace is on the same line as "define", the first BB
        // label (or first instruction) will follow.
      } else {
        InFunction = false;
      }
      continue;
    }

    if (!InFunction)
      continue;

    if (Trimmed == "}") {
      InFunction = false;
      continue;
    }

    if (Trimmed == "{") {
      // Opening brace on its own line (rare but valid).
      continue;
    }

    if (isBBLabel(Trimmed)) {
      // New basic block.
      if (FuncIdx < LM.size()) {
        LM[FuncIdx].BBInstLines.emplace_back(); // new BB slot
        BBIdx = LM[FuncIdx].BBInstLines.size() - 1;
      }
      continue;
    }

    // Everything else inside a function is an instruction.
    if (FuncIdx < LM.size()) {
      // If we see an instruction before any BB label (first BB may be unnamed),
      // create a slot for it implicitly.
      if (LM[FuncIdx].BBInstLines.empty())
        LM[FuncIdx].BBInstLines.emplace_back();

      BBIdx = LM[FuncIdx].BBInstLines.size() - 1;
      LM[FuncIdx].BBInstLines[BBIdx].push_back(LineNo);
    }
  }

  return LM;
}

static void addDebugInfoVersionIfMissing(Module &M) {
  if (!M.getModuleFlag("Debug Info Version"))
    M.addModuleFlag(Module::Warning, "Debug Info Version",
                    DEBUG_METADATA_VERSION);
}

static DISubprogram *createSyntheticSubprogram(DIBuilder &DIB, DICompileUnit *CU,
                                               DIFile *File, Function &F,
                                               unsigned FuncLine,
                                               DISubroutineType *SPType) {
  DISubprogram::DISPFlags SPFlags =
      DISubprogram::SPFlagDefinition | DISubprogram::SPFlagOptimized;
  if (F.hasPrivateLinkage() || F.hasInternalLinkage())
    SPFlags |= DISubprogram::SPFlagLocalToUnit;
  return DIB.createFunction(CU, F.getName(), F.getName(), File, FuncLine,
                            SPType, FuncLine, DINode::FlagZero, SPFlags);
}

static void applyOrdinalLocations(Module &M, LLVMContext &Ctx, DIBuilder &DIB,
                                  DICompileUnit *CU, DIFile *File,
                                  bool ForceOverwrite) {
  auto SPType = DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));
  unsigned NextOrdinal = 1;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    unsigned FuncLine = NextOrdinal;
    DISubprogram *SP =
        createSyntheticSubprogram(DIB, CU, File, F, FuncLine, SPType);
    F.setSubprogram(SP);

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (!ForceOverwrite && I.getDebugLoc())
          continue;
        I.setDebugLoc(DILocation::get(Ctx, NextOrdinal, 0, SP));
        ++NextOrdinal;
      }
    }
  }
}

} // namespace

void llvm::applyLLSourceDebugLoc(Module &M, StringRef LLFilePath,
                                 bool ForceOverwrite) {
  if (LLFilePath.empty()) {
    errs() << "warning: --add-ll-debugloc: no input file path known, "
              "skipping.\n";
    return;
  }

  // If the module has real debug info, skip unless forced.
  if (!ForceOverwrite && M.getNamedMetadata("llvm.dbg.cu")) {
    errs() << "note: --add-ll-debugloc: module already has debug info; "
              "use --add-ll-debugloc-force to overwrite.\n";
    return;
  }

  // Read the original input. For textual LLVM assembly we use the actual
  // source line numbers. For bitcode (or unreadable input), we fall back to
  // synthetic ordinal IDs derived from the parsed IR traversal order.
  auto BufOrErr = MemoryBuffer::getFile(LLFilePath);
  bool UseTextualLineMap = false;
  LineMap LM;
  if (BufOrErr) {
    StringRef Text = (*BufOrErr)->getBuffer();
    if (!isBitcodeBuffer(Text)) {
      LM = buildLineMap(Text);
      UseTextualLineMap = true;
    }
  } else {
    errs() << "warning: --add-ll-debugloc: cannot read '" << LLFilePath
           << "': " << BufOrErr.getError().message()
           << "; falling back to synthetic ordinal locations\n";
  }

  // Build debug info hierarchy.
  DIBuilder DIB(M);
  LLVMContext &Ctx = M.getContext();

  StringRef Dir = sys::path::parent_path(LLFilePath);
  if (Dir.empty())
    Dir = ".";
  StringRef Filename = sys::path::filename(LLFilePath);

  DIFile *File = DIB.createFile(Filename, Dir);
  DICompileUnit *CU = DIB.createCompileUnit(
      dwarf::DW_LANG_C, File, "llvm-opt-add-ll-debugloc",
      /*isOptimized=*/true, "", /*RV=*/0, /*SplitName=*/"",
      DICompileUnit::FullDebug);
  addDebugInfoVersionIfMissing(M);

  if (!UseTextualLineMap) {
    applyOrdinalLocations(M, Ctx, DIB, CU, File, ForceOverwrite);
    DIB.finalize();
    return;
  }

  auto SPType = DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));

  unsigned FuncIdx = 0;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    // Determine the line of this function's definition in the source. Use 1
    // if we don't have data (graceful fallback).
    unsigned FuncLine = 1;
    if (FuncIdx < LM.size())
      FuncLine = LM[FuncIdx].DefineLine;

    DISubprogram *SP =
        createSyntheticSubprogram(DIB, CU, File, F, FuncLine, SPType);
    F.setSubprogram(SP);

    unsigned BBIdx = 0;
    for (BasicBlock &BB : F) {
      unsigned InstIdx = 0;
      for (Instruction &I : BB) {
        // Skip if already has a location and not force-overwriting.
        if (!ForceOverwrite && I.getDebugLoc())
          continue;

        unsigned Line = FuncLine; // fallback
        if (FuncIdx < LM.size()) {
          const auto &FuncData = LM[FuncIdx].BBInstLines;
          if (BBIdx < FuncData.size()) {
            const auto &BBData = FuncData[BBIdx];
            if (InstIdx < BBData.size())
              Line = BBData[InstIdx];
          }
        }
        I.setDebugLoc(DILocation::get(Ctx, Line, 0, SP));
        ++InstIdx;
      }
      ++BBIdx;
    }

    ++FuncIdx;
  }

  DIB.finalize();
}
