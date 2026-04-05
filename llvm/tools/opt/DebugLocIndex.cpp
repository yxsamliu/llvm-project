//===- DebugLocIndex.cpp - SQLite debug-loc index for opt -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DebugLocIndex.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PrintPasses.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Twine.h"

#ifdef LLVM_ENABLE_DEBUG_LOC_INDEX
#include <sqlite3.h>
#endif

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>

using namespace llvm;

#ifdef LLVM_ENABLE_DEBUG_LOC_INDEX

namespace {

template <typename IRUnitT> static const IRUnitT *unwrapIR(Any IR) {
  const IRUnitT **IRPtr = llvm::any_cast<const IRUnitT *>(&IR);
  return IRPtr ? *IRPtr : nullptr;
}

/// Match \c StandardInstrumentations.cpp \c shouldPrintIR (anonymous there).
static bool moduleContainsFilterPrintFunc(const Module &M) {
  return any_of(M.functions(), [](const Function &F) {
           return isFunctionInPrintList(F.getName());
         }) ||
         isFunctionInPrintList("*");
}

static bool sccContainsFilterPrintFunc(const LazyCallGraph::SCC &C) {
  return any_of(C, [](const LazyCallGraph::Node &N) {
           return isFunctionInPrintList(N.getName());
         }) ||
         isFunctionInPrintList("*");
}

static bool shouldIndexIRUnit(Any IR) {
  if (const auto *M = unwrapIR<Module>(IR))
    return moduleContainsFilterPrintFunc(*M);
  if (const auto *F = unwrapIR<Function>(IR))
    return isFunctionInPrintList(F->getName());
  if (const auto *C = unwrapIR<LazyCallGraph::SCC>(IR))
    return sccContainsFilterPrintFunc(*C);
  if (const auto *L = unwrapIR<Loop>(IR))
    return isFunctionInPrintList(L->getHeader()->getParent()->getName());
  if (const auto *MF = unwrapIR<MachineFunction>(IR))
    return isFunctionInPrintList(MF->getName());
  llvm_unreachable("Unknown IR unit");
}

static bool isIgnoredPass(StringRef PassID) {
  return PassID == "PassManager" || PassID == "PassAdaptor" ||
         PassID == "AnalysisManagerProxy" || PassID == "DevirtSCCRepeatedPass" ||
         PassID == "ModuleInlinerWrapperPass" || PassID == "VerifierPass" ||
         PassID == "PrintModulePass" || PassID == "PrintMIRPass" ||
         PassID == "PrintMIRPreparePass";
}

static std::string getIRUnitLabel(Any IR) {
  if (unwrapIR<Module>(IR))
    return "[module]";
  if (const auto *F = unwrapIR<Function>(IR))
    return F->getName().str();
  if (const auto *C = unwrapIR<LazyCallGraph::SCC>(IR))
    return std::string(C->getName());
  if (const auto *L = unwrapIR<Loop>(IR))
    return (Twine("loop %") + L->getName() + " in " +
            L->getHeader()->getParent()->getName())
        .str();
  if (const auto *MF = unwrapIR<MachineFunction>(IR))
    return MF->getName().str();
  return "[unknown-ir-unit]";
}

class DebugLocIndexState {
  sqlite3 *DB = nullptr;
  sqlite3_stmt *StmtInsertPass = nullptr;
  sqlite3_stmt *StmtInsertFileIgnore = nullptr;
  sqlite3_stmt *StmtSelectFileId = nullptr;
  sqlite3_stmt *StmtInsertInst = nullptr;
  unsigned NextSeq = 0;
  std::unordered_map<std::string, int> FileIdCache;

  static int check(int RC, const char *Ctx) {
    if (RC == SQLITE_OK || RC == SQLITE_DONE || RC == SQLITE_ROW)
      return RC;
    report_fatal_error(Twine(Ctx) + ": " + sqlite3_errstr(RC));
  }

  int getOrCreateFileId(StringRef Path) {
    std::string Key = Path.str();
    auto I = FileIdCache.find(Key);
    if (I != FileIdCache.end())
      return I->second;

    check(sqlite3_bind_text(StmtInsertFileIgnore, 1, Key.data(),
                            (int)Key.size(), SQLITE_TRANSIENT),
          "sqlite3_bind_text(files)");
    check(sqlite3_step(StmtInsertFileIgnore), "sqlite3_step(insert file)");
    check(sqlite3_reset(StmtInsertFileIgnore), "sqlite3_reset(insert file)");

    check(sqlite3_bind_text(StmtSelectFileId, 1, Key.data(), (int)Key.size(),
                            SQLITE_TRANSIENT),
          "sqlite3_bind_text(select file)");
    if (sqlite3_step(StmtSelectFileId) != SQLITE_ROW)
      report_fatal_error("debug-loc-index: missing file row after insert");
    int Id = sqlite3_column_int(StmtSelectFileId, 0);
    check(sqlite3_reset(StmtSelectFileId), "sqlite3_reset(select file)");
    FileIdCache[Key] = Id;
    return Id;
  }

  void indexInstructionsInFunction(const Function &F, sqlite3_int64 PassRowId) {
    if (F.isDeclaration() || !isFunctionInPrintList(F.getName()))
      return;

    std::string FuncName = F.getName().str();
    for (const BasicBlock &BB : F) {
      std::string BBLabel =
          BB.hasName() ? BB.getName().str() : std::string("<unnamed>");
      unsigned InstSeq = 0;
      for (const Instruction &I : BB) {
        DebugLoc DL = I.getDebugLoc();
        if (!DL)
          continue;
        const DILocation *Loc = DL.get();
        unsigned Line = Loc->getLine();
        if (Line == 0)
          continue;
        unsigned Col = Loc->getColumn();
        std::string FilePath = Loc->getFilename().str();
        if (FilePath.empty())
          continue;
        int FileId = getOrCreateFileId(FilePath);

        check(sqlite3_bind_int64(StmtInsertInst, 1, PassRowId),
              "bind pass_rowid");
        check(sqlite3_bind_text(StmtInsertInst, 2, FuncName.data(),
                                (int)FuncName.size(), SQLITE_TRANSIENT),
              "bind function");
        check(sqlite3_bind_text(StmtInsertInst, 3, BBLabel.data(),
                                (int)BBLabel.size(), SQLITE_TRANSIENT),
              "bind bb");
        check(sqlite3_bind_int(StmtInsertInst, 4, (int)InstSeq), "bind seq");
        StringRef Opc = I.getOpcodeName();
        check(sqlite3_bind_text(StmtInsertInst, 5, Opc.data(),
                                (int)Opc.size(), SQLITE_TRANSIENT),
              "bind opcode");
        check(sqlite3_bind_int(StmtInsertInst, 6, FileId), "bind file");
        check(sqlite3_bind_int(StmtInsertInst, 7, (int)Line), "bind line");
        check(sqlite3_bind_int(StmtInsertInst, 8, (int)Col), "bind col");
        check(sqlite3_step(StmtInsertInst), "sqlite3_step(insert inst)");
        check(sqlite3_reset(StmtInsertInst), "sqlite3_reset(insert inst)");
        ++InstSeq;
      }
    }
  }

  void indexIR(Any IR) {
    if (const auto *M = unwrapIR<Module>(IR)) {
      for (const Function &F : *M)
        indexInstructionsInFunction(F, LastPassRowId);
      return;
    }
    if (const auto *F = unwrapIR<Function>(IR)) {
      indexInstructionsInFunction(*F, LastPassRowId);
      return;
    }
    if (const auto *C = unwrapIR<LazyCallGraph::SCC>(IR)) {
      for (const LazyCallGraph::Node &N : *C)
        indexInstructionsInFunction(N.getFunction(), LastPassRowId);
      return;
    }
    if (const auto *L = unwrapIR<Loop>(IR)) {
      indexInstructionsInFunction(*L->getHeader()->getParent(), LastPassRowId);
      return;
    }
  }

  sqlite3_int64 LastPassRowId = 0;

public:
  explicit DebugLocIndexState(StringRef Path) {
    std::string PathStr = Path.str();
    check(sqlite3_open_v2(PathStr.c_str(), &DB,
                          SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr),
          "sqlite3_open_v2");

    const char *Schema = R"SQL(
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS passes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  seq INTEGER NOT NULL,
  pass_class TEXT NOT NULL,
  ir_unit TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_passes_seq ON passes(seq);
CREATE TABLE IF NOT EXISTS instructions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  pass_id INTEGER NOT NULL REFERENCES passes(id),
  function TEXT NOT NULL,
  basicblock TEXT NOT NULL,
  inst_seq INTEGER NOT NULL,
  opcode TEXT NOT NULL,
  file_id INTEGER NOT NULL REFERENCES files(id),
  line INTEGER NOT NULL,
  col INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_instr_file_loc ON instructions(file_id, line, col);
CREATE INDEX IF NOT EXISTS idx_instr_pass ON instructions(pass_id);
)SQL";
    char *Err = nullptr;
    if (sqlite3_exec(DB, Schema, nullptr, nullptr, &Err) != SQLITE_OK) {
      std::string Msg = Err ? Err : "unknown error";
      sqlite3_free(Err);
      report_fatal_error(Twine("debug-loc-index schema: ") + Msg);
    }

    sqlite3_exec(DB, "DELETE FROM instructions", nullptr, nullptr, nullptr);
    sqlite3_exec(DB, "DELETE FROM passes", nullptr, nullptr, nullptr);
    sqlite3_exec(DB, "DELETE FROM files", nullptr, nullptr, nullptr);
    sqlite3_exec(DB, "DELETE FROM meta", nullptr, nullptr, nullptr);
    sqlite3_exec(DB,
                 "INSERT INTO meta(key,value) VALUES('schema_version','1')",
                 nullptr, nullptr, nullptr);

    const char *InsPass =
        "INSERT INTO passes(seq, pass_class, ir_unit) VALUES(?,?,?)";
    check(sqlite3_prepare_v2(DB, InsPass, -1, &StmtInsertPass, nullptr),
          "sqlite3_prepare(insert pass)");
    const char *InsFile = "INSERT OR IGNORE INTO files(path) VALUES(?)";
    check(sqlite3_prepare_v2(DB, InsFile, -1, &StmtInsertFileIgnore, nullptr),
          "sqlite3_prepare(insert file)");
    const char *SelFile = "SELECT id FROM files WHERE path = ?";
    check(sqlite3_prepare_v2(DB, SelFile, -1, &StmtSelectFileId, nullptr),
          "sqlite3_prepare(select file)");
    const char *InsInst =
        "INSERT INTO "
        "instructions(pass_id,function,basicblock,inst_seq,opcode,file_id,line,"
        "col) VALUES(?,?,?,?,?,?,?,?)";
    check(sqlite3_prepare_v2(DB, InsInst, -1, &StmtInsertInst, nullptr),
          "sqlite3_prepare(insert inst)");
  }

  ~DebugLocIndexState() {
    if (StmtInsertInst)
      sqlite3_finalize(StmtInsertInst);
    if (StmtSelectFileId)
      sqlite3_finalize(StmtSelectFileId);
    if (StmtInsertFileIgnore)
      sqlite3_finalize(StmtInsertFileIgnore);
    if (StmtInsertPass)
      sqlite3_finalize(StmtInsertPass);
    if (DB)
      sqlite3_close(DB);
  }

  void afterPass(StringRef PassID, Any IR, PassInstrumentationCallbacks &PIC) {
    if (isIgnoredPass(PassID))
      return;
    if (!shouldIndexIRUnit(IR))
      return;

    ++NextSeq;
    std::string IRUnit = getIRUnitLabel(IR);
    std::string PassName = PIC.getPassNameForClassName(PassID).str();
    if (PassName.empty())
      PassName = PassID.str();

    check(sqlite3_exec(DB, "BEGIN IMMEDIATE", nullptr, nullptr, nullptr),
          "sqlite3 BEGIN");

    check(sqlite3_bind_int(StmtInsertPass, 1, (int)NextSeq), "bind seq");
    check(sqlite3_bind_text(StmtInsertPass, 2, PassName.data(),
                            (int)PassName.size(), SQLITE_TRANSIENT),
          "bind pass_class");
    check(sqlite3_bind_text(StmtInsertPass, 3, IRUnit.data(),
                            (int)IRUnit.size(), SQLITE_TRANSIENT),
          "bind ir_unit");
    check(sqlite3_step(StmtInsertPass), "sqlite3_step(insert pass)");
    check(sqlite3_reset(StmtInsertPass), "sqlite3_reset(insert pass)");

    LastPassRowId = sqlite3_last_insert_rowid(DB);

    indexIR(IR);

    check(sqlite3_exec(DB, "COMMIT", nullptr, nullptr, nullptr),
          "sqlite3 COMMIT");
  }
};

} // namespace

void llvm::registerDebugLocIndexCallbacks(PassInstrumentationCallbacks &PIC) {
  StringRef Path = getDebugLocIndexDatabasePath();
  if (Path.empty())
    return;

  auto State = std::make_shared<DebugLocIndexState>(Path);
  PIC.registerAfterPassCallback(
      [State, &PIC](StringRef PassID, Any IR, const PreservedAnalyses &) {
        State->afterPass(PassID, IR, PIC);
      });
}

#else // !LLVM_ENABLE_DEBUG_LOC_INDEX

void llvm::registerDebugLocIndexCallbacks(PassInstrumentationCallbacks &) {
  if (!getDebugLocIndexDatabasePath().empty()) {
    errs() << "opt: -debug-loc-index-database requires an LLVM build linked "
              "against SQLite3 (e.g. install libsqlite3-dev and reconfigure).\n";
    exit(1);
  }
}

#endif
