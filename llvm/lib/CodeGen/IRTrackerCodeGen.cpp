//===- IRTrackerCodeGen.cpp - Final MIR / ISA tracker hooks ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Appends final MIR and textual ISA rows into the SQLite database selected by
// -ir-tracker-database.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/IR/PrintPasses.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#ifdef LLVM_ENABLE_IR_TRACKER_SQLITE
#include <sqlite3.h>
#endif

#include <cctype>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>

using namespace llvm;

#ifdef LLVM_ENABLE_IR_TRACKER_SQLITE
namespace {

static std::string getIRTrackerFilePath(const DILocation *Loc) {
  if (!Loc)
    return {};
  StringRef Dir = Loc->getDirectory();
  StringRef File = Loc->getFilename();
  if (File.empty())
    return {};
  if (Dir.empty())
    return File.str();
  SmallString<256> Path;
  Path = Dir;
  sys::path::append(Path, File);
  return std::string(Path);
}

class IRTrackerCodeGenDB {
  sqlite3 *DB = nullptr;
  sqlite3_stmt *StmtInsertPass = nullptr;
  sqlite3_stmt *StmtInsertFileIgnore = nullptr;
  sqlite3_stmt *StmtSelectFileId = nullptr;
  sqlite3_stmt *StmtInsertInst = nullptr;
  unsigned NextSeq = 0;
  std::unordered_map<std::string, int> FileIdCache;

  static int check(sqlite3 *DB, int RC, const char *Ctx) {
    if (RC == SQLITE_OK || RC == SQLITE_DONE || RC == SQLITE_ROW)
      return RC;
    StringRef Msg = DB ? sqlite3_errmsg(DB) : sqlite3_errstr(RC);
    report_fatal_error(Twine(Ctx) + ": " + Msg);
  }

  void ensureSchema() {
    const char *Schema = R"SQL(
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS ir_tracker_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ir_tracker_files (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS ir_tracker_passes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  seq INTEGER NOT NULL,
  phase TEXT NOT NULL,
  pass_class TEXT NOT NULL,
  ir_unit TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ir_tracker_idx_passes_seq
  ON ir_tracker_passes(seq);
CREATE TABLE IF NOT EXISTS ir_tracker_instructions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  pass_id INTEGER NOT NULL REFERENCES ir_tracker_passes(id),
  function TEXT NOT NULL,
  basicblock TEXT NOT NULL,
  inst_seq INTEGER NOT NULL,
  kind TEXT NOT NULL,
  opcode TEXT NOT NULL,
  inst_text TEXT NOT NULL,
  file_id INTEGER NOT NULL REFERENCES ir_tracker_files(id),
  line INTEGER NOT NULL,
  col INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ir_tracker_idx_instr_file_loc
  ON ir_tracker_instructions(kind, file_id, line, col);
CREATE INDEX IF NOT EXISTS ir_tracker_idx_instr_pass
  ON ir_tracker_instructions(pass_id);
)SQL";
    char *Err = nullptr;
    if (sqlite3_exec(DB, Schema, nullptr, nullptr, &Err) != SQLITE_OK) {
      std::string Msg = Err ? Err : "unknown error";
      sqlite3_free(Err);
      report_fatal_error(Twine("ir-tracker codegen schema: ") + Msg);
    }
    if (sqlite3_exec(DB,
                     "INSERT OR REPLACE INTO ir_tracker_meta(key, value) "
                     "VALUES('schema_version','4')",
                     nullptr, nullptr, &Err) != SQLITE_OK) {
      std::string Msg = Err ? Err : "unknown error";
      sqlite3_free(Err);
      report_fatal_error(Twine("ir-tracker codegen meta: ") + Msg);
    }
  }

  void prepareStatements() {
    const char *InsPass =
        "INSERT INTO ir_tracker_passes(seq, phase, pass_class, ir_unit) "
        "VALUES(?,?,?,?)";
    check(DB, sqlite3_prepare_v2(DB, InsPass, -1, &StmtInsertPass, nullptr),
          "sqlite3_prepare(codegen insert pass)");

    const char *InsFile = "INSERT OR IGNORE INTO ir_tracker_files(path) VALUES(?)";
    check(DB,
          sqlite3_prepare_v2(DB, InsFile, -1, &StmtInsertFileIgnore, nullptr),
          "sqlite3_prepare(codegen insert file)");

    const char *SelFile = "SELECT id FROM ir_tracker_files WHERE path = ?";
    check(DB, sqlite3_prepare_v2(DB, SelFile, -1, &StmtSelectFileId, nullptr),
          "sqlite3_prepare(codegen select file)");

    const char *InsInst =
        "INSERT INTO "
        "ir_tracker_instructions(pass_id,function,basicblock,inst_seq,kind,"
        "opcode,inst_text,file_id,line,col) VALUES(?,?,?,?,?,?,?,?,?,?)";
    check(DB, sqlite3_prepare_v2(DB, InsInst, -1, &StmtInsertInst, nullptr),
          "sqlite3_prepare(codegen insert inst)");
  }

  void initializeNextSeq() {
    sqlite3_stmt *Stmt = nullptr;
    std::string Q = "SELECT MAX(seq) FROM ir_tracker_passes";
    check(DB, sqlite3_prepare_v2(DB, Q.c_str(), -1, &Stmt, nullptr),
          "sqlite3_prepare(codegen max seq)");
    int MaxSeq = -1;
    if (sqlite3_step(Stmt) == SQLITE_ROW &&
        sqlite3_column_type(Stmt, 0) != SQLITE_NULL)
      MaxSeq = sqlite3_column_int(Stmt, 0);
    sqlite3_finalize(Stmt);
    NextSeq = (unsigned)(MaxSeq + 1);
  }

  int getOrCreateFileId(StringRef Path) {
    std::string Key = Path.str();
    auto It = FileIdCache.find(Key);
    if (It != FileIdCache.end())
      return It->second;

    check(DB, sqlite3_bind_text(StmtInsertFileIgnore, 1, Key.data(),
                                (int)Key.size(), SQLITE_TRANSIENT),
          "sqlite3_bind_text(codegen insert file)");
    check(DB, sqlite3_step(StmtInsertFileIgnore),
          "sqlite3_step(codegen insert file)");
    check(DB, sqlite3_reset(StmtInsertFileIgnore),
          "sqlite3_reset(codegen insert file)");

    check(DB, sqlite3_bind_text(StmtSelectFileId, 1, Key.data(), (int)Key.size(),
                                SQLITE_TRANSIENT),
          "sqlite3_bind_text(codegen select file)");
    if (sqlite3_step(StmtSelectFileId) != SQLITE_ROW)
      report_fatal_error("ir-tracker codegen: missing file row after insert");
    int Id = sqlite3_column_int(StmtSelectFileId, 0);
    check(DB, sqlite3_reset(StmtSelectFileId),
          "sqlite3_reset(codegen select file)");
    FileIdCache[Key] = Id;
    return Id;
  }

  sqlite3_int64 insertPassRow(StringRef Phase, StringRef PassName,
                              StringRef IRUnit) {
    check(DB, sqlite3_bind_int(StmtInsertPass, 1, (int)NextSeq++),
          "bind codegen seq");
    check(DB, sqlite3_bind_text(StmtInsertPass, 2, Phase.data(),
                                (int)Phase.size(), SQLITE_TRANSIENT),
          "bind codegen phase");
    check(DB, sqlite3_bind_text(StmtInsertPass, 3, PassName.data(),
                                (int)PassName.size(), SQLITE_TRANSIENT),
          "bind codegen pass");
    check(DB, sqlite3_bind_text(StmtInsertPass, 4, IRUnit.data(),
                                (int)IRUnit.size(), SQLITE_TRANSIENT),
          "bind codegen ir unit");
    check(DB, sqlite3_step(StmtInsertPass), "sqlite3_step(codegen insert pass)");
    check(DB, sqlite3_reset(StmtInsertPass),
          "sqlite3_reset(codegen insert pass)");
    return sqlite3_last_insert_rowid(DB);
  }

  void insertInstructionRow(sqlite3_int64 PassRowId, StringRef FunctionName,
                            StringRef BBLabel, int InstSeq, StringRef Kind,
                            StringRef Opcode, StringRef InstText, int FileId,
                            int Line, int Col) {
    check(DB, sqlite3_bind_int64(StmtInsertInst, 1, PassRowId),
          "bind codegen pass row");
    check(DB, sqlite3_bind_text(StmtInsertInst, 2, FunctionName.data(),
                                (int)FunctionName.size(), SQLITE_TRANSIENT),
          "bind codegen function");
    check(DB, sqlite3_bind_text(StmtInsertInst, 3, BBLabel.data(),
                                (int)BBLabel.size(), SQLITE_TRANSIENT),
          "bind codegen bb");
    check(DB, sqlite3_bind_int(StmtInsertInst, 4, InstSeq),
          "bind codegen inst seq");
    check(DB, sqlite3_bind_text(StmtInsertInst, 5, Kind.data(), (int)Kind.size(),
                                SQLITE_TRANSIENT),
          "bind codegen kind");
    check(DB, sqlite3_bind_text(StmtInsertInst, 6, Opcode.data(),
                                (int)Opcode.size(), SQLITE_TRANSIENT),
          "bind codegen opcode");
    check(DB, sqlite3_bind_text(StmtInsertInst, 7, InstText.data(),
                                (int)InstText.size(), SQLITE_TRANSIENT),
          "bind codegen inst text");
    check(DB, sqlite3_bind_int(StmtInsertInst, 8, FileId),
          "bind codegen file");
    check(DB, sqlite3_bind_int(StmtInsertInst, 9, Line),
          "bind codegen line");
    check(DB, sqlite3_bind_int(StmtInsertInst, 10, Col),
          "bind codegen col");
    check(DB, sqlite3_step(StmtInsertInst), "sqlite3_step(codegen insert inst)");
    check(DB, sqlite3_reset(StmtInsertInst),
          "sqlite3_reset(codegen insert inst)");
  }

  static std::string getMachineBlockLabel(const MachineBasicBlock &MBB) {
    if (MBB.hasName())
      return MBB.getName().str();
    std::string Label;
    raw_string_ostream OS(Label);
    MBB.printAsOperand(OS, /*PrintType=*/false);
    OS.flush();
    return Label;
  }

public:
  explicit IRTrackerCodeGenDB(StringRef Path) {
    std::string PathStr = Path.str();
    check(DB, sqlite3_open_v2(PathStr.c_str(), &DB,
                              SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
                              nullptr),
          "sqlite3_open_v2(codegen)");
    ensureSchema();
    prepareStatements();
    initializeNextSeq();
    check(DB, sqlite3_exec(DB, "BEGIN", nullptr, nullptr, nullptr),
          "sqlite3 BEGIN(codegen)");
  }

  ~IRTrackerCodeGenDB() {
    if (!DB)
      return;
    if (StmtInsertInst)
      sqlite3_finalize(StmtInsertInst);
    if (StmtSelectFileId)
      sqlite3_finalize(StmtSelectFileId);
    if (StmtInsertFileIgnore)
      sqlite3_finalize(StmtInsertFileIgnore);
    if (StmtInsertPass)
      sqlite3_finalize(StmtInsertPass);
    if (sqlite3_get_autocommit(DB) == 0)
      check(DB, sqlite3_exec(DB, "COMMIT", nullptr, nullptr, nullptr),
            "sqlite3 COMMIT(codegen)");
    sqlite3_close(DB);
  }

  void appendFinalMIR(MachineFunction &MF) {
    sqlite3_int64 PassRowId =
        insertPassRow("final-mir", "<final-mir>", MF.getName());
    ModuleSlotTracker MST(MF.getFunction().getParent());
    MST.incorporateFunction(MF.getFunction());
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    std::string FuncName = MF.getName().str();
    for (const MachineBasicBlock &MBB : MF) {
      std::string BBLabel = getMachineBlockLabel(MBB);
      int InstSeq = 0;
      for (const MachineInstr &MI : MBB) {
        DebugLoc DL = MI.getDebugLoc();
        if (!DL)
          continue;
        const DILocation *Loc = DL.get();
        unsigned Line = Loc->getLine();
        if (Line == 0)
          continue;
        std::string FilePath = getIRTrackerFilePath(Loc);
        if (FilePath.empty())
          continue;
        std::string InstText;
        raw_string_ostream OS(InstText);
        MI.print(OS, MST, /*IsStandalone=*/true, /*SkipOpers=*/false,
                 /*SkipDebugLoc=*/true, /*AddNewLine=*/false, TII);
        OS.flush();
        StringRef Opcode = TII ? TII->getName(MI.getOpcode()) : "<unknown>";
        insertInstructionRow(PassRowId, FuncName, BBLabel, InstSeq++, "mir",
                             Opcode, InstText, getOrCreateFileId(FilePath),
                             (int)Line, (int)Loc->getColumn());
      }
    }
  }

  void appendISAFromAssembly(StringRef AssemblyText) {
    sqlite3_int64 PassRowId =
        insertPassRow("final-isa", "<final-isa>", "[module]");
    std::unordered_map<unsigned, std::string> FilePaths;
    int CurrentFileId = 0;
    int CurrentLine = 0;
    int CurrentCol = 0;
    std::string CurrentFunction = "<unknown>";
    std::string CurrentBlock = "<asm>";
    int InstSeq = 0;

    auto parseQuotedStrings = [](StringRef Line) {
      SmallVector<std::string, 4> Strings;
      size_t Pos = 0;
      while (Pos < Line.size()) {
        size_t Start = Line.find('"', Pos);
        if (Start == StringRef::npos)
          break;
        size_t End = Line.find('"', Start + 1);
        if (End == StringRef::npos)
          break;
        Strings.push_back(Line.slice(Start + 1, End).str());
        Pos = End + 1;
      }
      return Strings;
    };

    auto parseToken = [](StringRef &Line) {
      Line = Line.ltrim();
      StringRef Tok = Line.take_until(
          [](char C) { return std::isspace(static_cast<unsigned char>(C)); });
      Line = Line.drop_front(Tok.size());
      return Tok;
    };

    StringRef Remaining = AssemblyText;
    while (!Remaining.empty()) {
      auto Split = Remaining.split('\n');
      StringRef Line =
          Split.first.take_until([](char C) { return C == ';'; }).trim();
      Remaining = Split.second;
      if (Line.empty())
        continue;

      if (Line.ends_with(":")) {
        InstSeq = 0;
        std::string Label = Line.drop_back().str();
        if (!Line.starts_with(".")) {
          CurrentFunction = Label;
          CurrentBlock = "<asm>";
        } else {
          CurrentBlock = Label;
        }
        continue;
      }

      if (Line.starts_with(".file")) {
        StringRef Rest = Line.drop_front(strlen(".file"));
        StringRef FileNoTok = parseToken(Rest);
        unsigned FileNo = 0;
        if (FileNoTok.getAsInteger(10, FileNo))
          continue;
        SmallVector<std::string, 4> Strings = parseQuotedStrings(Line);
        if (Strings.empty())
          continue;
        std::string Path;
        if (Strings.size() >= 2) {
          if (sys::path::is_absolute(Strings[1]))
            Path = Strings[1];
          else {
            SmallString<256> Joined(Strings[0]);
            sys::path::append(Joined, Strings[1]);
            Path = std::string(Joined);
          }
        } else {
          Path = Strings[0];
        }
        FilePaths[FileNo] = std::move(Path);
        continue;
      }

      if (Line.starts_with(".loc")) {
        StringRef Rest = Line.drop_front(strlen(".loc"));
        StringRef FileTok = parseToken(Rest);
        StringRef LineTok = parseToken(Rest);
        StringRef ColTok = parseToken(Rest);
        unsigned FileNo = 0;
        unsigned SrcLine = 0;
        unsigned SrcCol = 0;
        if (FileTok.getAsInteger(10, FileNo) || LineTok.getAsInteger(10, SrcLine))
          continue;
        if (ColTok.empty() || ColTok.getAsInteger(10, SrcCol))
          SrcCol = 0;
        auto It = FilePaths.find(FileNo);
        if (It == FilePaths.end())
          continue;
        CurrentFileId = getOrCreateFileId(It->second);
        CurrentLine = (int)SrcLine;
        CurrentCol = (int)SrcCol;
        InstSeq = 0;
        continue;
      }

      if (CurrentFileId == 0 || CurrentLine == 0)
        continue;
      if (Line.starts_with(".") || Line.starts_with(";") ||
          Line.starts_with("#"))
        continue;

      std::string InstText = Line.str();
      StringRef OpcodeLine = Line;
      StringRef Opcode = parseToken(OpcodeLine);
      if (Opcode.empty())
        continue;
      insertInstructionRow(PassRowId, CurrentFunction, CurrentBlock, InstSeq++,
                           "isa", Opcode, InstText, CurrentFileId, CurrentLine,
                           CurrentCol);
    }
  }
};

class IRTrackerFinalMIRPass : public MachineFunctionPass {
  std::unique_ptr<IRTrackerCodeGenDB> Appender;

public:
  static char ID;
  IRTrackerFinalMIRPass() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "IR tracker final MIR pass"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool doInitialization(Module &) override {
    StringRef Path = getIRTrackerDatabasePath();
    if (!Path.empty())
      Appender = std::make_unique<IRTrackerCodeGenDB>(Path);
    return false;
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (Appender)
      Appender->appendFinalMIR(MF);
    return false;
  }

  bool doFinalization(Module &) override {
    Appender.reset();
    return false;
  }
};

char IRTrackerFinalMIRPass::ID = 0;

} // namespace

MachineFunctionPass *llvm::createIRTrackerFinalMIRPass() {
  if (getIRTrackerDatabasePath().empty())
    return nullptr;
  return new IRTrackerFinalMIRPass();
}

void llvm::appendIRTrackerISAFromAssembly(StringRef DatabasePath,
                                          StringRef AssemblyText) {
  if (DatabasePath.empty() || AssemblyText.empty())
    return;
  IRTrackerCodeGenDB Appender(DatabasePath);
  Appender.appendISAFromAssembly(AssemblyText);
}

#else

MachineFunctionPass *llvm::createIRTrackerFinalMIRPass() { return nullptr; }

void llvm::appendIRTrackerISAFromAssembly(StringRef, StringRef) {}

#endif
