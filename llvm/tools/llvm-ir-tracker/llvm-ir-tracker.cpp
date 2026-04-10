//===- llvm-ir-tracker.cpp - IR tracker SQLite helper ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wraps opt -ir-tracker-database= and queries ir_tracker_* tables (schema v2).
// Query subcommands require LLVM to be built with SQLite (same as LLVMPasses).
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#ifdef LLVM_IR_TRACKER_TOOL_SQLITE
#include <sqlite3.h>
#endif

#include <cctype>
#include <string>
#include <vector>

using namespace llvm;

static constexpr StringRef T_FILES = "ir_tracker_files";
static constexpr StringRef T_PASSES = "ir_tracker_passes";
static constexpr StringRef T_INSTR = "ir_tracker_instructions";

cl::SubCommand BuildCmd("build",
                        "Run opt with -ir-tracker-database (forward args after --)");
cl::SubCommand PassesCmd("passes", "List recorded passes");
cl::SubCommand TraceCmd(
    "trace", "Find first pass with instructions matching a source line");
cl::SubCommand SqlCmd("sql", "Run a read-only SQL query");

static cl::opt<std::string>
    DbPath("db", cl::desc("SQLite database path"), cl::value_desc("path"),
           cl::Required, cl::sub(BuildCmd), cl::sub(PassesCmd),
           cl::sub(TraceCmd), cl::sub(SqlCmd));

static cl::opt<std::string>
    BuildOptPath("opt",
                 cl::desc("Path to opt (default: opt next to this executable)"),
                 cl::value_desc("path"), cl::init(""), cl::sub(BuildCmd));

static cl::list<std::string>
    BuildForwardArgs(cl::Positional,
                     cl::desc("Arguments for opt (use -- before flags)"),
                     cl::ConsumeAfter, cl::sub(BuildCmd));

static cl::opt<std::string>
    TraceFile("file", cl::desc("Substring or basename to match file path"),
              cl::Required, cl::sub(TraceCmd));
static cl::opt<std::string> TraceLine("line", cl::desc("1-based source line"),
                                      cl::Required, cl::sub(TraceCmd));
static cl::opt<int> TraceCol("col", cl::desc("Optional column"), cl::init(-1),
                             cl::sub(TraceCmd));
static cl::opt<std::string> TraceOpcode("opcode", cl::desc("Optional opcode"),
                                        cl::init(""), cl::sub(TraceCmd));

static cl::opt<std::string> SqlStatement(cl::Positional, cl::Required,
                                         cl::desc("<sql>"), cl::sub(SqlCmd));

namespace {

std::string getExecutablePath(const char *Argv0) {
  void *P = (void *)(intptr_t)getExecutablePath;
  return sys::fs::getMainExecutable(Argv0, P);
}

std::string makeAbsoluteDbPath(StringRef Db) {
  if (sys::path::is_absolute(Db))
    return std::string(Db);
  SmallString<256> Cwd;
  if (std::error_code EC = sys::fs::current_path(Cwd)) {
    errs() << "llvm-ir-tracker: cannot get working directory: " << EC.message()
           << '\n';
    return {};
  }
  sys::path::append(Cwd, Db);
  return std::string(Cwd);
}

int runBuild(const char *Argv0) {
  std::string OptExe = BuildOptPath;
  if (OptExe.empty()) {
    SmallString<256> OptPath(sys::path::parent_path(getExecutablePath(Argv0)));
    sys::path::append(OptPath, "opt");
    OptExe = std::string(OptPath);
#if defined(_WIN32)
    OptExe += ".exe";
#endif
  }

  if (!sys::fs::exists(OptExe)) {
    errs() << "llvm-ir-tracker: opt not found: " << OptExe
           << " (use --opt=PATH)\n";
    return 2;
  }

  std::string DbAbs = makeAbsoluteDbPath(DbPath);
  if (DbAbs.empty())
    return 1;

  std::vector<std::string> Storage;
  Storage.push_back(OptExe);
  Storage.push_back(std::string("-ir-tracker-database=") + DbAbs);

  for (const std::string &A : BuildForwardArgs) {
    if (Storage.size() == 2 && A == "--")
      continue;
    Storage.push_back(A);
  }

  SmallVector<StringRef, 32> ArgSR;
  for (const std::string &S : Storage)
    ArgSR.push_back(S);

  std::string ErrMsg;
  bool ExecFailed = false;
  int RC = sys::ExecuteAndWait(OptExe, ArgSR, std::nullopt, {}, 0, 0, &ErrMsg,
                               &ExecFailed);
  if (ExecFailed) {
    if (!ErrMsg.empty())
      errs() << "llvm-ir-tracker: " << ErrMsg << '\n';
    return 1;
  }
  if (RC < 0) {
    errs() << "llvm-ir-tracker: failed to execute opt\n";
    return 1;
  }
  return RC;
}

#ifdef LLVM_IR_TRACKER_TOOL_SQLITE

static void printSqliteError(sqlite3 *DB, const char *Ctx) {
  errs() << "llvm-ir-tracker: " << Ctx << ": "
         << (DB ? sqlite3_errmsg(DB) : "null db") << '\n';
}

static sqlite3 *openDbReadOnly(StringRef Path) {
  if (!sys::fs::exists(Path)) {
    errs() << "llvm-ir-tracker: database not found: " << Path << '\n';
    return nullptr;
  }
  sqlite3 *DB = nullptr;
  std::string PathStr(Path);
  int RC =
      sqlite3_open_v2(PathStr.c_str(), &DB, SQLITE_OPEN_READONLY, nullptr);
  if (RC != SQLITE_OK) {
    printSqliteError(DB, "sqlite3_open_v2");
    if (DB)
      sqlite3_close(DB);
    return nullptr;
  }
  return DB;
}

static std::string lowerString(StringRef S) {
  std::string R = S.str();
  for (char &C : R)
    C = static_cast<char>(tolower(static_cast<unsigned char>(C)));
  return R;
}

static std::vector<int> resolveFileIds(sqlite3 *DB, StringRef FilePat) {
  std::vector<int> Ids;
  sqlite3_stmt *Stmt = nullptr;
  std::string Q = "SELECT id, path FROM " + std::string(T_FILES);
  if (sqlite3_prepare_v2(DB, Q.c_str(), -1, &Stmt, nullptr) != SQLITE_OK) {
    printSqliteError(DB, "prepare(files)");
    return Ids;
  }
  std::string PatLower = lowerString(FilePat);
  while (sqlite3_step(Stmt) == SQLITE_ROW) {
    int Id = sqlite3_column_int(Stmt, 0);
    const char *PathPtr =
        reinterpret_cast<const char *>(sqlite3_column_text(Stmt, 1));
    std::string Path = PathPtr ? PathPtr : "";
    std::string PathLower = lowerString(Path);
    StringRef PathRef(Path);
    if (PathLower.find(PatLower) != std::string::npos ||
        PathRef.ends_with(FilePat) || PathRef == FilePat)
      Ids.push_back(Id);
  }
  sqlite3_finalize(Stmt);
  return Ids;
}

int runPasses() {
  sqlite3 *DB = openDbReadOnly(DbPath);
  if (!DB)
    return 1;
  sqlite3_stmt *Stmt = nullptr;
  std::string Q = "SELECT id, seq, pass_class, ir_unit FROM " +
                  std::string(T_PASSES) + " ORDER BY seq ASC";
  if (sqlite3_prepare_v2(DB, Q.c_str(), -1, &Stmt, nullptr) != SQLITE_OK) {
    printSqliteError(DB, "prepare(passes)");
    sqlite3_close(DB);
    return 1;
  }
  int Count = 0;
  while (sqlite3_step(Stmt) == SQLITE_ROW) {
    ++Count;
    sqlite3_int64 Id = sqlite3_column_int64(Stmt, 0);
    int Seq = sqlite3_column_int(Stmt, 1);
    const char *PC = reinterpret_cast<const char *>(sqlite3_column_text(Stmt, 2));
    const char *IU = reinterpret_cast<const char *>(sqlite3_column_text(Stmt, 3));
    outs() << format("%5d  id=%-6lld  '", Seq, (long long)Id)
           << (PC ? PC : "") << "'  on '" << (IU ? IU : "") << "'\n";
  }
  sqlite3_finalize(Stmt);
  outs() << "total passes recorded: " << Count << '\n';
  sqlite3_close(DB);
  return 0;
}

int runTrace() {
  sqlite3 *DB = openDbReadOnly(DbPath);
  if (!DB)
    return 1;

  std::vector<int> FileIds = resolveFileIds(DB, TraceFile);
  if (FileIds.empty()) {
    errs() << "llvm-ir-tracker: no " << T_FILES
           << " rows match --file (try a basename or substring)\n";
    sqlite3_close(DB);
    return 1;
  }

  int Line = 0;
  if (TraceLine.getAsInteger(0, Line) || Line <= 0) {
    errs() << "llvm-ir-tracker: invalid --line\n";
    sqlite3_close(DB);
    return 1;
  }

  std::string ColSql;
  std::string OpcSql;
  if (TraceCol >= 0)
    ColSql = " AND i.col = ?";
  if (!TraceOpcode.empty())
    OpcSql = " AND i.opcode = ?";

  sqlite3_stmt *MaxStmt = nullptr;
  std::string QMax = "SELECT MAX(seq) AS m FROM " + std::string(T_PASSES);
  if (sqlite3_prepare_v2(DB, QMax.c_str(), -1, &MaxStmt, nullptr) !=
      SQLITE_OK) {
    printSqliteError(DB, "prepare(max seq)");
    sqlite3_close(DB);
    return 1;
  }
  int MaxSeq = -1;
  if (sqlite3_step(MaxStmt) == SQLITE_ROW &&
      sqlite3_column_type(MaxStmt, 0) != SQLITE_NULL)
    MaxSeq = sqlite3_column_int(MaxStmt, 0);
  sqlite3_finalize(MaxStmt);
  if (MaxSeq < 0) {
    errs() << "llvm-ir-tracker: empty " << T_PASSES << " table\n";
    sqlite3_close(DB);
    return 1;
  }

  std::string InClause;
  for (size_t I = 0; I < FileIds.size(); ++I) {
    if (I)
      InClause += ',';
    InClause += '?';
  }

  auto bindFileIdsAndLine = [&](sqlite3_stmt *S, int StartIdx) {
    int Idx = StartIdx;
    for (int Fid : FileIds)
      sqlite3_bind_int(S, Idx++, Fid);
    sqlite3_bind_int(S, Idx++, Line);
    if (TraceCol >= 0)
      sqlite3_bind_int(S, Idx++, TraceCol);
    if (!TraceOpcode.empty()) {
      sqlite3_bind_text(S, Idx++, TraceOpcode.c_str(), -1, SQLITE_TRANSIENT);
    }
    return SQLITE_OK;
  };

  {
    std::string QCnt = "SELECT COUNT(*) AS c FROM " + std::string(T_INSTR) +
                       " i JOIN " + std::string(T_PASSES) +
                       " p ON i.pass_id = p.id WHERE p.seq = ? AND i.file_id IN (" +
                       InClause + ") AND i.line = ?" + ColSql + OpcSql;
    sqlite3_stmt *Stmt = nullptr;
    if (sqlite3_prepare_v2(DB, QCnt.c_str(), -1, &Stmt, nullptr) != SQLITE_OK) {
      printSqliteError(DB, "prepare(count)");
      sqlite3_close(DB);
      return 1;
    }
    int B = 1;
    sqlite3_bind_int(Stmt, B++, MaxSeq);
    bindFileIdsAndLine(Stmt, B);
    int FinalCount = 0;
    if (sqlite3_step(Stmt) == SQLITE_ROW)
      FinalCount = sqlite3_column_int(Stmt, 0);
    sqlite3_finalize(Stmt);

    outs() << "Matches at final pass (seq=" << MaxSeq << "): " << FinalCount
           << " instruction(s) (file id(s) ";
    for (size_t I = 0; I < FileIds.size(); ++I) {
      if (I)
        outs() << ',';
      outs() << FileIds[I];
    }
    outs() << ", line " << Line << ")\n";
    if (FinalCount == 0)
      errs() << "llvm-ir-tracker: no match in final IR — debug locations may "
                "have been dropped, or try different --file/--line/--col.\n";
  }

  {
    std::string QFirst =
        "SELECT p.seq, p.pass_class, p.ir_unit, COUNT(*) AS n FROM " +
        std::string(T_INSTR) + " i JOIN " + std::string(T_PASSES) +
        " p ON i.pass_id = p.id WHERE i.file_id IN (" + InClause +
        ") AND i.line = ?" + ColSql + OpcSql +
        " GROUP BY p.id ORDER BY p.seq ASC LIMIT 1";
    sqlite3_stmt *Stmt = nullptr;
    if (sqlite3_prepare_v2(DB, QFirst.c_str(), -1, &Stmt, nullptr) !=
        SQLITE_OK) {
      printSqliteError(DB, "prepare(first pass)");
      sqlite3_close(DB);
      return 1;
    }
    bindFileIdsAndLine(Stmt, 1);
    if (sqlite3_step(Stmt) == SQLITE_ROW) {
      int Seq = sqlite3_column_int(Stmt, 0);
      const char *PC =
          reinterpret_cast<const char *>(sqlite3_column_text(Stmt, 1));
      const char *IU =
          reinterpret_cast<const char *>(sqlite3_column_text(Stmt, 2));
      int N = sqlite3_column_int(Stmt, 3);
      outs() << "First pass with any matching instruction: seq=" << Seq << ' '
             << (PC ? PC : "") << " on " << (IU ? IU : "") << " (" << N
             << " row(s))\n";
    } else {
      outs() << "No pass recorded any matching instruction.\n";
    }
    sqlite3_finalize(Stmt);
  }

  sqlite3_close(DB);
  return 0;
}

int runSql() {
  sqlite3 *DB = openDbReadOnly(DbPath);
  if (!DB)
    return 1;
  sqlite3_stmt *Stmt = nullptr;
  if (sqlite3_prepare_v2(DB, SqlStatement.c_str(), -1, &Stmt, nullptr) !=
      SQLITE_OK) {
    printSqliteError(DB, "prepare(sql)");
    sqlite3_close(DB);
    return 1;
  }
  int Cols = sqlite3_column_count(Stmt);
  while (true) {
    int StepRC = sqlite3_step(Stmt);
    if (StepRC == SQLITE_DONE)
      break;
    if (StepRC != SQLITE_ROW) {
      printSqliteError(DB, "sqlite3_step(sql)");
      sqlite3_finalize(Stmt);
      sqlite3_close(DB);
      return 1;
    }
    outs() << '(';
    for (int C = 0; C < Cols; ++C) {
      if (C)
        outs() << ", ";
      if (sqlite3_column_type(Stmt, C) == SQLITE_NULL)
        outs() << "None";
      else {
        const char *T =
            reinterpret_cast<const char *>(sqlite3_column_text(Stmt, C));
        outs() << (T ? T : "");
      }
    }
    outs() << ")\n";
  }
  sqlite3_finalize(Stmt);
  sqlite3_close(DB);
  return 0;
}

#endif // LLVM_IR_TRACKER_TOOL_SQLITE

} // namespace

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  StringRef ProgName = sys::path::filename(argv[0]);

  if (argc < 2) {
    errs() << ProgName
           << ": no subcommand specified; run with --help for usage.\n";
    return 1;
  }

  cl::ParseCommandLineOptions(argc, argv, "LLVM IR tracker (SQLite)\n");

  if (BuildCmd)
    return runBuild(argv[0]);

#ifndef LLVM_IR_TRACKER_TOOL_SQLITE
  errs() << ProgName << ": query subcommands need SQLite (same dependency as "
                        "opt -ir-tracker-database); only 'build' is available.\n";
  return 1;
#else
  if (PassesCmd)
    return runPasses();
  if (TraceCmd)
    return runTrace();
  if (SqlCmd)
    return runSql();
#endif

  errs() << ProgName << ": unknown command; run with --help for usage.\n";
  return 1;
}
