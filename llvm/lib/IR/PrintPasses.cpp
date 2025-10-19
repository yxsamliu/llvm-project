#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include <optional>
#include <unordered_map>
//===- PrintPasses.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PrintPasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include <unordered_set>

using namespace llvm;

// Source location filtering should only impact debug-oriented IR printing,
// such as -print-before/-print-after and print-changed. To ensure regular
// outputs (e.g. opt -o foo.bc) are not affected, we gate filtering behind a
// per-thread debug mode flag that is enabled only within the printing
// instrumentation codepaths.
static thread_local bool SourceLocFilterDebugMode = false;

// Print IR out before/after specified passes.
static cl::list<std::string>
    PrintBefore("print-before",
                llvm::cl::desc("Print IR before specified passes"),
                cl::CommaSeparated, cl::Hidden);

static cl::list<std::string>
    PrintAfter("print-after", llvm::cl::desc("Print IR after specified passes"),
               cl::CommaSeparated, cl::Hidden);

static cl::opt<bool> PrintBeforeAll("print-before-all",
                                    llvm::cl::desc("Print IR before each pass"),
                                    cl::init(false), cl::Hidden);
static cl::opt<bool> PrintAfterAll("print-after-all",
                                   llvm::cl::desc("Print IR after each pass"),
                                   cl::init(false), cl::Hidden);

// Print out the IR after passes, similar to -print-after-all except that it
// only prints the IR after passes that change the IR. Those passes that do not
// make changes to the IR are reported as not making any changes. In addition,
// the initial IR is also reported.  Other hidden options affect the output from
// this option. -filter-passes will limit the output to the named passes that
// actually change the IR and other passes are reported as filtered out. The
// specified passes will either be reported as making no changes (with no IR
// reported) or the changed IR will be reported. Also, the -filter-print-funcs
// and -print-module-scope options will do similar filtering based on function
// name, reporting changed IRs as functions(or modules if -print-module-scope is
// specified) for a particular function or indicating that the IR has been
// filtered out. The extra options can be combined, allowing only changed IRs
// for certain passes on certain functions to be reported in different formats,
// with the rest being reported as filtered out.  The -print-before-changed
// option will print the IR as it was before each pass that changed it. The
// optional value of quiet will only report when the IR changes, suppressing all
// other messages, including the initial IR. The values "diff" and "diff-quiet"
// will present the changes in a form similar to a patch, in either verbose or
// quiet mode, respectively. The lines that are removed and added are prefixed
// with '-' and '+', respectively. The -filter-print-funcs and -filter-passes
// can be used to filter the output.  This reporter relies on the linux diff
// utility to do comparisons and insert the prefixes. For systems that do not
// have the necessary facilities, the error message will be shown in place of
// the expected output.
cl::opt<ChangePrinter> llvm::PrintChanged(
    "print-changed", cl::desc("Print changed IRs"), cl::Hidden,
    cl::ValueOptional, cl::init(ChangePrinter::None),
    cl::values(
        clEnumValN(ChangePrinter::Quiet, "quiet", "Run in quiet mode"),
        clEnumValN(ChangePrinter::DiffVerbose, "diff",
                   "Display patch-like changes"),
        clEnumValN(ChangePrinter::DiffQuiet, "diff-quiet",
                   "Display patch-like changes in quiet mode"),
        clEnumValN(ChangePrinter::ColourDiffVerbose, "cdiff",
                   "Display patch-like changes with color"),
        clEnumValN(ChangePrinter::ColourDiffQuiet, "cdiff-quiet",
                   "Display patch-like changes in quiet mode with color"),
        clEnumValN(ChangePrinter::DotCfgVerbose, "dot-cfg",
                   "Create a website with graphical changes"),
        clEnumValN(ChangePrinter::DotCfgQuiet, "dot-cfg-quiet",
                   "Create a website with graphical changes in quiet mode"),
        // Sentinel value for unspecified option.
        clEnumValN(ChangePrinter::Verbose, "", "")));

// An option for specifying the diff used by print-changed=[diff | diff-quiet]
static cl::opt<std::string>
    DiffBinary("print-changed-diff-path", cl::Hidden, cl::init("diff"),
               cl::desc("system diff used by change reporters"));

static cl::opt<bool>
    PrintModuleScope("print-module-scope",
                     cl::desc("When printing IR for print-[before|after]{-all} "
                              "always print a module IR"),
                     cl::init(false), cl::Hidden);

static cl::opt<bool> LoopPrintFuncScope(
    "print-loop-func-scope",
    cl::desc("When printing IR for print-[before|after]{-all} "
             "for a loop pass, always print function IR"),
    cl::init(false), cl::Hidden);

// See the description for -print-changed for an explanation of the use
// of this option.
static cl::list<std::string> FilterPasses(
    "filter-passes", cl::value_desc("pass names"),
    cl::desc("Only consider IR changes for passes whose names "
             "match the specified value. No-op without -print-changed"),
    cl::CommaSeparated, cl::Hidden);

static cl::list<std::string>
    PrintFuncsList("filter-print-funcs", cl::value_desc("function names"),
                   cl::desc("Only print IR for functions whose name "
                            "match this for all print-[before|after][-all] "
                            "options"),
                   cl::CommaSeparated, cl::Hidden);

static cl::list<std::string> PrintSourceLocations(
    "filter-print-src-locs", cl::Hidden, cl::CommaSeparated,
    cl::desc("Only print IR that contains instructions in the given source "
             "location ranges. Each entry uses 'path[:line[-line]][:col[-col]]'."));

namespace {

class SourceLocMatcher {
public:
  bool initialize(ArrayRef<std::string> Specs, raw_ostream &Errs);
  bool empty() const { return Filters.empty(); }
  bool matches(StringRef FilePath, unsigned Line, unsigned Column) const;

private:
  bool addFilter(StringRef Spec, raw_ostream &Errs);
  bool parseRange(StringRef Input, unsigned &Begin, unsigned &End,
                  raw_ostream &Errs, StringRef Kind) const;
  static std::string normalizePath(StringRef Path);
  bool matchFile(const SourceLocFilterSpec &Filter, StringRef NormalizedPath,
                 StringRef Basename) const;

  SmallVector<SourceLocFilterSpec, 4> Filters;
};

}

static SourceLocMatcher &getSourceLocMatcher() {
  static SourceLocMatcher Matcher;
  static bool Initialized = false;
  if (!Initialized) {
    std::string Buffer;
    raw_string_ostream OS(Buffer);
    Matcher.initialize(PrintSourceLocations, OS);
    if (!Buffer.empty())
      errs() << OS.str();
    Initialized = true;
  }
  return Matcher;
}

// ---- SourceLocMatcher implementation ----

bool SourceLocMatcher::parseRange(StringRef Input, unsigned &Begin,
                                  unsigned &End, raw_ostream &Errs,
                                  StringRef Kind) const {
  if (Input.empty())
    return true; // Unspecified is treated as wildcard

  // Accept single value or begin-end
  StringRef L = Input;
  StringRef R;
  std::tie(L, R) = Input.split('-');
  unsigned Val = 0;
  if (!L.getAsInteger(10, Val)) {
    Begin = Val;
    if (R.empty()) {
      End = Val;
      return true;
    }
    if (!R.getAsInteger(10, Val)) {
      End = Val;
      return true;
    }
  }
  Errs << "error: invalid " << Kind << " range '" << Input << "'\n";
  return false;
}

std::string SourceLocMatcher::normalizePath(StringRef PathRef) {
  SmallString<256> P(PathRef);
  sys::path::native(P);
  sys::path::remove_dots(P, /*remove_dot_dot=*/true);
  sys::path::remove_leading_dotslash(P);
  // Convert to posix-style for stable matching regardless of host
  return sys::path::convert_to_slash(StringRef(P), sys::path::Style::native);
}

bool SourceLocMatcher::matchFile(const SourceLocFilterSpec &Filter,
                                 StringRef NormalizedPath,
                                 StringRef Basename) const {
  if (Filter.IsWildcard)
    return true;
  if (!Filter.HasFile)
    return true;
  if (Filter.MatchBasenameOnly)
    return Basename == Filter.Basename;
  return NormalizedPath == Filter.NormalizedFile;
}

bool SourceLocMatcher::addFilter(StringRef Spec, raw_ostream &Errs) {
  SourceLocFilterSpec F;
  F.RawFile = Spec.str();

  // Split into path[:line[-line]][:col[-col]]
  StringRef PathPart = Spec;
  StringRef Rest;
  std::tie(PathPart, Rest) = Spec.split(':');

  if (PathPart == "*") {
    F.IsWildcard = true;
    F.HasFile = false;
  } else {
    F.HasFile = true;
    F.NormalizedFile = normalizePath(PathPart);
    F.Basename = sys::path::filename(F.NormalizedFile).str();
    // If only a basename is provided (no directory separators), match basename
    F.MatchBasenameOnly = (F.NormalizedFile.find('/') == std::string::npos);
  }

  // Parse optional :line[-line]
  if (!Rest.empty()) {
    StringRef LinePart = Rest;
    std::tie(LinePart, Rest) = Rest.split(':');
    unsigned B = 0, E = std::numeric_limits<unsigned>::max();
    if (!parseRange(LinePart, B, E, Errs, "line"))
      return false;
    F.HasLineRange = true;
    F.LineBegin = B;
    F.LineEnd = E;
  }

  // Parse optional :col[-col]
  if (!Rest.empty()) {
    unsigned B = 0, E = std::numeric_limits<unsigned>::max();
    if (!parseRange(Rest, B, E, Errs, "column"))
      return false;
    F.HasColumnRange = true;
    F.ColBegin = B;
    F.ColEnd = E;
  }

  Filters.push_back(std::move(F));
  return true;
}

bool SourceLocMatcher::initialize(ArrayRef<std::string> Specs,
                                  raw_ostream &Errs) {
  Filters.clear();

  // Support shorthand syntax where a single file path is followed by multiple
  // comma-separated line (and optional column) specs, e.g.:
  //   file.c:10,20,25-30,40:3-8
  // The command-line parsing splits on commas before we get here, so we may
  // see tokens like {"file.c:10", "20", "25-30", "40:3-8"}. We carry forward
  // the most recently seen file and create separate filters for each line/col
  // token that does not specify a file path.
  SourceLocFilterSpec LastFile;
  bool HaveLastFile = false;

  auto LooksLikeLineOrColSpec = [](StringRef Tok) {
    if (Tok.empty())
      return false;
    // Tokens starting with a digit or ':' (column-only) are treated as
    // additional range specs for the last seen file.
    char C0 = Tok.front();
    return (C0 >= '0' && C0 <= '9') || C0 == ':';
  };

  for (const std::string &S : Specs) {
    StringRef Tok(S);

    // Fast-path: tokens that clearly include a file path (contain a directory
    // separator or a dot before the first ':') start a new file context.
    StringRef PathPart, Rest;
    std::tie(PathPart, Rest) = Tok.split(':');

    bool StartsNewFile = false;
    if (!PathPart.empty()) {
      bool HasDirSep = PathPart.contains('/');
      bool HasDot = PathPart.contains('.');
      bool StartsWithDigit = PathPart.size() && isdigit(PathPart.front());
      // Heuristic: a new file context if it looks like a path (dirsep or dot)
      // and not purely a numeric range start.
      StartsNewFile = (HasDirSep || HasDot) && !StartsWithDigit;
    }

    if (StartsNewFile || !HaveLastFile) {
      // Treat the token as a full spec with file. If it succeeds, remember the
      // file context for possible subsequent shorthand tokens.
      if (!addFilter(Tok, Errs)) {
        Filters.clear();
        return false;
      }
      // Reconstruct LastFile from PathPart to support following tokens.
      if (!PathPart.empty()) {
        LastFile = SourceLocFilterSpec();
        LastFile.HasFile = true;
        LastFile.NormalizedFile = normalizePath(PathPart);
        LastFile.Basename = sys::path::filename(LastFile.NormalizedFile).str();
        LastFile.MatchBasenameOnly =
            (LastFile.NormalizedFile.find('/') == std::string::npos);
        HaveLastFile = true;
      } else {
        HaveLastFile = false;
      }
      continue;
    }

    // If we get here, we have a previous file context and this token is an
    // additional line/column spec (e.g. "20" or "40:3-8").
    if (!LooksLikeLineOrColSpec(Tok)) {
      // Fallback: attempt to parse as a standalone filter.
      if (!addFilter(Tok, Errs)) {
        Filters.clear();
        return false;
      }
      continue;
    }

    // Build a new filter using the last file and the given ranges.
    SourceLocFilterSpec F = LastFile;
    if (Tok.front() == ':') {
      // Column-only spec: ":col[-col]"
      StringRef ColPart = Tok.drop_front();
      unsigned B = 0, E = std::numeric_limits<unsigned>::max();
      if (!parseRange(ColPart, B, E, Errs, "column")) {
        Filters.clear();
        return false;
      }
      F.HasColumnRange = true;
      F.ColBegin = B;
      F.ColEnd = E;
    } else {
      // "line[-line]" or "line[-line]:col[-col]"
      StringRef LinePart = Tok;
      StringRef ColPart;
      std::tie(LinePart, ColPart) = Tok.split(':');
      unsigned LB = 0, LE = std::numeric_limits<unsigned>::max();
      if (!parseRange(LinePart, LB, LE, Errs, "line")) {
        Filters.clear();
        return false;
      }
      F.HasLineRange = true;
      F.LineBegin = LB;
      F.LineEnd = LE;
      if (!ColPart.empty()) {
        unsigned CB = 0, CE = std::numeric_limits<unsigned>::max();
        if (!parseRange(ColPart, CB, CE, Errs, "column")) {
          Filters.clear();
          return false;
        }
        F.HasColumnRange = true;
        F.ColBegin = CB;
        F.ColEnd = CE;
      }
    }

    Filters.push_back(std::move(F));
  }

  return true;
}

bool SourceLocMatcher::matches(StringRef FilePath, unsigned Line,
                               unsigned Column) const {
  if (Filters.empty())
    return true;
  std::string Norm = normalizePath(FilePath);
  StringRef Base = sys::path::filename(Norm);
  for (const SourceLocFilterSpec &F : Filters) {
    if (!matchFile(F, Norm, Base))
      continue;
    if (F.matches(Line, Column))
      return true;
  }
  return false;
}

// ---- Public filtering API ----

static bool matchesDILocationChain(const DILocation *Loc) {
  if (!Loc)
    return false;
  const auto &Matcher = getSourceLocMatcher();
  for (const DILocation *Cur = Loc; Cur != nullptr; Cur = Cur->getInlinedAt()) {
    StringRef File = Cur->getFilename();
    StringRef Dir = Cur->getDirectory();
    SmallString<256> FullPath;
    if (!Dir.empty()) {
      FullPath = Dir;
      sys::path::append(FullPath, File);
    } else {
      FullPath = File;
    }
    unsigned Line = Cur->getLine();
    unsigned Col = Cur->getColumn();
    if (Matcher.matches(FullPath, Line, Col))
      return true;
  }
  return false;
}

bool llvm::isSourceLocationFilteringEnabled() {
  return SourceLocFilterDebugMode && !getSourceLocMatcher().empty();
}
void llvm::setSourceLocationFilteringDebugMode(bool Enabled) {
  SourceLocFilterDebugMode = Enabled;
}

bool SourceLocFilterSpec::matches(unsigned Line, unsigned Column) const {
  // Line/Column of 0 are treated as unknown; require explicit range including 0
  bool LineOK = (LineBegin <= Line && Line <= LineEnd);
  bool ColOK = (ColBegin <= Column && Column <= ColEnd);
  return LineOK && ColOK;
}

bool llvm::instructionMatchesRequestedSourceLocation(const Instruction &I) {
  if (!isSourceLocationFilteringEnabled())
    return true;
  DebugLoc DL = I.getDebugLoc();
  if (!DL)
    return false;
  return matchesDILocationChain(DL.get());
}

bool llvm::functionContainsRequestedSourceLocation(const Function &F) {
  if (!isSourceLocationFilteringEnabled())
    return true;
  for (const BasicBlock &BB : F) {
    for (const Instruction &I : BB) {
      if (instructionMatchesRequestedSourceLocation(I))
        return true;
    }
  }
  return false;
}

bool llvm::moduleContainsRequestedSourceLocation(const Module &M) {
  if (!isSourceLocationFilteringEnabled())
    return true;
  for (const Function &F : M) {
    if (functionContainsRequestedSourceLocation(F))
      return true;
  }
  return false;
}

bool llvm::loopContainsRequestedSourceLocation(const Loop &L) {
  if (!isSourceLocationFilteringEnabled())
    return true;
  for (const BasicBlock *BB : L.blocks())
    for (const Instruction &I : *BB)
      if (instructionMatchesRequestedSourceLocation(I))
        return true;
  return false;
}

/// This is a helper to determine whether to print IR before or
/// after a pass.

bool llvm::shouldPrintBeforeSomePass() {
  return PrintBeforeAll || !PrintBefore.empty();
}

bool llvm::shouldPrintAfterSomePass() {
  return PrintAfterAll || !PrintAfter.empty();
}

static bool shouldPrintBeforeOrAfterPass(StringRef PassID,
                                         ArrayRef<std::string> PassesToPrint) {
  return llvm::is_contained(PassesToPrint, PassID);
}

bool llvm::shouldPrintBeforeAll() { return PrintBeforeAll; }

bool llvm::shouldPrintAfterAll() { return PrintAfterAll; }

bool llvm::shouldPrintBeforePass(StringRef PassID) {
  return PrintBeforeAll || shouldPrintBeforeOrAfterPass(PassID, PrintBefore);
}

bool llvm::shouldPrintAfterPass(StringRef PassID) {
  return PrintAfterAll || shouldPrintBeforeOrAfterPass(PassID, PrintAfter);
}

std::vector<std::string> llvm::printBeforePasses() {
  return std::vector<std::string>(PrintBefore);
}

std::vector<std::string> llvm::printAfterPasses() {
  return std::vector<std::string>(PrintAfter);
}

bool llvm::forcePrintModuleIR() { return PrintModuleScope; }

bool llvm::forcePrintFuncIR() { return LoopPrintFuncScope; }

bool llvm::isPassInPrintList(StringRef PassName) {
  static std::unordered_set<std::string> Set(FilterPasses.begin(),
                                             FilterPasses.end());
  return Set.empty() || Set.count(std::string(PassName));
}

bool llvm::isFilterPassesEmpty() { return FilterPasses.empty(); }

bool llvm::isFunctionInPrintList(StringRef FunctionName) {
  static std::unordered_set<std::string> PrintFuncNames(PrintFuncsList.begin(),
                                                        PrintFuncsList.end());
  return PrintFuncNames.empty() ||
         PrintFuncNames.count(std::string(FunctionName));
}

std::error_code cleanUpTempFilesImpl(ArrayRef<std::string> FileName,
                                     unsigned N) {
  std::error_code RC;
  for (unsigned I = 0; I < N; ++I) {
    std::error_code EC = sys::fs::remove(FileName[I]);
    if (EC)
      RC = EC;
  }
  return RC;
}

std::error_code llvm::prepareTempFiles(SmallVector<int> &FD,
                                       ArrayRef<StringRef> SR,
                                       SmallVector<std::string> &FileName) {
  assert(FD.size() >= SR.size() && FileName.size() == FD.size() &&
         "Unexpected array sizes");
  std::error_code EC;
  unsigned I = 0;
  for (; I < FD.size(); ++I) {
    if (FD[I] == -1) {
      SmallVector<char, 200> SV;
      EC = sys::fs::createTemporaryFile("tmpfile", "txt", FD[I], SV);
      if (EC)
        break;
      FileName[I] = Twine(SV).str();
    }
    if (I < SR.size()) {
      EC = sys::fs::openFileForWrite(FileName[I], FD[I]);
      if (EC)
        break;
      raw_fd_ostream OutStream(FD[I], /*shouldClose=*/true);
      if (FD[I] == -1) {
        EC = make_error_code(errc::io_error);
        break;
      }
      OutStream << SR[I];
    }
  }
  if (EC && I > 0)
    // clean up created temporary files
    cleanUpTempFilesImpl(FileName, I);
  return EC;
}

std::error_code llvm::cleanUpTempFiles(ArrayRef<std::string> FileName) {
  return cleanUpTempFilesImpl(FileName, FileName.size());
}

std::string llvm::doSystemDiff(StringRef Before, StringRef After,
                               StringRef OldLineFormat, StringRef NewLineFormat,
                               StringRef UnchangedLineFormat) {
  // Store the 2 bodies into temporary files and call diff on them
  // to get the body of the node.
  static SmallVector<int> FD{-1, -1, -1};
  SmallVector<StringRef> SR{Before, After};
  static SmallVector<std::string> FileName{"", "", ""};
  if (prepareTempFiles(FD, SR, FileName))
    return "Unable to create temporary file.";

  static ErrorOr<std::string> DiffExe = sys::findProgramByName(DiffBinary);
  if (!DiffExe)
    return "Unable to find diff executable.";

  SmallString<128> OLF, NLF, ULF;
  ("--old-line-format=" + OldLineFormat).toVector(OLF);
  ("--new-line-format=" + NewLineFormat).toVector(NLF);
  ("--unchanged-line-format=" + UnchangedLineFormat).toVector(ULF);

  StringRef Args[] = {DiffBinary, "-w", "-d",        OLF,
                      NLF,        ULF,  FileName[0], FileName[1]};
  std::optional<StringRef> Redirects[] = {std::nullopt, StringRef(FileName[2]),
                                          std::nullopt};
  int Result = sys::ExecuteAndWait(*DiffExe, Args, std::nullopt, Redirects);
  if (Result < 0)
    return "Error executing system diff.";
  std::string Diff;
  auto B = MemoryBuffer::getFile(FileName[2]);
  if (B && *B)
    Diff = (*B)->getBuffer().str();
  else
    return "Unable to read result.";

  if (cleanUpTempFiles(FileName))
    return "Unable to remove temporary file.";

  return Diff;
}
