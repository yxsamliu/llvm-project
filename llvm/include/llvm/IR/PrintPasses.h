//===- PrintPasses.h - Determining whether/when to print IR ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_PRINTPASSES_H
#define LLVM_IR_PRINTPASSES_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include <limits>
#include <string>
#include <vector>

namespace llvm {

class BasicBlock;
class Function;
class Loop;
class Instruction;
class MachineFunction;
class MachineFunction;
class Module;
class Instruction;

enum class ChangePrinter {
  None,
  Verbose,
  Quiet,
  DiffVerbose,
  DiffQuiet,
  ColourDiffVerbose,
  ColourDiffQuiet,
  DotCfgVerbose,
  DotCfgQuiet
};

extern cl::opt<ChangePrinter> PrintChanged;

// Returns true if printing before/after some pass is enabled, whether all
// passes or a specific pass.
bool shouldPrintBeforeSomePass();
bool shouldPrintAfterSomePass();

// Returns true if we should print before/after a specific pass. The argument
// should be the pass ID, e.g. "instcombine".
bool shouldPrintBeforePass(StringRef PassID);
bool shouldPrintAfterPass(StringRef PassID);

// Returns true if we should print before/after all passes.
bool shouldPrintBeforeAll();
bool shouldPrintAfterAll();

// The list of passes to print before/after, if we only want to print
// before/after specific passes.
std::vector<std::string> printBeforePasses();
std::vector<std::string> printAfterPasses();

// Returns true if we should always print the entire module.
bool forcePrintModuleIR();

// Returns true if we should print the entire function for loop passes.
bool forcePrintFuncIR();

// Return true if -filter-passes is empty or contains the pass name.
bool isPassInPrintList(StringRef PassName);
bool isFilterPassesEmpty();

// Returns true if we should print the function.
bool isFunctionInPrintList(StringRef FunctionName);

struct SourceLocFilterSpec {
  std::string RawFile;
  std::string NormalizedFile;
  std::string Basename;
  bool IsWildcard = false;
  bool MatchBasenameOnly = false;
  bool HasFile = false;
  bool HasLineRange = false;
  bool HasColumnRange = false;
  unsigned LineBegin = 0;
  unsigned LineEnd = std::numeric_limits<unsigned>::max();
  unsigned ColBegin = 0;
  unsigned ColEnd = std::numeric_limits<unsigned>::max();

  bool matches(unsigned Line, unsigned Column) const;
};

bool isSourceLocationFilteringEnabled();
// Enable or disable source location filtering for the current thread's
// debug-printing context. This allows filtering to affect only debugging
// outputs (e.g. -print-before/-print-after) while leaving normal output
// (-o) unaffected.
void setSourceLocationFilteringDebugMode(bool Enabled);
bool instructionMatchesRequestedSourceLocation(const Instruction &I);
bool functionContainsRequestedSourceLocation(const Function &F);
bool moduleContainsRequestedSourceLocation(const Module &M);
bool loopContainsRequestedSourceLocation(const Loop &L);

// Ensure temporary files exist, creating or re-using them.  \p FD contains
// file descriptors (-1 indicates that the file should be created) and
// \p SR contains the corresponding initial content.  \p FileName will have
// the filenames filled in when creating files.  Return first error code (if
// any) and stop.
std::error_code prepareTempFiles(SmallVector<int> &FD, ArrayRef<StringRef> SR,
                                 SmallVector<std::string> &FileName);

// Remove the temporary files in \p FileName.  Typically used in conjunction
// with prepareTempFiles.  Return first error code (if any) and stop..
std::error_code cleanUpTempFiles(ArrayRef<std::string> FileName);

// Perform a system based diff between \p Before and \p After, using \p
// OldLineFormat, \p NewLineFormat, and \p UnchangedLineFormat to control the
// formatting of the output. Return an error message for any failures instead
// of the diff.
std::string doSystemDiff(StringRef Before, StringRef After,
                         StringRef OldLineFormat, StringRef NewLineFormat,
                         StringRef UnchangedLineFormat);

} // namespace llvm

#endif // LLVM_IR_PRINTPASSES_H
