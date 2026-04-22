//===- llvm/IR/FunctionInstructionPrinter.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPIKE for measurement only -- do not upstream as is.
//
// FunctionInstructionPrinter amortizes the AssemblyWriter setup cost across
// many printInstruction calls. The standard Instruction::print(OS, MST) path
// constructs a fresh AssemblyWriter for every call (~32% of execution on
// hot-path consumers like the IR tracker, per callgrind). This wrapper lets a
// caller pay that cost once per function and reuse the writer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_FUNCTIONINSTRUCTIONPRINTER_H
#define LLVM_IR_FUNCTIONINSTRUCTIONPRINTER_H

#include "llvm/Support/Compiler.h"
#include <memory>

namespace llvm {

class Function;
class Instruction;
class Module;
class ModuleSlotTracker;
class raw_ostream;

class LLVM_ABI FunctionInstructionPrinter {
  struct Impl;
  std::unique_ptr<Impl> P;

public:
  /// Build a writer for ``F``. ``MST`` must already cover ``F``'s parent
  /// module; this constructor calls ``MST.incorporateFunction(F)``.
  FunctionInstructionPrinter(ModuleSlotTracker &MST, const Function &F);
  ~FunctionInstructionPrinter();

  FunctionInstructionPrinter(const FunctionInstructionPrinter &) = delete;
  FunctionInstructionPrinter &
  operator=(const FunctionInstructionPrinter &) = delete;

  /// Print ``I`` into ``OS`` using the cached AssemblyWriter. Many calls
  /// against the same ``OS`` or different ``OS`` are valid; the internal
  /// writer's buffer is cleared between calls.
  void printInstruction(raw_ostream &OS, const Instruction &I);
};

} // namespace llvm

#endif // LLVM_IR_FUNCTIONINSTRUCTIONPRINTER_H
