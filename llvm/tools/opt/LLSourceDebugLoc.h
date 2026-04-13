//===- LLSourceDebugLoc.h - Attach .ll file line numbers as !dbg ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OPT_LLSOURCEDEBUG_H
#define LLVM_TOOLS_OPT_LLSOURCEDEBUG_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class Module;

/// For each instruction in \p M that does not already have a debug location,
/// assign a synthetic \c DILocation whose file is derived from \p LLFilePath.
///
/// For textual LLVM assembly input (\c .ll), line numbers are inferred by
/// scanning \p LLFilePath positionally: instructions are matched in
/// declaration order (function, basic block, instruction within basic block)
/// to lines detected in the text.
///
/// For bitcode input (\c .bc) without existing debug info, line numbers fall
/// back to synthetic ordinal IDs derived from the parsed IR traversal order.
///
/// If the module already has a \c !llvm.dbg.cu metadata node (i.e. real debug
/// info is present), the function is a no-op unless \p ForceOverwrite is true.
///
void applyLLSourceDebugLoc(Module &M, StringRef LLFilePath,
                           bool ForceOverwrite = false);

} // namespace llvm

#endif
