//===- DebugLocIndex.h - SQLite debug-loc index for opt -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OPT_DEBUGLOCINDEX_H
#define LLVM_TOOLS_OPT_DEBUGLOCINDEX_H

namespace llvm {

class PassInstrumentationCallbacks;

/// When \c -debug-loc-index-database is set, register callbacks that record
/// instructions with \c !dbg locations after each pass.
void registerDebugLocIndexCallbacks(PassInstrumentationCallbacks &PIC);

} // namespace llvm

#endif
