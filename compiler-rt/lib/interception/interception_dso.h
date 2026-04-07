//===-- interception_dso.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Minimal dynamic library loading helpers shared by compiler-rt runtimes.
// The interface is C-compatible so C and C++ runtimes can both consume it.
//===----------------------------------------------------------------------===//

#ifndef INTERCEPTION_DSO_H
#define INTERCEPTION_DSO_H

#ifdef __cplusplus
extern "C" {
#endif

// Returns non-zero if optional dynamic library loading is available.
int __interception_is_dynamic_loader_available(void);

// Opens a shared library by name. Passing NULL returns a handle for the current
// process image when supported on the platform.
void *__interception_open_library(const char *name);

// Looks up a symbol in a previously opened dynamic library handle.
void *__interception_lookup_symbol(void *handle, const char *symbol);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // INTERCEPTION_DSO_H
