//===-- interception_dso.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Optional dynamic library loading helpers reusable by compiler-rt runtimes.
//===----------------------------------------------------------------------===//

#include "interception_dso.h"

#include "interception_defs.h"
#include "interception_platform.h"

#if SANITIZER_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

extern "C" int __interception_is_dynamic_loader_available(void) { return 1; }

extern "C" void *__interception_open_library(const char *name) {
  if (!name)
    return reinterpret_cast<void *>(GetModuleHandleA(nullptr));
  return reinterpret_cast<void *>(LoadLibraryA(name));
}

extern "C" void *__interception_lookup_symbol(void *handle,
                                              const char *symbol) {
  if (!handle)
    return nullptr;
  return reinterpret_cast<void *>(reinterpret_cast<uptr>(
      GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol)));
}

#else
#include <dlfcn.h>

#pragma weak dlopen
#pragma weak dlsym

extern "C" int __interception_is_dynamic_loader_available(void) {
  return dlopen != nullptr && dlsym != nullptr;
}

extern "C" void *__interception_open_library(const char *name) {
  if (!__interception_is_dynamic_loader_available())
    return nullptr;
  return dlopen(name, RTLD_LAZY | RTLD_LOCAL);
}

extern "C" void *__interception_lookup_symbol(void *handle,
                                              const char *symbol) {
  if (!__interception_is_dynamic_loader_available())
    return nullptr;
  return dlsym(handle, symbol);
}

#endif
