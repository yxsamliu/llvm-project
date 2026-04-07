//===-- interception_type_traits.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Minimal type traits for interception tests and support code.
//===----------------------------------------------------------------------===//

#ifndef INTERCEPTION_TYPE_TRAITS_H
#define INTERCEPTION_TYPE_TRAITS_H

#include "interception_defs.h"

namespace __sanitizer {

struct true_type {
  static const bool value = true;
};

struct false_type {
  static const bool value = false;
};

template <class T, class U> struct is_same : public false_type {};

template <class T> struct is_same<T, T> : public true_type {};

} // namespace __sanitizer

#endif // INTERCEPTION_TYPE_TRAITS_H
