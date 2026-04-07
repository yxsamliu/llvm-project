//===-- interception_defs.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interception-local low-level definitions.
// Keep this header small and free of sanitizer_common dependencies so the
// interception library can be reused outside sanitizers.
//===----------------------------------------------------------------------===//

#ifndef INTERCEPTION_DEFS_H
#define INTERCEPTION_DEFS_H

#include "interception_platform.h"

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if !defined(__has_cpp_attribute)
#define __has_cpp_attribute(x) 0
#endif

#define SANITIZER_STRINGIFY_(S) #S
#define SANITIZER_STRINGIFY(S) SANITIZER_STRINGIFY_(S)

namespace __sanitizer {

#if defined(__UINTPTR_TYPE__)
#if defined(__arm__) && defined(__linux__)
typedef unsigned int uptr;
typedef int sptr;
#else
typedef __UINTPTR_TYPE__ uptr;
typedef __INTPTR_TYPE__ sptr;
#endif
#elif defined(_WIN64)
typedef unsigned long long uptr;
typedef signed long long sptr;
#elif defined(_WIN32)
typedef unsigned int uptr;
typedef signed int sptr;
#else
#error Unsupported compiler, missing __UINTPTR_TYPE__
#endif

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef signed char s8;
typedef signed short s16;
typedef signed int s32;
typedef signed long long s64;

#if SANITIZER_FREEBSD || SANITIZER_NETBSD || SANITIZER_APPLE ||                \
    (SANITIZER_SOLARIS && (defined(_LP64) || _FILE_OFFSET_BITS == 64)) ||      \
    (SANITIZER_LINUX && !SANITIZER_GLIBC && !SANITIZER_ANDROID) ||             \
    (SANITIZER_LINUX && (defined(__x86_64__) || defined(__hexagon__)))
typedef u64 OFF_T;
#else
typedef uptr OFF_T;
#endif
typedef u64 OFF64_T;

#ifdef __SIZE_TYPE__
typedef __SIZE_TYPE__ usize;
#else
typedef uptr usize;
#endif

#if defined(__s390__) && !defined(__s390x__)
typedef long ssize;
#else
typedef sptr ssize;
#endif

} // namespace __sanitizer

#if defined(_MSC_VER)
#define ALIAS(x)
#define FORMAT(f, a)
#define NOINLINE __declspec(noinline)
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define WARN_UNUSED_RESULT
#else
#define ALIAS(x) __attribute__((alias(SANITIZER_STRINGIFY(x))))
#define FORMAT(f, a) __attribute__((format(printf, f, a)))
#define NOINLINE __attribute__((noinline))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define WARN_UNUSED_RESULT __attribute__((warn_unused_result))
#endif

#if !defined(_MSC_VER) || defined(__clang__)
#define UNUSED __attribute__((unused))
#else
#define UNUSED
#endif

#define COMPILER_CHECK(pred) static_assert(pred, "")
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

namespace __interception {
using namespace __sanitizer;
}

#endif // INTERCEPTION_DEFS_H
