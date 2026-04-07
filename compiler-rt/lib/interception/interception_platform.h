//===-- interception_platform.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Minimal platform detection for interception support.
// Keep this header independent of sanitizer_common so interception can be
// reused by non-sanitizer runtimes.
//===----------------------------------------------------------------------===//

#ifndef INTERCEPTION_PLATFORM_H
#define INTERCEPTION_PLATFORM_H

#ifndef __has_include
#define __has_include(x) 0
#endif

#if !defined(__linux__) && !defined(__FreeBSD__) && !defined(__NetBSD__) &&    \
    !defined(__APPLE__) && !defined(_WIN32) && !defined(__Fuchsia__) &&        \
    !(defined(__sun__) && defined(__svr4__)) && !defined(__HAIKU__) &&         \
    !defined(_AIX)
#error "This operating system is not supported"
#endif

#if __has_include(<features.h>) && !defined(__ANDROID__)
#include <features.h>
#endif

#if defined(__linux__)
#define SANITIZER_LINUX 1
#else
#define SANITIZER_LINUX 0
#endif

#if defined(__GLIBC__)
#define SANITIZER_GLIBC 1
#else
#define SANITIZER_GLIBC 0
#endif

#if defined(__FreeBSD__)
#define SANITIZER_FREEBSD 1
#else
#define SANITIZER_FREEBSD 0
#endif

#if defined(__NetBSD__)
#define SANITIZER_NETBSD 1
#else
#define SANITIZER_NETBSD 0
#endif

#if defined(__sun__) && defined(__svr4__)
#define SANITIZER_SOLARIS 1
#else
#define SANITIZER_SOLARIS 0
#endif

#if defined(__HAIKU__)
#define SANITIZER_HAIKU 1
#else
#define SANITIZER_HAIKU 0
#endif

#if defined(__APPLE__)
#define SANITIZER_APPLE 1
#include <TargetConditionals.h>
#else
#define SANITIZER_APPLE 0
#endif

#if defined(_WIN32)
#define SANITIZER_WINDOWS 1
#else
#define SANITIZER_WINDOWS 0
#endif

#if defined(_WIN64)
#define SANITIZER_WINDOWS64 1
#else
#define SANITIZER_WINDOWS64 0
#endif

#if defined(__ANDROID__)
#define SANITIZER_ANDROID 1
#else
#define SANITIZER_ANDROID 0
#endif

#if defined(__Fuchsia__)
#define SANITIZER_FUCHSIA 1
#else
#define SANITIZER_FUCHSIA 0
#endif

#if defined(_AIX)
#define SANITIZER_AIX 1
#else
#define SANITIZER_AIX 0
#endif

#if __LP64__ || defined(_WIN64)
#define SANITIZER_WORDSIZE 64
#else
#define SANITIZER_WORDSIZE 32
#endif

#if SANITIZER_WORDSIZE == 64
#define FIRST_32_SECOND_64(a, b) (b)
#else
#define FIRST_32_SECOND_64(a, b) (a)
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define SANITIZER_ARM64 1
#else
#define SANITIZER_ARM64 0
#endif

#if SANITIZER_WINDOWS64 && SANITIZER_ARM64
#define SANITIZER_WINDOWS_ARM64 1
#define SANITIZER_WINDOWS_x64 0
#elif SANITIZER_WINDOWS64 && !SANITIZER_ARM64
#define SANITIZER_WINDOWS_ARM64 0
#define SANITIZER_WINDOWS_x64 1
#else
#define SANITIZER_WINDOWS_ARM64 0
#define SANITIZER_WINDOWS_x64 0
#endif

#endif // INTERCEPTION_PLATFORM_H
