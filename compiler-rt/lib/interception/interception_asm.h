//===-- interception_asm.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interception-local asm/trampoline support.
// Keep this header free of sanitizer_common dependencies so interception can be
// reused outside sanitizers.
//===----------------------------------------------------------------------===//

#ifndef INTERCEPTION_ASM_H
#define INTERCEPTION_ASM_H

#include "interception_defs.h"

#if defined(__clang__) ||                                                      \
    (defined(__GNUC__) && defined(__GCC_HAVE_DWARF2_CFI_ASM))
#define CFI_STARTPROC .cfi_startproc
#define CFI_ENDPROC .cfi_endproc
#else
#define CFI_STARTPROC
#define CFI_ENDPROC
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_BTI_DEFAULT)
#define C_ASM_STARTPROC SANITIZER_STRINGIFY(CFI_STARTPROC) "\nhint #34"
#else
#define C_ASM_STARTPROC SANITIZER_STRINGIFY(CFI_STARTPROC)
#endif
#define C_ASM_ENDPROC SANITIZER_STRINGIFY(CFI_ENDPROC)

#if defined(__x86_64__) || defined(__i386__) || defined(__sparc__)
#define ASM_TAIL_CALL jmp
#elif defined(__arm__) || defined(__aarch64__) || defined(__mips__) ||         \
    defined(__powerpc__) || defined(__loongarch_lp64)
#define ASM_TAIL_CALL b
#elif defined(__s390__)
#define ASM_TAIL_CALL jg
#elif defined(__riscv)
#define ASM_TAIL_CALL tail
#elif defined(__hexagon__)
#define ASM_TAIL_CALL jump
#endif

#if defined(__mips64)
#define C_ASM_TAIL_CALL(t_func, i_func)                                        \
  "lui $t8, %hi(%neg(%gp_rel(" t_func ")))\n"                                  \
  "daddu $t8, $t8, $t9\n"                                                      \
  "daddiu $t8, $t8, %lo(%neg(%gp_rel(" t_func ")))\n"                          \
  "ld $t9, %got_disp(" i_func ")($t8)\n"                                       \
  "jr $t9\n"
#elif defined(__mips__)
#define C_ASM_TAIL_CALL(t_func, i_func)                                        \
  ".set    noreorder\n"                                                        \
  ".cpload $t9\n"                                                              \
  ".set    reorder\n"                                                          \
  "lw $t9, %got(" i_func ")($gp)\n"                                            \
  "jr $t9\n"
#elif defined(ASM_TAIL_CALL)
#define C_ASM_TAIL_CALL(t_func, i_func)                                        \
  SANITIZER_STRINGIFY(ASM_TAIL_CALL) " " i_func
#endif

#if (defined(__ELF__) && defined(__x86_64__)) || defined(__i386__) ||          \
    defined(__riscv)
#define ASM_PREEMPTIBLE_SYM(sym) sym @plt
#else
#define ASM_PREEMPTIBLE_SYM(sym) sym
#endif

#if !defined(__APPLE__)
#if defined(__i386__) || defined(__powerpc__) || defined(__s390__) ||          \
    defined(__sparc__)
#define ASM_INTERCEPTOR_TRAMPOLINE_SUPPORT 0
#else
#define ASM_INTERCEPTOR_TRAMPOLINE_SUPPORT 1
#endif
#else
#define ASM_INTERCEPTOR_TRAMPOLINE_SUPPORT 0
#endif

#endif // INTERCEPTION_ASM_H
