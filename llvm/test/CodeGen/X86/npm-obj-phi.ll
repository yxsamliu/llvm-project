; REQUIRES: object-emission
; Default llc (legacy codegen PM): object emission must include code and a global
; symbol for @f (ir-tracker / normal workflows should omit -enable-new-pm for .o).
;
; NewPM splits AsmPrinter across begin / per-function / end passes, each with its
; own MCStreamer. A stale MCSection state across those streamers used to crash
; object mode (SIGSEGV in MCStreamer::addFragment); MCObjectStreamer teardown
; now resets MCSection/MCSymbol emission state so llc does not crash.
;
; FIXME: NewPM -filetype=obj still produces an incomplete ELF (e.g. missing a
; real .text section with code) because intermediate streamers are destroyed
; without Finish() and only the end pass runs doFinalization on a fresh streamer.
; A proper fix is to share one MCStreamer across all three passes.
;
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t.o %s
; RUN: llvm-nm %t.o | FileCheck %s --check-prefix=NM
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-new-pm -filetype=obj -o %t.npm.o %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-new-pm -filetype=asm -o - %s | FileCheck %s
; RUN: rm -f %t.npm.db
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-new-pm -ir-tracker-database=%t.npm.db -filetype=obj -o %t.dbobj.o %s 2>&1 | FileCheck %s --check-prefix=DBOBJ
; RUN: test ! -f %t.npm.db

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @f(i1 %c, i32 %a, i32 %b) {
entry:
  br i1 %c, label %t, label %f

t:
  br label %m

f:
  br label %m

m:
  %x = phi i32 [ %a, %t ], [ %b, %f ]
  ret i32 %x
}

; NM: {{.*}} T f

; DBOBJ: note: ignoring -ir-tracker-database for object emission

; CHECK-LABEL: f:
; CHECK: retq
