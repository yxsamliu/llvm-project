; RUN: opt -S -passes=no-op-module < %s | FileCheck %s --check-prefix=NORMAL
; RUN: opt -S -passes=no-op-module -print-ir-fast < %s > %t
; RUN: FileCheck %s --check-prefix=FAST-MODULE < %t
; RUN: opt -disable-output < %t
; RUN: opt -disable-output -passes=print -print-ir-fast < %s 2>&1 | FileCheck %s --check-prefix=FAST-MODULE
; RUN: opt -disable-output -passes='function(print)' -filter-print-funcs=second \
; RUN:   -print-ir-fast < %s 2>&1 | FileCheck %s --check-prefix=FAST-FUNCTION
; RUN: opt -disable-output -passes='function(no-op-function)' \
; RUN:   -print-before=no-op-function -filter-print-funcs=second \
; RUN:   -print-ir-fast < %s 2>&1 | FileCheck %s --check-prefix=FAST-FUNCTION
; RUN: opt -disable-output -passes='function(no-op-function)' -print-after-all \
; RUN:   -filter-print-funcs=second -print-ir-fast < %s 2>&1 | FileCheck %s --check-prefix=FAST-FUNCTION
; RUN: opt -S -passes=no-op-module < %S/Inputs/print-ir-fast-labels.ll > %t.normal
; RUN: opt -S -passes=no-op-module -print-ir-fast \
; RUN:   < %S/Inputs/print-ir-fast-labels.ll > %t.fast
; RUN: diff %t.normal %t.fast

define void @first() {
  ret void, !annotation !1
}

define void @second() {
  ret void, !annotation !3
}

!named = !{!0}
!0 = !{!"named metadata"}
!1 = !{!2}
!2 = !{!"first metadata"}
!3 = !{!4}
!4 = !{!"second metadata"}

; NORMAL: ret void, !annotation !1
; NORMAL: ret void, !annotation !3
; NORMAL: !named = !{!0}

; FAST-MODULE: ret void, !annotation !0
; FAST-MODULE: ret void, !annotation !2
; FAST-MODULE: !named = !{!4}
; FAST-MODULE: !0 = !{!1}
; FAST-MODULE: !1 = !{!"first metadata"}
; FAST-MODULE: !2 = !{!3}
; FAST-MODULE: !3 = !{!"second metadata"}
; FAST-MODULE: !4 = !{!"named metadata"}

; FAST-FUNCTION: define void @second() {
; FAST-FUNCTION: ret void, !annotation !0
