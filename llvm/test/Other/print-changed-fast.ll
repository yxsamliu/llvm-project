; RUN: opt -S -passes=instsimplify -filter-print-funcs=second \
; RUN:   -print-changed=quiet -disable-output < %s 2>&1 | FileCheck %s --check-prefix=NORMAL
; RUN: opt -S -passes=instsimplify -filter-print-funcs=second \
; RUN:   -print-changed=fast-quiet -disable-output < %s 2>&1 | FileCheck %s --check-prefix=FAST
; RUN: opt -S -passes=instsimplify -filter-print-funcs=second \
; RUN:   -print-changed=quiet -print-ir-fast -disable-output < %s 2>&1 | FileCheck %s --check-prefix=FAST
; RUN: opt -S -passes=instsimplify -filter-print-funcs=second \
; RUN:   -print-changed=diff-quiet -print-ir-fast -disable-output < %s 2>&1 | FileCheck %s --check-prefix=FAST-DIFF

declare i32 @opaque(i32)

define i32 @first(i32 %arg) #0 {
  %keep = call i32 @opaque(i32 %arg), !annotation !0
  ret i32 %keep
}

define i32 @second(i32 %arg) #1 {
  %keep = call i32 @opaque(i32 %arg), !annotation !1, !other !3
  %constant = add i32 2, 3
  %result = add i32 %keep, %constant
  ret i32 %result
}

!0 = !{!"first metadata"}
!1 = !{!2}
!2 = !{!"second metadata"}
!3 = !{!"other metadata"}

attributes #0 = { nounwind }
attributes #1 = { noinline }

; NORMAL: *** IR Dump After InstSimplifyPass on second ***
; NORMAL: define i32 @second(i32 %arg) #1 {
; NORMAL: %keep = call i32 @opaque(i32 %arg), !annotation !1, !other !3

; FAST: *** IR Dump After InstSimplifyPass on second ***
; FAST: define i32 @second(i32 %arg) #0 {
; FAST: %keep = call i32 @opaque(i32 %arg), !annotation !0, !other !1

; FAST-DIFF: *** IR Dump After InstSimplifyPass on second ***
; FAST-DIFF: %keep = call i32 @opaque(i32 %arg), !annotation !0, !other !1
; FAST-DIFF: -  %constant = add i32 2, 3
; FAST-DIFF: +  %result = add i32 %keep, 5
