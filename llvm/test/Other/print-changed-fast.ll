; RUN: opt -passes=instsimplify -filter-print-funcs=second \
; RUN:   -print-changed=quiet -print-ir-fast -disable-output < %s 2>&1 | FileCheck %s --check-prefix=FAST
; RUN: opt -passes=instsimplify -filter-print-funcs=second \
; RUN:   -print-before=instsimplify -print-after=instsimplify -print-ir-fast \
; RUN:   -disable-output < %s 2>&1 | FileCheck %s --check-prefix=STABLE

declare i32 @opaque(i32)

define i32 @first(i32 %arg) #0 {
  %keep = call i32 @opaque(i32 %arg), !annotation !0
  ret i32 %keep
}

define i32 @second(i32 %arg) #1 {
  %constant = add i32 2, 3, !annotation !0
  %keep = call i32 @opaque(i32 %arg), !annotation !1, !other !3
  %result = add i32 %keep, %constant
  ret i32 %result
}

!0 = !{!"first metadata"}
!1 = !{!2}
!2 = !{!"second metadata"}
!3 = !{!"other metadata"}

attributes #0 = { nounwind }
attributes #1 = { noinline }

; FAST: *** IR Dump After InstSimplifyPass on second ***
; FAST: define i32 @second(i32 %arg) #1 {
; FAST: %keep = call i32 @opaque(i32 %arg)
; FAST-SAME: !annotation ![[ANNOTATION:[0-9]+]], !other ![[OTHER:[0-9]+]]

; STABLE: *** IR Dump Before InstSimplifyPass on second ***
; STABLE: %constant = add i32 2, 3, !annotation !{{[0-9]+}}
; STABLE: %keep = call i32 @opaque(i32 %arg)
; STABLE-SAME: !annotation ![[STABLE_ANNOTATION:[0-9]+]], !other ![[STABLE_OTHER:[0-9]+]]
; STABLE: *** IR Dump After InstSimplifyPass on second ***
; STABLE-NOT: %constant
; STABLE: %keep = call i32 @opaque(i32 %arg)
; STABLE-SAME: !annotation ![[STABLE_ANNOTATION]], !other ![[STABLE_OTHER]]
