; RUN: opt -S -passes=instsimplify -filter-print-funcs=second \
; RUN:   -print-changed=quiet -disable-output < %s 2>&1 | FileCheck %s --check-prefix=NORMAL
; RUN: opt -S -passes=instsimplify -filter-print-funcs=second \
; RUN:   -print-changed=fast-quiet -disable-output < %s 2>&1 | FileCheck %s --check-prefix=FAST

declare i32 @opaque(i32)

define i32 @first(i32 %arg) {
  %keep = call i32 @opaque(i32 %arg), !annotation !0
  ret i32 %keep
}

define i32 @second(i32 %arg) {
  %keep = call i32 @opaque(i32 %arg), !annotation !1
  %constant = add i32 2, 3
  %result = add i32 %keep, %constant
  ret i32 %result
}

!0 = !{!"first metadata"}
!1 = !{!"second metadata"}

; NORMAL: *** IR Dump After InstSimplifyPass on second ***
; NORMAL: %keep = call i32 @opaque(i32 %arg), !annotation !1

; FAST: *** IR Dump After InstSimplifyPass on second ***
; FAST: %keep = call i32 @opaque(i32 %arg), !annotation !0
