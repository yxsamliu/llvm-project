; RUN: opt -S -passes=simplifycfg -mtriple=amdgcn-amd-amdhsa < %s | FileCheck %s --check-prefixes=CHECK,KEEP
; RUN: opt -S -passes=simplifycfg -mtriple=amdgcn-amd-amdhsa -simplifycfg-branch-unanimity-prototype < %s | FileCheck %s --check-prefixes=CHECK,CHANGE
; RUN: opt -S -passes=simplifycfg -mtriple=x86_64-unknown-linux-gnu -simplifycfg-branch-unanimity-prototype < %s | FileCheck %s --check-prefixes=CHECK,KEEP
;
; Models an inlined conditional correction with a 127:1 lane profile. Half of
; wave visits split, so the rare lane path still executes on half of the waves.
; Profile evidence affects profitability only. It is not a uniformity proof.

; CHECK-LABEL: define i32 @split(
; KEEP: br i1
; CHANGE-NOT: br i1
; CHANGE: select i1
define i32 @split(i1 noundef %c, i32 %x) {
entry:
  br i1 %c, label %end, label %then, !prof !0, !branch.unanimity.prototype !1
then:
  %y = mul i32 %x, 1664525
  br label %end
end:
  %r = phi i32 [ %x, %entry ], [ %y, %then ]
  ret i32 %r
}

; CHECK-LABEL: define i32 @unanimous(
; CHECK: br i1
define i32 @unanimous(i1 noundef %c, i32 %x) {
entry:
  br i1 %c, label %end, label %then, !prof !0, !branch.unanimity.prototype !2
then:
  %y = mul i32 %x, 1664525
  br label %end
end:
  %r = phi i32 [ %x, %entry ], [ %y, %then ]
  ret i32 %r
}

; CHECK-LABEL: define i32 @sparse(
; CHECK: br i1
define i32 @sparse(i1 noundef %c, i32 %x) {
entry:
  br i1 %c, label %end, label %then, !prof !0, !branch.unanimity.prototype !3
then:
  %y = mul i32 %x, 1664525
  br label %end
end:
  %r = phi i32 [ %x, %entry ], [ %y, %then ]
  ret i32 %r
}

; CHECK-LABEL: define i32 @zero(
; CHECK: br i1
define i32 @zero(i1 noundef %c, i32 %x) {
entry:
  br i1 %c, label %end, label %then, !prof !0, !branch.unanimity.prototype !4
then:
  %y = mul i32 %x, 1664525
  br label %end
end:
  %r = phi i32 [ %x, %entry ], [ %y, %then ]
  ret i32 %r
}

; CHECK-LABEL: define i32 @few_splits(
; CHECK: br i1
define i32 @few_splits(i1 noundef %c, i32 %x) {
entry:
  br i1 %c, label %end, label %then, !prof !0, !branch.unanimity.prototype !5
then:
  %y = mul i32 %x, 1664525
  br label %end
end:
  %r = phi i32 [ %x, %entry ], [ %y, %then ]
  ret i32 %r
}

; CHECK-LABEL: define i32 @boundary(
; CHECK: br i1
define i32 @boundary(i1 noundef %c, i32 %x) {
entry:
  br i1 %c, label %end, label %then, !prof !0, !branch.unanimity.prototype !6
then:
  %y = mul i32 %x, 1664525
  br label %end
end:
  %r = phi i32 [ %x, %entry ], [ %y, %then ]
  ret i32 %r
}

; CHECK-LABEL: define i32 @invalid_order(
; CHECK: br i1
define i32 @invalid_order(i1 noundef %c, i32 %x) {
entry:
  br i1 %c, label %end, label %then, !prof !0, !branch.unanimity.prototype !7
then:
  %y = mul i32 %x, 1664525
  br label %end
end:
  %r = phi i32 [ %x, %entry ], [ %y, %then ]
  ret i32 %r
}

; CHECK-LABEL: define i32 @invalid_width(
; CHECK: br i1
define i32 @invalid_width(i1 noundef %c, i32 %x) {
entry:
  br i1 %c, label %end, label %then, !prof !0, !branch.unanimity.prototype !8
then:
  %y = mul i32 %x, 1664525
  br label %end
end:
  %r = phi i32 [ %x, %entry ], [ %y, %then ]
  ret i32 %r
}

; CHECK-LABEL: define i32 @invalid_arity(
; CHECK: br i1
define i32 @invalid_arity(i1 noundef %c, i32 %x) {
entry:
  br i1 %c, label %end, label %then, !prof !0, !branch.unanimity.prototype !9
then:
  %y = mul i32 %x, 1664525
  br label %end
end:
  %r = phi i32 [ %x, %entry ], [ %y, %then ]
  ret i32 %r
}

; CHECK-LABEL: define i32 @large(
; KEEP: br i1
; CHANGE-NOT: br i1
; CHANGE: select i1
define i32 @large(i1 noundef %c, i32 %x) {
entry:
  br i1 %c, label %end, label %then, !prof !0, !branch.unanimity.prototype !10
then:
  %y = mul i32 %x, 1664525
  br label %end
end:
  %r = phi i32 [ %x, %entry ], [ %y, %then ]
  ret i32 %r
}

; CHECK-LABEL: define i32 @missing(
; CHECK: br i1
define i32 @missing(i1 noundef %c, i32 %x) {
entry:
  br i1 %c, label %end, label %then, !prof !0
then:
  %y = mul i32 %x, 1664525
  br label %end
end:
  %r = phi i32 [ %x, %entry ], [ %y, %then ]
  ret i32 %r
}

; CHECK-LABEL: define i32 @unsafe_load(
; CHECK: br i1
; CHECK: load volatile
; CHECK: ret i32
define i32 @unsafe_load(i1 noundef %c, ptr %p) {
entry:
  br i1 %c, label %end, label %then, !prof !0, !branch.unanimity.prototype !1
then:
  %y = load volatile i32, ptr %p
  br label %end
end:
  %r = phi i32 [ 0, %entry ], [ %y, %then ]
  ret i32 %r
}

; CHECK-LABEL: define i32 @convergent_call(
; CHECK: br i1
; CHECK: call i32 @convergent
; CHECK: ret i32
define i32 @convergent_call(i1 noundef %c, i32 %x) {
entry:
  br i1 %c, label %end, label %then, !prof !0, !branch.unanimity.prototype !1
then:
  %y = call i32 @convergent(i32 %x)
  br label %end
end:
  %r = phi i32 [ %x, %entry ], [ %y, %then ]
  ret i32 %r
}
declare i32 @convergent(i32) convergent

!0 = !{!"branch_weights", i32 127, i32 1}
!1 = !{i64 10000, i64 5000}
!2 = !{i64 10000, i64 10000}
!3 = !{i64 99, i64 0}
!4 = !{i64 0, i64 0}
!5 = !{i64 10000, i64 9999}
!6 = !{i64 10000, i64 9900}
!7 = !{i64 100, i64 101}
!8 = !{i32 10000, i32 5000}
!9 = !{i64 10000}
!10 = !{i64 -1, i64 0}
