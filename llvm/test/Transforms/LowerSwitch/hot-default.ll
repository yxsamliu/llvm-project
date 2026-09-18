; RUN: opt -S -passes='lower-switch,verify' %s | FileCheck %s

; A dense set needs one range test. The source block still represents the same
; execution event; cold dispatch blocks must not inherit its wave measurement.
; CHECK-LABEL: define i32 @dense(
; CHECK-SAME: !wave.profile [[TABLE:![0-9]+]]
; CHECK: entry:
; CHECK-NEXT: %switch.offset = sub i32 %x, 1
; CHECK-NEXT: %switch.range = icmp ule i32 %switch.offset, 3
; CHECK-NEXT: br i1 %switch.range, label %switch.cold, label %default, !prof [[WEIGHTS:![0-9]+]], !block.uniformity.profile [[UNIFORM:![0-9]+]], !wave.profile.block [[ENTRY:![0-9]+]]
; CHECK: default:
; CHECK-NEXT: %r = phi i32 [ 99, %entry ]
; CHECK: switch.cold:
; CHECK-NEXT: br label {{.*}}, !wave.profile.block [[COLD:![0-9]+]]
define i32 @dense(i32 %x) !uniformity.profile !1 !wave.profile !2 {
entry:
  switch i32 %x, label %default [i32 1, label %a
                                i32 2, label %b
                                i32 3, label %c
                                i32 4, label %d], !prof !0, !block.uniformity.profile !1, !wave.profile.block !3
default:
  %r = phi i32 [99, %entry]
  ret i32 %r, !wave.profile.block !4
a: ret i32 1, !wave.profile.block !5
b: ret i32 2, !wave.profile.block !6
c: ret i32 3, !wave.profile.block !7
d: ret i32 4, !wave.profile.block !8
}

; The range select must suppress a poison shift for default values such as 100.
; CHECK-LABEL: define i32 @sparse(
; CHECK: %switch.offset = sub i32 %x, -3
; CHECK: %switch.range = icmp ule i32 %switch.offset, 9
; CHECK: [[SHIFT:%.*]] = lshr i32 553, %switch.offset
; CHECK-NEXT: [[BIT:%.*]] = trunc i32 [[SHIFT]] to i1
; CHECK-NEXT: %switch.member = select i1 %switch.range, i1 [[BIT]], i1 false
; CHECK-NEXT: br i1 %switch.member, label %switch.cold, label %default, !prof [[WEIGHTS]]
define i32 @sparse(i32 %x) {
entry:
  switch i32 %x, label %default [i32 -3, label %a
                                i32 0, label %b
                                i32 2, label %c
                                i32 6, label %d], !prof !0
default: ret i32 99
a: ret i32 1
b: ret i32 2
c: ret i32 3
d: ret i32 4
}

; CHECK-LABEL: define i32 @wide_mask(
; CHECK: %switch.range = icmp ule i64 %switch.offset, 63
; CHECK: lshr i64 -9223372036854775797, %switch.offset
define i32 @wide_mask(i64 %x) {
entry:
  switch i64 %x, label %default [i64 -32, label %a
                                i64 -31, label %b
                                i64 -29, label %c
                                i64 31, label %d], !prof !0
default: ret i32 99
a: ret i32 1
b: ret i32 2
c: ret i32 3
d: ret i32 4
}

; No profile, or observed case traffic, retains ordinary lowering.
; CHECK-LABEL: define i32 @unprofiled(
; CHECK-NOT: switch.range
; CHECK-NOT: switch.member
; CHECK: ret i32 99
define i32 @unprofiled(i32 %x) {
entry:
  switch i32 %x, label %default [i32 1, label %a
                                i32 2, label %b
                                i32 3, label %c
                                i32 4, label %d]
default: ret i32 99
a: ret i32 1
b: ret i32 2
c: ret i32 3
d: ret i32 4
}

; CHECK-LABEL: define i32 @case_traffic(
; CHECK-NOT: switch.range
; CHECK: ret i32 99
define i32 @case_traffic(i32 %x) {
entry:
  switch i32 %x, label %default [i32 1, label %a
                                i32 2, label %b
                                i32 3, label %c
                                i32 4, label %d], !prof !9
default: ret i32 99
a: ret i32 1
b: ret i32 2
c: ret i32 3
d: ret i32 4
}

; The bounded membership test does not handle a wider sparse range.
; CHECK-LABEL: define i32 @too_wide(
; CHECK-NOT: switch.range
; CHECK: ret i32 99
define i32 @too_wide(i32 %x) {
entry:
  switch i32 %x, label %default [i32 0, label %a
                                i32 1, label %b
                                i32 2, label %c
                                i32 64, label %d], !prof !0
default: ret i32 99
a: ret i32 1
b: ret i32 2
c: ret i32 3
d: ret i32 4
}

; CHECK-DAG: [[WEIGHTS]] = !{!"branch_weights", !"expected", i32 0, i32 1000}
; CHECK-DAG: [[UNIFORM]] = !{}
; CHECK-DAG: [[TABLE]] = !{i64 2, i64 543080270900059739, i64 10, i64 10, i64 0, i64 0, i64 0, i64 0, {{.*}}}
; CHECK-DAG: [[ENTRY]] = !{i64 2, i64 543080270900059739, i64 0, i64 1, i64 6, i64 1}
; CHECK-DAG: [[COLD]] = !{i64 2, i64 543080270900059739, i64 6, i64 0, {{.*}}}
!0 = !{!"branch_weights", !"expected", i32 1000, i32 0, i32 0, i32 0, i32 0}
!1 = !{}
!2 = !{i64 2, i64 543080270900059739, i64 10, i64 10, i64 0, i64 0, i64 0, i64 0}
!3 = !{i64 2, i64 543080270900059739, i64 0, i64 1, i64 1, i64 2, i64 3, i64 4, i64 5}
!4 = !{i64 2, i64 543080270900059739, i64 1, i64 1}
!5 = !{i64 2, i64 543080270900059739, i64 2, i64 1}
!6 = !{i64 2, i64 543080270900059739, i64 3, i64 1}
!7 = !{i64 2, i64 543080270900059739, i64 4, i64 1}
!8 = !{i64 2, i64 543080270900059739, i64 5, i64 1}
!9 = !{!"branch_weights", i32 1000, i32 1, i32 0, i32 0, i32 0}
