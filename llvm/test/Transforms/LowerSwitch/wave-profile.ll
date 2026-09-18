; RUN: opt -S -passes='lower-switch,verify' %s | FileCheck %s --implicit-check-not=block.uniformity.profile

; Keep the original invocation anchor even though original entry ID zero is
; absent. A synthetic comparison receives an identity but no measured count.
define void @if_else(i32 %v, ptr %out) !uniformity.profile !5 !wave.profile !0 {
; CHECK-LABEL: define void @if_else(
; CHECK-SAME: !wave.profile [[PROFILE:![0-9]+]]
entry:
; CHECK: entry:
; CHECK: br label %LeafBlock, !block.uniformity.profile {{![0-9]+}}, !wave.profile.block [[ENTRY:![0-9]+]]
  switch i32 %v, label %right [i32 0, label %left], !wave.profile.block !1, !block.uniformity.profile !5
; CHECK: LeafBlock:
; CHECK: br i1 %SwitchLeaf, label %left, label %right, !wave.profile.block [[LEAF:![0-9]+]]
left:
  store volatile i32 1, ptr %out
  br label %exit, !wave.profile.block !2
right:
  store volatile i32 2, ptr %out
  br label %exit, !wave.profile.block !3
exit:
  ret void, !wave.profile.block !4
}
!0 = !{i64 2, i64 2685589004101179296, i64 11, i64 10, i64 7, i64 3, i64 10}
!1 = !{i64 2, i64 2685589004101179296, i64 1, i64 1, i64 3, i64 2}
!2 = !{i64 2, i64 2685589004101179296, i64 2, i64 1, i64 4}
!3 = !{i64 2, i64 2685589004101179296, i64 3, i64 1, i64 4}
!4 = !{i64 2, i64 2685589004101179296, i64 4, i64 1}
!5 = !{}
; CHECK-DAG: [[PROFILE]] = !{i64 2, i64 2685589004101179296, i64 11, i64 10, i64 7, i64 3, i64 10, i64 0}
; CHECK-DAG: [[ENTRY]] = !{i64 2, i64 2685589004101179296, i64 1, i64 1, i64 5}
; CHECK-DAG: [[LEAF]] = !{i64 2, i64 2685589004101179296, i64 5, i64 0, i64 2, i64 3}
