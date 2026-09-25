; RUN: opt -S -passes='structurizecfg,verify' %s | FileCheck %s --implicit-check-not=branch.uniformity.profile --implicit-check-not=branch.unanimity.prototype

; Inverting the decision in the same block preserves its uniformity and swaps
; its weights, including the llvm.expect origin, and its direct vote pair. The
; PHI-driven Flow decision combines different incoming execution populations
; and receives none of these hints.
; CHECK-LABEL: define void @diamond(
; CHECK: entry:
; CHECK: br i1 {{.*}}, label %right, label %Flow, !prof [[WEIGHTS:![0-9]+]], !branch.uniformity.profile [[UNIFORM:![0-9]+]], !branch.unanimity.prototype [[VOTES:![0-9]+]]
; CHECK: Flow:
; CHECK: br i1 {{.*}}, label %left, label %exit{{$}}
define void @diamond(i1 %c, ptr %p) !uniformity.profile !1 {
entry:
  br i1 %c, label %left, label %right, !prof !0, !branch.uniformity.profile !1, !branch.unanimity.prototype !2
left:
  store volatile i32 1, ptr %p
  br label %exit
right:
  store volatile i32 2, ptr %p
  br label %exit
exit:
  ret void
}

!0 = !{!"branch_weights", !"expected", i32 90, i32 10}
!1 = !{}
!2 = !{i64 100, i64 100}
; CHECK-DAG: [[WEIGHTS]] = !{!"branch_weights", !"expected", i32 10, i32 90}
; CHECK-DAG: [[UNIFORM]] = !{}
; CHECK-DAG: [[VOTES]] = !{i64 100, i64 100}
