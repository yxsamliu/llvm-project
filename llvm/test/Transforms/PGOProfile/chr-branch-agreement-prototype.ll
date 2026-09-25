; RUN: opt < %s -passes='require<profile-summary>,function(chr,instcombine,simplifycfg)' -S | FileCheck %s --check-prefix=OFF
; RUN: opt < %s -passes='require<profile-summary>,function(chr,instcombine,simplifycfg)' -chr-use-branch-agreement-prototype -S | FileCheck %s --check-prefix=ON
; RUN: opt < %s -passes='require<profile-summary>,function(chr,instcombine,simplifycfg)' -chr-use-branch-agreement-prototype -chr-branch-agreement-min-samples=1 -S | FileCheck %s --check-prefix=ONE
; RUN: opt < %s -passes='require<profile-summary>,function(chr,instcombine,simplifycfg)' -chr-use-branch-agreement-prototype -chr-branch-agreement-min-percent=99 -S | FileCheck %s --check-prefix=NINETY_NINE

declare void @foo()

; Direct votes can admit a scope whose branches lack the older full-wave hint.
define void @vote_only(ptr %ptr) !prof !14 !uniformity.profile !16 {
; OFF-LABEL: define void @vote_only(
; OFF-NOT: split
; OFF: ret void
; ON-LABEL: define void @vote_only(
; ON: entry.split.nonchr:
; ON-NOT: !branch.agreement.prototype
; ON: ret void
entry:
  %value = load i32, ptr %ptr
  %bit0 = and i32 %value, 1
  %cond0 = icmp eq i32 %bit0, 0
  br i1 %cond0, label %bb1, label %bb0, !prof !15, !branch.agreement.prototype !17
bb0:
  call void @foo()
  br label %bb1
bb1:
  %bit1 = and i32 %value, 2
  %cond1 = icmp eq i32 %bit1, 0
  br i1 %cond1, label %exit, label %bb2, !prof !15, !branch.agreement.prototype !17
bb2:
  call void @foo()
  br label %exit
exit:
  ret void
}

; An observed split takes precedence over the older positive hint.
define void @conflicting(ptr %ptr) !prof !14 !uniformity.profile !16 {
; OFF-LABEL: define void @conflicting(
; OFF: entry.split.nonchr:
; ON-LABEL: define void @conflicting(
; ON-NOT: split
; ON: ret void
entry:
  %value = load i32, ptr %ptr
  %bit0 = and i32 %value, 1
  %cond0 = icmp eq i32 %bit0, 0
  br i1 %cond0, label %bb1, label %bb0, !prof !15, !branch.uniformity.profile !16, !branch.agreement.prototype !17
bb0:
  call void @foo()
  br label %bb1
bb1:
  %bit1 = and i32 %value, 2
  %cond1 = icmp eq i32 %bit1, 0
  br i1 %cond1, label %exit, label %bb2, !prof !15, !branch.uniformity.profile !16, !branch.agreement.prototype !18
bb2:
  call void @foo()
  br label %exit
exit:
  ret void
}

; One unanimous visit is insufficient evidence by default.
define void @too_few(ptr %ptr) !prof !14 !uniformity.profile !16 {
; ON-LABEL: define void @too_few(
; ON-NOT: split
; ON: ret void
; ONE-LABEL: define void @too_few(
; ONE: entry.split.nonchr:
entry:
  %value = load i32, ptr %ptr
  %bit0 = and i32 %value, 1
  %cond0 = icmp eq i32 %bit0, 0
  br i1 %cond0, label %bb1, label %bb0, !prof !15, !branch.agreement.prototype !19
bb0:
  call void @foo()
  br label %bb1
bb1:
  %bit1 = and i32 %value, 2
  %cond1 = icmp eq i32 %bit1, 0
  br i1 %cond1, label %exit, label %bb2, !prof !15, !branch.agreement.prototype !19
bb2:
  call void @foo()
  br label %exit
exit:
  ret void
}

; Allowing one observed split per 100 visits needs an explicit threshold.
define void @near_unanimous(ptr %ptr) !prof !14 !uniformity.profile !16 {
; ON-LABEL: define void @near_unanimous(
; ON-NOT: split
; ON: ret void
; NINETY_NINE-LABEL: define void @near_unanimous(
; NINETY_NINE: entry.split.nonchr:
entry:
  %value = load i32, ptr %ptr
  %bit0 = and i32 %value, 1
  %cond0 = icmp eq i32 %bit0, 0
  br i1 %cond0, label %bb1, label %bb0, !prof !15, !branch.agreement.prototype !17
bb0:
  call void @foo()
  br label %bb1
bb1:
  %bit1 = and i32 %value, 2
  %cond1 = icmp eq i32 %bit1, 0
  br i1 %cond1, label %exit, label %bb2, !prof !15, !branch.agreement.prototype !20
bb2:
  call void @foo()
  br label %exit
exit:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 1}
!8 = !{!"NumFunctions", i64 1}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11}
!11 = !{i32 999999, i64 1, i32 1}
!14 = !{!"function_entry_count", i64 100}
!15 = !{!"branch_weights", i32 0, i32 1}
!16 = !{}
!17 = !{i64 100, i64 100}
!18 = !{i64 100, i64 0}
!19 = !{i64 1, i64 1}
!20 = !{i64 100, i64 99}
