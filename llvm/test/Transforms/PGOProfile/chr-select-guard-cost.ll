; RUN: opt < %s -mtriple=amdgpu9.50-amd-amdhsa -mcpu=gfx950 -passes='require<profile-summary>,function(chr)' -chr-bias-threshold=0.5 -chr-select-guard-cost -S | FileCheck %s --check-prefixes=COMMON,COST
; RUN: opt < %s -mtriple=amdgpu9.50-amd-amdhsa -mcpu=gfx950 -passes='require<profile-summary>,function(chr)' -chr-bias-threshold=0.5 -S | FileCheck %s --check-prefixes=COMMON,ORIGINAL
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -passes='require<profile-summary>,function(chr)' -chr-bias-threshold=0.5 -chr-select-guard-cost -S | FileCheck %s --check-prefixes=COMMON,ORIGINAL

; distinct_conditions: guard cost exceeds profiled select savings.
define half @distinct_conditions(i1 %condition, i1 %second_condition, half %value0, half %value1) !prof !14 {
; COMMON-LABEL: define half @distinct_conditions(
; COST-NOT: br i1
; COST: ret half
; ORIGINAL: entry.split.nonchr:
entry:
  %select0 = select i1 %condition, half %value0, half 0.0, !prof !15
  %select1 = select i1 %second_condition, half %value1, half 0.0, !prof !15
  %sum1 = fadd half %select0, %select1
  ret half %sum1
}

; shared_hot: guard cost is covered by profiled select savings.
define half @shared_hot(i1 %condition, half %value0, half %value1, half %value2, half %value3, half %value4, half %value5, half %value6, half %value7) !prof !14 {
; COMMON-LABEL: define half @shared_hot(
; COMMON: entry.split.nonchr:
entry:
  %select0 = select i1 %condition, half %value0, half 0.0, !prof !15
  %select1 = select i1 %condition, half %value1, half 0.0, !prof !15
  %select2 = select i1 %condition, half %value2, half 0.0, !prof !15
  %select3 = select i1 %condition, half %value3, half 0.0, !prof !15
  %select4 = select i1 %condition, half %value4, half 0.0, !prof !15
  %select5 = select i1 %condition, half %value5, half 0.0, !prof !15
  %select6 = select i1 %condition, half %value6, half 0.0, !prof !15
  %select7 = select i1 %condition, half %value7, half 0.0, !prof !15
  %sum1 = fadd half %select0, %select1
  %sum2 = fadd half %sum1, %select2
  %sum3 = fadd half %sum2, %select3
  %sum4 = fadd half %sum3, %select4
  %sum5 = fadd half %sum4, %select5
  %sum6 = fadd half %sum5, %select6
  %sum7 = fadd half %sum6, %select7
  ret half %sum7
}

; shared_less_hot: guard cost exceeds profiled select savings.
define half @shared_less_hot(i1 %condition, half %value0, half %value1, half %value2, half %value3, half %value4, half %value5, half %value6, half %value7) !prof !14 {
; COMMON-LABEL: define half @shared_less_hot(
; COST-NOT: br i1
; COST: ret half
; ORIGINAL: entry.split.nonchr:
entry:
  %select0 = select i1 %condition, half %value0, half 0.0, !prof !18
  %select1 = select i1 %condition, half %value1, half 0.0, !prof !18
  %select2 = select i1 %condition, half %value2, half 0.0, !prof !18
  %select3 = select i1 %condition, half %value3, half 0.0, !prof !18
  %select4 = select i1 %condition, half %value4, half 0.0, !prof !18
  %select5 = select i1 %condition, half %value5, half 0.0, !prof !18
  %select6 = select i1 %condition, half %value6, half 0.0, !prof !18
  %select7 = select i1 %condition, half %value7, half 0.0, !prof !18
  %sum1 = fadd half %select0, %select1
  %sum2 = fadd half %sum1, %select2
  %sum3 = fadd half %sum2, %select3
  %sum4 = fadd half %sum3, %select4
  %sum5 = fadd half %sum4, %select5
  %sum6 = fadd half %sum5, %select6
  %sum7 = fadd half %sum6, %select7
  ret half %sum7
}

; false_hot: guard cost is covered by profiled select savings.
define half @false_hot(i1 %condition, half %value0, half %value1, half %value2, half %value3, half %value4, half %value5, half %value6, half %value7, half %value8) !prof !14 {
; COMMON-LABEL: define half @false_hot(
; COMMON: entry.split.nonchr:
entry:
  %select0 = select i1 %condition, half 0.0, half %value0, !prof !17
  %select1 = select i1 %condition, half 0.0, half %value1, !prof !17
  %select2 = select i1 %condition, half 0.0, half %value2, !prof !17
  %select3 = select i1 %condition, half 0.0, half %value3, !prof !17
  %select4 = select i1 %condition, half 0.0, half %value4, !prof !17
  %select5 = select i1 %condition, half 0.0, half %value5, !prof !17
  %select6 = select i1 %condition, half 0.0, half %value6, !prof !17
  %select7 = select i1 %condition, half 0.0, half %value7, !prof !17
  %select8 = select i1 %condition, half 0.0, half %value8, !prof !17
  %sum1 = fadd half %select0, %select1
  %sum2 = fadd half %sum1, %select2
  %sum3 = fadd half %sum2, %select3
  %sum4 = fadd half %sum3, %select4
  %sum5 = fadd half %sum4, %select5
  %sum6 = fadd half %sum5, %select6
  %sum7 = fadd half %sum6, %select7
  %sum8 = fadd half %sum7, %select8
  ret half %sum8
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
!15 = !{!"branch_weights", i32 1000, i32 0}
!16 = !{}
!17 = !{!"branch_weights", i32 0, i32 1000}
!18 = !{!"branch_weights", i32 800, i32 200}
