; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -O0 -stop-after=finalize-isel -o - %s | \
; RUN:   llc -mtriple=amdgcn-amd-amdhsa -passes='print<block-uniformity-profile>' -x mir -filetype=null 2>&1 | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -O0 -stop-after=finalize-isel -o - %s | \
; RUN:   llc -mtriple=amdgcn-amd-amdhsa -spill-use-branch-agreement-prototype -passes='print<block-uniformity-profile>' -x mir -filetype=null 2>&1 | FileCheck %s --check-prefix=ON
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -O3 -report-profiled-spill -filetype=null %s 2>&1 | FileCheck %s --check-prefix=SPILL-OFF
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -O3 -report-profiled-spill -spill-use-branch-agreement-prototype -filetype=null %s 2>&1 | FileCheck %s --check-prefix=SPILL-ON

; Direct votes only change the opt-in spill-cost fallback. A unanimous branch
; with enough observed visits suppresses static divergence for an equivalent
; rewritten decision. StructurizeCFG's new Flow decision has no vote, so its
; descendants retain the fallback. Split, insufficient, and missing votes also
; retain it.

declare i32 @llvm.amdgcn.workitem.id.x()

; OFF-LABEL: BlockUniformityProfile for function: @unanimous_votes
; OFF: (%Flow): no PGO annotation (statically may be divergent)
; OFF: (%else): no PGO annotation (statically may be divergent)
; ON-LABEL: BlockUniformityProfile for function: @unanimous_votes
; ON: (%Flow): no PGO annotation (not classified as divergent)
; ON: (%then): no PGO annotation (statically may be divergent)
; ON: (%else): no PGO annotation (not classified as divergent)
; SPILL-OFF: PROFILED_SPILL{{[[:space:]]+}}unanimous_votes{{[[:space:]]+}}1{{[[:space:]]+}}1{{[[:space:]]+}}5{{[[:space:]]+}}4{{[[:space:]]+}}2
; SPILL-ON: PROFILED_SPILL{{[[:space:]]+}}unanimous_votes{{[[:space:]]+}}1{{[[:space:]]+}}1{{[[:space:]]+}}5{{[[:space:]]+}}2{{[[:space:]]+}}1
define amdgpu_kernel void @unanimous_votes(ptr addrspace(1) %out) !uniformity.profile !0 {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp eq i32 %id, 0
  br i1 %cond, label %then, label %else, !branch.agreement.prototype !1
then:
  store volatile i32 1, ptr addrspace(1) %out
  ret void
else:
  store volatile i32 2, ptr addrspace(1) %out
  ret void
}

; OFF-LABEL: BlockUniformityProfile for function: @split_votes
; OFF: (%then): no PGO annotation (statically may be divergent)
; ON-LABEL: BlockUniformityProfile for function: @split_votes
; ON: (%then): no PGO annotation (statically may be divergent)
; SPILL-ON: PROFILED_SPILL{{[[:space:]]+}}split_votes{{[[:space:]]+}}1{{[[:space:]]+}}1{{[[:space:]]+}}5{{[[:space:]]+}}4{{[[:space:]]+}}2
define amdgpu_kernel void @split_votes(ptr addrspace(1) %out) !uniformity.profile !0 {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp eq i32 %id, 0
  br i1 %cond, label %then, label %else, !branch.agreement.prototype !2
then:
  store volatile i32 1, ptr addrspace(1) %out
  ret void
else:
  store volatile i32 2, ptr addrspace(1) %out
  ret void
}

; OFF-LABEL: BlockUniformityProfile for function: @insufficient_votes
; OFF: (%then): no PGO annotation (statically may be divergent)
; ON-LABEL: BlockUniformityProfile for function: @insufficient_votes
; ON: (%then): no PGO annotation (statically may be divergent)
; SPILL-ON: PROFILED_SPILL{{[[:space:]]+}}insufficient_votes{{[[:space:]]+}}1{{[[:space:]]+}}1{{[[:space:]]+}}5{{[[:space:]]+}}4{{[[:space:]]+}}2
define amdgpu_kernel void @insufficient_votes(ptr addrspace(1) %out) !uniformity.profile !0 {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp eq i32 %id, 0
  br i1 %cond, label %then, label %else, !branch.agreement.prototype !3
then:
  store volatile i32 1, ptr addrspace(1) %out
  ret void
else:
  store volatile i32 2, ptr addrspace(1) %out
  ret void
}

; ON-LABEL: BlockUniformityProfile for function: @missing_votes
; ON: (%then): no PGO annotation (statically may be divergent)
; SPILL-ON: PROFILED_SPILL{{[[:space:]]+}}missing_votes{{[[:space:]]+}}1{{[[:space:]]+}}1{{[[:space:]]+}}5{{[[:space:]]+}}4{{[[:space:]]+}}2
define amdgpu_kernel void @missing_votes(ptr addrspace(1) %out) !uniformity.profile !0 {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp eq i32 %id, 0
  br i1 %cond, label %then, label %else
then:
  store volatile i32 1, ptr addrspace(1) %out
  ret void
else:
  store volatile i32 2, ptr addrspace(1) %out
  ret void
}

!0 = !{}
!1 = !{i64 100, i64 100}
!2 = !{i64 100, i64 99}
!3 = !{i64 99, i64 99}
