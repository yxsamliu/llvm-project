; RUN: split-file %s %t
; RUN: opt -S -passes=jump-threading,jump-threading -verify-each -jump-threading-divergent-policy=uniform -report-uniform-jump-threading < %t/uniform-fork.ll > %t/uniform-fork.out 2> %t/uniform-fork.log
; RUN: FileCheck %s --check-prefix=PRESERVE < %t/uniform-fork.out
; RUN: FileCheck %s --check-prefix=PRESERVE-COUNT < %t/uniform-fork.log
; RUN: opt -S -passes=jump-threading,jump-threading -verify-each -jump-threading-divergent-policy=uniform -report-uniform-jump-threading < %t/divergent-fork.ll > %t/divergent-fork.out 2> %t/divergent-fork.log
; RUN: FileCheck %s --check-prefix=INVALIDATE < %t/divergent-fork.out
; RUN: FileCheck %s --check-prefix=INVALIDATE-COUNT < %t/divergent-fork.log
; RUN: opt -S -passes=jump-threading -verify-each -jump-threading-divergent-policy=uniform -jump-threading-require-disjoint-waves < %t/uniform-fork.ll | FileCheck %s --check-prefix=PRESERVE
; RUN: opt -S -passes=jump-threading -verify-each -jump-threading-divergent-policy=uniform -jump-threading-require-disjoint-waves < %t/divergent-fork.ll | FileCheck %s --check-prefix=REJECT

; REJECT-LABEL: define amdgpu_kernel void @if_else(
; REJECT-NOT: join.thread:
; REJECT: %v = phi i32 [ 11, %left ], [ 23, %right ]
; REJECT: br i1 %div, label %yes, label %no, !block.uniformity.profile


; A divergent guard under one arm of a statically uniform fork does not allow
; the same wave to enter both arms. Threading partitions the join's count, but
; leaves downstream events unchanged. A profiled (statically divergent) fork
; cannot establish that proof, so downstream wave and uniformity data is dropped.
; Run twice to ensure invalidated counts cannot become valid again.
;
; PRESERVE-COUNT: JT_WAVE_DISJOINT{{.*}}if_else{{.*}}join{{[[:space:]]}}1
; PRESERVE-COUNT: JT_CHANGED{{.*}}if_else{{.*}}join
; PRESERVE-COUNT: JT_UNIFORM{{.*}}if_else{{.*}}tail{{.*}}waves=12
; INVALIDATE-COUNT: JT_WAVE_DISJOINT{{.*}}if_else{{.*}}join{{[[:space:]]}}0
; INVALIDATE-COUNT: JT_CHANGED{{.*}}if_else{{.*}}join
; INVALIDATE-COUNT: JT_UNIFORM{{.*}}if_else{{.*}}tail{{.*}}waves=missing
;
; PRESERVE-LABEL: define amdgpu_kernel void @if_else(
; PRESERVE: join.thread:
; PRESERVE: br label %yes, !wave.profile.block ![[CLONE:[0-9]+]]{{$}}
; PRESERVE: tail:
; PRESERVE-NEXT: br i1 %second, label %done, label %exit, !prof ![[W:[0-9]+]], !block.uniformity.profile ![[U:[0-9]+]], !branch.uniformity.profile ![[U]], !wave.profile.block ![[TAIL:[0-9]+]]{{$}}
; PRESERVE: ![[CLONE]] = !{i64 2, i64 2685589004101179296, i64 10, i64 0, i64 5}
; PRESERVE: ![[W]] = !{!"branch_weights", i32 11, i32 1}
; PRESERVE: ![[TAIL]] = !{i64 2, i64 2685589004101179296, i64 7, i64 1, i64 8, i64 9}
;
; INVALIDATE-LABEL: define amdgpu_kernel void @if_else(
; INVALIDATE: join.thread:
; INVALIDATE: br label %yes, !wave.profile.block ![[CLONE:[0-9]+]]{{$}}
; INVALIDATE: tail:
; INVALIDATE-NEXT: br i1 %second, label %done, label %exit, !prof ![[W:[0-9]+]], !wave.profile.block ![[TAIL:[0-9]+]]{{$}}
; INVALIDATE: ![[CLONE]] = !{i64 2, i64 2685589004101179296, i64 10, i64 0, i64 5}
; INVALIDATE: ![[W]] = !{!"branch_weights", i32 11, i32 1}
; INVALIDATE: ![[TAIL]] = !{i64 2, i64 2685589004101179296, i64 7, i64 0, i64 8, i64 9}

;--- uniform-fork.ll
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @if_else(i1 %argcond, i1 %second, i32 %x, ptr addrspace(1) %out) !uniformity.profile !20 !wave.profile !0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div = icmp ult i32 %tid, 32
  br i1 %argcond, label %guard, label %right, !branch.uniformity.profile !20, !wave.profile.block !1
guard:
  br i1 %div, label %left, label %exit, !wave.profile.block !2
left:
  br label %join, !wave.profile.block !3
right:
  br label %join, !wave.profile.block !4
join:
  %v = phi i32 [ 11, %left ], [ 23, %right ]
  %r = udiv i32 %v, %x
  br i1 %argcond, label %yes, label %no, !block.uniformity.profile !20, !branch.uniformity.profile !20, !wave.profile.block !5
yes:
  store volatile i32 %r, ptr addrspace(1) %out
  br label %tail, !wave.profile.block !6
no:
  store volatile i32 %r, ptr addrspace(1) %out
  br label %tail, !wave.profile.block !7
tail:
  br i1 %second, label %done, label %exit, !prof !21, !block.uniformity.profile !20, !branch.uniformity.profile !20, !wave.profile.block !8
done:
  ret void, !wave.profile.block !9
exit:
  ret void, !wave.profile.block !10
}
declare i32 @llvm.amdgcn.workitem.id.x()
!0 = !{i64 2, i64 2685589004101179296, i64 12, i64 6, i64 6, i64 6, i64 12, i64 6, i64 6, i64 12, i64 11, i64 1}
!20 = !{}
!21 = !{!"branch_weights", i32 11, i32 1}
!1 = !{i64 2, i64 2685589004101179296, i64 0, i64 1, i64 1, i64 3}
!2 = !{i64 2, i64 2685589004101179296, i64 1, i64 1, i64 2, i64 9}
!3 = !{i64 2, i64 2685589004101179296, i64 2, i64 1, i64 4}
!4 = !{i64 2, i64 2685589004101179296, i64 3, i64 1, i64 4}
!5 = !{i64 2, i64 2685589004101179296, i64 4, i64 1, i64 5, i64 6}
!6 = !{i64 2, i64 2685589004101179296, i64 5, i64 1, i64 7}
!7 = !{i64 2, i64 2685589004101179296, i64 6, i64 1, i64 7}
!8 = !{i64 2, i64 2685589004101179296, i64 7, i64 1, i64 8, i64 9}
!9 = !{i64 2, i64 2685589004101179296, i64 8, i64 1}
!10 = !{i64 2, i64 2685589004101179296, i64 9, i64 1}

;--- divergent-fork.ll
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @if_else(i1 %argcond, i1 %second, i32 %x, ptr addrspace(1) %out) !uniformity.profile !20 !wave.profile !0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div = icmp ult i32 %tid, 32
  br i1 %div, label %guard, label %right, !branch.uniformity.profile !20, !wave.profile.block !1
guard:
  br i1 %div, label %left, label %exit, !wave.profile.block !2
left:
  br label %join, !wave.profile.block !3
right:
  br label %join, !wave.profile.block !4
join:
  %v = phi i32 [ 11, %left ], [ 23, %right ]
  %r = udiv i32 %v, %x
  br i1 %div, label %yes, label %no, !block.uniformity.profile !20, !branch.uniformity.profile !20, !wave.profile.block !5
yes:
  store volatile i32 %r, ptr addrspace(1) %out
  br label %tail, !wave.profile.block !6
no:
  store volatile i32 %r, ptr addrspace(1) %out
  br label %tail, !wave.profile.block !7
tail:
  br i1 %second, label %done, label %exit, !prof !21, !block.uniformity.profile !20, !branch.uniformity.profile !20, !wave.profile.block !8
done:
  ret void, !wave.profile.block !9
exit:
  ret void, !wave.profile.block !10
}
declare i32 @llvm.amdgcn.workitem.id.x()
!0 = !{i64 2, i64 2685589004101179296, i64 12, i64 6, i64 6, i64 6, i64 12, i64 6, i64 6, i64 12, i64 11, i64 1}
!20 = !{}
!21 = !{!"branch_weights", i32 11, i32 1}
!1 = !{i64 2, i64 2685589004101179296, i64 0, i64 1, i64 1, i64 3}
!2 = !{i64 2, i64 2685589004101179296, i64 1, i64 1, i64 2, i64 9}
!3 = !{i64 2, i64 2685589004101179296, i64 2, i64 1, i64 4}
!4 = !{i64 2, i64 2685589004101179296, i64 3, i64 1, i64 4}
!5 = !{i64 2, i64 2685589004101179296, i64 4, i64 1, i64 5, i64 6}
!6 = !{i64 2, i64 2685589004101179296, i64 5, i64 1, i64 7}
!7 = !{i64 2, i64 2685589004101179296, i64 6, i64 1, i64 7}
!8 = !{i64 2, i64 2685589004101179296, i64 7, i64 1, i64 8, i64 9}
!9 = !{i64 2, i64 2685589004101179296, i64 8, i64 1}
!10 = !{i64 2, i64 2685589004101179296, i64 9, i64 1}
