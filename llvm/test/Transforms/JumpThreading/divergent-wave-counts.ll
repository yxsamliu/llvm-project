; RUN: split-file %s %t
; RUN: opt -S -passes=jump-threading -verify-each -jump-threading-divergent-policy=uniform-wave < %t/positive.ll | FileCheck %s --check-prefix=THREAD
; RUN: opt -S -passes=jump-threading -verify-each -jump-threading-divergent-policy=uniform-wave < %t/zero.ll | FileCheck %s --check-prefix=KEEP
; RUN: opt -S -passes=jump-threading -verify-each -jump-threading-divergent-policy=uniform-wave < %t/stale.ll | FileCheck %s --check-prefix=KEEP
; RUN: opt -S -passes=jump-threading -verify-each -jump-threading-divergent-policy=uniform-wave < %t/missing.ll | FileCheck %s --check-prefix=KEEP

; Only a valid, mapped, positive count admits threading in the wave-gated mode.
; Function identity and topology must be validated by the actual wave consumer.
; THREAD-LABEL: define amdgpu_kernel void @if_else(
; THREAD: join.thread:
; THREAD: udiv i32 11, %x
; THREAD: br label %yes
; KEEP-LABEL: define amdgpu_kernel void @if_else(
; KEEP-NOT: join.thread:
; KEEP: %v = phi i32 [ 11, %left ], [ 23, %right ]

;--- positive.ll
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @if_else(i1 %argcond, i32 %x, ptr addrspace(1) %out) !uniformity.profile !9 !wave.profile !0 {
entry:
  %cond = xor i1 %argcond, false
  br i1 %cond, label %left, label %right, !wave.profile.block !1
left:
  br label %join, !wave.profile.block !2
right:
  br label %join, !wave.profile.block !3
join:
  %v = phi i32 [ 11, %left ], [ 23, %right ]
  %r = udiv i32 %v, %x

  br i1 %cond, label %yes, label %no, !wave.profile.block !4
yes:
  store volatile i32 %r, ptr addrspace(1) %out
  ret void, !wave.profile.block !5
no:
  store volatile i32 %r, ptr addrspace(1) %out
  ret void, !wave.profile.block !6
}
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()
!0 = !{i64 2, i64 2685589004101179296, i64 12, i64 6, i64 6, i64 12, i64 6, i64 6}
!1 = !{i64 2, i64 2685589004101179296, i64 0, i64 1, i64 1, i64 2}
!2 = !{i64 2, i64 2685589004101179296, i64 1, i64 1, i64 3}
!3 = !{i64 2, i64 2685589004101179296, i64 2, i64 1, i64 3}
!4 = !{i64 2, i64 2685589004101179296, i64 3, i64 1, i64 4, i64 5}
!5 = !{i64 2, i64 2685589004101179296, i64 4, i64 1}
!6 = !{i64 2, i64 2685589004101179296, i64 5, i64 1}
!9 = !{}

;--- zero.ll
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @if_else(i1 %argcond, i32 %x, ptr addrspace(1) %out) !uniformity.profile !9 !wave.profile !0 {
entry:
  %cond = xor i1 %argcond, false
  br i1 %cond, label %left, label %right, !wave.profile.block !1
left:
  br label %join, !wave.profile.block !2
right:
  br label %join, !wave.profile.block !3
join:
  %v = phi i32 [ 11, %left ], [ 23, %right ]
  %r = udiv i32 %v, %x

  br i1 %cond, label %yes, label %no, !wave.profile.block !4
yes:
  store volatile i32 %r, ptr addrspace(1) %out
  ret void, !wave.profile.block !5
no:
  store volatile i32 %r, ptr addrspace(1) %out
  ret void, !wave.profile.block !6
}
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()
!0 = !{i64 2, i64 2685589004101179296, i64 12, i64 6, i64 6, i64 0, i64 6, i64 6}
!1 = !{i64 2, i64 2685589004101179296, i64 0, i64 1, i64 1, i64 2}
!2 = !{i64 2, i64 2685589004101179296, i64 1, i64 1, i64 3}
!3 = !{i64 2, i64 2685589004101179296, i64 2, i64 1, i64 3}
!4 = !{i64 2, i64 2685589004101179296, i64 3, i64 1, i64 4, i64 5}
!5 = !{i64 2, i64 2685589004101179296, i64 4, i64 1}
!6 = !{i64 2, i64 2685589004101179296, i64 5, i64 1}
!9 = !{}

;--- stale.ll
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @if_else(i1 %argcond, i32 %x, ptr addrspace(1) %out) !uniformity.profile !9 !wave.profile !0 {
entry:
  %cond = xor i1 %argcond, false
  br i1 %cond, label %left, label %right, !wave.profile.block !1
left:
  br label %join, !wave.profile.block !2
right:
  br label %join, !wave.profile.block !3
join:
  %v = phi i32 [ 11, %left ], [ 23, %right ]
  %r = udiv i32 %v, %x

  br i1 %cond, label %yes, label %no, !wave.profile.block !4
yes:
  store volatile i32 %r, ptr addrspace(1) %out
  ret void, !wave.profile.block !5
no:
  store volatile i32 %r, ptr addrspace(1) %out
  ret void, !wave.profile.block !6
}
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()
!0 = !{i64 2, i64 99, i64 12, i64 6, i64 6, i64 12, i64 6, i64 6}
!1 = !{i64 2, i64 99, i64 0, i64 1, i64 1, i64 2}
!2 = !{i64 2, i64 99, i64 1, i64 1, i64 3}
!3 = !{i64 2, i64 99, i64 2, i64 1, i64 3}
!4 = !{i64 2, i64 99, i64 3, i64 1, i64 4, i64 5}
!5 = !{i64 2, i64 99, i64 4, i64 1}
!6 = !{i64 2, i64 99, i64 5, i64 1}
!9 = !{}

;--- missing.ll
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @if_else(i1 %argcond, i32 %x, ptr addrspace(1) %out) !uniformity.profile !9 !wave.profile !0 {
entry:
  %cond = xor i1 %argcond, false
  br i1 %cond, label %left, label %right, !wave.profile.block !1
left:
  br label %join, !wave.profile.block !2
right:
  br label %join, !wave.profile.block !3
join:
  %v = phi i32 [ 11, %left ], [ 23, %right ]
  %r = udiv i32 %v, %x

  br i1 %cond, label %yes, label %no, !wave.profile.block !4
yes:
  store volatile i32 %r, ptr addrspace(1) %out
  ret void, !wave.profile.block !5
no:
  store volatile i32 %r, ptr addrspace(1) %out
  ret void, !wave.profile.block !6
}
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()
!0 = !{i64 2, i64 2685589004101179296, i64 12, i64 6, i64 6, i64 12, i64 6, i64 6}
!1 = !{i64 2, i64 2685589004101179296, i64 0, i64 1, i64 1, i64 2}
!2 = !{i64 2, i64 2685589004101179296, i64 1, i64 1, i64 3}
!3 = !{i64 2, i64 2685589004101179296, i64 2, i64 1, i64 3}
!4 = !{i64 2, i64 2685589004101179296, i64 3, i64 0, i64 4, i64 5}
!5 = !{i64 2, i64 2685589004101179296, i64 4, i64 1}
!6 = !{i64 2, i64 2685589004101179296, i64 5, i64 1}
!9 = !{}
