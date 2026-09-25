; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -passes=loop-unroll -unroll-runtime \
; RUN:   -unroll-static-uniformity-prototype -verify-each -S %s | FileCheck %s --check-prefix=STATIC
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -passes=loop-unroll -unroll-runtime \
; RUN:   -unroll-branch-unanimity-prototype -verify-each -S %s | FileCheck %s --check-prefix=VOTES
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -passes=loop-unroll -unroll-runtime \
; RUN:   -verify-each -S %s | FileCheck %s --check-prefix=LEGACY
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -passes=loop-unroll \
; RUN:   -unroll-count=4 -unroll-runtime=false -verify-each -S %s | FileCheck %s --check-prefix=PARTIAL
;
; Only profitability is guided by direct votes. Missing, split, undersampled,
; malformed, and non-rare cases stay conservative. A static proof needs no votes.

declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier() convergent

; PARTIAL-LABEL: define amdgpu_kernel void @unanimous(
; PARTIAL-NOT: !branch.unanimity.prototype
; PARTIAL: %iv.next.3 =
; PARTIAL-NOT: !branch.unanimity.prototype
; STATIC-LABEL: define amdgpu_kernel void @unanimous(
; STATIC-NOT:    %xtraiter =
; STATIC:    ret void
; VOTES-LABEL: define amdgpu_kernel void @unanimous(
; VOTES:    %xtraiter =
; VOTES-NOT: !branch.unanimity.prototype
; VOTES:    ret void
; VOTES-NOT: !branch.unanimity.prototype
; LEGACY-LABEL: define amdgpu_kernel void @unanimous(
; LEGACY:    %xtraiter =
; LEGACY:    ret void
define amdgpu_kernel void @unanimous(ptr addrspace(1) %p, i32 %n, i32 %bound) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %header

header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %latch ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %latch ]
  %val = add i32 %acc, %tid
  %sidecond = icmp sgt i32 %val, %bound
  br i1 %sidecond, label %sideexit, label %body, !prof !0, !branch.unanimity.prototype !2

body:
  %gep = getelementptr i32, ptr addrspace(1) %p, i32 %iv
  %ld = load i32, ptr addrspace(1) %gep
  %acc.next = add i32 %acc, %ld
  br label %latch

latch:
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %header

sideexit:
  store i32 %acc, ptr addrspace(1) %p
  ret void

exit:
  ret void
}

; PARTIAL-LABEL: define amdgpu_kernel void @split(
; STATIC-LABEL: define amdgpu_kernel void @split(
; STATIC-NOT:    %xtraiter =
; STATIC:    ret void
; VOTES-LABEL: define amdgpu_kernel void @split(
; VOTES-NOT:    %xtraiter =
; VOTES:    ret void
; LEGACY-LABEL: define amdgpu_kernel void @split(
; LEGACY:    %xtraiter =
; LEGACY:    ret void
define amdgpu_kernel void @split(ptr addrspace(1) %p, i32 %n, i32 %bound) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %header

header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %latch ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %latch ]
  %val = add i32 %acc, %tid
  %sidecond = icmp sgt i32 %val, %bound
  br i1 %sidecond, label %sideexit, label %body, !prof !0, !branch.unanimity.prototype !3

body:
  %gep = getelementptr i32, ptr addrspace(1) %p, i32 %iv
  %ld = load i32, ptr addrspace(1) %gep
  %acc.next = add i32 %acc, %ld
  br label %latch

latch:
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %header

sideexit:
  store i32 %acc, ptr addrspace(1) %p
  ret void

exit:
  ret void
}

; STATIC-LABEL: define amdgpu_kernel void @undersampled(
; STATIC-NOT:    %xtraiter =
; STATIC:    ret void
; VOTES-LABEL: define amdgpu_kernel void @undersampled(
; VOTES-NOT:    %xtraiter =
; VOTES:    ret void
; LEGACY-LABEL: define amdgpu_kernel void @undersampled(
; LEGACY:    %xtraiter =
; LEGACY:    ret void
define amdgpu_kernel void @undersampled(ptr addrspace(1) %p, i32 %n, i32 %bound) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %header

header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %latch ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %latch ]
  %val = add i32 %acc, %tid
  %sidecond = icmp sgt i32 %val, %bound
  br i1 %sidecond, label %sideexit, label %body, !prof !0, !branch.unanimity.prototype !4

body:
  %gep = getelementptr i32, ptr addrspace(1) %p, i32 %iv
  %ld = load i32, ptr addrspace(1) %gep
  %acc.next = add i32 %acc, %ld
  br label %latch

latch:
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %header

sideexit:
  store i32 %acc, ptr addrspace(1) %p
  ret void

exit:
  ret void
}

; STATIC-LABEL: define amdgpu_kernel void @zero(
; STATIC-NOT:    %xtraiter =
; STATIC:    ret void
; VOTES-LABEL: define amdgpu_kernel void @zero(
; VOTES-NOT:    %xtraiter =
; VOTES:    ret void
; LEGACY-LABEL: define amdgpu_kernel void @zero(
; LEGACY:    %xtraiter =
; LEGACY:    ret void
define amdgpu_kernel void @zero(ptr addrspace(1) %p, i32 %n, i32 %bound) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %header

header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %latch ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %latch ]
  %val = add i32 %acc, %tid
  %sidecond = icmp sgt i32 %val, %bound
  br i1 %sidecond, label %sideexit, label %body, !prof !0, !branch.unanimity.prototype !5

body:
  %gep = getelementptr i32, ptr addrspace(1) %p, i32 %iv
  %ld = load i32, ptr addrspace(1) %gep
  %acc.next = add i32 %acc, %ld
  br label %latch

latch:
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %header

sideexit:
  store i32 %acc, ptr addrspace(1) %p
  ret void

exit:
  ret void
}

; STATIC-LABEL: define amdgpu_kernel void @invalid(
; STATIC-NOT:    %xtraiter =
; STATIC:    ret void
; VOTES-LABEL: define amdgpu_kernel void @invalid(
; VOTES-NOT:    %xtraiter =
; VOTES:    ret void
; LEGACY-LABEL: define amdgpu_kernel void @invalid(
; LEGACY:    %xtraiter =
; LEGACY:    ret void
define amdgpu_kernel void @invalid(ptr addrspace(1) %p, i32 %n, i32 %bound) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %header

header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %latch ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %latch ]
  %val = add i32 %acc, %tid
  %sidecond = icmp sgt i32 %val, %bound
  br i1 %sidecond, label %sideexit, label %body, !prof !0, !branch.unanimity.prototype !6

body:
  %gep = getelementptr i32, ptr addrspace(1) %p, i32 %iv
  %ld = load i32, ptr addrspace(1) %gep
  %acc.next = add i32 %acc, %ld
  br label %latch

latch:
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %header

sideexit:
  store i32 %acc, ptr addrspace(1) %p
  ret void

exit:
  ret void
}

; STATIC-LABEL: define amdgpu_kernel void @wrong_type(
; STATIC-NOT:    %xtraiter =
; STATIC:    ret void
; VOTES-LABEL: define amdgpu_kernel void @wrong_type(
; VOTES-NOT:    %xtraiter =
; VOTES:    ret void
; LEGACY-LABEL: define amdgpu_kernel void @wrong_type(
; LEGACY:    %xtraiter =
; LEGACY:    ret void
define amdgpu_kernel void @wrong_type(ptr addrspace(1) %p, i32 %n, i32 %bound) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %header

header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %latch ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %latch ]
  %val = add i32 %acc, %tid
  %sidecond = icmp sgt i32 %val, %bound
  br i1 %sidecond, label %sideexit, label %body, !prof !0, !branch.unanimity.prototype !7

body:
  %gep = getelementptr i32, ptr addrspace(1) %p, i32 %iv
  %ld = load i32, ptr addrspace(1) %gep
  %acc.next = add i32 %acc, %ld
  br label %latch

latch:
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %header

sideexit:
  store i32 %acc, ptr addrspace(1) %p
  ret void

exit:
  ret void
}

; STATIC-LABEL: define amdgpu_kernel void @missing(
; STATIC-NOT:    %xtraiter =
; STATIC:    ret void
; VOTES-LABEL: define amdgpu_kernel void @missing(
; VOTES-NOT:    %xtraiter =
; VOTES:    ret void
; LEGACY-LABEL: define amdgpu_kernel void @missing(
; LEGACY:    %xtraiter =
; LEGACY:    ret void
define amdgpu_kernel void @missing(ptr addrspace(1) %p, i32 %n, i32 %bound) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %header

header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %latch ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %latch ]
  %val = add i32 %acc, %tid
  %sidecond = icmp sgt i32 %val, %bound
  br i1 %sidecond, label %sideexit, label %body, !prof !0

body:
  %gep = getelementptr i32, ptr addrspace(1) %p, i32 %iv
  %ld = load i32, ptr addrspace(1) %gep
  %acc.next = add i32 %acc, %ld
  br label %latch

latch:
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %header

sideexit:
  store i32 %acc, ptr addrspace(1) %p
  ret void

exit:
  ret void
}

; STATIC-LABEL: define amdgpu_kernel void @nonrare(
; STATIC-NOT:    %xtraiter =
; STATIC:    ret void
; VOTES-LABEL: define amdgpu_kernel void @nonrare(
; VOTES-NOT:    %xtraiter =
; VOTES:    ret void
; LEGACY-LABEL: define amdgpu_kernel void @nonrare(
; LEGACY-NOT:    %xtraiter =
; LEGACY:    ret void
define amdgpu_kernel void @nonrare(ptr addrspace(1) %p, i32 %n, i32 %bound) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %header

header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %latch ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %latch ]
  %val = add i32 %acc, %tid
  %sidecond = icmp sgt i32 %val, %bound
  br i1 %sidecond, label %sideexit, label %body, !prof !1, !branch.unanimity.prototype !9

body:
  %gep = getelementptr i32, ptr addrspace(1) %p, i32 %iv
  %ld = load i32, ptr addrspace(1) %gep
  %acc.next = add i32 %acc, %ld
  br label %latch

latch:
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %header

sideexit:
  store i32 %acc, ptr addrspace(1) %p
  ret void

exit:
  ret void
}

; STATIC-LABEL: define amdgpu_kernel void @static_uniform(
; STATIC:    %xtraiter =
; STATIC:    ret void
; VOTES-LABEL: define amdgpu_kernel void @static_uniform(
; VOTES:    %xtraiter =
; VOTES:    ret void
; LEGACY-LABEL: define amdgpu_kernel void @static_uniform(
; LEGACY:    %xtraiter =
; LEGACY:    ret void
define amdgpu_kernel void @static_uniform(ptr addrspace(1) %p, i32 %n, i32 %bound) {
entry:
  br label %header

header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %latch ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %latch ]
  %sidecond = icmp sgt i32 %bound, %n
  br i1 %sidecond, label %sideexit, label %body, !prof !0

body:
  %gep = getelementptr i32, ptr addrspace(1) %p, i32 %iv
  %ld = load i32, ptr addrspace(1) %gep
  %acc.next = add i32 %acc, %ld
  br label %latch

latch:
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %header

sideexit:
  store i32 %acc, ptr addrspace(1) %p
  ret void

exit:
  ret void
}

; Votes cannot make a divergent trip count safe for a convergent operation.
; LEGACY-LABEL: define amdgpu_kernel void @convergent_divergent_trip(
; LEGACY-NOT: %xtraiter =
; LEGACY: ret void
; STATIC-LABEL: define amdgpu_kernel void @convergent_divergent_trip(
; STATIC-NOT: %xtraiter =
; STATIC: ret void
; VOTES-LABEL: define amdgpu_kernel void @convergent_divergent_trip(
; VOTES-NOT: %xtraiter =
; VOTES: ret void
define amdgpu_kernel void @convergent_divergent_trip(ptr addrspace(1) %p, i32 %boundn, i32 %bound) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %n = add i32 %boundn, %tid
  br label %header

header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %latch ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %latch ]
  %val = add i32 %acc, %tid
  %sidecond = icmp sgt i32 %val, %bound
  br i1 %sidecond, label %sideexit, label %body, !prof !0, !branch.unanimity.prototype !2

body:
  call void @llvm.amdgcn.s.barrier()
  %gep = getelementptr i32, ptr addrspace(1) %p, i32 %iv
  %ld = load i32, ptr addrspace(1) %gep
  %acc.next = add i32 %acc, %ld
  br label %latch

latch:
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %header

sideexit:
  store i32 %acc, ptr addrspace(1) %p
  ret void

exit:
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 1000000}
!1 = !{!"branch_weights", i32 1, i32 1}
!2 = !{i64 1000, i64 1000}
!3 = !{i64 1000, i64 999}
!4 = !{i64 99, i64 99}
!5 = !{i64 0, i64 0}
!6 = !{i64 100, i64 101}
!7 = !{i32 1000, i32 1000}
!9 = !{i64 1000, i64 1000}
