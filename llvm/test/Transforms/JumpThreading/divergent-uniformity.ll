; RUN: opt -S -passes=jump-threading -verify-each < %s | FileCheck %s --check-prefix=DISABLED
; RUN: opt -S -passes=jump-threading -verify-each -jump-threading-divergent-policy=uniform < %s | FileCheck %s --check-prefix=UNIFORM
; RUN: opt -S -passes=jump-threading -verify-each -jump-threading-divergent-policy=profile < %s | FileCheck %s --check-prefix=PROFILE

target triple = "amdgcn-amd-amdhsa"

; A static proof works without profiling, but only in the uniform policy.
; DISABLED-LABEL: define amdgpu_kernel void @static_uniform(
; DISABLED-NOT: join.thread:
; DISABLED: %v = phi i32 [ 11, %left ], [ 23, %right ]
; PROFILE-LABEL: define amdgpu_kernel void @static_uniform(
; PROFILE-NOT: join.thread:
; PROFILE: %v = phi i32 [ 11, %left ], [ 23, %right ]
; UNIFORM-LABEL: define amdgpu_kernel void @static_uniform(
; UNIFORM: join.thread:
; UNIFORM: udiv i32 11, %x
; UNIFORM: br label %yes
define amdgpu_kernel void @static_uniform(i1 %cond, i32 %x, ptr addrspace(1) %out) {
entry:
  br i1 %cond, label %left, label %right
left:
  br label %join
right:
  br label %join
join:
  %v = phi i32 [ 11, %left ], [ 23, %right ]
  %r = udiv i32 %v, %x
  br i1 %cond, label %yes, label %no
yes:
  store volatile i32 %r, ptr addrspace(1) %out
  ret void
no:
  store volatile i32 %r, ptr addrspace(1) %out
  ret void
}

; Observed uniformity is a profitability hint, even if static analysis cannot
; establish uniformity. LVI must still prove the condition on the incoming edge.
; DISABLED-LABEL: define amdgpu_kernel void @profile_uniform(
; DISABLED-NOT: join.thread:
; DISABLED: %v = phi i32 [ 11, %left ], [ 23, %right ]
; PROFILE-LABEL: define amdgpu_kernel void @profile_uniform(
; PROFILE: join.thread:
; PROFILE: udiv i32 11, %x
; PROFILE: br label %yes
; UNIFORM-LABEL: define amdgpu_kernel void @profile_uniform(
; UNIFORM: join.thread:
; UNIFORM: udiv i32 11, %x
; UNIFORM: br label %yes
define amdgpu_kernel void @profile_uniform(i32 %x, ptr addrspace(1) %out) !uniformity.profile !0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp ult i32 %tid, 32
  br i1 %cond, label %left, label %right
left:
  br label %join
right:
  br label %join
join:
  %v = phi i32 [ 11, %left ], [ 23, %right ]
  %r = udiv i32 %v, %x
  br i1 %cond, label %yes, label %no, !branch.uniformity.profile !0
yes:
  store volatile i32 %r, ptr addrspace(1) %out
  ret void
no:
  store volatile i32 %r, ptr addrspace(1) %out
  ret void
}

; A branch attachment without the function marker is not a usable profile.
; DISABLED-LABEL: define amdgpu_kernel void @missing_function_profile(
; DISABLED-NOT: join.thread:
; DISABLED: %v = phi
; PROFILE-LABEL: define amdgpu_kernel void @missing_function_profile(
; PROFILE-NOT: join.thread:
; PROFILE: %v = phi
; UNIFORM-LABEL: define amdgpu_kernel void @missing_function_profile(
; UNIFORM-NOT: join.thread:
; UNIFORM: %v = phi
define amdgpu_kernel void @missing_function_profile(i32 %x, ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp ult i32 %tid, 32
  br i1 %cond, label %left, label %right
left:
  br label %join
right:
  br label %join
join:
  %v = phi i32 [ 11, %left ], [ 23, %right ]
  %r = udiv i32 %v, %x
  br i1 %cond, label %yes, label %no, !branch.uniformity.profile !0
yes:
  store volatile i32 %r, ptr addrspace(1) %out
  ret void
no:
  store volatile i32 %r, ptr addrspace(1) %out
  ret void
}

; A uniform branch does not override the ordinary prohibition on duplicating
; convergent operations.
; DISABLED-LABEL: define amdgpu_kernel void @convergent_call(
; DISABLED-NOT: join.thread:
; DISABLED: call void @llvm.amdgcn.s.barrier()
; PROFILE-LABEL: define amdgpu_kernel void @convergent_call(
; PROFILE-NOT: join.thread:
; PROFILE: call void @llvm.amdgcn.s.barrier()
; UNIFORM-LABEL: define amdgpu_kernel void @convergent_call(
; UNIFORM-NOT: join.thread:
; UNIFORM: call void @llvm.amdgcn.s.barrier()
define amdgpu_kernel void @convergent_call(i1 %cond) !uniformity.profile !0 {
entry:
  br i1 %cond, label %left, label %right
left:
  br label %join
right:
  br label %join
join:
  call void @llvm.amdgcn.s.barrier()
  br i1 %cond, label %yes, label %no, !branch.uniformity.profile !0
yes:
  ret void
no:
  ret void
}

; Folding a proven condition preserves the block population, but removes the
; branch decision and its weights/hint.
; DISABLED-LABEL: define amdgpu_kernel void @folded_branch(
; DISABLED: br i1 %cond, label %yes, label %no, !prof
; PROFILE-LABEL: define amdgpu_kernel void @folded_branch(
; PROFILE: join:
; PROFILE-NEXT: br label %yes, !block.uniformity.profile ![[U:[0-9]+]]{{$}}
; UNIFORM-LABEL: define amdgpu_kernel void @folded_branch(
; UNIFORM: join:
; UNIFORM-NEXT: br label %yes, !block.uniformity.profile ![[U:[0-9]+]]{{$}}
define amdgpu_kernel void @folded_branch(i1 %cond) !uniformity.profile !0 {
entry:
  br i1 %cond, label %join, label %no
join:
  br i1 %cond, label %yes, label %no, !prof !1, !block.uniformity.profile !0, !branch.uniformity.profile !0
yes:
  ret void
no:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()
!0 = !{}
!1 = !{!"branch_weights", i32 100, i32 0}
