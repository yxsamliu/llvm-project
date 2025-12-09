; RUN: opt -S -passes=amdgpu-attributor < %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@lds.data = internal addrspace(10) global i32 0, align 4

; Test 1: Basic rank specialization with workgroup.id.x
define protected amdgpu_kernel void @dispatch_kernel(ptr addrspace(1) %ptr) #0 !reqd_work_group_size !0 {
; CHECK-LABEL: define {{[^@]+}}@dispatch_kernel
; CHECK-SAME: #[[ATTR0:[0-9]+]]
  tail call void @llvm.amdgcn.wavegroup.rank.p0(i32 0, ptr nonnull @dispatch_kernel.rank_0)
  tail call void @llvm.amdgcn.wavegroup.rank.p0(i32 1, ptr nonnull @dispatch_kernel.rank_1)
  ret void
}

define internal amdgpu_kernel void @dispatch_kernel.rank_0(ptr addrspace(1) %ptr) #1 !reqd_work_group_size !0 {
; CHECK-LABEL: define {{[^@]+}}@dispatch_kernel.rank_0
; CHECK-SAME: #[[ATTR1:[0-9]+]]
  %wgid = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %wave_id = tail call i32 @llvm.amdgcn.wavegroup.id()
  %lane_id = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %val = load i32, ptr addrspace(1) %ptr, align 4
  %result = add i32 %val, %lane_id
  store i32 %result, ptr addrspace(10) @lds.data, align 4
  ret void
}

define internal amdgpu_kernel void @dispatch_kernel.rank_1(ptr addrspace(1) %ptr) #1 !reqd_work_group_size !0 {
; CHECK-LABEL: define {{[^@]+}}@dispatch_kernel.rank_1
; CHECK-SAME: #[[ATTR1]]
  %wgid = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %wave_id = tail call i32 @llvm.amdgcn.wavegroup.id()
  %lane_id = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %val = load i32, ptr addrspace(10) @lds.data, align 4
  %offset = shl i32 %lane_id, 3
  %idx = zext i32 %offset to i64
  %gep = getelementptr inbounds i32, ptr addrspace(1) %ptr, i64 %idx
  store i32 %val, ptr addrspace(1) %gep, align 4
  ret void
}

; Test 2: Mixed implicit argument requirements (workgroup.id.y and .z)
define protected amdgpu_kernel void @mixed_dispatch_kernel(ptr addrspace(1) %ptr) #0 !reqd_work_group_size !0 {
; CHECK-LABEL: define {{[^@]+}}@mixed_dispatch_kernel
; CHECK-SAME: #[[ATTR2:[0-9]+]]
  tail call void @llvm.amdgcn.wavegroup.rank.p0(i32 0, ptr nonnull @mixed_dispatch_kernel.rank_0)
  tail call void @llvm.amdgcn.wavegroup.rank.p0(i32 1, ptr nonnull @mixed_dispatch_kernel.rank_1)
  ret void
}

; This rank function uses workgroup.id.y
define internal amdgpu_kernel void @mixed_dispatch_kernel.rank_0(ptr addrspace(1) %ptr) #1 !reqd_work_group_size !0 {
; CHECK-LABEL: define {{[^@]+}}@mixed_dispatch_kernel.rank_0
; CHECK-SAME: #[[ATTR3:[0-9]+]]
  %wgid.y = tail call i32 @llvm.amdgcn.workgroup.id.y()
  %val = load i32, ptr addrspace(1) %ptr, align 4
  %result = add i32 %val, %wgid.y
  store i32 %result, ptr addrspace(10) @lds.data, align 4
  ret void
}

; This rank function uses workgroup.id.z
define internal amdgpu_kernel void @mixed_dispatch_kernel.rank_1(ptr addrspace(1) %ptr) #1 !reqd_work_group_size !0 {
; CHECK-LABEL: define {{[^@]+}}@mixed_dispatch_kernel.rank_1
; CHECK-SAME: #[[ATTR3]]
  %wgid.z = tail call i32 @llvm.amdgcn.workgroup.id.z()
  %val = load i32, ptr addrspace(10) @lds.data, align 4
  %result = add i32 %val, %wgid.z
  store i32 %result, ptr addrspace(1) %ptr, align 4
  ret void
}

; Test 3: Unknown intrinsic without NoCallback - should pessimistically assume it needs all implicit args
define protected amdgpu_kernel void @unknown_intrinsic_dispatch(ptr addrspace(1) %ptr) #0 !reqd_work_group_size !0 {
; CHECK-LABEL: define {{[^@]+}}@unknown_intrinsic_dispatch
; CHECK-SAME: #[[ATTR4:[0-9]+]]
  tail call void @llvm.amdgcn.wavegroup.rank.p0(i32 0, ptr nonnull @unknown_intrinsic_dispatch.rank_0)
  tail call void @llvm.amdgcn.wavegroup.rank.p0(i32 1, ptr nonnull @unknown_intrinsic_dispatch.rank_1)  
  ret void
}

define internal amdgpu_kernel void @unknown_intrinsic_dispatch.rank_0(ptr addrspace(1) %ptr) #1 !reqd_work_group_size !0 {
; CHECK-LABEL: define {{[^@]+}}@unknown_intrinsic_dispatch.rank_0
; CHECK-SAME: #[[ATTR5:[0-9]+]]
  %wgid = tail call i32 @llvm.amdgcn.workgroup.id.x()
  ; Call an unknown intrinsic without nocallback - this should cause pessimistic fixpoint
  %val = call i32 @llvm.experimental.foo()
  %result = add i32 %val, %wgid
  store i32 %result, ptr addrspace(10) @lds.data, align 4
  ret void
}

define internal amdgpu_kernel void @unknown_intrinsic_dispatch.rank_1(ptr addrspace(1) %ptr) #1 !reqd_work_group_size !0 {
; CHECK-LABEL: define {{[^@]+}}@unknown_intrinsic_dispatch.rank_1
; CHECK-SAME: #[[ATTR5]]
  %val = load i32, ptr addrspace(10) @lds.data, align 4
  store i32 %val, ptr addrspace(1) %ptr, align 4
  ret void
}

declare !callback !1 void @llvm.amdgcn.wavegroup.rank.p0(i32 immarg, ptr) #2
declare i32 @llvm.amdgcn.workgroup.id.x() #3
declare i32 @llvm.amdgcn.workgroup.id.y() #3
declare i32 @llvm.amdgcn.workgroup.id.z() #3
declare i32 @llvm.amdgcn.wavegroup.id() #3
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #4
; Unknown intrinsic without nocallback
declare i32 @llvm.experimental.foo() #5

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-wavegroup-enable" }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-wavegroup-enable" "amdgpu-wavegroup-rank-function" }
attributes #2 = { convergent nounwind }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #5 = { nounwind }

; CHECK: attributes #[[ATTR0]] = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-wavegroup-enable" "uniform-work-group-size"="false" }
; CHECK: attributes #[[ATTR1]] = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-wavegroup-enable" "amdgpu-wavegroup-rank-function" "uniform-work-group-size"="false" }
; CHECK: attributes #[[ATTR2]] = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-wavegroup-enable" "uniform-work-group-size"="false" }
; CHECK: attributes #[[ATTR3]] = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-wavegroup-enable" "amdgpu-wavegroup-rank-function" "uniform-work-group-size"="false" }
; CHECK: attributes #[[ATTR4]] = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-wavegroup-enable" "uniform-work-group-size"="false" }
; CHECK: attributes #[[ATTR5]] = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-wavegroup-enable" "amdgpu-wavegroup-rank-function" "uniform-work-group-size"="false" }

!0 = !{i32 256, i32 1, i32 1}
!1 = !{!2}
!2 = !{i64 1, i64 -1, i1 false}
