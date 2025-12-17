; RUN: opt -mtriple=amdgcn-amd-amdhsa -verify-each -passes=amdgpu-rank-specialization,instcombine,simplifycfg -S -o - %s | FileCheck --check-prefixes=CHECK,NODSE %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -verify-each -passes=amdgpu-rank-specialization,instcombine,simplifycfg,dse -S -o - %s | FileCheck --check-prefixes=CHECK,DSE %s

target datalayout = "A5"

@vx = external local_unnamed_addr addrspace(10) global [7 x float], align 4

declare float @dummy_common1()
declare float @dummy_common2()

; Second store to @vx overwrites the data from the first store, and gets eliminated by DSE.
define amdgpu_kernel void @test_kernel_1() local_unnamed_addr "amdgpu-wavegroup-enable" !reqd_work_group_size !{i32 32, i32 4, i32 1} {
entry:
  %0 = call i32 @llvm.amdgcn.wave.id.in.wavegroup()
  %1 = call float @dummy_common1()
  %2 = call float @dummy_common2()
  %cmp0 = icmp eq i32 %0, 0
  br i1 %cmp0, label %if.then0, label %if.end0

if.then0:
  store float %1, ptr addrspace(10) @vx, align 4
  br label %if.end0

if.end0:
  %cmp1 = icmp eq i32 %0, 1
  br i1 %cmp1, label %if.then1, label %if.end1

if.then1:
  %3 = load float, ptr addrspace(10) @vx, align 4
  br label %if.end1

if.end1:
  %cmp2 = icmp eq i32 %0, 0
  br i1 %cmp2, label %if.then2, label %if.end2

if.then2:
  store float %2, ptr addrspace(10) @vx, align 4
  br label %if.end2

if.end2:
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @test_kernel_1(
; CHECK-SAME: ) local_unnamed_addr #[[ATTR0:[0-9]+]] !reqd_work_group_size [[META0:![0-9]+]] {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    call void @llvm.amdgcn.wavegroup.rank.p0(i32 0, ptr nonnull @test_kernel_1.rank_0)
; CHECK-NEXT:    ret void
;
; NODSE-LABEL: define internal amdgpu_kernel void @test_kernel_1.rank_0(
; NODSE-SAME: ) local_unnamed_addr #[[ATTR2:[0-9]+]] !reqd_work_group_size [[META0]] {
; NODSE-NEXT:  [[ENTRY:.*:]]
; NODSE-NEXT:    [[TMP0:%.*]] = call float @dummy_common1()
; NODSE-NEXT:    [[TMP1:%.*]] = call float @dummy_common2()
; NODSE-NEXT:    store float [[TMP0]], ptr addrspace(10) @vx, align 4
; NODSE-NEXT:    store float [[TMP1]], ptr addrspace(10) @vx, align 4
; NODSE-NEXT:    ret void
;
; DSE-LABEL: define internal amdgpu_kernel void @test_kernel_1.rank_0(
; DSE-SAME: ) local_unnamed_addr #[[ATTR2:[0-9]+]] !reqd_work_group_size [[META0]] {
; DSE-NEXT:  [[ENTRY:.*:]]
; DSE-NEXT:    [[TMP0:%.*]] = call float @dummy_common1()
; DSE-NEXT:    [[TMP1:%.*]] = call float @dummy_common2()
; DSE-NEXT:    store float [[TMP1]], ptr addrspace(10) @vx, align 4
; DSE-NEXT:    ret void
;
; CHECK: [[META0]] = !{i32 32, i32 4, i32 1}
;
