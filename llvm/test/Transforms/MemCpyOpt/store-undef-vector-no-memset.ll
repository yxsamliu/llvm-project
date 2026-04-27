; RUN: opt -passes='sroa,instcombine,memcpyopt' -S -verify-memoryssa < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%class.aiMatrix4x4t = type { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }

declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg)

; CHECK-LABEL: define void @store_only_fp_tail(
; CHECK: store <4 x float> <float 0.000000e+00, float undef, float undef, float undef>, ptr %dst, align 1
; CHECK: %.sroa.3.0.dst.sroa_idx = getelementptr inbounds nuw i8, ptr %dst, i64 16
; CHECK: store float 0.000000e+00, ptr %.sroa.3.0.dst.sroa_idx, align 1
; CHECK-NOT: call void @llvm.memset
define void @store_only_fp_tail(ptr noalias %dst) {
  %1 = alloca %class.aiMatrix4x4t, align 4
  %2 = alloca %class.aiMatrix4x4t, align 4
  %3 = getelementptr i8, ptr %2, i64 16
  store float 0.000000e+00, ptr %3, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr %1, ptr %2, i64 64, i1 false)
  store float 0.000000e+00, ptr %1, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %1, i64 64, i1 false)
  ret void
}
