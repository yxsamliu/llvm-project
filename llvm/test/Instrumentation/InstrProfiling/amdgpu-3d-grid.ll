;; Test that AMDGPU PGO instrumentation generates contiguous counter arrays
;; and profile section symbols with CUID-based naming. The __gpu_pgo_is_sampled
;; library function handles 3D block linearization internally.

; RUN: opt %s -mtriple=amdgcn-amd-amdhsa -passes=instrprof -S | FileCheck %s

@__hip_cuid_abcdef789 = addrspace(1) global i8 0
@__profn_kernel_3d = private constant [9 x i8] c"kernel_3d"

define amdgpu_kernel void @kernel_3d() {
  call void @llvm.instrprof.increment(ptr @__profn_kernel_3d, i64 12345, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)

;; Check contiguous counter array with CUID suffix
; CHECK: @__llvm_prf_c_abcdef789 = protected addrspace(1) global [1 x i64] zeroinitializer

;; Check uniform counter array
; CHECK: @__profu_all_abcdef789 = protected addrspace(1) global [1 x i64] zeroinitializer

;; Check profile section symbol
; CHECK: @__llvm_offload_prf_abcdef789 = addrspace(1) constant

;; Check sampling guard calls library function
; CHECK: call i32 @__gpu_pgo_is_sampled(i32 3)
