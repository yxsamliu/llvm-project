;; Test that AMDGPU PGO instrumentation generates library calls for Wave64.
;; Wave64 targets (e.g., gfx908) should embed wave size 64 in profile data.

; RUN: opt %s -mtriple=amdgcn-amd-amdhsa -passes=instrprof -S | FileCheck %s

@__hip_cuid_abcdef123 = addrspace(1) global i8 0
@__profn_kernel_w64 = private constant [10 x i8] c"kernel_w64"

define amdgpu_kernel void @kernel_w64() #0 {
  call void @llvm.instrprof.increment(ptr @__profn_kernel_w64, i64 12345, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)

attributes #0 = { "target-cpu"="gfx908" }

;; Check wave size 64 in profile data
; CHECK: @__llvm_prf_d_abcdef123 = {{.*}}i16 0, i16 64, i32 0

;; Check sampling guard
; CHECK: %pgo.sampled = call i32 @__gpu_pgo_is_sampled(i32 3)
; CHECK: %pgo.matched = icmp ne i32 %pgo.sampled, 0
; CHECK: br i1 %pgo.matched, label %po_then, label %po_cont

;; Check library call
; CHECK: po_then:
; CHECK: call void @__gpu_pgo_increment(ptr addrspace(1) @__llvm_prf_c_abcdef123, ptr addrspace(1) @__profu_all_abcdef123, i64 1)
