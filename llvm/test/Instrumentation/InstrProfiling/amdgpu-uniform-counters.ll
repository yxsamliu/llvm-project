;; Test that AMDGPU targets generate uniform counter arrays alongside regular
;; counters. The uniform counter is passed to __gpu_pgo_increment which
;; updates it when all lanes in the wave are active.

; RUN: opt %s -mtriple=amdgcn-amd-amdhsa -passes=instrprof -S | FileCheck %s

@__hip_cuid_test123 = addrspace(1) global i8 0
@__profn_test_kernel = private constant [11 x i8] c"test_kernel"

define amdgpu_kernel void @test_kernel() {
  call void @llvm.instrprof.increment(ptr @__profn_test_kernel, i64 12345, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)

;; Check that uniform counter array is created
; CHECK: @__profu_all_test123 = protected addrspace(1) global

;; Check that __gpu_pgo_increment receives both counter and uniform counter
; CHECK: call void @__gpu_pgo_increment(ptr addrspace(1) @__llvm_prf_c_test123, ptr addrspace(1) @__profu_all_test123, i64 1)
