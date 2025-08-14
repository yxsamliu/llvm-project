;; Check that HIP device profile start/stop symbols are registered as device variables
;; when __hip_register_globals function exists in the module.
; RUN: opt %s -passes='instrprof' -S | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__start___llvm_prf_cnts_offload = internal global [0 x i8] undef
; CHECK: @__stop___llvm_prf_cnts_offload = internal global [0 x i8] undef
; CHECK: @__start___llvm_prf_cnts.name = private constant [24 x i8] c"__start___llvm_prf_cnts\00"
; CHECK: @__stop___llvm_prf_cnts.name = private constant [23 x i8] c"__stop___llvm_prf_cnts\00"
; CHECK: @__start___llvm_prf_data_offload = internal global [0 x i8] undef
; CHECK: @__stop___llvm_prf_data_offload = internal global [0 x i8] undef
; CHECK: @__start___llvm_prf_data.name = private constant [24 x i8] c"__start___llvm_prf_data\00"
; CHECK: @__stop___llvm_prf_data.name = private constant [23 x i8] c"__stop___llvm_prf_data\00"
; CHECK: @__start___llvm_prf_names_offload = internal global [0 x i8] undef
; CHECK: @__stop___llvm_prf_names_offload = internal global [0 x i8] undef
; CHECK: @__start___llvm_prf_names.name = private constant [25 x i8] c"__start___llvm_prf_names\00"
; CHECK: @__stop___llvm_prf_names.name = private constant [24 x i8] c"__stop___llvm_prf_names\00"

declare void @llvm.instrprof.increment(ptr %0, i64 %1, i32 %2, i32 %3)
; CHECK: declare void @__hipRegisterVar(ptr, ptr, ptr, ptr, i32, i64, i32, i32)
declare void @__hipRegisterVar(ptr, ptr, ptr, ptr, i32, i64, i32, i32)

@__profn_hip_kernel = private constant [10 x i8] c"hip_kernel"

define void @hip_kernel() {
  call void @llvm.instrprof.increment(ptr @__profn_hip_kernel, i64 123456, i32 1, i32 0)
  ret void
}

; Existing HIP registration function that should be modified
define internal void @__hip_register_globals(ptr %handle) {
entry:
  ret void
}

; CHECK: define internal void @__hip_register_globals(ptr %handle) {
; CHECK: entry:
; CHECK:   call void @__hipRegisterVar(ptr %handle, ptr @__start___llvm_prf_cnts_offload, ptr @__start___llvm_prf_cnts.name, ptr @__start___llvm_prf_cnts.name, i32 0, i64 0, i32 0, i32 0)
; CHECK:   call void @__hipRegisterVar(ptr %handle, ptr @__stop___llvm_prf_cnts_offload, ptr @__stop___llvm_prf_cnts.name, ptr @__stop___llvm_prf_cnts.name, i32 0, i64 0, i32 0, i32 0)
; CHECK:   call void @__hipRegisterVar(ptr %handle, ptr @__start___llvm_prf_data_offload, ptr @__start___llvm_prf_data.name, ptr @__start___llvm_prf_data.name, i32 0, i64 0, i32 0, i32 0)
; CHECK:   call void @__hipRegisterVar(ptr %handle, ptr @__stop___llvm_prf_data_offload, ptr @__stop___llvm_prf_data.name, ptr @__stop___llvm_prf_data.name, i32 0, i64 0, i32 0, i32 0)
; CHECK:   call void @__hipRegisterVar(ptr %handle, ptr @__start___llvm_prf_names_offload, ptr @__start___llvm_prf_names.name, ptr @__start___llvm_prf_names.name, i32 0, i64 0, i32 0, i32 0)
; CHECK:   call void @__hipRegisterVar(ptr %handle, ptr @__stop___llvm_prf_names_offload, ptr @__stop___llvm_prf_names.name, ptr @__stop___llvm_prf_names.name, i32 0, i64 0, i32 0, i32 0)
; CHECK:   ret void
; CHECK: }