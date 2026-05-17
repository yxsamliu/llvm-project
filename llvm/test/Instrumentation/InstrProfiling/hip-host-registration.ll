;; Verify that HIP-host shadow-variable registration is gated on the "hip"
;; module flag. With the flag, the pass emits the per-TU shadow pointer
;; (__llvm_offload_prf_<CUID>), a per-TU ctor that registers it with the
;; profile runtime, and a declaration of the runtime hook. Without the
;; flag, none of those globals or calls are emitted.

; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -passes=instrprof -S \
; RUN:   | FileCheck %s --check-prefixes=COMMON,HIP
; RUN: cat %s | sed -e 's/^!1 = .*"hip".*$/!1 = !{i32 1, !"unused", i32 0}/' \
; RUN:   | opt -mtriple=x86_64-unknown-linux-gnu -passes=instrprof -S \
; RUN:   | FileCheck %s --check-prefixes=COMMON,NOHIP

@__hip_cuid_abc123 = external global i8
@__profn_foo = private constant [3 x i8] c"foo"

define void @foo() {
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)

!llvm.module.flags = !{!0, !1}
!0 = !{i32 2, !"EnableValueProfiling", i32 0}
!1 = !{i32 1, !"hip", i32 1}

;; Counter / data globals are emitted in both modes (instrprof always runs).
; COMMON-DAG: @__profc_foo
; COMMON-DAG: @__profd_foo

;; The HIP-only host registration plumbing.
; HIP-DAG: @__llvm_offload_prf_abc123
; HIP-DAG: declare void @__llvm_profile_offload_register_shadow_variable(ptr)
; HIP-DAG: define internal void @__llvm_pgo_register_abc123()
; HIP-DAG: call void @__llvm_profile_offload_register_shadow_variable(ptr @__llvm_offload_prf_abc123)

; NOHIP-NOT: __llvm_offload_prf_
; NOHIP-NOT: __llvm_pgo_register_
; NOHIP-NOT: __llvm_profile_offload_register_shadow_variable
