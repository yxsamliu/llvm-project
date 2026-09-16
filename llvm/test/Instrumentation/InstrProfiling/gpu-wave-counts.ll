; RUN: opt -passes=instrprof -offload-pgo-sampling=0 -S %s | FileCheck %s
; RUN: opt -passes=instrprof -offload-pgo-sampling=3 -S %s | FileCheck %s --check-prefix=SAMPLE

target triple = "amdgcn-amd-amdhsa"
@__profn_test = private constant [4 x i8] c"test"

; CHECK: @__profc_test = {{.*}}[5 x i64] zeroinitializer
; CHECK: @__llvm_prf_unifcnt_test = {{.*}}[1 x i64] zeroinitializer
; CHECK: @__profd_test = {{.*}}i32 5, [3 x i16] zeroinitializer, i16 0, i32 0, i32 4 }
; CHECK-LABEL: define amdgpu_kernel void @test
; CHECK: call void @__llvm_profile_instrument_gpu_wave(ptr {{.*}}i32 1
; CHECK: call void @__llvm_profile_instrument_gpu(ptr
; CHECK: call void @__llvm_profile_instrument_gpu_wave(ptr {{.*}}i32 2
; CHECK: call void @__llvm_profile_instrument_gpu_wave(ptr {{.*}}i32 3
; CHECK: call void @__llvm_profile_instrument_gpu_wave(ptr {{.*}}i32 4
; CHECK-NOT: call void @llvm.instrprof
; SAMPLE: call i32 @__llvm_profile_sampling_gpu(i32 3)
; SAMPLE: br i1
; SAMPLE: call void @__llvm_profile_instrument_gpu_wave
define amdgpu_kernel void @test(i1 %cond) {
entry:
  call void @llvm.instrprof.increment.wave(ptr @__profn_test, i64 123, i32 1, i32 0, i32 4)
  call void @llvm.instrprof.increment(ptr @__profn_test, i64 123, i32 1, i32 0)
  br i1 %cond, label %a, label %b
a:
  call void @llvm.instrprof.increment.wave(ptr @__profn_test, i64 123, i32 1, i32 1, i32 4)
  br label %exit
b:
  call void @llvm.instrprof.increment.wave(ptr @__profn_test, i64 123, i32 1, i32 2, i32 4)
  br label %exit
exit:
  call void @llvm.instrprof.increment.wave(ptr @__profn_test, i64 123, i32 1, i32 3, i32 4)
  ret void
}
declare void @llvm.instrprof.increment(ptr, i64, i32, i32)
declare void @llvm.instrprof.increment.wave(ptr, i64, i32, i32, i32)
