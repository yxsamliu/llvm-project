; RUN: opt -passes=pgo-instr-gen -offload-pgo-wave-counts -S %s | FileCheck %s
; RUN: opt -passes=pgo-instr-gen -S %s | FileCheck %s --check-prefix=DEFAULT
; RUN: opt -passes=pgo-instr-gen -offload-pgo-wave-counts -mtriple=x86_64-unknown-linux-gnu -S %s | FileCheck %s --check-prefix=DEFAULT

target triple = "amdgcn-amd-amdhsa"

; DEFAULT-NOT: call void @llvm.instrprof.increment.wave
; CHECK-LABEL: define void @diamond
; CHECK: entry:
; CHECK-NEXT: call void @llvm.instrprof.increment.wave({{.*}}i32 2, i32 0, i32 4)
; CHECK: a:
; CHECK-NEXT: call void @llvm.instrprof.increment.wave({{.*}}i32 2, i32 1, i32 4)
; CHECK: b:
; CHECK-NEXT: call void @llvm.instrprof.increment.wave({{.*}}i32 2, i32 2, i32 4)
; CHECK: exit:
; CHECK-NEXT: call void @llvm.instrprof.increment.wave({{.*}}i32 2, i32 3, i32 4)
define void @diamond(i1 %cond, ptr %p) {
entry:
  br i1 %cond, label %a, label %b
a:
  store volatile i32 1, ptr %p
  br label %exit
b:
  store volatile i32 2, ptr %p
  br label %exit
exit:
  ret void
}
