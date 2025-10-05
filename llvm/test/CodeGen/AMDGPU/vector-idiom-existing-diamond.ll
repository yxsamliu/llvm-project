; RUN: opt -passes=amdgpu-vector-idiom -S %s | FileCheck %s

target triple = "amdgcn-amd-amdhsa"

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1 immarg)

define amdgpu_kernel void @diamond(ptr %dst, i1 %cond) {
; CHECK-LABEL: @diamond(
entry:
  %true.stack = alloca [4 x i8], align 16, addrspace(5)
  %false.stack = alloca [4 x i8], align 16, addrspace(5)
  %true.ascast = addrspacecast ptr addrspace(5) %true.stack to ptr
  %false.ascast = addrspacecast ptr addrspace(5) %false.stack to ptr
  br i1 %cond, label %then, label %else

then:                                             ; preds = %entry
; CHECK: then:
; CHECK: load <16 x i8>
; CHECK: store <16 x i8>
; CHECK-NOT: memcpy.src.then
  br label %join

else:                                             ; preds = %entry
; CHECK: else:
; CHECK: load <16 x i8>
; CHECK: store <16 x i8>
; CHECK-NOT: memcpy.src.else
  br label %join

join:                                             ; preds = %else, %then
  %choose = select i1 %cond, ptr %true.ascast, ptr %false.ascast
; CHECK: join:
; CHECK-NOT: call void @llvm.memcpy
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %choose, i64 16, i1 false)
  ret void
}
