; RUN: opt -passes=slp-vectorizer -S -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

define void @buildvector_store_blocks_store_chain(ptr %p, float %a0, float %a1, float %a2, float %a3, float %a4, float %a5, float %a6, float %a7) {
; CHECK-LABEL: @buildvector_store_blocks_store_chain(
; CHECK-NOT: store <2 x float>
; CHECK: store <4 x float> {{.*}}, ptr %p, align 4
; CHECK: [[P4:%.*]] = getelementptr i8, ptr [[P3:%.*]], i64 4
; CHECK: store <4 x float> {{.*}}, ptr [[P4]], align 4
; CHECK-NOT: store <2 x float>
; CHECK: ret void
entry:
  %v0 = fadd float %a0, 1.000000e+00
  %v1 = fadd float %a1, 1.000000e+00
  %v2 = fadd float %a2, 1.000000e+00
  %v3 = fadd float %a3, 1.000000e+00
  %v4 = fadd float %a4, 1.000000e+00
  %v5 = fadd float %a5, 1.000000e+00
  %v6 = fadd float %a6, 1.000000e+00
  %v7 = fadd float %a7, 1.000000e+00
  store float %v0, ptr %p, align 4
  %p1 = getelementptr inbounds float, ptr %p, i64 1
  store float %v1, ptr %p1, align 4
  %p2 = getelementptr inbounds float, ptr %p, i64 2
  store float %v2, ptr %p2, align 4
  %p3 = getelementptr inbounds float, ptr %p, i64 3
  %b0 = insertelement <4 x float> poison, float %v3, i32 0
  %b1 = insertelement <4 x float> %b0, float %v4, i32 1
  %b2 = insertelement <4 x float> %b1, float %v5, i32 2
  %b3 = insertelement <4 x float> %b2, float %v6, i32 3
  store <4 x float> %b3, ptr %p3, align 4
  %p7 = getelementptr inbounds float, ptr %p, i64 7
  store float %v7, ptr %p7, align 4
  ret void
}
