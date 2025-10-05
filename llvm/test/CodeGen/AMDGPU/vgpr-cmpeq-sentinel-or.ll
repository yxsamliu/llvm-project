; Good-shape counterpart: use OR of bad conditions
; (idx >= limit) || (cond == -1) to guard the value select,
; avoiding the local v_cmp_ne ...,-1 at the selection site.
;
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -O2 < %s | FileCheck %s
;
; Expect:
; - Presence of an equality compare against -1 (v_cmp_eq)
; - Absence of v_cmp_ne against -1
;
; CHECK: v_cmp_eq_u32_e64 {{[^,]+}}, -1, v{{[0-9]+}}
; CHECK-NOT: v_cmp_ne_u32_e64 {{[^,]+}}, -1, v{{[0-9]+}}

; NOTE: Opaque pointers IR

target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

define amdgpu_kernel void @repro_good(ptr addrspace(1) %idx_v,
                                      ptr addrspace(1) %row_data_v,
                                      ptr addrspace(1) %out,
                                      i32 %limit, i32 %flag) nounwind {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()

  ; Create PHI(%loaded, -1) to model the sentinel source
  %use = icmp ne i32 %flag, 0
  br i1 %use, label %cond_true, label %cond_false

cond_true:
  %idxext = zext i32 %tid to i64
  %gptr = getelementptr inbounds i32, ptr addrspace(1) %idx_v, i64 %idxext
  %loaded = load i32, ptr addrspace(1) %gptr, align 4
  br label %cond_end

cond_false:
  br label %cond_end

cond_end:
  %cond = phi i32 [ %loaded, %cond_true ], [ -1, %cond_false ]

  ; Bad conditions to OR: (tid >= limit) OR (cond == -1)
  %cmpBoundBad = icmp uge i32 %tid, %limit
  %cmpSentinelEq = icmp eq i32 %cond, -1
  ; Use select form to match canonical good shape seen in logs
  %orBad = select i1 %cmpBoundBad, i1 true, i1 %cmpSentinelEq

  ; Load vector and select zero vs data under OR-of-bad
  %rowidx = zext i32 %tid to i64
  %vptr = getelementptr inbounds <4 x i32>, ptr addrspace(1) %row_data_v, i64 %rowidx
  %vec = load <4 x i32>, ptr addrspace(1) %vptr, align 16

  %sel = select i1 %orBad, <4 x i32> zeroinitializer, <4 x i32> %vec
  %outptr = getelementptr inbounds <4 x i32>, ptr addrspace(1) %out, i64 0
  store <4 x i32> %sel, ptr addrspace(1) %outptr, align 16
  ret void
}
