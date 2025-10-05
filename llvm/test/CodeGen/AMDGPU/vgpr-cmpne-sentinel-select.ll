; Minimal reproducer for a v_cmp_ne_u32_e64 against -1
; arising from a PHI-fed sentinel check kept in value form
; and used via a boolean select (A && (cond != -1)) to gate
; a memcpy(select-src) which later becomes a vector select.
;
; First, verify AMDGPUVectorIdiom rewrites memcpy(select-src):
; RUN: opt -passes=amdgpu-vector-idiom -S %s | FileCheck %s --check-prefix=IDIOM
;
; Then, verify backend still shows the v_cmp_ne in this minimal form:
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -O2 < %s | FileCheck %s
;
; The key is the explicit `%cmpSentinel = icmp ne i32 %cond, -1` combined
; as `%valid = select i1 %cmpBound, i1 %cmpSentinel, i1 false`, which tends
; to lower to a `v_cmp_ne_u32_e64 ..., -1, v...` on AMDGPU when %cond is a VGPR.
;
; CHECK: v_cmp_ne_u32_e64 {{[^,]+}}, -1, v{{[0-9]+}}

; NOTE: Opaque pointers IR

target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly,
                                    ptr noalias nocapture readonly,
                                    i64, i1 immarg)

define amdgpu_kernel void @repro(ptr addrspace(1) %idx_v,
                                 ptr addrspace(1) %row_data_v,
                                 ptr addrspace(1) %out,
                                 i32 %iv, i32 %limit, i32 %flag) nounwind {
entry:
  ; Use per-lane workitem id to force VGPR addressing/loads
  %tid = call i32 @llvm.amdgcn.workitem.id.x()

  ; Create the PHI(%loaded, -1) using a uniform branch so PHI lives in VGPR
  %use = icmp ne i32 %flag, 0
  br i1 %use, label %cond_true, label %cond_false

cond_true:                                        ; load dynamic i32 for the sentinel PHI (VGPR)
  %idxext = zext i32 %tid to i64
  %gptr = getelementptr inbounds i32, ptr addrspace(1) %idx_v, i64 %idxext
  %loaded = load i32, ptr addrspace(1) %gptr, align 4
  br label %cond_end

cond_false:
  br label %cond_end

cond_end:
  %cond = phi i32 [ %loaded, %cond_true ], [ -1, %cond_false ]

  ; Bound and sentinel compares (both per-lane)
  %cmpBound = icmp ult i32 %tid, %limit
  %cmpSentinel = icmp ne i32 %cond, -1

  ; Materialize boolean via select (A && B)
  %valid = select i1 %cmpBound, i1 %cmpSentinel, i1 false

  ; Use memcpy(select-src) to mirror real-world problematic shape
  ; Two allocas in addrspace(5), casted to generic ptr, so vector idiom applies
  %row.stack = alloca [16 x i8], align 16, addrspace(5)
  %zero.stack = alloca [16 x i8], align 16, addrspace(5)
  %row.ascast = addrspacecast ptr addrspace(5) %row.stack to ptr
  %zero.ascast = addrspacecast ptr addrspace(5) %zero.stack to ptr

  ; Select pointer source by the boolean valid
  %choose = select i1 %valid, ptr %row.ascast, ptr %zero.ascast

  ; The idiom pass should rewrite this memcpy into branch-local loads/stores
  %out.ascast = addrspacecast ptr addrspace(1) %out to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %out.ascast, ptr %choose, i64 16, i1 false)
  ret void
}

; IDIOM-LABEL: @repro(
; IDIOM: br i1
; IDIOM: load <16 x i8>
; IDIOM: store <16 x i8>
; IDIOM: load <16 x i8>
; IDIOM: store <16 x i8>
; IDIOM-NOT: call void @llvm.memcpy
