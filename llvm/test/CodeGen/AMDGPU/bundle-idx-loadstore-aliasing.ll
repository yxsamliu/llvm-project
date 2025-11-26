; RUN: llc -march=amdgcn -mcpu=gfx1300 -verify-machineinstrs -debug-only=bundle-indexed-load-store -stop-after=bundle-indexed-load-store < %s 2> %t | FileCheck --check-prefixes=SINGLEBB %s
; RUN: FileCheck --check-prefixes=DBG %s < %t
;
; REQUIRES: asserts

; The two tests in this file demonstrate alias analysis (AA) working in two different contexts for the
; AMDGPUBundleIdxLdSt pass: the single basic block use case (bundling), and the multi basic block use
; case (sinking phase). In both test cases, both of the AA outcomes (presence and absence of an alias
; conflict) are shown.

target triple = "amdgcn-amd-amdhsa"
@weights = external local_unnamed_addr addrspace(10) global [256 x i32], align 4
@out = external local_unnamed_addr addrspace(10) global [32 x i32], align 4

define dso_local amdgpu_kernel void @amdgcn_aa_singlebb() "amdgpu-wavegroup-enable" !reqd_work_group_size !{i32 128, i32 1, i32 1} {
; SINGLEBB-LABEL: name:            amdgcn_aa_singlebb
; SINGLEBB: [[S_MOV_B32_:%[0-9]+]]:sgpr_32 = S_MOV_B32 0
; SINGLEBB-NEXT:    [[V_LOAD_IDX_:%[0-9]+]]:vgpr_32 = V_LOAD_IDX_B32 [[S_MOV_B32_]], 50, implicit $exec :: (dereferenceable load (s32) from `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 200)`, align 8, addrspace 10)
; SINGLEBB-NEXT:    BUNDLE implicit-def $stg_dsta, implicit $exec, implicit [[S_MOV_B32_]]
; SINGLEBB-NEXT:      $stg_dsta = V_MOV_B32_e32 5, implicit $exec
; SINGLEBB-NEXT:      V_STORE_IDX_B32 internal $stg_dsta, [[S_MOV_B32_]], 50, implicit $exec :: (store (s32) into `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 200)`, align 8, addrspace 10)
; SINGLEBB-NEXT:    }
; SINGLEBB-NEXT:    BUNDLE implicit-def $stg_dsta, implicit $exec, implicit [[S_MOV_B32_]]
; SINGLEBB-NEXT:      $stg_dsta = V_MOV_B32_e32 7, implicit $exec
; SINGLEBB-NEXT:      V_STORE_IDX_B32 internal $stg_dsta, [[S_MOV_B32_]], 48, implicit $exec :: (store (s32) into `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 192)`, align 64, addrspace 10)
; SINGLEBB-NEXT:    }
; SINGLEBB-NEXT:    BUNDLE implicit-def dead $stg_srca, implicit-def $stg_dsta, implicit killed [[S_MOV_B32_]], implicit $exec, implicit killed [[V_LOAD_IDX_]]
; SINGLEBB-NEXT:      $stg_srca = V_LOAD_IDX_B32 [[S_MOV_B32_]], 55, implicit $exec :: (dereferenceable load (s32) from `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 220)`, addrspace 10)
; SINGLEBB-NEXT:      $stg_dsta = nsw V_ADD_U32_e64 internal killed $stg_srca, killed [[V_LOAD_IDX_]], 0, implicit $exec
; SINGLEBB-NEXT:      V_STORE_IDX_B32 internal $stg_dsta, killed [[S_MOV_B32_]], 272, implicit $exec :: (store (s32) into `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @out, i32 64)`, align 64, addrspace 10)
; SINGLEBB-NEXT:    }
; SINGLEBB-NEXT:    S_ENDPGM 0
entry:
  %0 = load i32, ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 200), align 4
  store i32 5, ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 200), align 4
  %1 = load i32, ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 220), align 4
  store i32 7, ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 192), align 4
  %add3 = add nsw i32 %1, %0
  store i32 %add3, ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @out, i32 64), align 4
  ret void
}

define dso_local amdgpu_kernel void @amdgcn_aa_multibb() "amdgpu-wavegroup-enable" !reqd_work_group_size !{i32 128, i32 1, i32 1} {
; DBG: ===== AMDGPUBundleIdxLdSt :: Sinking Phase =====
;     Skip first kernel.
; DBG: ===== AMDGPUBundleIdxLdSt :: Sinking Phase =====
; DBG:  *** Conflict with V_STORE_IDX_B32 [[V_STORE_IDX_:%[0-9]+]]:vgpr_32, killed %43:sgpr_32, 50, implicit $exec :: (store (s32) into `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 200)`, align 8, addrspace 10)
; DBG: BB.2 :: [[V_STORE_IDX_]]:vgpr_32 = V_MOV_B32_e32 5, implicit $exec
; DBG-NEXT:  *** Found 1 use(s)
; DBG-NEXT:  *** Use is in MI's current block. Leaving a copy in block 2
; DBG: BB.4 :: [[V_LOAD_IDX_:%[0-9]+]]:vgpr_32 = V_LOAD_IDX_B32 [[V_LOAD_IDX_1:%[0-9]+]]:sreg_32_xm0, 50, implicit $exec :: (load (s32) from %ir.arrayidx2, addrspace 10)
; DBG-NEXT:  *** Found 1 use(s)
; DBG-NEXT:  *** Sinking MI to block [[BLOCK_:[0-9]+]]
; DBG: BB.5 :: [[V_ADD_:%[0-9]+]]:vgpr_32 = nsw V_ADD_U32_e64 killed [[V_LOAD_IDX_]]:vgpr_32, killed [[_:%[0-9]+]]:vgpr_32, 0, implicit $exec
; DBG-NEXT:  *** CoreMI sinking to larger cycle depth is not profitable
; DBG: BB.5 :: [[_:%[0-9]+]]:vgpr_32 = V_MOV_B32_e32 7, implicit $exec
; DBG-NEXT:  *** Found 1 use(s)
; DBG-NEXT:  *** Use is in MI's current block. Leaving a copy in block [[BLOCK_]]
; DBG: BB.5 :: [[V_LOAD_IDX_]]:vgpr_32 = V_LOAD_IDX_B32 [[V_LOAD_IDX_1]]:sreg_32_xm0, 50, implicit $exec :: (load (s32) from %ir.arrayidx2, addrspace 10)
; DBG-NEXT:  *** Found 1 use(s)
; DBG-NEXT:  *** Use is in MI's current block. Leaving a copy in block [[BLOCK_]]
; DBG: BB.7 :: V_STORE_IDX_B32 [[V_ADD_]]:vgpr_32, [[_:%[0-9]+]]:sreg_32_xm0, 256, implicit $exec :: (store (s32) into %ir.arrayidx6, addrspace 10)
entry:
  %0 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %add = add nuw nsw i32 %0, 50
  %arrayidx = getelementptr inbounds nuw [256 x i32], ptr addrspace(10) @weights, i32 0, i32 %add
  %1 = load i32, ptr addrspace(10) %arrayidx, align 4
  store i32 5, ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 200), align 4
  %2 = load i32, ptr addrspace(10) %arrayidx, align 4
  store i32 7, ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 192), align 4
  %add4 = add nsw i32 %2, %1
  %arrayidx6 = getelementptr inbounds nuw [32 x i32], ptr addrspace(10) @out, i32 0, i32 %0
  store i32 %add4, ptr addrspace(10) %arrayidx6, align 4
  ret void
}
