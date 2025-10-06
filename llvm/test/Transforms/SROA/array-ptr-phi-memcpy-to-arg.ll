; RUN: opt -S -passes='sroa<modify-cfg>' -sroa-max-struct-to-vector-bytes=16 < %s | FileCheck %s
; RUN: opt -S -passes='sroa<modify-cfg>' -sroa-max-struct-to-vector-bytes=0  < %s | FileCheck %s --check-prefix=DISABLED

target triple = "amdgcn-amd-amdhsa"

; Scalar replacement should be able to eliminate the array alloca when
; struct-to-vector promotion is enabled, mirroring the pattern seen in
; vgpr/test/fbgemm_g/bad.ll.

%v4i32 = type { i32, i32, i32, i32 }

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare void @llvm.lifetime.start.p5(i64 immarg, ptr addrspace(5) nocapture)
declare void @llvm.lifetime.end.p5(i64 immarg, ptr addrspace(5) nocapture)
declare i32 @sum_array(ptr)

define void @test_array_ptr_phi_memcpy_to_arg(ptr %data_out, i1 %valid_arg, i32 %idx_init, i32 %idx_use, ptr %src) {
; CHECK-LABEL: @test_array_ptr_phi_memcpy_to_arg(
; CHECK-NOT: alloca
; CHECK-NOT: call void @llvm.memcpy
; CHECK: [[IDX_INIT_Z:%.*]] = zext i32 [[IDX_INIT:%.*]] to i64
; CHECK: [[GEP_INIT:%.*]] = getelementptr inbounds [4 x { i32, i32, i32, i32 }], ptr [[ROW_CAST:%.*]], i64 0, i64 [[IDX_INIT_Z]]
; CHECK: [[VLOAD_INIT:%.*]] = load <4 x i32>, ptr [[GEP_INIT]], align 16
; CHECK: [[IDX_TRUE_Z:%.*]] = zext i32 [[IDX_TRUE:%.*]] to i64
; CHECK: [[GEP_TRUE:%.*]] = getelementptr inbounds [4 x { i32, i32, i32, i32 }], ptr [[ROW_CAST]], i64 0, i64 [[IDX_TRUE_Z]]
; CHECK: [[VLOAD_TRUE:%.*]] = load <4 x i32>, ptr [[GEP_TRUE]], align 16
; CHECK: [[VPHI:%.*]] = phi <4 x i32>
; CHECK: store <4 x i32> [[VPHI]], ptr [[DATA_OUT:%.*]], align 16
;
; DISABLED-LABEL: @test_array_ptr_phi_memcpy_to_arg(
; DISABLED: alloca
; DISABLED: call void @llvm.memcpy
entry:
  %row_data_v = alloca [4 x %v4i32], align 16, addrspace(5)
  %zeros = alloca %v4i32, align 16, addrspace(5)
  %scratch = alloca %v4i32, align 16, addrspace(5)
  %valid_slot = alloca i8, align 1, addrspace(5)
  %row_cast = addrspacecast ptr addrspace(5) %row_data_v to ptr
  %zeros_cast = addrspacecast ptr addrspace(5) %zeros to ptr
  %scratch_cast = addrspacecast ptr addrspace(5) %scratch to ptr
  %valid_cast = addrspacecast ptr addrspace(5) %valid_slot to ptr
  call void @llvm.lifetime.start.p5(i64 64, ptr addrspace(5) %row_data_v)
  call void @llvm.lifetime.start.p5(i64 16, ptr addrspace(5) %zeros)
  call void @llvm.lifetime.start.p5(i64 16, ptr addrspace(5) %scratch)
  call void @llvm.lifetime.start.p5(i64 1, ptr addrspace(5) %valid_slot)
  call void @llvm.memset.p0.i64(ptr align 16 %zeros_cast, i8 0, i64 16, i1 false)
  %idx_init_z = zext i32 %idx_init to i64
  %arrayidx_init = getelementptr inbounds [4 x %v4i32], ptr %row_cast, i64 0, i64 %idx_init_z
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %arrayidx_init, ptr align 16 %src, i64 16, i1 false)
  store i8 0, ptr %valid_cast, align 1
  %row_sum = call i32 @sum_array(ptr %row_cast)
  %rot_sum = add i32 %row_sum, %idx_use
  %idx_seed = and i32 %rot_sum, 3
  br i1 %valid_arg, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  store i8 1, ptr %valid_cast, align 1
  %idx_true = add i32 %idx_seed, 1
  %idx_true_mod = and i32 %idx_true, 3
  %idx_true_z = zext i32 %idx_true_mod to i64
  %arrayidx_true = getelementptr inbounds [4 x %v4i32], ptr %row_cast, i64 0, i64 %idx_true_z
  %vec_true = load <4 x i32>, ptr %arrayidx_true, align 16
  store <4 x i32> %vec_true, ptr %scratch_cast, align 16
  br label %cond.end

cond.false:                                       ; preds = %entry
  %vec_false = load <4 x i32>, ptr %zeros_cast, align 16
  store <4 x i32> %vec_false, ptr %scratch_cast, align 16
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond_ptr = phi ptr [ %arrayidx_true, %cond.true ], [ %zeros_cast, %cond.false ]
  %vec_phi = phi <4 x i32> [ %vec_true, %cond.true ], [ %vec_false, %cond.false ]
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %data_out, ptr align 16 %cond_ptr, i64 16, i1 false)
  store <4 x i32> %vec_phi, ptr %data_out, align 16
  call void @llvm.lifetime.end.p5(i64 1, ptr addrspace(5) %valid_slot)
  call void @llvm.lifetime.end.p5(i64 16, ptr addrspace(5) %scratch)
  call void @llvm.lifetime.end.p5(i64 16, ptr addrspace(5) %zeros)
  call void @llvm.lifetime.end.p5(i64 64, ptr addrspace(5) %row_data_v)
  ret void
}

define void @test_single_struct_ptr_phi_memcpy_to_arg(ptr %data_out, i1 %valid, ptr %src) {
; CHECK-LABEL: @test_single_struct_ptr_phi_memcpy_to_arg(
; CHECK-NOT: alloca
; CHECK-NOT: call void @llvm.memcpy
; CHECK: [[VLOAD:%.*]] = load <4 x i32>, ptr [[SRC:%.*]], align 16
; CHECK: [[VSEL:%.*]] = select i1 [[VALID:%.*]], <4 x i32> [[VLOAD]], <4 x i32> zeroinitializer
; CHECK: store <4 x i32> [[VSEL]], ptr [[DATA_OUT:%.*]], align 16
;
; DISABLED-LABEL: @test_single_struct_ptr_phi_memcpy_to_arg(
; DISABLED: alloca
; DISABLED: call void @llvm.memcpy
entry:
  %row = alloca %v4i32, align 16, addrspace(5)
  %zeros = alloca %v4i32, align 16, addrspace(5)
  %row_cast = addrspacecast ptr addrspace(5) %row to ptr
  %zeros_cast = addrspacecast ptr addrspace(5) %zeros to ptr
  call void @llvm.lifetime.start.p5(i64 16, ptr addrspace(5) %row)
  call void @llvm.lifetime.start.p5(i64 16, ptr addrspace(5) %zeros)
  call void @llvm.memset.p0.i64(ptr align 16 %zeros_cast, i8 0, i64 16, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %row_cast, ptr align 16 %src, i64 16, i1 false)
  br i1 %valid, label %cond.true, label %cond.false

cond.true:                                          ; preds = %entry
  br label %cond.end

cond.false:                                         ; preds = %entry
  br label %cond.end

cond.end:                                           ; preds = %cond.false, %cond.true
  %cond190 = phi ptr [ %row_cast, %cond.true ], [ %zeros_cast, %cond.false ]
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %data_out, ptr align 16 %cond190, i64 16, i1 false)
  call void @llvm.lifetime.end.p5(i64 16, ptr addrspace(5) %zeros)
  call void @llvm.lifetime.end.p5(i64 16, ptr addrspace(5) %row)
  ret void
}
