; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1300 < %s

@bar = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison

; CHECK-LABEL: {{^}}func:
; CHECK:           .amdgpu_pal_metadata
; CHECK-NEXT: ---
; CHECK-NEXT:    .hardware_stages:
; CHECK-NEXT:      .cs:
; CHECK-NEXT:        .entry_point:    _amdgpu_cs_main
; CHECK-NEXT:        .entry_point_symbol: kernel2
; CHECK-NEXT:         .named_bar_cnt:  0x1
; CHECK-NEXT:        .scratch_memory_size: 0
; CHECK-NEXT:        .sgpr_count:     0x21
; CHECK-NEXT:        .vgpr_count:     0x20
; CHECK:    .registers:      {}
; CHECK-NEXT:  '0x2e12 (COMPUTE_PGM_RSRC1)': 0x600f0003
; CHECK-NEXT:  '0x2e13 (COMPUTE_PGM_RSRC2)': 0x1391
; CHECK-NEXT:...
; CHECK-NEXT:        .end_amdgpu_pal_metadata

define void @func() {
    call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar)
    call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar, i32 7)
    call void @llvm.amdgcn.s.barrier.wait(i16 1)
    ret void
}


define amdgpu_kernel void @kernel(ptr addrspace(1) %out, ptr addrspace(3) %in) #0 {
    call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar)
    call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar, i32 7)
    call void @llvm.amdgcn.s.barrier.wait(i16 1)

    call void @func()
    ret void
}

declare void @llvm.amdgcn.s.barrier() #1
declare void @llvm.amdgcn.s.barrier.wait(i16) #1
declare void @llvm.amdgcn.s.barrier.signal(i32) #1
declare void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3), i32) #1
declare i1 @llvm.amdgcn.s.barrier.signal.isfirst(i32) #1
declare void @llvm.amdgcn.s.barrier.init(ptr addrspace(3), i32) #1
declare void @llvm.amdgcn.s.barrier.join(ptr addrspace(3)) #1
declare void @llvm.amdgcn.s.barrier.leave(i16) #1
declare i32 @llvm.amdgcn.s.get.barrier.state(i32) #1
declare i32 @llvm.amdgcn.s.get.named.barrier.state(ptr addrspace(3)) #1

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }
attributes #2 = { nounwind readnone }


!0 = !{!"\82\B0amdpal.pipelines\91\8A\A4.api\A6Vulkan\B2.compute_registers\85\AB.tg_size_en\C3\AA.tgid_x_en\C2\AA.tgid_y_en\C2\AA.tgid_z_en\C2\AF.tidig_comp_cnt\01\B0.hardware_stages\81\A3.cs\8C\AF.checksum_value\CE\94D\D7\D0\AB.debug_mode\00\AB.float_mode\CC\C0\A9.image_op\C2\AC.mem_ordered\C3\AB.sgpr_limitj\B7.threadgroup_dimensions\93\01\CD\04\00\01\AD.trap_present\00\B2.user_data_reg_map\DC\00 \CE\10\00\00\00\CE\FF\FF\FF\FF\00\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\AB.user_sgprs\03\AB.vgpr_limit\CD\01\00\AF.wavefront_size@\B7.internal_pipeline_hash\92\CF\E7\10k\A6:\A6%\F7\CF\B2\1F\1A\D4{\DA\E1T\AA.registers\80\A8.shaders\81\A8.compute\82\B0.api_shader_hash\92\CF\E9Zn7}\1E\B9\E7\00\B1.hardware_mapping\91\A3.cs\B0.spill_threshold\CE\FF\FF\FF\FF\A5.type\A2Cs\B0.user_data_limit\01\AF.xgl_cache_info\82\B3.128_bit_cache_hash\92\CF\B4X\B8\11[\A4\88P\CF\A0;\B0\AF\FF\B4\BE\C0\AD.llpc_version\A461.1\AEamdpal.version\92\03\06"}
!1 = !{i32 7}
