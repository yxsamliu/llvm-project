; RUN: llc -mtriple=amdgpu9.0a-amd-amdhsa -mcpu=gfx90a -amdgpu-stress-sgpr=12 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,GFX90A
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -mcpu=gfx950 -amdgpu-stress-sgpr=12 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,GFX950
; RUN: llc -mtriple=amdgpu9.0a-amd-amdhsa -mcpu=gfx90a -amdgpu-stress-sgpr=12 -verify-machineinstrs -enable-new-pm < %s | FileCheck %s --check-prefixes=CHECK,GFX90A
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -mcpu=gfx950 -amdgpu-stress-sgpr=12 -verify-machineinstrs -enable-new-pm < %s | FileCheck %s --check-prefixes=CHECK,GFX950
;
; SGPR pressure creates a WWM spill register. Leaving WWM allocation reserves
; that register and clears the temporary per-lane register mask. The shared
; RegisterClassInfo must be refreshed at this transition: after rewriting the
; MFMA to use AGPRs, spill elimination consults its VGPR allocation order.
; A stale order tries to assign an ordinary spill to the reserved WWM register.
;
; CHECK-LABEL: wwm_cache:
; CHECK: v_writelane_b32 [[WWM:v[0-9]+]],
; GFX90A: v_mfma_i32_4x4x4i8 a[
; GFX950: v_mfma_i32_4x4x4_16b_i8 a[
; CHECK: v_readlane_b32 {{[^,]+}}, [[WWM]],
; CHECK: s_endpgm

target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @wwm_cache(ptr addrspace(1) %ptr) #0 {
  ; Keep scalar values live across the vector-pressure region.
  %s0 = call i32 asm sideeffect "; scalar", "=s"()
  %s1 = call i32 asm sideeffect "; scalar", "=s"()
  %s2 = call i32 asm sideeffect "; scalar", "=s"()
  %s3 = call i32 asm sideeffect "; scalar", "=s"()
  %s4 = call i32 asm sideeffect "; scalar", "=s"()
  %s5 = call i32 asm sideeffect "; scalar", "=s"()
  %s6 = call i32 asm sideeffect "; scalar", "=s"()
  %s7 = call i32 asm sideeffect "; scalar", "=s"()
  %s8 = call i32 asm sideeffect "; scalar", "=s"()
  %s9 = call i32 asm sideeffect "; scalar", "=s"()
  %s10 = call i32 asm sideeffect "; scalar", "=s"()
  %s11 = call i32 asm sideeffect "; scalar", "=s"()
  %s12 = call i32 asm sideeffect "; scalar", "=s"()
  %mai = call <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32 0, i32 0, <4 x i32> zeroinitializer, i32 0, i32 0, i32 0)

  ; These scalar VGPR values allow individual spills to be reassigned.
  %spill0 = call i32 asm sideeffect "; spill", "=v"()
  %spill1 = call i32 asm sideeffect "; spill", "=v"()
  %spill2 = call i32 asm sideeffect "; spill", "=v"()
  %spill3 = call i32 asm sideeffect "; spill", "=v"()
  %spill4 = call i32 asm sideeffect "; spill", "=v"()
  %spill5 = call i32 asm sideeffect "; spill", "=v"()
  %spill6 = call i32 asm sideeffect "; spill", "=v"()
  %spill7 = call i32 asm sideeffect "; spill", "=v"()
  %spill8 = call i32 asm sideeffect "; spill", "=v"()
  %spill9 = call i32 asm sideeffect "; spill", "=v"()
  %spill10 = call i32 asm sideeffect "; spill", "=v"()
  %spill11 = call i32 asm sideeffect "; spill", "=v"()
  %spill12 = call i32 asm sideeffect "; spill", "=v"()
  %spill13 = call i32 asm sideeffect "; spill", "=v"()
  %spill14 = call i32 asm sideeffect "; spill", "=v"()
  %spill15 = call i32 asm sideeffect "; spill", "=v"()
  %spill16 = call i32 asm sideeffect "; spill", "=v"()
  %spill17 = call i32 asm sideeffect "; spill", "=v"()
  %spill18 = call i32 asm sideeffect "; spill", "=v"()
  %spill19 = call i32 asm sideeffect "; spill", "=v"()
  %spill20 = call i32 asm sideeffect "; spill", "=v"()
  %spill21 = call i32 asm sideeffect "; spill", "=v"()
  %spill22 = call i32 asm sideeffect "; spill", "=v"()
  %spill23 = call i32 asm sideeffect "; spill", "=v"()
  %spill24 = call i32 asm sideeffect "; spill", "=v"()
  %spill25 = call i32 asm sideeffect "; spill", "=v"()

  ; Exhaust the remaining vector registers, then require the MFMA in AGPRs.
  %v0 = call <16 x i32> asm sideeffect "; pressure", "=v"()
  %v1 = call <16 x i32> asm sideeffect "; pressure", "=v"()
  call void asm sideeffect "; mfma in agpr", "a"(<4 x i32> %mai)
  store volatile <16 x i32> %v0, ptr addrspace(1) %ptr
  store volatile <16 x i32> %v1, ptr addrspace(1) %ptr
  store volatile i32 %spill0, ptr addrspace(1) %ptr
  store volatile i32 %spill1, ptr addrspace(1) %ptr
  store volatile i32 %spill2, ptr addrspace(1) %ptr
  store volatile i32 %spill3, ptr addrspace(1) %ptr
  store volatile i32 %spill4, ptr addrspace(1) %ptr
  store volatile i32 %spill5, ptr addrspace(1) %ptr
  store volatile i32 %spill6, ptr addrspace(1) %ptr
  store volatile i32 %spill7, ptr addrspace(1) %ptr
  store volatile i32 %spill8, ptr addrspace(1) %ptr
  store volatile i32 %spill9, ptr addrspace(1) %ptr
  store volatile i32 %spill10, ptr addrspace(1) %ptr
  store volatile i32 %spill11, ptr addrspace(1) %ptr
  store volatile i32 %spill12, ptr addrspace(1) %ptr
  store volatile i32 %spill13, ptr addrspace(1) %ptr
  store volatile i32 %spill14, ptr addrspace(1) %ptr
  store volatile i32 %spill15, ptr addrspace(1) %ptr
  store volatile i32 %spill16, ptr addrspace(1) %ptr
  store volatile i32 %spill17, ptr addrspace(1) %ptr
  store volatile i32 %spill18, ptr addrspace(1) %ptr
  store volatile i32 %spill19, ptr addrspace(1) %ptr
  store volatile i32 %spill20, ptr addrspace(1) %ptr
  store volatile i32 %spill21, ptr addrspace(1) %ptr
  store volatile i32 %spill22, ptr addrspace(1) %ptr
  store volatile i32 %spill23, ptr addrspace(1) %ptr
  store volatile i32 %spill24, ptr addrspace(1) %ptr
  store volatile i32 %spill25, ptr addrspace(1) %ptr
  store volatile i32 %s0, ptr addrspace(1) %ptr
  store volatile i32 %s1, ptr addrspace(1) %ptr
  store volatile i32 %s2, ptr addrspace(1) %ptr
  store volatile i32 %s3, ptr addrspace(1) %ptr
  store volatile i32 %s4, ptr addrspace(1) %ptr
  store volatile i32 %s5, ptr addrspace(1) %ptr
  store volatile i32 %s6, ptr addrspace(1) %ptr
  store volatile i32 %s7, ptr addrspace(1) %ptr
  store volatile i32 %s8, ptr addrspace(1) %ptr
  store volatile i32 %s9, ptr addrspace(1) %ptr
  store volatile i32 %s10, ptr addrspace(1) %ptr
  store volatile i32 %s11, ptr addrspace(1) %ptr
  store volatile i32 %s12, ptr addrspace(1) %ptr
  ret void
}
declare <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32, i32, <4 x i32>, i32 immarg, i32 immarg, i32 immarg)
attributes #0 = { nounwind "amdgpu-waves-per-eu"="8,8" }
