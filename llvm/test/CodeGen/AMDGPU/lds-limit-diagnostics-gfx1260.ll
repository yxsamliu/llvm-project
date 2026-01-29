; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1260 < %s 2>&1 | FileCheck -check-prefix=ERROR %s

; GFX1260 supports upto 660 KB LDS memory.
; This is a negative test to check when the LDS size exceeds the max usable limit.

; ERROR: error: <unknown>:0:0: local memory (675844) exceeds limit (675840) in function 'test_lds_limit'
@dst = addrspace(3) global [168961 x i32] undef

define amdgpu_kernel void @test_lds_limit(i32 %val) {
  %gep = getelementptr [168961 x i32], ptr addrspace(3) @dst, i32 0, i32 100
  store i32 %val, ptr addrspace(3) %gep
  ret void
}
