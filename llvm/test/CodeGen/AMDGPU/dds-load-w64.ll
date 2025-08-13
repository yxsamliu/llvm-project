; RUN: not llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize64 < %s 2<&1 | FileCheck --check-prefix=ERROR-GISEL %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize64 < %s 2<&1 | FileCheck --check-prefix=ERROR-SDAG %s

; ERROR-GISEL: LLVM ERROR: cannot select: {{.*}}
; ERROR-SDAG: LLVM ERROR: Cannot select: {{.*}}

define amdgpu_ps float @dds_load_uniform(ptr addrspace(11) inreg %ptr) {
    %load = load i32, ptr addrspace(11) %ptr
    %to.vgpr = bitcast i32 %load to float
    ret float %to.vgpr
}
