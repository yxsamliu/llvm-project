; RUN: split-file %s %t

; RUN: not llc -mtriple=amdgcn -mcpu=gfx1300 -filetype=null %t/asymmetric-cluster-clamp-1d-err.ll 2>&1 | FileCheck -check-prefix=CHECK-ERROR1 %s
; RUN: not llc -mtriple=amdgcn -mcpu=gfx1300 -filetype=null %t/asymmetric-cluster-clamp-fixed-dims-err.ll 2>&1 | FileCheck -check-prefix=CHECK-ERROR2 %s

;--- asymmetric-cluster-clamp-1d-err.ll
; CHECK-ERROR1: error: Asymmetric cluster clamp kernel is not 1D
define amdgpu_kernel void @non_1d_dims() #0 !reqd_work_group_size !{i32 32, i32 8, i32 1} {
entry:
  ret void
}
attributes #0 = { "amdgpu-cluster-dims"="2,2,1" "amdgpu-asymmetric-cluster-clamp" }

;--- asymmetric-cluster-clamp-fixed-dims-err.ll
; CHECK-ERROR2: error: Asymmetric cluster clamp kernel has non fixed cluster dims
define amdgpu_kernel void @non_fixed_dims() #1 !reqd_work_group_size !{i32 32, i32 8, i32 1} {
entry:
  ret void
}
attributes #1 = { "amdgpu-asymmetric-cluster-clamp" }
