; RUN: split-file %s %t

; RUN: not llc -mtriple=amdgcn -mcpu=gfx1300 -filetype=null %t/asymmetric-cluster-clamp-1d-err.ll 2>&1 | FileCheck -check-prefix=CHECK-ERROR1 %s

;--- asymmetric-cluster-clamp-1d-err.ll
; CHECK-ERROR1: error: Asymmetric cluster clamp kernel is not 1D
define amdgpu_kernel void @non_1d_dims() "amdgpu-cluster-dims"="2,2,1" "amdgpu-asymmetric-cluster-clamp" !reqd_work_group_size !{i32 32, i32 8, i32 1} {
entry:
  ret void
}
