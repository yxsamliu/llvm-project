; RUN: split-file %s %t

; RUN: not llc -mtriple=amdgcn -mcpu=gfx1300 -filetype=null %t/spatial-cluster-1d-z-err.ll 2>&1 | FileCheck -check-prefix=CHECK-ERROR1 %s
; RUN: not llc -mtriple=amdgcn -mcpu=gfx1300 -filetype=null %t/spatial-cluster-1d-x-err.ll 2>&1 | FileCheck -check-prefix=CHECK-ERROR2 %s
; RUN: not llc -mtriple=amdgcn -mcpu=gfx1300 -filetype=null %t/spatial-cluster-wg-err.ll 2>&1 | FileCheck -check-prefix=CHECK-ERROR3 %s

;--- spatial-cluster-1d-z-err.ll
; CHECK-ERROR1: error: Spatial cluster kernel is not 1D
define amdgpu_kernel void @non_1d_z_dims() "amdgpu-cluster-dims"="2,2,1" "amdgpu-wavegroup-enable" "amdgpu-spatial-cluster" !reqd_work_group_size !{i32 32, i32 8, i32 1} {
entry:
  ret void
}

;--- spatial-cluster-1d-x-err.ll
; CHECK-ERROR2: error: Spatial cluster kernel is not 1D
define amdgpu_kernel void @non_1d_dims_x() "amdgpu-cluster-dims"="1,2,2" "amdgpu-wavegroup-enable" "amdgpu-spatial-cluster" !reqd_work_group_size !{i32 32, i32 8, i32 1} {
entry:
  ret void
}

;--- spatial-cluster-wg-err.ll
; CHECK-ERROR3: error: Spatial cluster kernel is not wavegroup kernel
define amdgpu_kernel void @non_wavegroup_kernel() "amdgpu-cluster-dims"="2,1,1" "amdgpu-spatial-cluster" !reqd_work_group_size !{i32 32, i32 8, i32 1} {
entry:
  ret void
}
