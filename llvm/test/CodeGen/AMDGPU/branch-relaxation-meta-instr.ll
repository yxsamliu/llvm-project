; RUN: llc -mtriple=amdgcn-amd-amdhsa -O3 -filetype=null < %s

; Branch relaxation queries instruction sizes after CFI pseudos have been
; inserted. Those meta instructions must be treated as size 0.

define fastcc void @dynamic_alloca(i32 %size) {
entry:
  %alloca = alloca i8, i32 %size, align 8, addrspace(5)
  store volatile i8 0, ptr addrspace(5) %alloca, align 8
  ret void
}
