; RUN: opt -passes="amdgpu-promote-alloca-to-vector,verify" -disable-output %s
;
; Regression test: amdgpu-promote-alloca-to-vector must not leave dangling
; placeholder references (e.g. "<badref>") when promoting an alloca used in a
; loop, even if an in-block store is effectively a no-op (store of `undef`).
;
; The failure mode was: SSAUpdater recorded a temporary placeholder as the
; block's available value (because inserting `undef` can be simplified away),
; then the placeholder got deleted, leaving a bad reference used by a PHI.

target triple = "amdgcn-amd-amdhsa"

define void @test(i1 %cmp) {
entry:
  %keys = alloca [4 x float], align 16, addrspace(5)
  br i1 %cmp, label %if.else, label %ret

if.else:
  ; This may be treated as a no-op during promotion/simplification.
  store float undef, ptr addrspace(5) %keys, align 4
  br label %loop

loop:                                              ; preds = %loop, %if.else
  %i = phi i32 [ 0, %if.else ], [ %inc, %loop ]
  %idx = zext i32 %i to i64
  %p = getelementptr float, ptr addrspace(5) %keys, i64 %idx
  store float 0.000000e+00, ptr addrspace(5) %p, align 4
  %inc = add i32 %i, 1
  br label %loop

ret:
  ret void
}

