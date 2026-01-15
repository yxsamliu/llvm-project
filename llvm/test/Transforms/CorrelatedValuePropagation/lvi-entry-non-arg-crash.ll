; RUN: opt -passes="default<O3>" -disable-output %s
;
; Regression test: CVP should not crash when querying LazyValueInfo for a value
; that is not live-in to the entry block.
;
; Reduced from LCOMPILER-9 investigation: inlining a callee containing an
; infinite loop can create unreachable code paths that cause LazyValueInfo to
; trace back to the entry block for a non-Argument value. This used to trip:
;   LazyValueInfo.cpp: Assertion `isa<Argument>(Val) && "Unknown live-in to the entry block"' failed.

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @test(i1 %cmp) {
entry:
  %keys = alloca [4 x float], align 16, addrspace(5)
  br i1 %cmp, label %if.else, label %common.ret

common.ret:
  ret void

if.else:
  %keys.ascast = addrspacecast ptr addrspace(5) %keys to ptr
  %0 = load float, ptr null, align 4
  call void @callee(ptr %keys.ascast, float %0)
  br label %common.ret
}

define void @callee(ptr %items, float %out_of_bounds) {
entry:
  store float %out_of_bounds, ptr %items, align 4
  br label %for.cond

for.cond:                                          ; preds = %for.cond, %entry
  %item = phi i32 [ 0, %entry ], [ %inc, %for.cond ]
  %idxprom = zext i32 %item to i64
  %arrayidx = getelementptr float, ptr %items, i64 %idxprom
  store float 0.000000e+00, ptr %arrayidx, align 4
  %inc = add i32 %item, 1
  br label %for.cond
}

