; AArch64: Demonstrate that a hot virtual register is favored over a cold one
; by weighting blocks with branch_weights and constraining available physregs
; with an inline asm regmask. Run with the spill debug env var to observe
; allocator decisions.
;
; Example run:
;   LLVM_DBG_SPILL=1 llc -mtriple=aarch64-apple-ios7.0 -aarch64-neon-syntax=apple \
;     %s -o - 2>&1 | less
;
; The hot loop carries %hot across many iterations (high frequency).
; The cold path defines %cold just once. At the join, both are live and
; contend for the constrained GPRs, causing the allocator to prefer keeping
; the hot value in a register and spilling/splitting the cold one.

; Function Attrs: noinline nounwind
define i32 @hot_eviction(i32 %n) #0 {
entry:
  ; Bias control flow: hot loop (weight 1000000) vs cold path (weight 1)
  %go_hot = icmp ne i32 %n, 0
  br i1 %go_hot, label %loop.preheader, label %cold, !prof !0

loop.preheader:
  br label %loop

loop:                                             ; preds = %loop, %loop.preheader
  %iv = phi i32 [ 0, %loop.preheader ], [ %iv.next, %loop ]
  %hot = add i32 %iv, 1

  ; Constrain many allocatable GPRs to increase pressure.
  call void asm sideeffect "", "~{w8},~{w9},~{w10},~{w11},~{w12},~{w13},~{w14},~{w15},~{w16},~{w17},~{w19},~{w20},~{w21},~{w22},~{w23},~{w24},~{w25},~{w26},~{w27},~{w28},~{w30}"()

  %iv.next = add i32 %iv, 1
  %more = icmp slt i32 %iv.next, %n
  br i1 %more, label %loop, label %join, !prof !1

cold:                                             ; preds = %entry
  ; Single definition on a very cold path
  %coldv = add i32 %n, 42
  br label %join

join:                                             ; preds = %cold, %loop
  ; Both %hot (from loop) and %coldv (from cold) are live-in here
  %x = phi i32 [ %hot, %loop ], [ %coldv, %cold ]
  ret i32 %x
}

attributes #0 = { noinline nounwind }

!0 = !{!"branch_weights", i32 1000000, i32 1}
!1 = !{!"branch_weights", i32 1000000, i32 1}
