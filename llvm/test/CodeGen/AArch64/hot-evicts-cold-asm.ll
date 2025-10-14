; AArch64: Hot vreg should evict a cold vreg at a high-pressure join
; Use many "r" asm inputs to force simultaneous register assignment.
; Run:
;   LLVM_DBG_SPILL=1 llc -mtriple=arm64-apple-ios7.0 -aarch64-neon-syntax=apple %s -o - 2>&1 | less

; Function Attrs: noinline nounwind
define i32 @hot_evicts_cold_asm(i32 %n) #0 {
entry:
  %go_hot = icmp ne i32 %n, 0
  br i1 %go_hot, label %loop.preheader, label %cold, !prof !0

loop.preheader:
  br label %loop

loop:
  %iv = phi i32 [ 0, %loop.preheader ], [ %iv.next, %loop ]
  %hot = add i32 %iv, 1
  %iv.next = add i32 %iv, 1
  %more = icmp slt i32 %iv.next, %n
  br i1 %more, label %loop, label %join, !prof !1

cold:
  ; Build a bunch of cold values kept alive to the join
  %c0 = add i32 %n, 1
  %c1 = add i32 %n, 2
  %c2 = add i32 %n, 3
  %c3 = add i32 %n, 4
  %c4 = add i32 %n, 5
  %c5 = add i32 %n, 6
  %c6 = add i32 %n, 7
  %c7 = add i32 %n, 8
  %c8 = add i32 %n, 9
  %c9 = add i32 %n, 10
  %c10 = add i32 %n, 11
  %c11 = add i32 %n, 12
  %c12 = add i32 %n, 13
  %c13 = add i32 %n, 14
  %c14 = add i32 %n, 15
  %c15 = add i32 %n, 16
  br label %join

join:
  %x = phi i32 [ %hot, %loop ], [ %c0, %cold ]
  ; Force many regs at once; include the hot value first so it tends to be kept
  call void asm sideeffect "", "r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r"(
    i32 %x, i32 %c1, i32 %c2, i32 %c3, i32 %c4, i32 %c5, i32 %c6, i32 %c7,
    i32 %c8, i32 %c9, i32 %c10, i32 %c11, i32 %c12, i32 %c13, i32 %c14, i32 %c15, i32 %c0)
  ret i32 %x
}

attributes #0 = { noinline nounwind }

!0 = !{!"branch_weights", i32 1000000, i32 1}
!1 = !{!"branch_weights", i32 1000000, i32 1}
