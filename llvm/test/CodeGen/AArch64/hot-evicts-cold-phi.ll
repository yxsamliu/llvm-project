; AArch64: Try to create a case where a hot vreg evicts a cold vreg
; Hot loop feeds %h, many cold-only values feed phis at join. Join applies
; heavy parallel "r" uses to raise pressure.
;
; Run:
;   LLVM_DBG_SPILL=1 llc -mtriple=arm64-apple-ios7.0 -aarch64-neon-syntax=apple %s -o - 2>&1 | less

; Function Attrs: noinline nounwind
define i32 @hot_evicts_cold_phi(i32 %n) #0 {
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
  br label %join

join:
  %h = phi i32 [ %hot, %loop ], [ %n, %cold ]
  %p0 = phi i32 [ poison, %loop ], [ %c0, %cold ]
  %p1 = phi i32 [ poison, %loop ], [ %c1, %cold ]
  %p2 = phi i32 [ poison, %loop ], [ %c2, %cold ]
  %p3 = phi i32 [ poison, %loop ], [ %c3, %cold ]
  %p4 = phi i32 [ poison, %loop ], [ %c4, %cold ]
  %p5 = phi i32 [ poison, %loop ], [ %c5, %cold ]
  %p6 = phi i32 [ poison, %loop ], [ %c6, %cold ]
  %p7 = phi i32 [ poison, %loop ], [ %c7, %cold ]
  %p8 = phi i32 [ poison, %loop ], [ %c8, %cold ]
  %p9 = phi i32 [ poison, %loop ], [ %c9, %cold ]
  %p10 = phi i32 [ poison, %loop ], [ %c10, %cold ]
  %p11 = phi i32 [ poison, %loop ], [ %c11, %cold ]
  %p12 = phi i32 [ poison, %loop ], [ %c12, %cold ]
  ; Consume many at once to maximize simultaneous reg needs.
  call void asm sideeffect "", "r,r,r,r,r,r,r,r,r,r,r,r,r"(
    i32 %h, i32 %p0, i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5,
    i32 %p6, i32 %p7, i32 %p8, i32 %p9, i32 %p10, i32 %p11)
  ret i32 %h
}

attributes #0 = { noinline nounwind }

!0 = !{!"branch_weights", i32 1000000, i32 1}
!1 = !{!"branch_weights", i32 1000000, i32 1}
