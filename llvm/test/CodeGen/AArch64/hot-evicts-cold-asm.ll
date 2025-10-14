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
  ; Constrain many allocatable GPRs in the hot loop to raise baseline pressure.
  call void asm sideeffect "", "~{w8},~{w9},~{w10},~{w11},~{w12},~{w13},~{w14},~{w15},~{w16},~{w17},~{w18},~{w19},~{w20},~{w21},~{w22},~{w23},~{w24},~{w25},~{w26},~{w27},~{w28},~{w29},~{w30}"()
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
  %c16 = add i32 %n, 17
  %c17 = add i32 %n, 18
  %c18 = add i32 %n, 19
  %c19 = add i32 %n, 20
  %c20 = add i32 %n, 21
  %c21 = add i32 %n, 22
  %c22 = add i32 %n, 23
  %c23 = add i32 %n, 24
  %c24 = add i32 %n, 25
  %c25 = add i32 %n, 26
  %c26 = add i32 %n, 27
  %c27 = add i32 %n, 28
  %c28 = add i32 %n, 29
  %c29 = add i32 %n, 30
  %c30 = add i32 %n, 31
  %c31 = add i32 %n, 32
  br label %join

join:
  %x = phi i32 [ %hot, %loop ], [ %c0, %cold ]
  ; Merge cold values from both paths (undef from hot path since they're not used there)
  %c0_join = phi i32 [ undef, %loop ], [ %c0, %cold ]
  %c1_join = phi i32 [ undef, %loop ], [ %c1, %cold ]
  %c2_join = phi i32 [ undef, %loop ], [ %c2, %cold ]
  %c3_join = phi i32 [ undef, %loop ], [ %c3, %cold ]
  %c4_join = phi i32 [ undef, %loop ], [ %c4, %cold ]
  %c5_join = phi i32 [ undef, %loop ], [ %c5, %cold ]
  %c6_join = phi i32 [ undef, %loop ], [ %c6, %cold ]
  %c7_join = phi i32 [ undef, %loop ], [ %c7, %cold ]
  %c8_join = phi i32 [ undef, %loop ], [ %c8, %cold ]
  %c9_join = phi i32 [ undef, %loop ], [ %c9, %cold ]
  %c10_join = phi i32 [ undef, %loop ], [ %c10, %cold ]
  %c11_join = phi i32 [ undef, %loop ], [ %c11, %cold ]
  %c12_join = phi i32 [ undef, %loop ], [ %c12, %cold ]
  %c13_join = phi i32 [ undef, %loop ], [ %c13, %cold ]
  %c14_join = phi i32 [ undef, %loop ], [ %c14, %cold ]
  %c15_join = phi i32 [ undef, %loop ], [ %c15, %cold ]
  %c16_join = phi i32 [ undef, %loop ], [ %c16, %cold ]
  %c17_join = phi i32 [ undef, %loop ], [ %c17, %cold ]
  %c18_join = phi i32 [ undef, %loop ], [ %c18, %cold ]
  %c19_join = phi i32 [ undef, %loop ], [ %c19, %cold ]
  %c20_join = phi i32 [ undef, %loop ], [ %c20, %cold ]
  %c21_join = phi i32 [ undef, %loop ], [ %c21, %cold ]
  %c22_join = phi i32 [ undef, %loop ], [ %c22, %cold ]
  %c23_join = phi i32 [ undef, %loop ], [ %c23, %cold ]
  %c24_join = phi i32 [ undef, %loop ], [ %c24, %cold ]
  %c25_join = phi i32 [ undef, %loop ], [ %c25, %cold ]
  %c26_join = phi i32 [ undef, %loop ], [ %c26, %cold ]
  %c27_join = phi i32 [ undef, %loop ], [ %c27, %cold ]
  %c28_join = phi i32 [ undef, %loop ], [ %c28, %cold ]
  %c29_join = phi i32 [ undef, %loop ], [ %c29, %cold ]
  %c30_join = phi i32 [ undef, %loop ], [ %c30, %cold ]
  %c31_join = phi i32 [ undef, %loop ], [ %c31, %cold ]
  ; Force many regs at once; include the hot value first so it tends to be kept
  call void asm sideeffect "", "r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r"(
    i32 %x,
    i32 %c1_join,  i32 %c2_join,  i32 %c3_join,  i32 %c4_join,  i32 %c5_join,  i32 %c6_join,  i32 %c7_join,
    i32 %c8_join,  i32 %c9_join,  i32 %c10_join, i32 %c11_join, i32 %c12_join, i32 %c13_join, i32 %c14_join, i32 %c15_join,
    i32 %c16_join, i32 %c17_join, i32 %c18_join, i32 %c19_join, i32 %c20_join, i32 %c21_join, i32 %c22_join, i32 %c23_join,
    i32 %c24_join, i32 %c25_join, i32 %c26_join, i32 %c27_join, i32 %c28_join, i32 %c29_join, i32 %c30_join, i32 %c31_join,
    i32 %c0_join)
  ret i32 %x
}

attributes #0 = { noinline nounwind }

!0 = !{!"branch_weights", i32 1000000, i32 1}
!1 = !{!"branch_weights", i32 1000000, i32 1}
