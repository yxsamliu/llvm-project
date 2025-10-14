; X86-64: Demonstrate hot vs cold spills in a loop branch
; This is meant as a debugging aid together with LLVM_DBG_SPILL, not a
; FileCheck-style codegen test.
;
; Pseudo-code:
;   int x = 0, y = 100, z = 200;          // loop-carried state shared by hot/cold
;   for (int i = 0; i < n; ++i) {
;     if (i % 100000 != 0) {              // hot path
;       x = hot_update(x, i);
;       y = y + i;                        // shared loop-carried
;       z = z + f_hot(x);                 // shared loop-carried
;     } else {                            // cold path
;       x = cold_update(x, i);
;       y = y - i;                        // shared loop-carried
;       z = z + f_cold(x);                // shared loop-carried
;     }
;   }
;   return x;
;
; Example run:
;   LLVM_DBG_SPILL=1 llc -mtriple=x86_64-unknown-linux-gnu %s -o - 2>&1 | less
;
; Function Attrs: noinline nounwind
define i32 @hot_cold_spill_loop(i32 %n) #0 {
entry:
  ; Handle n <= 0 quickly.
  %n_nonpos = icmp sle i32 %n, 0
  br i1 %n_nonpos, label %exit, label %loop

; Loop header: carries induction variable and loop state. x is shared,
; yh/zh are hot-specific loop-carried, yc/zc are cold-specific
; loop-carried values. The structure is similar but the actual
; state is distinct between hot and cold.
loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %x  = phi i32 [ 0,   %entry ], [ %x.next,  %loop.latch ]
  %yh = phi i32 [ 100, %entry ], [ %yh.next, %loop.latch ]
  %zh = phi i32 [ 200, %entry ], [ %zh.next, %loop.latch ]
  %yc = phi i32 [ 100, %entry ], [ %yc.next, %loop.latch ]
  %zc = phi i32 [ 200, %entry ], [ %zc.next, %loop.latch ]

  ; Canonical loop condition.
  %cmp = icmp slt i32 %iv, %n
  br i1 %cmp, label %body, label %exit, !prof !0

; Loop body with hot/cold if-else based on iv % 100000.
body:
  %mod = srem i32 %iv, 100000
  %is_hot = icmp ne i32 %mod, 0
  br i1 %is_hot, label %hot, label %cold, !prof !1

; Hot branch: only loop-carried values (x, iv, yh, zh) live here.
hot:
  ; Compute several temporaries from the loop-carried values to
  ; create real dataflow-based register pressure.
  %hx0 = add i32 %x, %iv
  %hx1 = add i32 %hx0, %yh
  %hx2 = add i32 %hx1, %zh
  %hx3 = xor i32 %hx2, %x
  ; Apply clobbers (excluding rax, r14, r15) so both branches
  ; see similar register pressure characteristics.
  call void asm sideeffect "", "~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11}"()
  ; Update hot-specific loop-carried state using the temporaries.
  %x.hot  = add i32 %x, %hx2
  %yh.hot = add i32 %yh, %iv
  %zh.hot = add i32 %zh, %hx3
  br label %loop.latch

; Cold branch: only loop-carried values (x, iv, yc, zc) live here.
cold:
  ; Compute several temporaries from the loop-carried values to
  ; create real dataflow-based register pressure.
  %cx0 = sub i32 %x, %iv
  %cx1 = add i32 %cx0, %yc
  %cx2 = add i32 %cx1, %zc
  %cx3 = xor i32 %cx2, %x
  ; Constrain most registers here to push spills to the cold path, but leave
  ; rax and several GPRs (including r12, r13, r14, r15) available across the clobber.
  call void asm sideeffect "", "~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11}"()
  ; Update cold-specific loop-carried state using the temporaries.
  %x.cold  = sub i32 %x, %cx2
  %yc.cold = sub i32 %yc, 1
  %zc.cold = add i32 %zc, %cx3
  br label %loop.latch

; Loop latch merges hot/cold updates to x and increments iv.
loop.latch:
  %x.next  = phi i32 [ %x.hot,  %hot ], [ %x.cold,  %cold ]
  %yh.next = phi i32 [ %yh.hot, %hot ], [ %yh,      %cold ]
  %zh.next = phi i32 [ %zh.hot, %hot ], [ %zh,      %cold ]
  %yc.next = phi i32 [ %yc,     %hot ], [ %yc.cold, %cold ]
  %zc.next = phi i32 [ %zc,     %hot ], [ %zc.cold, %cold ]
  %iv.next = add i32 %iv, 1
  br label %loop

; Exit: from entry (n <= 0) or from loop when iv >= n.
exit:
  %x.exit = phi i32 [ 0, %entry ], [ %x, %loop ]
  ret i32 %x.exit
}

attributes #0 = { noinline nounwind }

!0 = !{!"branch_weights", i32 1000000, i32 1}
!1 = !{!"branch_weights", i32 1000000, i32 1}
