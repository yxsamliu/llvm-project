; X86-64: Simple accumulator loop for regalloc exploration
; This is intended as a minimal, easy-to-reason-about test for studying
; how the greedy register allocator works (live intervals, spill weights,
; splitting, etc.), similar in spirit to hot-evicts-cold-asm.ll but with
; a much simpler control-flow shape.
;
; Pseudo-code:
;   int simple_accum_loop(int n) {
;     int acc = 0;
;     for (int i = 0; i < n; ++i)
;       acc = acc * 3 + i;
;     return acc;
;   }
;
; Structure:
;   - single canonical loop
;   - loop-carried state: induction variable i and accumulator acc
;   - the final value of acc is returned
;
; How to experiment with register allocation:
;   LLVM_DBG_SPILL=1 llc -mtriple=x86_64-unknown-linux-gnu %s -debug-only=regalloc -o /dev/null 2>&1 | less
;
; This test does not use FileCheck; the RUN line only ensures llc can
; compile the IR. The interesting content is in the regalloc debug logs.
;
; RUN: llc -mtriple=x86_64-unknown-linux-gnu %s -o /dev/null

define i32 @simple_accum_loop(i32 %n) #0 {
entry:
  ; Fast-exit for n <= 0.
  %n_nonpos = icmp sle i32 %n, 0
  br i1 %n_nonpos, label %exit, label %loop.preheader

loop.preheader:
  br label %loop

; Canonical loop header: induction variable iv and accumulator acc.
loop:
  %iv  = phi i32 [ 0,          %loop.preheader ],
                [ %iv.next,    %body ]
  %acc = phi i32 [ 0,          %loop.preheader ],
                [ %acc.next,   %body ]

  ; Canonical loop condition iv < n.
  %cmp = icmp slt i32 %iv, %n
  br i1 %cmp, label %body, label %exit

body:
  ; Simple arithmetic on the accumulator to give it some non-trivial uses.
  %mul      = mul nsw i32 %acc, 3
  %acc.next = add nsw i32 %mul, %iv
  %iv.next  = add nsw i32 %iv, 1
  br label %loop

; Exit: from entry (n <= 0) or from loop when iv >= n.
exit:
  ; If we exited early from entry, the accumulator is 0.
  %acc.exit = phi i32 [ 0,   %entry ],
                     [ %acc, %loop ]
  ret i32 %acc.exit
}

attributes #0 = { noinline nounwind }

