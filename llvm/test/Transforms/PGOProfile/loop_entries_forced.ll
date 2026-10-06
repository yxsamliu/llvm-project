; RUN: split-file %s %t
; RUN: opt %t/main.ll -passes=pgo-instr-gen -pgo-instrument-loop-entries -S | FileCheck %s --check-prefixes=GEN,LOOP --implicit-check-not="call void @llvm.instrprof.increment"
; RUN: opt %t/main.ll -passes=pgo-instr-gen -pgo-instrument-loop-entries -pgo-instrument-entry -S | FileCheck %s --check-prefixes=GEN,BOTH --implicit-check-not="call void @llvm.instrprof.increment"
; RUN: llvm-profdata merge %t/loop.proftext -o %t/loop.profdata
; RUN: opt %t/main.ll -passes=pgo-instr-use -pgo-test-profile-file=%t/loop.profdata -S | FileCheck %s --check-prefix=USE
; RUN: llvm-profdata merge %t/both.proftext -o %t/both.profdata
; RUN: opt %t/main.ll -passes=pgo-instr-use -pgo-test-profile-file=%t/both.profdata -S | FileCheck %s --check-prefix=USE

; Forced edges can disconnect the graph. Cover function entry immediately
; preceding a loop, a loop without an exit, and sequential and nested loops.
; The function-entry and loop-entry edges can require counters in the same BB.

; GEN-LABEL: define void @loop(
; GEN: entry:
; LOOP: call void @llvm.instrprof.increment(ptr @__profn_loop, i64 {{[0-9]+}}, i32 2, i32 1)
; BOTH: call void @llvm.instrprof.increment(ptr @__profn_loop, i64 {{[0-9]+}}, i32 3, i32 2)
; BOTH: call void @llvm.instrprof.increment(ptr @__profn_loop, i64 {{[0-9]+}}, i32 3, i32 0)
; GEN: header:
; GEN: body:
; LOOP: call void @llvm.instrprof.increment(ptr @__profn_loop, i64 {{[0-9]+}}, i32 2, i32 0)
; BOTH: call void @llvm.instrprof.increment(ptr @__profn_loop, i64 {{[0-9]+}}, i32 3, i32 1)
; GEN: exit:
; GEN-LABEL: define void @noexit(
; GEN: entry:
; LOOP: call void @llvm.instrprof.increment(ptr @__profn_noexit, i64 {{[0-9]+}}, i32 3, i32 2)
; LOOP: call void @llvm.instrprof.increment(ptr @__profn_noexit, i64 {{[0-9]+}}, i32 3, i32 1)
; BOTH: call void @llvm.instrprof.increment(ptr @__profn_noexit, i64 {{[0-9]+}}, i32 3, i32 2)
; BOTH: call void @llvm.instrprof.increment(ptr @__profn_noexit, i64 {{[0-9]+}}, i32 3, i32 0)
; GEN: header:
; GEN: body:
; LOOP: call void @llvm.instrprof.increment(ptr @__profn_noexit, i64 {{[0-9]+}}, i32 3, i32 0)
; BOTH: call void @llvm.instrprof.increment(ptr @__profn_noexit, i64 {{[0-9]+}}, i32 3, i32 1)
; GEN-LABEL: define void @sequential(
; GEN: entry:
; LOOP: call void @llvm.instrprof.increment(ptr @__profn_sequential, i64 {{[0-9]+}}, i32 4, i32 2)
; BOTH: call void @llvm.instrprof.increment(ptr @__profn_sequential, i64 {{[0-9]+}}, i32 5, i32 2)
; BOTH: call void @llvm.instrprof.increment(ptr @__profn_sequential, i64 {{[0-9]+}}, i32 5, i32 0)
; GEN: first:
; GEN: first.first_crit_edge:
; LOOP: call void @llvm.instrprof.increment(ptr @__profn_sequential, i64 {{[0-9]+}}, i32 4, i32 0)
; BOTH: call void @llvm.instrprof.increment(ptr @__profn_sequential, i64 {{[0-9]+}}, i32 5, i32 4)
; GEN: between:
; LOOP: call void @llvm.instrprof.increment(ptr @__profn_sequential, i64 {{[0-9]+}}, i32 4, i32 3)
; BOTH: call void @llvm.instrprof.increment(ptr @__profn_sequential, i64 {{[0-9]+}}, i32 5, i32 3)
; GEN: second:
; GEN: second.second_crit_edge:
; LOOP: call void @llvm.instrprof.increment(ptr @__profn_sequential, i64 {{[0-9]+}}, i32 4, i32 1)
; BOTH: call void @llvm.instrprof.increment(ptr @__profn_sequential, i64 {{[0-9]+}}, i32 5, i32 1)
; GEN: exit:
; GEN-LABEL: define void @nested(
; GEN: entry:
; LOOP: call void @llvm.instrprof.increment(ptr @__profn_nested, i64 {{[0-9]+}}, i32 3, i32 1)
; BOTH: call void @llvm.instrprof.increment(ptr @__profn_nested, i64 {{[0-9]+}}, i32 4, i32 1)
; BOTH: call void @llvm.instrprof.increment(ptr @__profn_nested, i64 {{[0-9]+}}, i32 4, i32 0)
; GEN: outer:
; LOOP: call void @llvm.instrprof.increment(ptr @__profn_nested, i64 {{[0-9]+}}, i32 3, i32 2)
; BOTH: call void @llvm.instrprof.increment(ptr @__profn_nested, i64 {{[0-9]+}}, i32 4, i32 2)
; GEN: inner:
; GEN: inner.inner_crit_edge:
; LOOP: call void @llvm.instrprof.increment(ptr @__profn_nested, i64 {{[0-9]+}}, i32 3, i32 0)
; BOTH: call void @llvm.instrprof.increment(ptr @__profn_nested, i64 {{[0-9]+}}, i32 4, i32 3)
; GEN: latch:
; GEN: exit:

; USE-LABEL: define void @loop(
; USE-SAME: !prof ![[ENTRY:[0-9]+]]
; USE: br i1 %done, label %exit, label %body, !prof ![[SIMPLE:[0-9]+]]
; USE-LABEL: define void @noexit(
; USE-LABEL: define void @sequential(
; USE-SAME: !prof ![[ENTRY3:[0-9]+]]
; USE: br i1 %done, label %between, label %first.first_crit_edge, !prof ![[FIRST:[0-9]+]]
; USE: br i1 %donej, label %exit, label %second.second_crit_edge, !prof ![[SECOND:[0-9]+]]
; USE-LABEL: define void @nested(
; USE-SAME: !prof ![[ENTRY3]]
; USE: br i1 %donej, label %latch, label %inner.inner_crit_edge, !prof ![[INNER:[0-9]+]]
; USE: br i1 %donei, label %exit, label %outer, !prof ![[FIRST]]
; USE-DAG: ![[ENTRY]] = !{!"function_entry_count", i64 5}
; USE-DAG: ![[ENTRY3]] = !{!"function_entry_count", i64 3}
; USE-DAG: ![[SIMPLE]] = !{!"branch_weights", i32 5, i32 25}
; USE-DAG: ![[FIRST]] = !{!"branch_weights", i32 3, i32 9}
; USE-DAG: ![[SECOND]] = !{!"branch_weights", i32 3, i32 18}
; USE-DAG: ![[INNER]] = !{!"branch_weights", i32 12, i32 48}

;--- main.ll
define void @loop(ptr %p, i32 %n) {
entry:
  br label %header
header:
  %i = phi i32 [0, %entry], [%next, %body]
  %done = icmp eq i32 %i, %n
  br i1 %done, label %exit, label %body
body:
  store volatile i32 %i, ptr %p
  %next = add i32 %i, 1
  br label %header
exit:
  ret void
}

define void @noexit(ptr %p) {
entry:
  br label %header
header:
  %x = load volatile i32, ptr %p
  br label %body
body:
  store volatile i32 %x, ptr %p
  br label %header
}

define void @sequential(ptr %p, i32 %n) {
entry:
  br label %first
first:
  %i = phi i32 [0, %entry], [%next, %first]
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %between, label %first
between:
  br label %second
second:
  %j = phi i32 [0, %between], [%nextj, %second]
  store volatile i32 %j, ptr %p
  %nextj = add i32 %j, 1
  %donej = icmp eq i32 %nextj, %n
  br i1 %donej, label %exit, label %second
exit:
  ret void
}

define void @nested(ptr %p, i32 %n) {
entry:
  br label %outer
outer:
  %i = phi i32 [0, %entry], [%nexti, %latch]
  br label %inner
inner:
  %j = phi i32 [0, %outer], [%nextj, %inner]
  store volatile i32 %j, ptr %p
  %nextj = add i32 %j, 1
  %donej = icmp eq i32 %nextj, %n
  br i1 %donej, label %latch, label %inner
latch:
  %nexti = add i32 %i, 1
  %donei = icmp eq i32 %nexti, %n
  br i1 %donei, label %exit, label %outer
exit:
  ret void
}

;--- loop.proftext
:ir
:instrument_loop_entries

loop
146835646621254984
2
25
5

noexit
444724187393124455
3
17
3
3

sequential
238984482353853237
4
9
18
3
3

nested
238984482393877634
3
48
3
12

;--- both.proftext
:ir
:entry_first
:instrument_loop_entries

loop
146835646621254984
3
5
25
5

noexit
444724187393124455
3
3
17
3

sequential
238984482353853237
5
3
18
3
3
9

nested
238984482393877634
4
3
3
12
48
