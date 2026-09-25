; RUN: rm -rf %t && split-file %s %t
; RUN: opt -passes=pgo-instr-gen -offload-pgo-branch-votes -S %t/main.ll | FileCheck %s --check-prefix=GEN
; RUN: opt -passes=pgo-instr-gen -offload-pgo-branch-votes -offload-pgo-branch-votes-function=diamond -S %t/main.ll | FileCheck %s --check-prefix=FILTER-MATCH
; RUN: opt -passes=pgo-instr-gen -offload-pgo-branch-votes -offload-pgo-branch-votes-function=other -S %t/main.ll | FileCheck %s --check-prefix=FILTER-NOMATCH
; RUN: opt -passes='pgo-instr-gen,instrprof' -offload-pgo-branch-votes -offload-pgo-branch-votes-max-sites=0 -S %t/main.ll | FileCheck %s --check-prefix=CAP-ZERO
; RUN: opt -passes='pgo-instr-gen,instrprof' -offload-pgo-branch-votes -S %t/main.ll | FileCheck %s --check-prefix=LOWER
; RUN: opt -passes=pgo-instr-gen -S %t/main.ll | FileCheck %s --check-prefix=DEFAULT
; RUN: llvm-profdata merge %t/unanimous.proftext -o %t/unanimous.profdata
; RUN: opt -passes=pgo-instr-use -pgo-test-profile-file=%t/unanimous.profdata -S %t/main.ll | FileCheck %s --check-prefix=UNANIMOUS
; RUN: llvm-profdata merge %t/divergent.proftext -o %t/divergent.profdata
; RUN: opt -passes=pgo-instr-use -pgo-test-profile-file=%t/divergent.profdata -S %t/main.ll | FileCheck %s --check-prefix=DIVERGENT
; RUN: llvm-profdata merge %t/ordinary.proftext -o %t/ordinary.profdata
; RUN: opt -passes=pgo-instr-use -pgo-test-profile-file=%t/ordinary.profdata -S %t/main.ll | FileCheck %s --check-prefix=ORDINARY

; GEN: call void @llvm.instrprof.branch.vote({{.*}}i32 4, i32 2, i1 %cond)
; FILTER-MATCH: call void @llvm.instrprof.branch.vote({{.*}}i32 4, i32 2, i1 %cond)
; FILTER-NOMATCH-NOT: call void @llvm.instrprof.branch.vote
; CAP-ZERO: @__profc_diamond = {{.*}}[4 x i64]
; CAP-ZERO-NOT: call void @__llvm_profile_instrument_gpu_branch
; LOWER: @__profc_diamond = {{.*}}[4 x i64]
; LOWER: call void @__llvm_profile_instrument_gpu_branch(ptr {{.*}}, ptr {{.*}}, i32 {{.*}})
; DEFAULT-NOT: call void @llvm.instrprof.branch.vote
; UNANIMOUS: br i1 %cond, label %a, label %b, {{.*}}!branch.unanimity.prototype ![[UNANIMOUS_MD:[0-9]+]]
; UNANIMOUS: ![[UNANIMOUS_MD]] = !{i64 1, i64 1}
; DIVERGENT: br i1 %cond, label %a, label %b, {{.*}}!branch.unanimity.prototype ![[DIVERGENT_MD:[0-9]+]]
; DIVERGENT: ![[DIVERGENT_MD]] = !{i64 1, i64 0}
; ORDINARY: br i1 %cond, label %a, label %b, !prof
; ORDINARY-NOT: !branch.unanimity.prototype

;--- main.ll
target triple = "amdgcn-amd-amdhsa"

define void @diamond(i1 %cond, ptr %p) {
entry:
  br i1 %cond, label %a, label %b, !branch.unanimity.prototype !0
a:
  store volatile i32 1, ptr %p
  br label %exit
b:
  store volatile i32 2, ptr %p
  br label %exit
exit:
  ret void
}

!0 = !{i64 9, i64 9}

;--- unanimous.proftext
:ir
diamond
# Func Hash:
146835647075900052
# Num Counters:
4
# Counter Values:
32
0
1
1

;--- divergent.proftext
:ir
diamond
# Func Hash:
146835647075900052
# Num Counters:
4
# Counter Values:
16
16
1
0

;--- ordinary.proftext
:ir
diamond
# Func Hash:
146835647075900052
# Num Counters:
2
# Counter Values:
16
16
