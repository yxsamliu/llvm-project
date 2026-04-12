; RUN: opt -disable-output -passes=no-op-module -ir-tracker-database=%t.db --add-ir-tracker-locs-force %s
; RUN: llvm-ir-tracker trace --db %t.db --file %s --line 9 | FileCheck %s --check-prefix=TRACE
; RUN: llvm-ir-tracker show --db %t.db --file %s --line 9 --seq 0 | FileCheck %s --check-prefix=SEQ0
; RUN: llvm-ir-tracker show --db %t.db --file %s --line 9 | FileCheck %s --check-prefix=FINAL
; RUN: llvm-ir-tracker show --db %t.db --file %s --line 9 --all-passes | FileCheck %s --check-prefix=ALL

define i32 @f(i32 %x) {
entry:
  %a = add i32 %x, 1
  ret i32 %a
}

; TRACE: First pass with any matching instruction: seq=0 <initial> on [module] (1 row(s))

; SEQ0: seq=0 '<initial>' on '[module]'
; SEQ0-NEXT:   function f, block entry:
; SEQ0-NEXT:     %a = add i32 %x, 1{{.*}}

; FINAL: seq=1 '{{.*}}' on '[module]'
; FINAL-NEXT:   function f, block entry:
; FINAL-NEXT:     %a = add i32 %x, 1{{.*}}

; ALL: seq=0 '<initial>' on '[module]'
; ALL-NEXT:   function f, block entry:
; ALL-NEXT:     %a = add i32 %x, 1{{.*}}
; ALL: seq=1 '{{.*}}' on '[module]'
; ALL-NEXT:   function f, block entry:
; ALL-NEXT:     %a = add i32 %x, 1{{.*}}
