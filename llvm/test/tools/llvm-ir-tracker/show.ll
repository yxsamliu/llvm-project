; RUN: opt -disable-output -passes=no-op-module -ir-tracker-database=%t.db --add-ir-tracker-locs-force %s
; RUN: %ir-tracker trace --db %t.db --file %s --line 11 --kind ir | FileCheck %s --check-prefix=TRACE
; RUN: %ir-tracker trace --db %t.db --file %s --line 11 | FileCheck %s --check-prefix=TRACE-DEFAULT
; RUN: %ir-tracker show --db %t.db --file %s --line 11 --kind ir --seq 0 | FileCheck %s --check-prefix=SEQ0
; RUN: %ir-tracker show --db %t.db --file %s --line 11 --kind ir | FileCheck %s --check-prefix=CHANGED-DEFAULT
; RUN: %ir-tracker show --db %t.db --file %s --line 11 | FileCheck %s --check-prefix=SHOW-DEFAULT
; RUN: %ir-tracker show --db %t.db --file %s --line 11 --kind ir --all-passes | FileCheck %s --check-prefix=ALL

define i32 @f(i32 %x) {
entry:
  %a = add i32 %x, 1
  ret i32 %a
}

; TRACE: First pass with any matching instruction: seq=0 <initial> on [module] (1 row(s))

; TRACE-DEFAULT: kind=ir:
; TRACE-DEFAULT: Matches at final pass (seq={{[0-9]+}}): 1 instruction(s) (file id(s) 1, line 11)
; TRACE-DEFAULT: First pass with any matching instruction: seq=0 <initial> on [module] (1 row(s))

; SEQ0: seq=0 '<initial>' on '[module]'
; SEQ0-NEXT:   function f, block entry:
; SEQ0-NEXT:     {{\[[0-9]+\]}} %a = add i32 %x, 1{{.*}}

; CHANGED-DEFAULT: seq=0 '<initial>' on '[module]'
; CHANGED-DEFAULT-NEXT:   function f, block entry:
; CHANGED-DEFAULT-NEXT:     {{\[[0-9]+\]}} %a = add i32 %x, 1{{.*}}

; SHOW-DEFAULT: seq=0 '<initial>' on '[module]' kind=ir
; SHOW-DEFAULT-NEXT:   function f, block entry:
; SHOW-DEFAULT-NEXT:     {{\[[0-9]+\]}} %a = add i32 %x, 1{{.*}}

; ALL: seq=0 '<initial>' on '[module]'
; ALL-NEXT:   function f, block entry:
; ALL-NEXT:     {{\[[0-9]+\]}} %a = add i32 %x, 1{{.*}}
; ALL: seq=1 '{{.*}}' on '[module]'
; ALL-NEXT:   function f, block entry:
; ALL-NEXT:     {{\[[0-9]+\]}} %a = add i32 %x, 1{{.*}}
