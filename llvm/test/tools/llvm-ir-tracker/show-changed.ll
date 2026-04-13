; RUN: opt -disable-output -passes='no-op-module,instcombine' -ir-tracker-database=%t.db --add-ir-tracker-locs-force %s
; RUN: %ir-tracker show --db %t.db --file %s --line 7 --kind ir | FileCheck %s --check-prefix=CHANGED
; RUN: %ir-tracker show --db %t.db --file %s --line 7 --kind ir --all-passes | FileCheck %s --check-prefix=ALLPASSES

define i32 @f(i32 %x) {
entry:
  %a = add i32 %x, %x
  ret i32 %a
}

; With --add-ir-tracker-locs-force, the add maps to source line 7 (blank + RUN lines above).
; Default show skips consecutive passes with identical inst_text; all-passes lists every seq.
; CHANGED-LABEL: seq=0 '<initial>' on '[module]'
; CHANGED: add i32 %x, %x
; CHANGED-LABEL: seq=
; CHANGED-SAME: {{'instcombine'}}
; CHANGED: shl i32 %x, 1

; ALLPASSES-LABEL: seq=0 '<initial>' on '[module]'
; ALLPASSES: add i32 %x, %x
; ALLPASSES-LABEL: seq=
; ALLPASSES-SAME: {{'no-op-module'}}
; ALLPASSES: add i32 %x, %x
; ALLPASSES-LABEL: seq=
; ALLPASSES-SAME: {{'instcombine'}}
; ALLPASSES: shl i32 %x, 1
