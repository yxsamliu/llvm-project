; RUN: opt -disable-output -passes=no-op-module -ir-tracker-database=%t.db %s
; RUN: %ir-tracker passes --db %t.db | FileCheck %s --check-prefix=PASSES
; RUN: %ir-tracker trace --db %t.db --file show.c --line 8 | FileCheck %s --check-prefix=TRACE
; RUN: %ir-tracker show --db %t.db --file show.c --line 8 --seq 0 | FileCheck %s --check-prefix=SEQ0
; RUN: %ir-tracker show --db %t.db --file show.c --line 8 | FileCheck %s --check-prefix=CHANGED
; RUN: %ir-tracker show --db %t.db --file show.c --line 8 --all-passes | FileCheck %s --check-prefix=ALL

define i32 @f(i32 %x) !dbg !6 {
entry:
  %a = add i32 %x, 1, !dbg !8
  ret i32 %a, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "ir-tracker-test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "show.c", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!4 = !DISubroutineType(types: !5)
!5 = !{!3, !3}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 7, type: !4, scopeLine: 7, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 8, column: 3, scope: !6)
!9 = !DILocation(line: 9, column: 3, scope: !6)

; PASSES: 0  id={{[0-9]+}}  initial  '<initial>'  on '[module]'
; PASSES: 1  id={{[0-9]+}}  after  'no-op-module'  on '[module]'
; PASSES: total passes recorded: 2

; TRACE: Matches at final pass (seq=1): 1 instruction(s)
; TRACE: First pass with any matching instruction: seq=0 <initial> on [module] (1 row(s))

; SEQ0: seq=0 '<initial>' on '[module]'
; SEQ0-NEXT:   function f, block entry:
; SEQ0-NEXT:     %a = add i32 %x, 1, !dbg !{{[0-9]+}}

; CHANGED: seq=0 '<initial>' on '[module]'
; CHANGED-NEXT:   function f, block entry:
; CHANGED-NEXT:     %a = add i32 %x, 1, !dbg !{{[0-9]+}}
; CHANGED-NOT: seq=1

; ALL: seq=0 '<initial>' on '[module]'
; ALL-NEXT:   function f, block entry:
; ALL-NEXT:     %a = add i32 %x, 1, !dbg !{{[0-9]+}}
; ALL: seq=1 'no-op-module' on '[module]'
; ALL-NEXT:   function f, block entry:
; ALL-NEXT:     %a = add i32 %x, 1, !dbg !{{[0-9]+}}
