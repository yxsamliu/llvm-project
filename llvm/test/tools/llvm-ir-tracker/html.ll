; RUN: opt -disable-output -passes=no-op-module -ir-tracker-json-output=%t.jsonl %s
; RUN: %ir-tracker build --input %t.jsonl --db %t.db
; RUN: rm -rf %t.html
; RUN: %ir-tracker html --db %t.db -o %t.html --no-highlight | FileCheck %s --check-prefix=LOG
; RUN: ls %t.html | FileCheck %s --check-prefix=FILES
; RUN: FileCheck %s --check-prefix=INDEX --input-file=%t.html/index.html
; RUN: FileCheck %s --check-prefix=PAGE --input-file=%t.html/show.c.html

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

; LOG: ir-tracker: wrote 1 file page(s) + index

; FILES-DAG: index.html
; FILES-DAG: show.c.html
; FILES-DAG: style.css

; INDEX: ir-tracker report
; INDEX: show.c.html
; INDEX: show.c

; PAGE: <title>show.c</title>
; PAGE: trackerToggle('p8')
; PAGE: seq=0
; PAGE: function f, block entry
; PAGE: %a = add i32 %x, 1
