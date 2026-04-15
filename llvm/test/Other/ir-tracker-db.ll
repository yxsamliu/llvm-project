; RUN: opt -disable-output -passes=instcombine -ir-tracker-json-output=%t.jsonl %s
; RUN: FileCheck %s --input-file=%t.jsonl --check-prefix=ALL
; RUN: opt -disable-output -passes=instcombine -filter-print-funcs=f -ir-tracker-json-output=%t-filter.jsonl %s
; RUN: FileCheck %s --input-file=%t-filter.jsonl --check-prefix=FILTER

define i32 @f(i32 %x) !dbg !6 {
entry:
  %add = add i32 %x, 1, !dbg !8
  ret i32 %add, !dbg !9
}

define i32 @g(i32 %x) !dbg !7 {
entry:
  %mul = mul i32 %x, 2, !dbg !10
  ret i32 %mul, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "ir-tracker-test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "ir-tracker.c", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!4 = !DISubroutineType(types: !5)
!5 = !{!3, !3}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 7, type: !4, scopeLine: 7, spFlags: DISPFlagDefinition, unit: !0)
!7 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 13, type: !4, scopeLine: 13, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 8, column: 3, scope: !6)
!9 = !DILocation(line: 9, column: 3, scope: !6)
!10 = !DILocation(line: 14, column: 3, scope: !7)
!11 = !DILocation(line: 15, column: 3, scope: !7)

; ALL: {"kind":"pass","phase":"initial","pass":"<initial>","ir_unit":"[module]","seq":0}
; ALL-NEXT: {"block":"entry","col":3,"file":"/tmp/ir-tracker.c","function":"f","inst_seq":0,"kind":"inst","line":8,"opcode":"add","text":"  %add = add i32 %x, 1, !dbg !{{[0-9]+}}"}
; ALL-NEXT: {"block":"entry","col":3,"file":"/tmp/ir-tracker.c","function":"f","inst_seq":1,"kind":"inst","line":9,"opcode":"ret","text":"  ret i32 %add, !dbg !{{[0-9]+}}"}
; ALL-NEXT: {"block":"entry","col":3,"file":"/tmp/ir-tracker.c","function":"g","inst_seq":0,"kind":"inst","line":14,"opcode":"mul","text":"  %mul = mul i32 %x, 2, !dbg !{{[0-9]+}}"}
; ALL-NEXT: {"block":"entry","col":3,"file":"/tmp/ir-tracker.c","function":"g","inst_seq":1,"kind":"inst","line":15,"opcode":"ret","text":"  ret i32 %mul, !dbg !{{[0-9]+}}"}
; ALL: {"kind":"pass","phase":"after","pass":"instcombine","ir_unit":"f","seq":1}
; ALL-NEXT: {"block":"entry","col":3,"file":"/tmp/ir-tracker.c","function":"f","inst_seq":0,"kind":"inst","line":8,"opcode":"add","text":"  %add = add i32 %x, 1, !dbg !{{[0-9]+}}"}
; ALL-NEXT: {"block":"entry","col":3,"file":"/tmp/ir-tracker.c","function":"f","inst_seq":1,"kind":"inst","line":9,"opcode":"ret","text":"  ret i32 %add, !dbg !{{[0-9]+}}"}
; ALL: {"kind":"pass","phase":"after","pass":"instcombine","ir_unit":"g","seq":2}
; ALL-NEXT: {"block":"entry","col":3,"file":"/tmp/ir-tracker.c","function":"g","inst_seq":0,"kind":"inst","line":14,"opcode":"shl","text":"  %mul = shl i32 %x, 1, !dbg !{{[0-9]+}}"}
; ALL-NEXT: {"block":"entry","col":3,"file":"/tmp/ir-tracker.c","function":"g","inst_seq":1,"kind":"inst","line":15,"opcode":"ret","text":"  ret i32 %mul, !dbg !{{[0-9]+}}"}

; FILTER: {"kind":"pass","phase":"initial","pass":"<initial>","ir_unit":"[module]","seq":0}
; FILTER-NEXT: {"block":"entry","col":3,"file":"/tmp/ir-tracker.c","function":"f","inst_seq":0,"kind":"inst","line":8,"opcode":"add","text":"  %add = add i32 %x, 1, !dbg !{{[0-9]+}}"}
; FILTER-NEXT: {"block":"entry","col":3,"file":"/tmp/ir-tracker.c","function":"f","inst_seq":1,"kind":"inst","line":9,"opcode":"ret","text":"  ret i32 %add, !dbg !{{[0-9]+}}"}
; FILTER-NOT: "function":"g"
; FILTER: {"kind":"pass","phase":"after","pass":"instcombine","ir_unit":"f","seq":1}
; FILTER-NEXT: {"block":"entry","col":3,"file":"/tmp/ir-tracker.c","function":"f","inst_seq":0,"kind":"inst","line":8,"opcode":"add","text":"  %add = add i32 %x, 1, !dbg !{{[0-9]+}}"}
; FILTER-NEXT: {"block":"entry","col":3,"file":"/tmp/ir-tracker.c","function":"f","inst_seq":1,"kind":"inst","line":9,"opcode":"ret","text":"  ret i32 %add, !dbg !{{[0-9]+}}"}
; FILTER-NOT: "ir_unit":"g"
