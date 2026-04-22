; RUN: opt -disable-output -passes=instcombine -ir-tracker-json-output=%t.tsv %s
; RUN: FileCheck %s --input-file=%t.tsv --check-prefix=ALL
; RUN: opt -disable-output -passes=instcombine -filter-print-funcs=f -ir-tracker-json-output=%t-filter.tsv %s
; RUN: FileCheck %s --input-file=%t-filter.tsv --check-prefix=FILTER

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

; Output is the cost-improvement TSV form: P/T/I rows. T rows carry source
; locations once per tracker ID; I rows reference tracker IDs.

; ALL: P{{	}}0{{	}}initial{{	}}<initial>{{	}}f
; ALL-NEXT: T{{	}}{{[0-9]+}}{{	}}/tmp{{[/\\]}}ir-tracker.c{{	}}8{{	}}3
; ALL-NEXT: I{{	}}f{{	}}entry{{	}}0{{	}}add{{	}}{{[0-9]+}}{{	}}%add = add i32 %x, 1
; ALL-NEXT: T{{	}}{{[0-9]+}}{{	}}/tmp{{[/\\]}}ir-tracker.c{{	}}9{{	}}3
; ALL-NEXT: I{{	}}f{{	}}entry{{	}}1{{	}}ret{{	}}{{[0-9]+}}{{	}}ret i32 %add
; ALL: P{{	}}1{{	}}after{{	}}instcombine{{	}}f
; ALL: P{{	}}2{{	}}after{{	}}instcombine{{	}}g
; ALL-NEXT: T{{	}}{{[0-9]+}}{{	}}/tmp{{[/\\]}}ir-tracker.c{{	}}14{{	}}3
; ALL-NEXT: I{{	}}g{{	}}entry{{	}}0{{	}}shl{{	}}{{[0-9]+}}{{	}}%mul = shl i32 %x, 1
; ALL-NEXT: T{{	}}{{[0-9]+}}{{	}}/tmp{{[/\\]}}ir-tracker.c{{	}}15{{	}}3
; ALL-NEXT: I{{	}}g{{	}}entry{{	}}1{{	}}ret{{	}}{{[0-9]+}}{{	}}ret i32 %mul

; FILTER: P{{	}}0{{	}}initial{{	}}<initial>{{	}}f
; FILTER: I{{	}}f{{	}}entry{{	}}0{{	}}add
; FILTER: I{{	}}f{{	}}entry{{	}}1{{	}}ret
; FILTER-NOT: I{{	}}g{{	}}
; FILTER-NOT: ir_unit{{[" :]+}}g
