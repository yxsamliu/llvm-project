; RUN: opt -passes=no-op-function -disable-output -print-before-all -filter-print-src-locs=file.c:2 %s 2>&1 | FileCheck %s

; The source-location filter should suppress non-matching instructions while
; preserving function/module structure. Only the instruction at file.c:2 should appear.

define i32 @foo() !dbg !2 {
entry:
  %a = add i32 1, 2, !dbg !4
  %b = add i32 3, 4, !dbg !5
  ret i32 %b, !dbg !5
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "file.c", directory: "/src")
!2 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !3, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!3 = !DISubroutineType(types: !7)
!4 = !DILocation(line: 1, column: 1, scope: !2)
!5 = !DILocation(line: 2, column: 1, scope: !2)
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{}

; CHECK: define i32 @foo()
; CHECK-NOT: add i32 1, 2
; CHECK: add i32 3, 4
