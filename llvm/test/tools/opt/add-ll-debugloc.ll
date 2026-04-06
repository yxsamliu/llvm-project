; RUN: opt %s --add-ll-debugloc-force -passes=no-op-module -S -o - | FileCheck %s
; RUN: opt %s --add-ll-debugloc-force -passes=no-op-module -S -o %t.ll
; RUN: opt -disable-output -passes=no-op-module %t.ll 2>&1 | FileCheck %s --check-prefix=NOWARN --allow-empty
; NOWARN-NOT: invalid version

declare void @decl()

define i32 @f1(i32 %x) {
entry:
  %a = add i32 %x, 1
  ret i32 %a
}

define i32 @f2(i32 %x) {
entry:
  %m = mul i32 %x, 2
  ret i32 %m
}

; CHECK: define i32 @f1(i32 %x) !dbg ![[F1SP:[0-9]+]]
; CHECK: define i32 @f2(i32 %x) !dbg ![[F2SP:[0-9]+]]
; CHECK: !llvm.dbg.cu = !{![[CU:[0-9]+]]}
; CHECK: !llvm.module.flags = !{![[VER:[0-9]+]]}
; CHECK: ![[VER]] = !{i32 2, !"Debug Info Version", i32 3}
; CHECK: ![[F1SP]] = distinct !DISubprogram(name: "f1", linkageName: "f1", scope: null, file: ![[FILE:[0-9]+]], line: 8
; CHECK: ![[F1ADDLOC:[0-9]+]] = !DILocation(line: 10, scope: ![[F1SP]])
; CHECK: ![[F2SP]] = distinct !DISubprogram(name: "f2", linkageName: "f2", scope: null, file: ![[FILE]], line: 14
; CHECK: ![[F2MULLOC:[0-9]+]] = !DILocation(line: 16, scope: ![[F2SP]])
