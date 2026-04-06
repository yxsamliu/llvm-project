; RUN: opt %s --add-ll-debugloc-force -passes=no-op-module -S -o - | FileCheck %s
; RUN: opt %s --add-ll-debugloc-force -passes=no-op-module -S -o %t.ll
; RUN: opt -disable-output -passes=no-op-module %t.ll 2>&1 | FileCheck %s --check-prefix=NOWARN --allow-empty
; RUN: llvm-as %s -o %t.bc
; RUN: opt %t.bc --add-ll-debugloc-force -passes=no-op-module -S -o - | FileCheck %s --check-prefix=BC
; RUN: opt %t.bc --add-ll-debugloc-force -passes=no-op-module -S -o %t-bc.ll
; RUN: opt -disable-output -passes=no-op-module %t-bc.ll 2>&1 | FileCheck %s --check-prefix=BCNOWARN --allow-empty
; NOWARN-NOT: invalid version
; BCNOWARN-NOT: invalid version

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
; CHECK: ![[F1SP]] = distinct !DISubprogram(name: "f1", linkageName: "f1", scope: null, file: ![[FILE:[0-9]+]], line: 13
; CHECK: ![[F1ADDLOC:[0-9]+]] = !DILocation(line: 15, scope: ![[F1SP]])
; CHECK: ![[F2SP]] = distinct !DISubprogram(name: "f2", linkageName: "f2", scope: null, file: ![[FILE]], line: 19
; CHECK: ![[F2MULLOC:[0-9]+]] = !DILocation(line: 21, scope: ![[F2SP]])

; BC: define i32 @f1(i32 %x) !dbg ![[BCF1SP:[0-9]+]]
; BC: define i32 @f2(i32 %x) !dbg ![[BCF2SP:[0-9]+]]
; BC: !llvm.dbg.cu = !{![[BCCU:[0-9]+]]}
; BC: !llvm.module.flags = !{![[BCVER:[0-9]+]]}
; BC: ![[BCFILE:[0-9]+]] = !DIFile(filename: "{{.*}}.bc", directory: "{{.*}}")
; BC: ![[BCVER]] = !{i32 2, !"Debug Info Version", i32 3}
; BC: ![[BCF1SP]] = distinct !DISubprogram(name: "f1", linkageName: "f1", scope: null, file: ![[BCFILE]], line: 1
; BC: ![[BCF1ADDLOC:[0-9]+]] = !DILocation(line: 1, scope: ![[BCF1SP]])
; BC: ![[BCF2SP]] = distinct !DISubprogram(name: "f2", linkageName: "f2", scope: null, file: ![[BCFILE]], line: 3
; BC: ![[BCF2MULLOC:[0-9]+]] = !DILocation(line: 3, scope: ![[BCF2SP]])
