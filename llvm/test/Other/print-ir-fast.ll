; RUN: opt -S -passes=no-op-module < %s > %t.normal
; RUN: opt -S -passes=no-op-module -print-ir-fast < %s > %t.fast
; RUN: diff %t.normal %t.fast
; RUN: FileCheck %s --check-prefix=NORMAL < %t.fast
; RUN: opt -disable-output -passes=print -print-ir-fast < %s 2>&1 | FileCheck %s --check-prefix=NORMAL
; RUN: opt -disable-output -passes='function(print)' -filter-print-funcs=second \
; RUN:   -print-ir-fast < %s 2>&1 | FileCheck %s --check-prefix=FAST-FUNCTION
; RUN: opt -disable-output -passes='function(no-op-function)' \
; RUN:   -print-before=no-op-function -filter-print-funcs=second \
; RUN:   -print-ir-fast < %s 2>&1 | FileCheck %s --check-prefix=FAST-FUNCTION
; RUN: opt -disable-output -passes='function(no-op-function)' -print-after-all \
; RUN:   -filter-print-funcs=second -print-ir-fast < %s 2>&1 | FileCheck %s --check-prefix=FAST-FUNCTION
; RUN: opt -disable-output -passes=no-op-module -print-before=no-op-module \
; RUN:   -filter-print-funcs=first,second -print-ir-fast < %s 2>&1 | FileCheck %s --check-prefix=FAST-MULTI
; RUN: opt -disable-output -passes='loop(no-op-loop)' -print-before=no-op-loop \
; RUN:   -filter-print-funcs=loop -print-ir-fast < %s 2>&1 | FileCheck %s --check-prefix=FAST-LOOP
$group = comdat any

@named = global ptr @0, comdat($group), !annotation !5
@0 = global i32 0
@1 = global i32 1

declare void @callee(ptr)

define void @first() #0 {
  call void @callee(ptr @0) #1
  call void @callee(ptr @1) #1
  ret void, !annotation !1
}

define void @second() {
  call void @callee(ptr @1) #1, !annotation !3
  ret void, !annotation !3
}

define void @loop() {
entry:
  call void @callee(ptr @0) #1
  call void @callee(ptr @1) #1
  br label %loop

loop:
  call void @callee(ptr @1) #1
  br i1 false, label %loop, label %exit

exit:
  ret void
}

attributes #0 = { noinline }
attributes #1 = { nounwind }

!named = !{!0}
!0 = !{!"named metadata"}
!1 = !{!2}
!2 = !{!"first metadata"}
!3 = !{!4}
!4 = !{!"second metadata"}
!5 = !{!6}
!6 = !{!"global metadata"}

; NORMAL: @named = global ptr @0, comdat($group), !annotation ![[NORMAL_GLOBAL:[0-9]+]]
; NORMAL: ret void, !annotation ![[NORMAL_FIRST:[0-9]+]]
; NORMAL: call void @callee(ptr @1) #1, !annotation ![[NORMAL_SECOND:[0-9]+]]
; NORMAL: ret void, !annotation ![[NORMAL_SECOND]]
; NORMAL: !named = !{![[NORMAL_NAMED:[0-9]+]]}

; FAST-FUNCTION: define void @second() {
; FAST-FUNCTION: call void @callee(ptr @1) #1, !annotation ![[SECOND:[0-9]+]]
; FAST-FUNCTION: ret void, !annotation ![[SECOND]]

; FAST-MULTI: define void @first() #0 {
; FAST-MULTI: call void @callee(ptr @0) #1
; FAST-MULTI: call void @callee(ptr @1) #1
; FAST-MULTI: define void @second() {
; FAST-MULTI: call void @callee(ptr @1) #1

; FAST-LOOP: ; Preheader:
; FAST-LOOP: call void @callee(ptr @0) #1
; FAST-LOOP: call void @callee(ptr @1) #1
; FAST-LOOP: ; Loop:
; FAST-LOOP: call void @callee(ptr @1) #1
