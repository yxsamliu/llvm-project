; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/noinline.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc

; Attempt the import now, ensure below that file containing noinline
; is not imported by default but imported with -import-noinline.

; RUN: opt -function-import -summary-file %t3.thinlto.bc %t.bc -S 2>&1 \
; RUN:   | FileCheck -check-prefix=NOIMPORT %s
; RUN: opt -function-import -import-noinline -summary-file %t3.thinlto.bc \
; RUN:   %t.bc -S 2>&1 | FileCheck -check-prefix=IMPORT %s

define i32 @main() #0 {
entry:
  %f = alloca i64, align 8
  call void @foo(i64* %f)
  ret i32 0
}

; NOIMPORT: declare void @foo(i64*)
; IMPORT: define available_externally void @foo
declare void @foo(i64*) #1
