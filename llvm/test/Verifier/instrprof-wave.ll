; RUN: not opt -passes=verify -disable-output %s 2>&1 | FileCheck %s
target triple = "amdgcn-amd-amdhsa"
@name = private constant [1 x i8] c"f"
declare void @llvm.instrprof.increment.wave(ptr, i64, i32, i32, i32)
; CHECK: wave profiling index out of bounds
define void @bad_index() {
  call void @llvm.instrprof.increment.wave(ptr @name, i64 1, i32 1, i32 2, i32 2)
  ret void
}
; CHECK: invalid wave profiling counter counts
define void @overflow() {
  call void @llvm.instrprof.increment.wave(ptr @name, i64 1, i32 -1, i32 0, i32 1)
  ret void
}
; CHECK: invalid wave profiling counter counts
define void @no_lanes() {
  call void @llvm.instrprof.increment.wave(ptr @name, i64 1, i32 0, i32 0, i32 1)
  ret void
}
