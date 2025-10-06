; RUN: opt -passes=sroa -sroa-skip-mem2reg -sroa-max-struct-to-vector-bytes=8 -S < %s | FileCheck %s --check-prefix=ENABLE
; RUN: opt -passes=sroa -sroa-skip-mem2reg -S < %s | FileCheck %s --check-prefix=DISABLE

; Verify that SROA canonicalizes a homogeneous 2-element struct to a vector
; when enabled via -sroa-max-struct-to-vector-bytes, and leaves it as a struct
; when the option is disabled (default).

%Inner = type { i32, i32 }

%Outer = type { i8, %Inner }

define void @test(ptr noalias %sink) {
; ENABLE-LABEL: @test(
; ENABLE: alloca <2 x i32>
; DISABLE-LABEL: @test(
; DISABLE-NOT: alloca <2 x i32>
entry:
  %a = alloca %Outer, align 4
  %inner.ptr = getelementptr inbounds %Outer, ptr %a, i32 0, i32 1
  ; Escape the inner pointer so SROA creates a partition alloca for it
  store ptr %inner.ptr, ptr %sink
  ret void
}


