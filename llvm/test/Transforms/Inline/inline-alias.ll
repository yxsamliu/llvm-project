; RUN: opt < %s -passes=inline -S | FileCheck %s

@foo2 = alias i32 (), ptr @foo

define i32 @foo() { ret i32 1 }

define i32 @test() {
	%ret = call i32 @foo2()
	ret i32 %ret
}
