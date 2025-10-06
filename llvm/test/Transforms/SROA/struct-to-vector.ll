; RUN: split-file %s %t
; RUN: opt -passes='sroa,gvn,instcombine,simplifycfg' -S -sroa-max-struct-to-vector-bytes=16 %t/flat.ll | FileCheck %s --check-prefix=FLAT
; RUN: opt -passes='sroa,gvn,instcombine,simplifycfg' -S -sroa-max-struct-to-vector-bytes=16 %t/nested.ll | FileCheck %s --check-prefix=NESTED

;--- flat.ll
%struct.myint4 = type { i32, i32, i32, i32 }

; FLAT-LABEL: define dso_local void @_Z3fooP6myint4S_i(
; FLAT-NOT: alloca
; FLAT-NOT: llvm.memcpy
; FLAT-NOT: llvm.memset
; FLAT: insertelement <2 x i64>
; FLAT: bitcast <2 x i64> %{{[^ ]+}} to <4 x i32>
; FLAT: select i1 %{{[^,]+}}, <4 x i32> zeroinitializer, <4 x i32> %{{[^)]+}}
; FLAT: store <4 x i32> %{{[^,]+}}, ptr %x, align 16
; FLAT: ret void
define dso_local void @_Z3fooP6myint4S_i(ptr noundef %x, i64 %y.coerce0, i64 %y.coerce1, i32 noundef %cond) {
entry:
  %y = alloca %struct.myint4, align 16
  %x.addr = alloca ptr, align 8
  %cond.addr = alloca i32, align 4
  %temp = alloca %struct.myint4, align 16
  %zero = alloca %struct.myint4, align 16
  %data = alloca %struct.myint4, align 16
  %0 = getelementptr inbounds nuw { i64, i64 }, ptr %y, i32 0, i32 0
  store i64 %y.coerce0, ptr %0, align 16
  %1 = getelementptr inbounds nuw { i64, i64 }, ptr %y, i32 0, i32 1
  store i64 %y.coerce1, ptr %1, align 8
  store ptr %x, ptr %x.addr, align 8
  store i32 %cond, ptr %cond.addr, align 4
  call void @llvm.lifetime.start.p0(ptr %temp)
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %temp, ptr align 16 %y, i64 16, i1 false)
  call void @llvm.lifetime.start.p0(ptr %zero)
  call void @llvm.memset.p0.i64(ptr align 16 %zero, i8 0, i64 16, i1 false)
  call void @llvm.lifetime.start.p0(ptr %data)
  %2 = load i32, ptr %cond.addr, align 4
  %tobool = icmp ne i32 %2, 0
  br i1 %tobool, label %cond.true, label %cond.false

cond.true:
  br label %cond.end

cond.false:
  br label %cond.end

cond.end:
  %cond1 = phi ptr [ %temp, %cond.true ], [ %zero, %cond.false ]
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %data, ptr align 16 %cond1, i64 16, i1 false)
  %3 = load ptr, ptr %x.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %3, ptr align 16 %data, i64 16, i1 false)
  call void @llvm.lifetime.end.p0(ptr %data)
  call void @llvm.lifetime.end.p0(ptr %zero)
  call void @llvm.lifetime.end.p0(ptr %temp)
  ret void
}

 

;--- nested.ll
%struct.myint4_base = type { i32, i32, i32, i32 }
%struct.myint4 = type { %struct.myint4_base }

; NESTED-LABEL: define dso_local void @_Z3fooP6myint4S_i(
; NESTED-NOT: alloca
; NESTED-NOT: llvm.memcpy
; NESTED-NOT: llvm.memset
; NESTED: insertelement <2 x i64>
; NESTED: bitcast <2 x i64> %{{[^ ]+}} to <4 x i32>
; NESTED: select i1 %{{[^,]+}}, <4 x i32> zeroinitializer, <4 x i32> %{{[^)]+}}
; NESTED: store <4 x i32> %{{[^,]+}}, ptr %x, align 16
; NESTED: ret void
define dso_local void @_Z3fooP6myint4S_i(ptr noundef %x, i64 %y.coerce0, i64 %y.coerce1, i32 noundef %cond) {
entry:
  %y = alloca %struct.myint4, align 16
  %x.addr = alloca ptr, align 8
  %cond.addr = alloca i32, align 4
  %temp = alloca %struct.myint4, align 16
  %zero = alloca %struct.myint4, align 16
  %data = alloca %struct.myint4, align 16
  %0 = getelementptr inbounds nuw { i64, i64 }, ptr %y, i32 0, i32 0
  store i64 %y.coerce0, ptr %0, align 16
  %1 = getelementptr inbounds nuw { i64, i64 }, ptr %y, i32 0, i32 1
  store i64 %y.coerce1, ptr %1, align 8
  store ptr %x, ptr %x.addr, align 8
  store i32 %cond, ptr %cond.addr, align 4
  call void @llvm.lifetime.start.p0(ptr %temp)
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %temp, ptr align 16 %y, i64 16, i1 false)
  call void @llvm.lifetime.start.p0(ptr %zero)
  call void @llvm.memset.p0.i64(ptr align 16 %zero, i8 0, i64 16, i1 false)
  call void @llvm.lifetime.start.p0(ptr %data)
  %2 = load i32, ptr %cond.addr, align 4
  %tobool = icmp ne i32 %2, 0
  br i1 %tobool, label %cond.true, label %cond.false

cond.true:
  br label %cond.end

cond.false:
  br label %cond.end

cond.end:
  %cond1 = phi ptr [ %temp, %cond.true ], [ %zero, %cond.false ]
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %data, ptr align 16 %cond1, i64 16, i1 false)
  %3 = load ptr, ptr %x.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %3, ptr align 16 %data, i64 16, i1 false)
  call void @llvm.lifetime.end.p0(ptr %data)
  call void @llvm.lifetime.end.p0(ptr %zero)
  call void @llvm.lifetime.end.p0(ptr %temp)
  ret void
}

 
