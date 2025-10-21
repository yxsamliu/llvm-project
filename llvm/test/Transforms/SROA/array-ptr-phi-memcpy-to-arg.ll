; ModuleID = '/app/example.cu'
source_filename = "/app/example.cu"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%0 = type { i64, i64, i32, i32 }
%1 = type { [64 x [8 x i64]] }
%struct.HIP_vector_type = type { %struct.HIP_vector_base }
%struct.HIP_vector_base = type { i32, i32, i32, i32 }

@__const.__assert_fail.fmt = private unnamed_addr addrspace(4) constant [47 x i8] c"%s:%u: %s: Device-side assertion `%s' failed.\0A\00", align 16
@__hip_cuid_a3171da8b7ab0ae9 = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_a3171da8b7ab0ae9 to ptr)], section "llvm.metadata"
@__oclc_ISA_version = internal local_unnamed_addr addrspace(4) constant i32 11000, align 4
@__oclc_ABI_version = internal local_unnamed_addr addrspace(4) constant i32 600, align 4

; Function Attrs: convergent mustprogress noinline noreturn nounwind optnone
define weak void @__cxa_pure_virtual() #0 {
entry:
  call void @llvm.trap()
  unreachable
}

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #1

; Function Attrs: convergent mustprogress noinline noreturn nounwind optnone
define weak void @__cxa_deleted_virtual() #0 {
entry:
  call void @llvm.trap()
  unreachable
}

; Function Attrs: convergent mustprogress noinline nounwind optnone
define weak hidden void @__assert_fail(ptr noundef %assertion, ptr noundef %file, i32 noundef %line, ptr noundef %function) #2 {
entry:
  %assertion.addr = alloca ptr, align 8, addrspace(5)
  %file.addr = alloca ptr, align 8, addrspace(5)
  %line.addr = alloca i32, align 4, addrspace(5)
  %function.addr = alloca ptr, align 8, addrspace(5)
  %fmt = alloca [47 x i8], align 16, addrspace(5)
  %msg = alloca i64, align 8, addrspace(5)
  %len = alloca i32, align 4, addrspace(5)
  %tmp = alloca ptr, align 8, addrspace(5)
  %tmp6 = alloca ptr, align 8, addrspace(5)
  %tmp22 = alloca ptr, align 8, addrspace(5)
  %tmp36 = alloca ptr, align 8, addrspace(5)
  %assertion.addr.ascast = addrspacecast ptr addrspace(5) %assertion.addr to ptr
  %file.addr.ascast = addrspacecast ptr addrspace(5) %file.addr to ptr
  %line.addr.ascast = addrspacecast ptr addrspace(5) %line.addr to ptr
  %function.addr.ascast = addrspacecast ptr addrspace(5) %function.addr to ptr
  %fmt.ascast = addrspacecast ptr addrspace(5) %fmt to ptr
  %msg.ascast = addrspacecast ptr addrspace(5) %msg to ptr
  %len.ascast = addrspacecast ptr addrspace(5) %len to ptr
  %tmp.ascast = addrspacecast ptr addrspace(5) %tmp to ptr
  %tmp6.ascast = addrspacecast ptr addrspace(5) %tmp6 to ptr
  %tmp22.ascast = addrspacecast ptr addrspace(5) %tmp22 to ptr
  %tmp36.ascast = addrspacecast ptr addrspace(5) %tmp36 to ptr
  store ptr %assertion, ptr %assertion.addr.ascast, align 8
  store ptr %file, ptr %file.addr.ascast, align 8
  store i32 %line, ptr %line.addr.ascast, align 4
  store ptr %function, ptr %function.addr.ascast, align 8
  call void @llvm.memcpy.p0.p4.i64(ptr align 16 %fmt.ascast, ptr addrspace(4) align 16 @__const.__assert_fail.fmt, i64 47, i1 false)
  %call = call i64 @__ockl_fprintf_stderr_begin() #14
  store i64 %call, ptr %msg.ascast, align 8
  store i32 0, ptr %len.ascast, align 4
  br label %do.body

do.body:                                          ; preds = %entry
  %arraydecay = getelementptr inbounds [47 x i8], ptr %fmt.ascast, i64 0, i64 0
  store ptr %arraydecay, ptr %tmp.ascast, align 8
  br label %while.cond

while.cond:                                       ; preds = %while.body, %do.body
  %0 = load ptr, ptr %tmp.ascast, align 8
  %incdec.ptr = getelementptr inbounds nuw i8, ptr %0, i32 1
  store ptr %incdec.ptr, ptr %tmp.ascast, align 8
  %1 = load i8, ptr %0, align 1
  %tobool = icmp ne i8 %1, 0
  br i1 %tobool, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  br label %while.cond, !llvm.loop !8

while.end:                                        ; preds = %while.cond
  %2 = load ptr, ptr %tmp.ascast, align 8
  %arraydecay1 = getelementptr inbounds [47 x i8], ptr %fmt.ascast, i64 0, i64 0
  %sub.ptr.lhs.cast = ptrtoint ptr %2 to i64
  %sub.ptr.rhs.cast = ptrtoint ptr %arraydecay1 to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %conv = trunc i64 %sub.ptr.sub to i32
  store i32 %conv, ptr %len.ascast, align 4
  br label %do.end

do.end:                                           ; preds = %while.end
  %3 = load i64, ptr %msg.ascast, align 8
  %arraydecay2 = getelementptr inbounds [47 x i8], ptr %fmt.ascast, i64 0, i64 0
  %4 = load i32, ptr %len.ascast, align 4
  %conv3 = sext i32 %4 to i64
  %call4 = call i64 @__ockl_fprintf_append_string_n(i64 noundef %3, ptr noundef %arraydecay2, i64 noundef %conv3, i32 noundef 0) #14
  store i64 %call4, ptr %msg.ascast, align 8
  br label %do.body5

do.body5:                                         ; preds = %do.end
  %5 = load ptr, ptr %file.addr.ascast, align 8
  store ptr %5, ptr %tmp6.ascast, align 8
  br label %while.cond7

while.cond7:                                      ; preds = %while.body10, %do.body5
  %6 = load ptr, ptr %tmp6.ascast, align 8
  %incdec.ptr8 = getelementptr inbounds nuw i8, ptr %6, i32 1
  store ptr %incdec.ptr8, ptr %tmp6.ascast, align 8
  %7 = load i8, ptr %6, align 1
  %tobool9 = icmp ne i8 %7, 0
  br i1 %tobool9, label %while.body10, label %while.end11

while.body10:                                     ; preds = %while.cond7
  br label %while.cond7, !llvm.loop !10

while.end11:                                      ; preds = %while.cond7
  %8 = load ptr, ptr %tmp6.ascast, align 8
  %9 = load ptr, ptr %file.addr.ascast, align 8
  %sub.ptr.lhs.cast12 = ptrtoint ptr %8 to i64
  %sub.ptr.rhs.cast13 = ptrtoint ptr %9 to i64
  %sub.ptr.sub14 = sub i64 %sub.ptr.lhs.cast12, %sub.ptr.rhs.cast13
  %conv15 = trunc i64 %sub.ptr.sub14 to i32
  store i32 %conv15, ptr %len.ascast, align 4
  br label %do.end16

do.end16:                                         ; preds = %while.end11
  %10 = load i64, ptr %msg.ascast, align 8
  %11 = load ptr, ptr %file.addr.ascast, align 8
  %12 = load i32, ptr %len.ascast, align 4
  %conv17 = sext i32 %12 to i64
  %call18 = call i64 @__ockl_fprintf_append_string_n(i64 noundef %10, ptr noundef %11, i64 noundef %conv17, i32 noundef 0) #14
  store i64 %call18, ptr %msg.ascast, align 8
  %13 = load i64, ptr %msg.ascast, align 8
  %14 = load i32, ptr %line.addr.ascast, align 4
  %conv19 = zext i32 %14 to i64
  %call20 = call i64 @__ockl_fprintf_append_args(i64 noundef %13, i32 noundef 1, i64 noundef %conv19, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0, i32 noundef 0) #14
  store i64 %call20, ptr %msg.ascast, align 8
  br label %do.body21

do.body21:                                        ; preds = %do.end16
  %15 = load ptr, ptr %function.addr.ascast, align 8
  store ptr %15, ptr %tmp22.ascast, align 8
  br label %while.cond23

while.cond23:                                     ; preds = %while.body26, %do.body21
  %16 = load ptr, ptr %tmp22.ascast, align 8
  %incdec.ptr24 = getelementptr inbounds nuw i8, ptr %16, i32 1
  store ptr %incdec.ptr24, ptr %tmp22.ascast, align 8
  %17 = load i8, ptr %16, align 1
  %tobool25 = icmp ne i8 %17, 0
  br i1 %tobool25, label %while.body26, label %while.end27

while.body26:                                     ; preds = %while.cond23
  br label %while.cond23, !llvm.loop !11

while.end27:                                      ; preds = %while.cond23
  %18 = load ptr, ptr %tmp22.ascast, align 8
  %19 = load ptr, ptr %function.addr.ascast, align 8
  %sub.ptr.lhs.cast28 = ptrtoint ptr %18 to i64
  %sub.ptr.rhs.cast29 = ptrtoint ptr %19 to i64
  %sub.ptr.sub30 = sub i64 %sub.ptr.lhs.cast28, %sub.ptr.rhs.cast29
  %conv31 = trunc i64 %sub.ptr.sub30 to i32
  store i32 %conv31, ptr %len.ascast, align 4
  br label %do.end32

do.end32:                                         ; preds = %while.end27
  %20 = load i64, ptr %msg.ascast, align 8
  %21 = load ptr, ptr %function.addr.ascast, align 8
  %22 = load i32, ptr %len.ascast, align 4
  %conv33 = sext i32 %22 to i64
  %call34 = call i64 @__ockl_fprintf_append_string_n(i64 noundef %20, ptr noundef %21, i64 noundef %conv33, i32 noundef 0) #14
  store i64 %call34, ptr %msg.ascast, align 8
  br label %do.body35

do.body35:                                        ; preds = %do.end32
  %23 = load ptr, ptr %assertion.addr.ascast, align 8
  store ptr %23, ptr %tmp36.ascast, align 8
  br label %while.cond37

while.cond37:                                     ; preds = %while.body40, %do.body35
  %24 = load ptr, ptr %tmp36.ascast, align 8
  %incdec.ptr38 = getelementptr inbounds nuw i8, ptr %24, i32 1
  store ptr %incdec.ptr38, ptr %tmp36.ascast, align 8
  %25 = load i8, ptr %24, align 1
  %tobool39 = icmp ne i8 %25, 0
  br i1 %tobool39, label %while.body40, label %while.end41

while.body40:                                     ; preds = %while.cond37
  br label %while.cond37, !llvm.loop !12

while.end41:                                      ; preds = %while.cond37
  %26 = load ptr, ptr %tmp36.ascast, align 8
  %27 = load ptr, ptr %assertion.addr.ascast, align 8
  %sub.ptr.lhs.cast42 = ptrtoint ptr %26 to i64
  %sub.ptr.rhs.cast43 = ptrtoint ptr %27 to i64
  %sub.ptr.sub44 = sub i64 %sub.ptr.lhs.cast42, %sub.ptr.rhs.cast43
  %conv45 = trunc i64 %sub.ptr.sub44 to i32
  store i32 %conv45, ptr %len.ascast, align 4
  br label %do.end46

do.end46:                                         ; preds = %while.end41
  %28 = load i64, ptr %msg.ascast, align 8
  %29 = load ptr, ptr %assertion.addr.ascast, align 8
  %30 = load i32, ptr %len.ascast, align 4
  %conv47 = sext i32 %30 to i64
  %call48 = call i64 @__ockl_fprintf_append_string_n(i64 noundef %28, ptr noundef %29, i64 noundef %conv47, i32 noundef 1) #14
  call void @llvm.trap()
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p4.i64(ptr noalias writeonly captures(none), ptr addrspace(4) noalias readonly captures(none), i64, i1 immarg) #3

; Function Attrs: convergent mustprogress noinline nounwind optnone
define weak hidden void @__assertfail() #2 {
entry:
  call void @llvm.trap()
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define protected amdgpu_kernel void @_Z4testP15HIP_vector_typeIiLj4EES1_i(ptr addrspace(1) noundef %out.coerce, ptr addrspace(1) noundef %in.coerce, i32 noundef %cond) #4 {
entry:
  %out = alloca ptr, align 8, addrspace(5)
  %in = alloca ptr, align 8, addrspace(5)
  %out.addr = alloca ptr, align 8, addrspace(5)
  %in.addr = alloca ptr, align 8, addrspace(5)
  %cond.addr = alloca i32, align 4, addrspace(5)
  %temp = alloca %struct.HIP_vector_type, align 16, addrspace(5)
  %zero = alloca %struct.HIP_vector_type, align 16, addrspace(5)
  %out.ascast = addrspacecast ptr addrspace(5) %out to ptr
  %in.ascast = addrspacecast ptr addrspace(5) %in to ptr
  %out.addr.ascast = addrspacecast ptr addrspace(5) %out.addr to ptr
  %in.addr.ascast = addrspacecast ptr addrspace(5) %in.addr to ptr
  %cond.addr.ascast = addrspacecast ptr addrspace(5) %cond.addr to ptr
  %temp.ascast = addrspacecast ptr addrspace(5) %temp to ptr
  %zero.ascast = addrspacecast ptr addrspace(5) %zero to ptr
  store ptr addrspace(1) %out.coerce, ptr %out.ascast, align 8
  %out1 = load ptr, ptr %out.ascast, align 8
  store ptr addrspace(1) %in.coerce, ptr %in.ascast, align 8
  %in2 = load ptr, ptr %in.ascast, align 8
  store ptr %out1, ptr %out.addr.ascast, align 8
  store ptr %in2, ptr %in.addr.ascast, align 8
  store i32 %cond, ptr %cond.addr.ascast, align 4
  %0 = load ptr, ptr %in.addr.ascast, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %temp.ascast, ptr align 16 %0, i64 16, i1 false)
  call void @llvm.memset.p0.i64(ptr align 16 %zero.ascast, i8 0, i64 16, i1 false)
  %1 = load i32, ptr %cond.addr.ascast, align 4
  %tobool = icmp ne i32 %1, 0
  br i1 %tobool, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  br label %cond.end

cond.false:                                       ; preds = %entry
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond3 = phi ptr [ %temp.ascast, %cond.true ], [ %zero.ascast, %cond.false ]
  %2 = load ptr, ptr %out.addr.ascast, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %2, ptr align 16 %cond3, i64 16, i1 false)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #5

; Function Attrs: convergent norecurse nounwind
define internal i64 @__ockl_fprintf_stderr_begin() #6 {
  %1 = tail call <2 x i64> @__ockl_hostcall_preview(i32 noundef 2, i64 noundef 33, i64 noundef 1, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0) #14
  %2 = extractelement <2 x i64> %1, i64 0
  ret i64 %2
}

; Function Attrs: cold convergent norecurse nounwind
define internal <2 x i64> @__ockl_hostcall_preview(i32 noundef %0, i64 noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i64 noundef %6, i64 noundef %7, i64 noundef %8) local_unnamed_addr #7 {
  %10 = load i32, ptr addrspace(4) @__oclc_ABI_version, align 4, !tbaa !13
  %11 = icmp slt i32 %10, 500
  %12 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %13 = select i1 %11, i64 24, i64 80
  %14 = getelementptr inbounds nuw i8, ptr addrspace(4) %12, i64 %13
  %15 = load i64, ptr addrspace(4) %14, align 8, !tbaa !17
  %16 = inttoptr i64 %15 to ptr addrspace(1)
  %17 = addrspacecast ptr addrspace(1) %16 to ptr
  %18 = tail call <2 x i64> @__ockl_hostcall_internal(ptr noundef %17, i32 noundef %0, i64 noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i64 noundef %6, i64 noundef %7, i64 noundef %8) #15
  ret <2 x i64> %18
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #8

; Function Attrs: convergent norecurse nounwind
define internal <2 x i64> @__ockl_hostcall_internal(ptr noundef captures(none) %0, i32 noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i64 noundef %6, i64 noundef %7, i64 noundef %8, i64 noundef %9) local_unnamed_addr #6 {
  %11 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %12 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %11)
  %13 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %12)
  %14 = addrspacecast ptr %0 to ptr addrspace(1)
  %15 = icmp eq i32 %12, %13
  br i1 %15, label %16, label %38

16:                                               ; preds = %10
  %17 = getelementptr inbounds nuw i8, ptr addrspace(1) %14, i64 24
  %18 = load atomic i64, ptr addrspace(1) %17 syncscope("one-as") acquire, align 8
  %19 = getelementptr i8, ptr addrspace(1) %14, i64 40
  %20 = load ptr addrspace(1), ptr addrspace(1) %14, align 8, !tbaa !19
  %21 = load i64, ptr addrspace(1) %19, align 8, !tbaa !23
  %22 = and i64 %21, %18
  %23 = getelementptr inbounds nuw %0, ptr addrspace(1) %20, i64 %22
  %24 = load atomic i64, ptr addrspace(1) %23 syncscope("one-as") monotonic, align 8
  %25 = cmpxchg ptr addrspace(1) %17, i64 %18, i64 %24 syncscope("one-as") acquire monotonic, align 8
  %26 = extractvalue { i64, i1 } %25, 1
  %27 = extractvalue { i64, i1 } %25, 0
  br i1 %26, label %38, label %28

28:                                               ; preds = %28, %16
  %29 = phi i64 [ %37, %28 ], [ %27, %16 ]
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  %30 = load ptr addrspace(1), ptr addrspace(1) %14, align 8, !tbaa !19
  %31 = load i64, ptr addrspace(1) %19, align 8, !tbaa !23
  %32 = and i64 %31, %29
  %33 = getelementptr inbounds nuw %0, ptr addrspace(1) %30, i64 %32
  %34 = load atomic i64, ptr addrspace(1) %33 syncscope("one-as") monotonic, align 8
  %35 = cmpxchg ptr addrspace(1) %17, i64 %29, i64 %34 syncscope("one-as") acquire monotonic, align 8
  %36 = extractvalue { i64, i1 } %35, 1
  %37 = extractvalue { i64, i1 } %35, 0
  br i1 %36, label %38, label %28

38:                                               ; preds = %28, %16, %10
  %39 = phi i64 [ 0, %10 ], [ %27, %16 ], [ %37, %28 ]
  %40 = trunc i64 %39 to i32
  %41 = lshr i64 %39, 32
  %42 = trunc nuw i64 %41 to i32
  %43 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %40)
  %44 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %42)
  %45 = zext i32 %44 to i64
  %46 = shl nuw i64 %45, 32
  %47 = zext i32 %43 to i64
  %48 = or disjoint i64 %46, %47
  %49 = load ptr addrspace(1), ptr addrspace(1) %14, align 8, !tbaa !19
  %50 = getelementptr i8, ptr addrspace(1) %14, i64 40
  %51 = load i64, ptr addrspace(1) %50, align 8, !tbaa !23
  %52 = and i64 %48, %51
  %53 = getelementptr inbounds nuw %0, ptr addrspace(1) %49, i64 %52
  %54 = getelementptr i8, ptr addrspace(1) %14, i64 8
  %55 = load ptr addrspace(1), ptr addrspace(1) %54, align 8, !tbaa !24
  %56 = getelementptr inbounds nuw %1, ptr addrspace(1) %55, i64 %52
  %57 = tail call i64 @llvm.amdgcn.ballot.i64(i1 true)
  br i1 %15, label %58, label %62

58:                                               ; preds = %38
  %59 = getelementptr inbounds nuw i8, ptr addrspace(1) %53, i64 16
  %60 = getelementptr inbounds nuw i8, ptr addrspace(1) %53, i64 8
  %61 = getelementptr inbounds nuw i8, ptr addrspace(1) %53, i64 20
  store i32 %1, ptr addrspace(1) %59, align 8, !tbaa !25
  store i64 %57, ptr addrspace(1) %60, align 8, !tbaa !27
  store i32 1, ptr addrspace(1) %61, align 4, !tbaa !28
  br label %62

62:                                               ; preds = %58, %38
  %63 = zext i32 %12 to i64
  %64 = getelementptr inbounds nuw [64 x [8 x i64]], ptr addrspace(1) %56, i64 0, i64 %63
  store i64 %2, ptr addrspace(1) %64, align 8, !tbaa !17
  %65 = getelementptr inbounds nuw i8, ptr addrspace(1) %64, i64 8
  store i64 %3, ptr addrspace(1) %65, align 8, !tbaa !17
  %66 = getelementptr inbounds nuw i8, ptr addrspace(1) %64, i64 16
  store i64 %4, ptr addrspace(1) %66, align 8, !tbaa !17
  %67 = getelementptr inbounds nuw i8, ptr addrspace(1) %64, i64 24
  store i64 %5, ptr addrspace(1) %67, align 8, !tbaa !17
  %68 = getelementptr inbounds nuw i8, ptr addrspace(1) %64, i64 32
  store i64 %6, ptr addrspace(1) %68, align 8, !tbaa !17
  %69 = getelementptr inbounds nuw i8, ptr addrspace(1) %64, i64 40
  store i64 %7, ptr addrspace(1) %69, align 8, !tbaa !17
  %70 = getelementptr inbounds nuw i8, ptr addrspace(1) %64, i64 48
  store i64 %8, ptr addrspace(1) %70, align 8, !tbaa !17
  %71 = getelementptr inbounds nuw i8, ptr addrspace(1) %64, i64 56
  store i64 %9, ptr addrspace(1) %71, align 8, !tbaa !17
  br i1 %15, label %72, label %88

72:                                               ; preds = %62
  %73 = getelementptr inbounds nuw i8, ptr addrspace(1) %14, i64 32
  %74 = load atomic i64, ptr addrspace(1) %73 syncscope("one-as") monotonic, align 8
  %75 = load i64, ptr addrspace(1) %50, align 8, !tbaa !23
  %76 = and i64 %75, %48
  %77 = getelementptr inbounds nuw %0, ptr addrspace(1) %49, i64 %76
  store i64 %74, ptr addrspace(1) %77, align 8, !tbaa !29
  %78 = cmpxchg ptr addrspace(1) %73, i64 %74, i64 %48 syncscope("one-as") release monotonic, align 8
  %79 = extractvalue { i64, i1 } %78, 1
  br i1 %79, label %85, label %80

80:                                               ; preds = %80, %72
  %81 = phi { i64, i1 } [ %83, %80 ], [ %78, %72 ]
  %82 = extractvalue { i64, i1 } %81, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %82, ptr addrspace(1) %77, align 8, !tbaa !29
  %83 = cmpxchg ptr addrspace(1) %73, i64 %82, i64 %48 syncscope("one-as") release monotonic, align 8
  %84 = extractvalue { i64, i1 } %83, 1
  br i1 %84, label %85, label %80

85:                                               ; preds = %80, %72
  %86 = getelementptr inbounds nuw i8, ptr addrspace(1) %14, i64 16
  %87 = load i64, ptr addrspace(1) %86, align 8
  tail call void @__ockl_hsa_signal_add(i64 %87, i64 noundef 1, i32 noundef 3) #14
  br label %88

88:                                               ; preds = %85, %62
  %89 = getelementptr inbounds nuw i8, ptr addrspace(1) %53, i64 20
  br label %90

90:                                               ; preds = %98, %88
  br i1 %15, label %91, label %94

91:                                               ; preds = %90
  %92 = load atomic i32, ptr addrspace(1) %89 syncscope("one-as") acquire, align 4
  %93 = and i32 %92, 1
  br label %94

94:                                               ; preds = %91, %90
  %95 = phi i32 [ %93, %91 ], [ 1, %90 ]
  %96 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %95)
  %97 = icmp eq i32 %96, 0
  br i1 %97, label %99, label %98

98:                                               ; preds = %94
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  br label %90

99:                                               ; preds = %94
  %100 = load i64, ptr addrspace(1) %64, align 8, !tbaa !17
  %101 = load i64, ptr addrspace(1) %65, align 8, !tbaa !17
  br i1 %15, label %102, label %120

102:                                              ; preds = %99
  %103 = load i64, ptr addrspace(1) %50, align 8, !tbaa !23
  %104 = add i64 %103, 1
  %105 = add i64 %104, %48
  %106 = icmp eq i64 %105, 0
  %107 = select i1 %106, i64 %104, i64 %105
  %108 = getelementptr inbounds nuw i8, ptr addrspace(1) %14, i64 24
  %109 = load atomic i64, ptr addrspace(1) %108 syncscope("one-as") monotonic, align 8
  %110 = load ptr addrspace(1), ptr addrspace(1) %14, align 8, !tbaa !19
  %111 = and i64 %107, %103
  %112 = getelementptr inbounds nuw %0, ptr addrspace(1) %110, i64 %111
  store i64 %109, ptr addrspace(1) %112, align 8, !tbaa !29
  %113 = cmpxchg ptr addrspace(1) %108, i64 %109, i64 %107 syncscope("one-as") release monotonic, align 8
  %114 = extractvalue { i64, i1 } %113, 1
  br i1 %114, label %120, label %115

115:                                              ; preds = %115, %102
  %116 = phi { i64, i1 } [ %118, %115 ], [ %113, %102 ]
  %117 = extractvalue { i64, i1 } %116, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %117, ptr addrspace(1) %112, align 8, !tbaa !29
  %118 = cmpxchg ptr addrspace(1) %108, i64 %117, i64 %107 syncscope("one-as") release monotonic, align 8
  %119 = extractvalue { i64, i1 } %118, 1
  br i1 %119, label %120, label %115

120:                                              ; preds = %115, %102, %99
  %121 = insertelement <2 x i64> poison, i64 %100, i64 0
  %122 = insertelement <2 x i64> %121, i64 %101, i64 1
  ret <2 x i64> %122
}

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #9

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.amdgcn.s.sleep(i32 immarg) #10

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i64 @llvm.amdgcn.ballot.i64(i1) #9

; Function Attrs: convergent norecurse nounwind
define internal void @__ockl_hsa_signal_add(i64 %0, i64 noundef %1, i32 noundef %2) local_unnamed_addr #11 {
  %4 = inttoptr i64 %0 to ptr addrspace(1)
  %5 = getelementptr inbounds nuw i8, ptr addrspace(1) %4, i64 8
  switch i32 %2, label %6 [
    i32 1, label %8
    i32 2, label %8
    i32 3, label %10
    i32 4, label %12
    i32 5, label %14
  ]

6:                                                ; preds = %3
  %7 = atomicrmw add ptr addrspace(1) %5, i64 %1 syncscope("one-as") monotonic, align 8
  br label %16

8:                                                ; preds = %3, %3
  %9 = atomicrmw add ptr addrspace(1) %5, i64 %1 syncscope("one-as") acquire, align 8
  br label %16

10:                                               ; preds = %3
  %11 = atomicrmw add ptr addrspace(1) %5, i64 %1 syncscope("one-as") release, align 8
  br label %16

12:                                               ; preds = %3
  %13 = atomicrmw add ptr addrspace(1) %5, i64 %1 syncscope("one-as") acq_rel, align 8
  br label %16

14:                                               ; preds = %3
  %15 = atomicrmw add ptr addrspace(1) %5, i64 %1 seq_cst, align 8
  br label %16

16:                                               ; preds = %14, %12, %10, %8, %6
  %17 = getelementptr inbounds nuw i8, ptr addrspace(1) %4, i64 16
  %18 = load i64, ptr addrspace(1) %17, align 16, !tbaa !30
  %19 = icmp eq i64 %18, 0
  br i1 %19, label %34, label %20

20:                                               ; preds = %16
  %21 = inttoptr i64 %18 to ptr addrspace(1)
  %22 = getelementptr inbounds nuw i8, ptr addrspace(1) %4, i64 24
  %23 = load i32, ptr addrspace(1) %22, align 8, !tbaa !32
  %24 = zext i32 %23 to i64
  store atomic i64 %24, ptr addrspace(1) %21 syncscope("one-as") release, align 8
  %25 = load i32, ptr addrspace(4) @__oclc_ISA_version, align 4, !tbaa !13
  %26 = icmp slt i32 %25, 9000
  %27 = icmp samesign ult i32 %25, 10000
  %28 = icmp samesign ult i32 %25, 11000
  %29 = select i1 %28, i32 8388607, i32 16777215
  %30 = select i1 %27, i32 16777215, i32 %29
  %31 = select i1 %26, i32 255, i32 %30
  %32 = and i32 %31, %23
  %33 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %32)
  tail call void @llvm.amdgcn.s.sendmsg(i32 1, i32 %33)
  br label %34

34:                                               ; preds = %20, %16
  ret void
}

; Function Attrs: nocallback nounwind willreturn
declare void @llvm.amdgcn.s.sendmsg(i32 immarg, i32) #12

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #13

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #13

; Function Attrs: convergent norecurse nounwind
define internal i64 @__ockl_fprintf_append_args(i64 noundef %0, i32 noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i64 noundef %6, i64 noundef %7, i64 noundef %8, i32 noundef %9) #6 {
  %11 = icmp eq i32 %9, 0
  %12 = or i64 %0, 2
  %13 = select i1 %11, i64 %0, i64 %12
  %14 = and i64 %13, -225
  %15 = zext i32 %1 to i64
  %16 = shl nuw nsw i64 %15, 5
  %17 = or i64 %14, %16
  %18 = tail call <2 x i64> @__ockl_hostcall_preview(i32 noundef 2, i64 noundef %17, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i64 noundef %6, i64 noundef %7, i64 noundef %8) #14
  %19 = extractelement <2 x i64> %18, i64 0
  ret i64 %19
}

; Function Attrs: convergent norecurse nounwind
define internal i64 @__ockl_fprintf_append_string_n(i64 noundef %0, ptr noundef readonly %1, i64 noundef %2, i32 noundef %3) #6 {
  %5 = icmp eq i32 %3, 0
  %6 = or i64 %0, 2
  %7 = select i1 %5, i64 %0, i64 %6
  %8 = icmp eq ptr %1, null
  br i1 %8, label %9, label %13

9:                                                ; preds = %4
  %10 = and i64 %7, -225
  %11 = or disjoint i64 %10, 32
  %12 = tail call <2 x i64> @__ockl_hostcall_preview(i32 noundef 2, i64 noundef %11, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0, i64 noundef 0) #14
  br label %452

13:                                               ; preds = %4
  %14 = and i64 %7, 2
  %15 = and i64 %7, -3
  %16 = insertelement <2 x i64> <i64 poison, i64 0>, i64 %15, i64 0
  br label %17

17:                                               ; preds = %440, %13
  %18 = phi i64 [ %2, %13 ], [ %449, %440 ]
  %19 = phi ptr [ %1, %13 ], [ %450, %440 ]
  %20 = phi <2 x i64> [ %16, %13 ], [ %448, %440 ]
  %21 = icmp ugt i64 %18, 56
  %22 = extractelement <2 x i64> %20, i64 0
  %23 = tail call i64 @llvm.umin.i64(i64 %18, i64 56)
  %24 = trunc nuw nsw i64 %23 to i32
  %25 = select i1 %21, i64 0, i64 %14
  %26 = icmp ugt i64 %18, 7
  br i1 %26, label %29, label %27

27:                                               ; preds = %17
  %28 = icmp eq i64 %18, 0
  br i1 %28, label %82, label %69

29:                                               ; preds = %17
  %30 = load i8, ptr %19, align 1, !tbaa !33
  %31 = zext i8 %30 to i64
  %32 = getelementptr inbounds nuw i8, ptr %19, i64 1
  %33 = load i8, ptr %32, align 1, !tbaa !33
  %34 = zext i8 %33 to i64
  %35 = shl nuw nsw i64 %34, 8
  %36 = or disjoint i64 %35, %31
  %37 = getelementptr inbounds nuw i8, ptr %19, i64 2
  %38 = load i8, ptr %37, align 1, !tbaa !33
  %39 = zext i8 %38 to i64
  %40 = shl nuw nsw i64 %39, 16
  %41 = or disjoint i64 %36, %40
  %42 = getelementptr inbounds nuw i8, ptr %19, i64 3
  %43 = load i8, ptr %42, align 1, !tbaa !33
  %44 = zext i8 %43 to i64
  %45 = shl nuw nsw i64 %44, 24
  %46 = or disjoint i64 %41, %45
  %47 = getelementptr inbounds nuw i8, ptr %19, i64 4
  %48 = load i8, ptr %47, align 1, !tbaa !33
  %49 = zext i8 %48 to i64
  %50 = shl nuw nsw i64 %49, 32
  %51 = or disjoint i64 %46, %50
  %52 = getelementptr inbounds nuw i8, ptr %19, i64 5
  %53 = load i8, ptr %52, align 1, !tbaa !33
  %54 = zext i8 %53 to i64
  %55 = shl nuw nsw i64 %54, 40
  %56 = or i64 %51, %55
  %57 = getelementptr inbounds nuw i8, ptr %19, i64 6
  %58 = load i8, ptr %57, align 1, !tbaa !33
  %59 = zext i8 %58 to i64
  %60 = shl nuw nsw i64 %59, 48
  %61 = or i64 %56, %60
  %62 = getelementptr inbounds nuw i8, ptr %19, i64 7
  %63 = load i8, ptr %62, align 1, !tbaa !33
  %64 = zext i8 %63 to i64
  %65 = shl nuw i64 %64, 56
  %66 = or i64 %61, %65
  %67 = add nsw i32 %24, -8
  %68 = getelementptr inbounds nuw i8, ptr %19, i64 8
  br label %82

69:                                               ; preds = %69, %27
  %70 = phi i32 [ %80, %69 ], [ 0, %27 ]
  %71 = phi i64 [ %79, %69 ], [ 0, %27 ]
  %72 = zext nneg i32 %70 to i64
  %73 = getelementptr inbounds nuw i8, ptr %19, i64 %72
  %74 = load i8, ptr %73, align 1, !tbaa !33
  %75 = zext i8 %74 to i64
  %76 = shl i32 %70, 3
  %77 = zext nneg i32 %76 to i64
  %78 = shl nuw i64 %75, %77
  %79 = or i64 %78, %71
  %80 = add nuw nsw i32 %70, 1
  %81 = icmp eq i32 %80, %24
  br i1 %81, label %82, label %69

82:                                               ; preds = %69, %29, %27
  %83 = phi ptr [ %68, %29 ], [ %19, %27 ], [ %19, %69 ]
  %84 = phi i32 [ %67, %29 ], [ 0, %27 ], [ 0, %69 ]
  %85 = phi i64 [ %66, %29 ], [ 0, %27 ], [ %79, %69 ]
  %86 = icmp ugt i32 %84, 7
  br i1 %86, label %89, label %87

87:                                               ; preds = %82
  %88 = icmp eq i32 %84, 0
  br i1 %88, label %142, label %129

89:                                               ; preds = %82
  %90 = load i8, ptr %83, align 1, !tbaa !33
  %91 = zext i8 %90 to i64
  %92 = getelementptr inbounds nuw i8, ptr %83, i64 1
  %93 = load i8, ptr %92, align 1, !tbaa !33
  %94 = zext i8 %93 to i64
  %95 = shl nuw nsw i64 %94, 8
  %96 = or disjoint i64 %95, %91
  %97 = getelementptr inbounds nuw i8, ptr %83, i64 2
  %98 = load i8, ptr %97, align 1, !tbaa !33
  %99 = zext i8 %98 to i64
  %100 = shl nuw nsw i64 %99, 16
  %101 = or disjoint i64 %96, %100
  %102 = getelementptr inbounds nuw i8, ptr %83, i64 3
  %103 = load i8, ptr %102, align 1, !tbaa !33
  %104 = zext i8 %103 to i64
  %105 = shl nuw nsw i64 %104, 24
  %106 = or disjoint i64 %101, %105
  %107 = getelementptr inbounds nuw i8, ptr %83, i64 4
  %108 = load i8, ptr %107, align 1, !tbaa !33
  %109 = zext i8 %108 to i64
  %110 = shl nuw nsw i64 %109, 32
  %111 = or disjoint i64 %106, %110
  %112 = getelementptr inbounds nuw i8, ptr %83, i64 5
  %113 = load i8, ptr %112, align 1, !tbaa !33
  %114 = zext i8 %113 to i64
  %115 = shl nuw nsw i64 %114, 40
  %116 = or i64 %111, %115
  %117 = getelementptr inbounds nuw i8, ptr %83, i64 6
  %118 = load i8, ptr %117, align 1, !tbaa !33
  %119 = zext i8 %118 to i64
  %120 = shl nuw nsw i64 %119, 48
  %121 = or i64 %116, %120
  %122 = getelementptr inbounds nuw i8, ptr %83, i64 7
  %123 = load i8, ptr %122, align 1, !tbaa !33
  %124 = zext i8 %123 to i64
  %125 = shl nuw i64 %124, 56
  %126 = or i64 %121, %125
  %127 = add nsw i32 %84, -8
  %128 = getelementptr inbounds nuw i8, ptr %83, i64 8
  br label %142

129:                                              ; preds = %129, %87
  %130 = phi i32 [ %140, %129 ], [ 0, %87 ]
  %131 = phi i64 [ %139, %129 ], [ 0, %87 ]
  %132 = zext nneg i32 %130 to i64
  %133 = getelementptr inbounds nuw i8, ptr %83, i64 %132
  %134 = load i8, ptr %133, align 1, !tbaa !33
  %135 = zext i8 %134 to i64
  %136 = shl i32 %130, 3
  %137 = zext nneg i32 %136 to i64
  %138 = shl nuw i64 %135, %137
  %139 = or i64 %138, %131
  %140 = add nuw nsw i32 %130, 1
  %141 = icmp eq i32 %140, %84
  br i1 %141, label %142, label %129

142:                                              ; preds = %129, %89, %87
  %143 = phi ptr [ %128, %89 ], [ %83, %87 ], [ %83, %129 ]
  %144 = phi i32 [ %127, %89 ], [ 0, %87 ], [ 0, %129 ]
  %145 = phi i64 [ %126, %89 ], [ 0, %87 ], [ %139, %129 ]
  %146 = icmp ugt i32 %144, 7
  br i1 %146, label %149, label %147

147:                                              ; preds = %142
  %148 = icmp eq i32 %144, 0
  br i1 %148, label %202, label %189

149:                                              ; preds = %142
  %150 = load i8, ptr %143, align 1, !tbaa !33
  %151 = zext i8 %150 to i64
  %152 = getelementptr inbounds nuw i8, ptr %143, i64 1
  %153 = load i8, ptr %152, align 1, !tbaa !33
  %154 = zext i8 %153 to i64
  %155 = shl nuw nsw i64 %154, 8
  %156 = or disjoint i64 %155, %151
  %157 = getelementptr inbounds nuw i8, ptr %143, i64 2
  %158 = load i8, ptr %157, align 1, !tbaa !33
  %159 = zext i8 %158 to i64
  %160 = shl nuw nsw i64 %159, 16
  %161 = or disjoint i64 %156, %160
  %162 = getelementptr inbounds nuw i8, ptr %143, i64 3
  %163 = load i8, ptr %162, align 1, !tbaa !33
  %164 = zext i8 %163 to i64
  %165 = shl nuw nsw i64 %164, 24
  %166 = or disjoint i64 %161, %165
  %167 = getelementptr inbounds nuw i8, ptr %143, i64 4
  %168 = load i8, ptr %167, align 1, !tbaa !33
  %169 = zext i8 %168 to i64
  %170 = shl nuw nsw i64 %169, 32
  %171 = or disjoint i64 %166, %170
  %172 = getelementptr inbounds nuw i8, ptr %143, i64 5
  %173 = load i8, ptr %172, align 1, !tbaa !33
  %174 = zext i8 %173 to i64
  %175 = shl nuw nsw i64 %174, 40
  %176 = or i64 %171, %175
  %177 = getelementptr inbounds nuw i8, ptr %143, i64 6
  %178 = load i8, ptr %177, align 1, !tbaa !33
  %179 = zext i8 %178 to i64
  %180 = shl nuw nsw i64 %179, 48
  %181 = or i64 %176, %180
  %182 = getelementptr inbounds nuw i8, ptr %143, i64 7
  %183 = load i8, ptr %182, align 1, !tbaa !33
  %184 = zext i8 %183 to i64
  %185 = shl nuw i64 %184, 56
  %186 = or i64 %181, %185
  %187 = add nsw i32 %144, -8
  %188 = getelementptr inbounds nuw i8, ptr %143, i64 8
  br label %202

189:                                              ; preds = %189, %147
  %190 = phi i32 [ %200, %189 ], [ 0, %147 ]
  %191 = phi i64 [ %199, %189 ], [ 0, %147 ]
  %192 = zext nneg i32 %190 to i64
  %193 = getelementptr inbounds nuw i8, ptr %143, i64 %192
  %194 = load i8, ptr %193, align 1, !tbaa !33
  %195 = zext i8 %194 to i64
  %196 = shl i32 %190, 3
  %197 = zext nneg i32 %196 to i64
  %198 = shl nuw i64 %195, %197
  %199 = or i64 %198, %191
  %200 = add nuw nsw i32 %190, 1
  %201 = icmp eq i32 %200, %144
  br i1 %201, label %202, label %189

202:                                              ; preds = %189, %149, %147
  %203 = phi ptr [ %188, %149 ], [ %143, %147 ], [ %143, %189 ]
  %204 = phi i32 [ %187, %149 ], [ 0, %147 ], [ 0, %189 ]
  %205 = phi i64 [ %186, %149 ], [ 0, %147 ], [ %199, %189 ]
  %206 = icmp ugt i32 %204, 7
  br i1 %206, label %209, label %207

207:                                              ; preds = %202
  %208 = icmp eq i32 %204, 0
  br i1 %208, label %262, label %249

209:                                              ; preds = %202
  %210 = load i8, ptr %203, align 1, !tbaa !33
  %211 = zext i8 %210 to i64
  %212 = getelementptr inbounds nuw i8, ptr %203, i64 1
  %213 = load i8, ptr %212, align 1, !tbaa !33
  %214 = zext i8 %213 to i64
  %215 = shl nuw nsw i64 %214, 8
  %216 = or disjoint i64 %215, %211
  %217 = getelementptr inbounds nuw i8, ptr %203, i64 2
  %218 = load i8, ptr %217, align 1, !tbaa !33
  %219 = zext i8 %218 to i64
  %220 = shl nuw nsw i64 %219, 16
  %221 = or disjoint i64 %216, %220
  %222 = getelementptr inbounds nuw i8, ptr %203, i64 3
  %223 = load i8, ptr %222, align 1, !tbaa !33
  %224 = zext i8 %223 to i64
  %225 = shl nuw nsw i64 %224, 24
  %226 = or disjoint i64 %221, %225
  %227 = getelementptr inbounds nuw i8, ptr %203, i64 4
  %228 = load i8, ptr %227, align 1, !tbaa !33
  %229 = zext i8 %228 to i64
  %230 = shl nuw nsw i64 %229, 32
  %231 = or disjoint i64 %226, %230
  %232 = getelementptr inbounds nuw i8, ptr %203, i64 5
  %233 = load i8, ptr %232, align 1, !tbaa !33
  %234 = zext i8 %233 to i64
  %235 = shl nuw nsw i64 %234, 40
  %236 = or i64 %231, %235
  %237 = getelementptr inbounds nuw i8, ptr %203, i64 6
  %238 = load i8, ptr %237, align 1, !tbaa !33
  %239 = zext i8 %238 to i64
  %240 = shl nuw nsw i64 %239, 48
  %241 = or i64 %236, %240
  %242 = getelementptr inbounds nuw i8, ptr %203, i64 7
  %243 = load i8, ptr %242, align 1, !tbaa !33
  %244 = zext i8 %243 to i64
  %245 = shl nuw i64 %244, 56
  %246 = or i64 %241, %245
  %247 = add nsw i32 %204, -8
  %248 = getelementptr inbounds nuw i8, ptr %203, i64 8
  br label %262

249:                                              ; preds = %249, %207
  %250 = phi i32 [ %260, %249 ], [ 0, %207 ]
  %251 = phi i64 [ %259, %249 ], [ 0, %207 ]
  %252 = zext nneg i32 %250 to i64
  %253 = getelementptr inbounds nuw i8, ptr %203, i64 %252
  %254 = load i8, ptr %253, align 1, !tbaa !33
  %255 = zext i8 %254 to i64
  %256 = shl i32 %250, 3
  %257 = zext nneg i32 %256 to i64
  %258 = shl nuw i64 %255, %257
  %259 = or i64 %258, %251
  %260 = add nuw nsw i32 %250, 1
  %261 = icmp eq i32 %260, %204
  br i1 %261, label %262, label %249

262:                                              ; preds = %249, %209, %207
  %263 = phi ptr [ %248, %209 ], [ %203, %207 ], [ %203, %249 ]
  %264 = phi i32 [ %247, %209 ], [ 0, %207 ], [ 0, %249 ]
  %265 = phi i64 [ %246, %209 ], [ 0, %207 ], [ %259, %249 ]
  %266 = icmp ugt i32 %264, 7
  br i1 %266, label %269, label %267

267:                                              ; preds = %262
  %268 = icmp eq i32 %264, 0
  br i1 %268, label %322, label %309

269:                                              ; preds = %262
  %270 = load i8, ptr %263, align 1, !tbaa !33
  %271 = zext i8 %270 to i64
  %272 = getelementptr inbounds nuw i8, ptr %263, i64 1
  %273 = load i8, ptr %272, align 1, !tbaa !33
  %274 = zext i8 %273 to i64
  %275 = shl nuw nsw i64 %274, 8
  %276 = or disjoint i64 %275, %271
  %277 = getelementptr inbounds nuw i8, ptr %263, i64 2
  %278 = load i8, ptr %277, align 1, !tbaa !33
  %279 = zext i8 %278 to i64
  %280 = shl nuw nsw i64 %279, 16
  %281 = or disjoint i64 %276, %280
  %282 = getelementptr inbounds nuw i8, ptr %263, i64 3
  %283 = load i8, ptr %282, align 1, !tbaa !33
  %284 = zext i8 %283 to i64
  %285 = shl nuw nsw i64 %284, 24
  %286 = or disjoint i64 %281, %285
  %287 = getelementptr inbounds nuw i8, ptr %263, i64 4
  %288 = load i8, ptr %287, align 1, !tbaa !33
  %289 = zext i8 %288 to i64
  %290 = shl nuw nsw i64 %289, 32
  %291 = or disjoint i64 %286, %290
  %292 = getelementptr inbounds nuw i8, ptr %263, i64 5
  %293 = load i8, ptr %292, align 1, !tbaa !33
  %294 = zext i8 %293 to i64
  %295 = shl nuw nsw i64 %294, 40
  %296 = or i64 %291, %295
  %297 = getelementptr inbounds nuw i8, ptr %263, i64 6
  %298 = load i8, ptr %297, align 1, !tbaa !33
  %299 = zext i8 %298 to i64
  %300 = shl nuw nsw i64 %299, 48
  %301 = or i64 %296, %300
  %302 = getelementptr inbounds nuw i8, ptr %263, i64 7
  %303 = load i8, ptr %302, align 1, !tbaa !33
  %304 = zext i8 %303 to i64
  %305 = shl nuw i64 %304, 56
  %306 = or i64 %301, %305
  %307 = add nsw i32 %264, -8
  %308 = getelementptr inbounds nuw i8, ptr %263, i64 8
  br label %322

309:                                              ; preds = %309, %267
  %310 = phi i32 [ %320, %309 ], [ 0, %267 ]
  %311 = phi i64 [ %319, %309 ], [ 0, %267 ]
  %312 = zext nneg i32 %310 to i64
  %313 = getelementptr inbounds nuw i8, ptr %263, i64 %312
  %314 = load i8, ptr %313, align 1, !tbaa !33
  %315 = zext i8 %314 to i64
  %316 = shl i32 %310, 3
  %317 = zext nneg i32 %316 to i64
  %318 = shl nuw i64 %315, %317
  %319 = or i64 %318, %311
  %320 = add nuw nsw i32 %310, 1
  %321 = icmp eq i32 %320, %264
  br i1 %321, label %322, label %309

322:                                              ; preds = %309, %269, %267
  %323 = phi ptr [ %308, %269 ], [ %263, %267 ], [ %263, %309 ]
  %324 = phi i32 [ %307, %269 ], [ 0, %267 ], [ 0, %309 ]
  %325 = phi i64 [ %306, %269 ], [ 0, %267 ], [ %319, %309 ]
  %326 = icmp ugt i32 %324, 7
  br i1 %326, label %329, label %327

327:                                              ; preds = %322
  %328 = icmp eq i32 %324, 0
  br i1 %328, label %382, label %369

329:                                              ; preds = %322
  %330 = load i8, ptr %323, align 1, !tbaa !33
  %331 = zext i8 %330 to i64
  %332 = getelementptr inbounds nuw i8, ptr %323, i64 1
  %333 = load i8, ptr %332, align 1, !tbaa !33
  %334 = zext i8 %333 to i64
  %335 = shl nuw nsw i64 %334, 8
  %336 = or disjoint i64 %335, %331
  %337 = getelementptr inbounds nuw i8, ptr %323, i64 2
  %338 = load i8, ptr %337, align 1, !tbaa !33
  %339 = zext i8 %338 to i64
  %340 = shl nuw nsw i64 %339, 16
  %341 = or disjoint i64 %336, %340
  %342 = getelementptr inbounds nuw i8, ptr %323, i64 3
  %343 = load i8, ptr %342, align 1, !tbaa !33
  %344 = zext i8 %343 to i64
  %345 = shl nuw nsw i64 %344, 24
  %346 = or disjoint i64 %341, %345
  %347 = getelementptr inbounds nuw i8, ptr %323, i64 4
  %348 = load i8, ptr %347, align 1, !tbaa !33
  %349 = zext i8 %348 to i64
  %350 = shl nuw nsw i64 %349, 32
  %351 = or disjoint i64 %346, %350
  %352 = getelementptr inbounds nuw i8, ptr %323, i64 5
  %353 = load i8, ptr %352, align 1, !tbaa !33
  %354 = zext i8 %353 to i64
  %355 = shl nuw nsw i64 %354, 40
  %356 = or i64 %351, %355
  %357 = getelementptr inbounds nuw i8, ptr %323, i64 6
  %358 = load i8, ptr %357, align 1, !tbaa !33
  %359 = zext i8 %358 to i64
  %360 = shl nuw nsw i64 %359, 48
  %361 = or i64 %356, %360
  %362 = getelementptr inbounds nuw i8, ptr %323, i64 7
  %363 = load i8, ptr %362, align 1, !tbaa !33
  %364 = zext i8 %363 to i64
  %365 = shl nuw i64 %364, 56
  %366 = or i64 %361, %365
  %367 = add nsw i32 %324, -8
  %368 = getelementptr inbounds nuw i8, ptr %323, i64 8
  br label %382

369:                                              ; preds = %369, %327
  %370 = phi i32 [ %380, %369 ], [ 0, %327 ]
  %371 = phi i64 [ %379, %369 ], [ 0, %327 ]
  %372 = zext nneg i32 %370 to i64
  %373 = getelementptr inbounds nuw i8, ptr %323, i64 %372
  %374 = load i8, ptr %373, align 1, !tbaa !33
  %375 = zext i8 %374 to i64
  %376 = shl i32 %370, 3
  %377 = zext nneg i32 %376 to i64
  %378 = shl nuw i64 %375, %377
  %379 = or i64 %378, %371
  %380 = add nuw nsw i32 %370, 1
  %381 = icmp eq i32 %380, %324
  br i1 %381, label %382, label %369

382:                                              ; preds = %369, %329, %327
  %383 = phi ptr [ %368, %329 ], [ %323, %327 ], [ %323, %369 ]
  %384 = phi i32 [ %367, %329 ], [ 0, %327 ], [ 0, %369 ]
  %385 = phi i64 [ %366, %329 ], [ 0, %327 ], [ %379, %369 ]
  %386 = icmp ugt i32 %384, 7
  br i1 %386, label %389, label %387

387:                                              ; preds = %382
  %388 = icmp eq i32 %384, 0
  br i1 %388, label %440, label %427

389:                                              ; preds = %382
  %390 = load i8, ptr %383, align 1, !tbaa !33
  %391 = zext i8 %390 to i64
  %392 = getelementptr inbounds nuw i8, ptr %383, i64 1
  %393 = load i8, ptr %392, align 1, !tbaa !33
  %394 = zext i8 %393 to i64
  %395 = shl nuw nsw i64 %394, 8
  %396 = or disjoint i64 %395, %391
  %397 = getelementptr inbounds nuw i8, ptr %383, i64 2
  %398 = load i8, ptr %397, align 1, !tbaa !33
  %399 = zext i8 %398 to i64
  %400 = shl nuw nsw i64 %399, 16
  %401 = or disjoint i64 %396, %400
  %402 = getelementptr inbounds nuw i8, ptr %383, i64 3
  %403 = load i8, ptr %402, align 1, !tbaa !33
  %404 = zext i8 %403 to i64
  %405 = shl nuw nsw i64 %404, 24
  %406 = or disjoint i64 %401, %405
  %407 = getelementptr inbounds nuw i8, ptr %383, i64 4
  %408 = load i8, ptr %407, align 1, !tbaa !33
  %409 = zext i8 %408 to i64
  %410 = shl nuw nsw i64 %409, 32
  %411 = or disjoint i64 %406, %410
  %412 = getelementptr inbounds nuw i8, ptr %383, i64 5
  %413 = load i8, ptr %412, align 1, !tbaa !33
  %414 = zext i8 %413 to i64
  %415 = shl nuw nsw i64 %414, 40
  %416 = or i64 %411, %415
  %417 = getelementptr inbounds nuw i8, ptr %383, i64 6
  %418 = load i8, ptr %417, align 1, !tbaa !33
  %419 = zext i8 %418 to i64
  %420 = shl nuw nsw i64 %419, 48
  %421 = or i64 %416, %420
  %422 = getelementptr inbounds nuw i8, ptr %383, i64 7
  %423 = load i8, ptr %422, align 1, !tbaa !33
  %424 = zext i8 %423 to i64
  %425 = shl nuw i64 %424, 56
  %426 = or i64 %421, %425
  br label %440

427:                                              ; preds = %427, %387
  %428 = phi i32 [ %438, %427 ], [ 0, %387 ]
  %429 = phi i64 [ %437, %427 ], [ 0, %387 ]
  %430 = zext nneg i32 %428 to i64
  %431 = getelementptr inbounds nuw i8, ptr %383, i64 %430
  %432 = load i8, ptr %431, align 1, !tbaa !33
  %433 = zext i8 %432 to i64
  %434 = shl i32 %428, 3
  %435 = zext nneg i32 %434 to i64
  %436 = shl nuw i64 %433, %435
  %437 = or i64 %436, %429
  %438 = add nuw nsw i32 %428, 1
  %439 = icmp eq i32 %438, %384
  br i1 %439, label %440, label %427

440:                                              ; preds = %427, %389, %387
  %441 = phi i64 [ %426, %389 ], [ 0, %387 ], [ %437, %427 ]
  %442 = shl nuw nsw i64 %23, 2
  %443 = add nuw nsw i64 %442, 28
  %444 = and i64 %443, 480
  %445 = and i64 %22, -225
  %446 = or i64 %445, %25
  %447 = or i64 %446, %444
  %448 = tail call <2 x i64> @__ockl_hostcall_preview(i32 noundef 2, i64 noundef %447, i64 noundef %85, i64 noundef %145, i64 noundef %205, i64 noundef %265, i64 noundef %325, i64 noundef %385, i64 noundef %441) #14
  %449 = sub i64 %18, %23
  %450 = getelementptr inbounds nuw i8, ptr %19, i64 %23
  %451 = icmp eq i64 %449, 0
  br i1 %451, label %452, label %17

452:                                              ; preds = %440, %9
  %453 = phi <2 x i64> [ %12, %9 ], [ %448, %440 ]
  %454 = extractelement <2 x i64> %453, i64 0
  ret i64 %454
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #8

attributes #0 = { convergent mustprogress noinline noreturn nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1100" "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+atomic-fmin-fmax-global-f32,+ci-insts,+dl-insts,+dot10-insts,+dot12-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32" }
attributes #1 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #2 = { convergent mustprogress noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1100" "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+atomic-fmin-fmax-global-f32,+ci-insts,+dl-insts,+dot10-insts,+dot12-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32" }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { convergent mustprogress noinline norecurse nounwind optnone "amdgpu-flat-work-group-size"="1,1024" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1100" "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+atomic-fmin-fmax-global-f32,+ci-insts,+dl-insts,+dot10-insts,+dot12-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32" "uniform-work-group-size"="true" }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #6 = { convergent norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1100" "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+atomic-fmin-fmax-global-f32,+ci-insts,+dl-insts,+dot10-insts,+dot12-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+gws,+image-insts,+wavefrontsize32" "uniform-work-group-size"="false" }
attributes #7 = { cold convergent norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1100" "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+atomic-fmin-fmax-global-f32,+ci-insts,+dl-insts,+dot10-insts,+dot12-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+gws,+image-insts,+wavefrontsize32" "uniform-work-group-size"="false" }
attributes #8 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #10 = { nocallback nofree nosync nounwind willreturn }
attributes #11 = { convergent norecurse nounwind "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1100" "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+atomic-fmin-fmax-global-f32,+ci-insts,+dl-insts,+dot10-insts,+dot12-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+gws,+image-insts,+wavefrontsize32" "uniform-work-group-size"="false" }
attributes #12 = { nocallback nounwind willreturn }
attributes #13 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #14 = { convergent nounwind }
attributes #15 = { cold convergent nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5, !6}
!opencl.ocl.version = !{!7}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 22.0.0git (https://github.com/llvm/llvm-project.git 9351ad638be5f5cb2f7de300f0518f5ff0923fbf)"}
!6 = !{!"clang version 20.0.0git (https://github.com/ROCm/llvm-project.git f4087f6b428f0e6f575ebac8a8a724dab123d06e)"}
!7 = !{i32 2, i32 0}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.mustprogress"}
!10 = distinct !{!10, !9}
!11 = distinct !{!11, !9}
!12 = distinct !{!12, !9}
!13 = !{!14, !14, i64 0}
!14 = !{!"int", !15, i64 0}
!15 = !{!"omnipotent char", !16, i64 0}
!16 = !{!"Simple C/C++ TBAA"}
!17 = !{!18, !18, i64 0}
!18 = !{!"long", !15, i64 0}
!19 = !{!20, !21, i64 0}
!20 = !{!"", !21, i64 0, !21, i64 8, !22, i64 16, !18, i64 24, !18, i64 32, !18, i64 40}
!21 = !{!"any pointer", !15, i64 0}
!22 = !{!"hsa_signal_s", !18, i64 0}
!23 = !{!20, !18, i64 40}
!24 = !{!20, !21, i64 8}
!25 = !{!26, !14, i64 16}
!26 = !{!"", !18, i64 0, !18, i64 8, !14, i64 16, !14, i64 20}
!27 = !{!26, !18, i64 8}
!28 = !{!26, !14, i64 20}
!29 = !{!26, !18, i64 0}
!30 = !{!31, !18, i64 16}
!31 = !{!"amd_signal_s", !18, i64 0, !15, i64 8, !18, i64 16, !14, i64 24, !14, i64 28, !18, i64 32, !18, i64 40, !15, i64 48, !15, i64 56}
!32 = !{!31, !14, i64 24}
!33 = !{!15, !15, i64 0}