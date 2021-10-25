// RUN: echo "GPU binary would be here" > %t

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-include-gpubinary %t -o - -x hip\
// RUN:   | FileCheck -check-prefixes=CHECK,GNU %s

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-include-gpubinary %t -o - -x hip\
// RUN:   | FileCheck -check-prefix=GNUNEG %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-llvm %s \
// RUN:     -aux-triple amdgcn-amd-amdhsa -fcuda-include-gpubinary \
// RUN:     %t -o - -x hip\
// RUN:   | FileCheck -check-prefixes=CHECK,MSVC %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-llvm %s \
// RUN:     -aux-triple amdgcn-amd-amdhsa -fcuda-include-gpubinary \
// RUN:     %t -o - -x hip\
// RUN:   | FileCheck -check-prefix=MSVCNEG %s

#include "Inputs/cuda.h"

// Check kernel handles are emitted for non-MSVC target but not for MSVC target.

// GNU: @[[HCKERN:ckernel]] = constant void ()* @[[CSTUB:__device_stub__ckernel]], align 8
// GNU: @[[HNSKERN:_ZN2ns8nskernelEv]] = constant void ()* @[[NSSTUB:_ZN2ns23__device_stub__nskernelEv]], align 8
// GNU: @[[HTKERN:_Z10kernelfuncIiEvv]] = linkonce_odr constant void ()* @[[TSTUB:_Z25__device_stub__kernelfuncIiEvv]], align 8
// GNU: @[[HDKERN:_Z11kernel_declv]] = external constant void ()*, align 8

// MSVCNEG-NOT: @ckernel =
// MSVCNEG-NOT: @{{"\?nskernel@ns@@YAXXZ.*"}} =
// MSVCNEG-NOT: @{{"\?\?\$kernelfunc@H@@YAXXZ.*"}} =
// MSVCNEG-NOT: @{{"\?kernel_decl@@YAXXZ.*"}} =

extern "C" __global__ void ckernel() {}

namespace ns {
__global__ void nskernel() {}
} // namespace ns

template<class T>
__global__ void kernelfunc() {}

__global__ void kernel_decl();

extern "C" void (*kernel_ptr)();
extern "C" void *void_ptr;

extern "C" void launch(void *kern);

// Device side kernel names

// CHECK: @[[CKERN:[0-9]*]] = {{.*}} c"ckernel\00"
// CHECK: @[[NSKERN:[0-9]*]] = {{.*}} c"_ZN2ns8nskernelEv\00"
// CHECK: @[[TKERN:[0-9]*]] = {{.*}} c"_Z10kernelfuncIiEvv\00"

// Non-template kernel stub functions

// GNU: define{{.*}}@[[CSTUB]]
// GNU: call{{.*}}@hipLaunchByPtr{{.*}}@[[HCKERN]]
// MSVC: define{{.*}}@[[CSTUB:ckernel]]
// MSVC: call{{.*}}@hipLaunchByPtr{{.*}}@[[CSTUB]]

// GNU: define{{.*}}@[[NSSTUB]]
// GNU: call{{.*}}@hipLaunchByPtr{{.*}}@[[HNSKERN]]
// MSVC: define{{.*}}@[[NSSTUB:"\?nskernel@ns@@YAXXZ"]]
// MSVC: call{{.*}}@hipLaunchByPtr{{.*}}@[[NSSTUB]]

// Check kernel stub is called for triple chevron.

// CHECK-LABEL: define{{.*}}@fun1()
// CHECK: call void @[[CSTUB]]()
// CHECK: call void @[[NSSTUB]]()
// GNU: call void @[[TSTUB]]()
// GNU: call void @[[DSTUB:_Z26__device_stub__kernel_declv]]()
// MSVC: call void @[[TSTUB:"\?\?\$kernelfunc@H@@YAXXZ"]]()
// MSVC: call void @[[DSTUB:"\?kernel_decl@@YAXXZ"]]()

extern "C" void fun1(void) {
  ckernel<<<1, 1>>>();
  ns::nskernel<<<1, 1>>>();
  kernelfunc<int><<<1, 1>>>();
  kernel_decl<<<1, 1>>>();
}

// Template kernel stub functions

// CHECK: define{{.*}}@[[TSTUB]]
// GNU: call{{.*}}@hipLaunchByPtr{{.*}}@[[HTKERN]]
// MSVC: call{{.*}}@hipLaunchByPtr{{.*}}@[[TSTUB]]

// Check declaration of stub function for external kernel.

// CHECK: declare{{.*}}@[[DSTUB]]

// Check kernel handle is used for passing the kernel as a function pointer
// for non-MSVC target but kernel stub is used for MSVC target.

// CHECK-LABEL: define{{.*}}@fun2()
// GNU: call void @launch({{.*}}[[HCKERN]]
// GNU: call void @launch({{.*}}[[HNSKERN]]
// GNU: call void @launch({{.*}}[[HTKERN]]
// GNU: call void @launch({{.*}}[[HDKERN]]
// MSVC: call void @launch({{.*}}[[CSTUB]]
// MSVC: call void @launch({{.*}}[[NSSTUB]]
// MSVC: call void @launch({{.*}}[[TSTUB]]
// MSVC: call void @launch({{.*}}[[DSTUB]]
extern "C" void fun2() {
  launch((void *)ckernel);
  launch((void *)ns::nskernel);
  launch((void *)kernelfunc<int>);
  launch((void *)kernel_decl);
}

// Check kernel handle is used for assigning a kernel to a function pointer for
// non-MSVC target but kernel stub is used for MSVC target.

// CHECK-LABEL: define{{.*}}@fun3()
// GNU:  store void ()* bitcast (void ()** @[[HCKERN]] to void ()*), void ()** @kernel_ptr, align 8
// GNU:  store void ()* bitcast (void ()** @[[HCKERN]] to void ()*), void ()** @kernel_ptr, align 8
// GNU:  store i8* bitcast (void ()** @[[HCKERN]] to i8*), i8** @void_ptr, align 8
// GNU:  store i8* bitcast (void ()** @[[HCKERN]] to i8*), i8** @void_ptr, align 8
// MSVC:  store void ()* @[[CSTUB]], void ()** @kernel_ptr, align 8
// MSVC:  store void ()* @[[CSTUB]], void ()** @kernel_ptr, align 8
// MSVC:  store i8* bitcast (void ()* @[[CSTUB]] to i8*), i8** @void_ptr, align 8
// MSVC:  store i8* bitcast (void ()* @[[CSTUB]] to i8*), i8** @void_ptr, align 8
extern "C" void fun3() {
  kernel_ptr = ckernel;
  kernel_ptr = &ckernel;
  void_ptr = (void *)ckernel;
  void_ptr = (void *)&ckernel;
}

// Check kernel stub is loaded from kernel handle when function pointer is
// used with triple chevron for non-MSVC target but kernel stub is directly
// used without extra indirection for MSVC target.

// CHECK-LABEL: define{{.*}}@fun4()
// GNU:  store void ()* bitcast (void ()** @[[HCKERN]] to void ()*), void ()** @kernel_ptr
// GNU:  call i32 @{{.*hipConfigureCall}}
// GNU:  %[[HANDLE:.*]] = load void ()*, void ()** @kernel_ptr, align 8
// GNU:  %[[CAST:.*]] = bitcast void ()* %[[HANDLE]] to void ()**
// GNU:  %[[STUB:.*]] = load void ()*, void ()** %[[CAST]], align 8
// GNU:  call void %[[STUB]]()
// MSVC:  store void ()* @[[CSTUB]], void ()** @kernel_ptr
// MSVC:  call i32 @{{.*hipConfigureCall}}
// MSVC:  %[[STUB:.*]] = load void ()*, void ()** @kernel_ptr, align 8
// MSVC:  call void %[[STUB]]()
extern "C" void fun4() {
  kernel_ptr = ckernel;
  kernel_ptr<<<1,1>>>();
}

// Check kernel handle is passed to a function for non-MSVC target but
// kernel stub is passed for MSVC target.

// CHECK-LABEL: define{{.*}}@fun5()
// GNU:  store void ()* bitcast (void ()** @[[HCKERN]] to void ()*), void ()** @kernel_ptr
// MSVC:  store void ()* @[[CSTUB]], void ()** @kernel_ptr
// CHECK:  %[[HANDLE:.*]] = load void ()*, void ()** @kernel_ptr, align 8
// CHECK:  %[[CAST:.*]] = bitcast void ()* %[[HANDLE]] to i8*
// CHECK:  call void @launch(i8* %[[CAST]])
extern "C" void fun5() {
  kernel_ptr = ckernel;
  launch((void *)kernel_ptr);
}

// Check kernel handle is registered for non-MSVC target but kernel stub
// is registered for MSVC target.

// CHECK-LABEL: define{{.*}}@__hip_register_globals
// GNU: call{{.*}}@__hipRegisterFunction{{.*}}@[[HCKERN]]{{.*}}@[[CKERN]]
// GNU: call{{.*}}@__hipRegisterFunction{{.*}}@[[HNSKERN]]{{.*}}@[[NSKERN]]
// GNU: call{{.*}}@__hipRegisterFunction{{.*}}@[[HTKERN]]{{.*}}@[[TKERN]]
// GNUNEG-NOT: call{{.*}}@__hipRegisterFunction{{.*}}@__device_stub__ckernel{{.*}}@{{[0-9]*}}
// GNUNEG-NOT: call{{.*}}@__hipRegisterFunction{{.*}}@_ZN2ns23__device_stub__nskernelEv{{.*}}@{{[0-9]*}}
// GNUNEG-NOT: call{{.*}}@__hipRegisterFunction{{.*}}@_Z25__device_stub__kernelfuncIiEvv{{.*}}@{{[0-9]*}}
// GNUNEG-NOT: call{{.*}}@__hipRegisterFunction{{.*}}@_Z26__device_stub__kernel_declv{{.*}}@{{[0-9]*}}
// MSVC: call{{.*}}@__hipRegisterFunction{{.*}}@[[CSTUB]]{{.*}}@[[CKERN]]
// MSVC: call{{.*}}@__hipRegisterFunction{{.*}}@[[NSSTUB]]{{.*}}@[[NSKERN]]
// MSVC: call{{.*}}@__hipRegisterFunction{{.*}}@[[TSTUB]]{{.*}}@[[TKERN]]
// MSVCNEG-NOT: call{{.*}}@__hipRegisterFunction{{.*}}@"\?kernel_decl@@YAXXZ"{{.*}}@{{[0-9]*}}
