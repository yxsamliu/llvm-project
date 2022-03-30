// RUN: %clang_cc1 -emit-llvm -o - -aux-triple x86_64-pc-windows-msvc \
// RUN:   -o %t.dev -fms-extensions -triple amdgcn-amd-amdhsa \
// RUN:   -target-cpu gfx1030 -fcuda-is-device -x hip %s

// RUN: %clang_cc1 -emit-llvm -o - -triple x86_64-pc-windows-msvc \
// RUN:   -o %t.host -fms-extensions -aux-triple amdgcn-amd-amdhsa \
// RUN:   -aux-target-cpu gfx1030 -x hip %s

// RUN: %clang_cc1 -emit-llvm -o - -triple x86_64-pc-windows-msvc \
// RUN:   -o %t.as_cpp -fms-extensions -x c++ %s

// RUN: cat %t.dev %t.host | FileCheck %s

// RUN: cat %t.host %t.as_cpp | FileCheck -check-prefix=CPP %s

#if __HIP__
#include "Inputs/cuda.h"
#endif

// Check local struct 'Op' uses Itanium mangling number instead of MSVC mangling
// number in device side name mangling. It is the same in device and host
// compilation.

// CHECK: define amdgpu_kernel void @[[KERN:_Z6kernelIZN4TestIiE3runEvE2OpEvv]](
// CHECK: @{{.*}} = {{.*}}c"[[KERN]]\00"

// CHECK-NOT: @{{.*}} = {{.*}}c"_Z6kernelIZN4TestIiE3runEvE2Op_1Evv\00"
#if __HIP__
template<typename T>
__attribute__((global)) void kernel()
{
}
#endif

// Check local struct 'Op' uses MSVC mangling number in host function name mangling.
// It is the same when compiled as HIP or C++ program.

// CPP: call void @[[FUN:"\?\?\$fun@UOp@\?2\?\?run@\?\$Test@H@@QEAAXXZ@@@YAXXZ"]]()
// CPP: call void @[[FUN]]()
template<typename T>
void fun()
{
}

template <typename T>
class Test {
public:
  void run()
  {
    struct Op
    {
    };
#if __HIP__
    kernel<Op><<<1, 1>>>();
#endif
    fun<Op>();
  }
};

int main() {
  Test<int> A;
  A.run();
}
