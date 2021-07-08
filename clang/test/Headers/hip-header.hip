// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -target-cpu gfx906 -emit-llvm %s -fcuda-is-device -o - \
// RUN:   -D__HIPCC_RTC__ | FileCheck %s

// expected-no-diagnostics

// Check support for pure and deleted virtual functions
struct base {
  __host__
  __device__
  virtual void pv() = 0;
  __host__
  __device__
  virtual void dv() = delete;
};
struct derived:base {
  __host__
  __device__
  virtual void pv() override {};
};
__device__ void test_vf() {
    derived d;
}
// CHECK: @_ZTV7derived = linkonce_odr unnamed_addr addrspace(1) constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* null, i8* bitcast (void (%struct.derived*)* @_ZN7derived2pvEv to i8*), i8* bitcast (void ()* @__cxa_deleted_virtual to i8*)] }, comdat, align 8
// CHECK: @_ZTV4base = linkonce_odr unnamed_addr addrspace(1) constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* null, i8* bitcast (void ()* @__cxa_pure_virtual to i8*), i8* bitcast (void ()* @__cxa_deleted_virtual to i8*)] }, comdat, align 8

// CHECK: define{{.*}}void @__cxa_pure_virtual()
// CHECK: define{{.*}}void @__cxa_deleted_virtual()

// CHECK-LABEL: amdgpu_kernel void @_Z4kernPff
__global__ void kern(float *x, float y) {
  *x = sin(y);
}

// CHECK-LABEL: define{{.*}} i64 @_Z11test_size_tv
// CHEC: ret i64 8
__device__ size_t test_size_t() {
  return sizeof(size_t);
}

// Check that device malloc and free do not conflict with std headers.
#include <cstdlib>
// CHECK-LABEL: define{{.*}}@_Z11test_malloc
// CHECK: call {{.*}}i8* @malloc(i64
// CHECK: define weak {{.*}}i8* @malloc(i64
__device__ void test_malloc(void *a) {
  a = malloc(42);
}

// CHECK-LABEL: define{{.*}}@_Z9test_free
// CHECK: call {{.*}}i8* @free(i8*
// CHECK: define weak {{.*}}i8* @free(i8*
__device__ void test_free(void *a) {
  free(a);
}
