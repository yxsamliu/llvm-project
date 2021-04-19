// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -target-cpu gfx906 -emit-llvm %s -fcuda-is-device -o - \
// RUN:   -D__HIPCC_RTC__ | FileCheck %s
// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -target-cpu gfx906 -emit-llvm %s -fcuda-is-device -o - \
// RUN:   -D__HIPCC_RTC__ -std=c++14 | FileCheck %s


// expected-no-diagnostics

// CHECK-LABEL: amdgpu_kernel void @_Z4kernPff
__global__ void kern(float *x, float y) {
  *x = sin(y);
}

// CHECK-LABEL: define{{.*}} i64 @_Z11test_size_tv
// CHECK: ret i64 8
__device__ size_t test_size_t() {
  return sizeof(size_t);
}

// Check there is no ambiguity when calling overloaded math functions.

// CHECK-LABEL: define{{.*}}@_Z10test_floorv
// CHECK: call {{.*}}double @__ocml_floor_f64(double
__device__ float test_floor() {
  return floor(5);
}

// CHECK-LABEL: define{{.*}}@_Z8test_maxv
// CHECK: call {{.*}}double @__ocml_fmax_f64(double {{.*}}, double
__device__ float test_max() {
  return max(5, 6.0);
}
