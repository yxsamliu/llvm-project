// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple amdgcn-amd-amdhsa -x hip -DHOST_TEST -emit-llvm -o - %s | FileCheck --check-prefix=HOST %s

#include "Inputs/cuda.h"

__device__ int device_var;
__device__ int device_array[4];
__device__ int *device_ptr;
__constant__ int constant_var;
__device__ const int const_device_var = 1;

#ifdef HOST_TEST

template <class T> constexpr int get_host_as(T *p) {
  return __builtin_pointee_address_space(p);
}

static_assert(__builtin_pointee_address_space(&device_var) == 1);
static_assert(__builtin_pointee_address_space(&constant_var) == 4);
static_assert(__builtin_pointee_address_space(&const_device_var) == 4);
static_assert(get_host_as(&device_var) == 1);
static_assert(get_host_as(&constant_var) == 4);
static_assert(get_host_as(&const_device_var) == 4);

extern "C" int test_host_device_var() {
  return __builtin_pointee_address_space(&device_var);
}

// HOST-LABEL: define{{.*}} i32 @test_host_device_var(
// HOST: ret i32 1

extern "C" int test_host_constant_var() {
  return __builtin_pointee_address_space(&constant_var);
}

// HOST-LABEL: define{{.*}} i32 @test_host_constant_var(
// HOST: ret i32 4

extern "C" int test_host_const_device_var() {
  return __builtin_pointee_address_space(&const_device_var);
}

// HOST-LABEL: define{{.*}} i32 @test_host_const_device_var(
// HOST: ret i32 4

#else

constexpr int get_as(int *p) {
  return __builtin_pointee_address_space(p);
}

template <class T> constexpr int get_template_as(T *p) {
  return __builtin_pointee_address_space(p);
}

template <int AS> struct AddressSpaceSpecialization;
template <> struct AddressSpaceSpecialization<0> {
  static constexpr int value = 0;
};
template <> struct AddressSpaceSpecialization<1> {
  static constexpr int value = 1;
};
template <> struct AddressSpaceSpecialization<3> {
  static constexpr int value = 3;
};
template <> struct AddressSpaceSpecialization<4> {
  static constexpr int value = 4;
};

static_assert(get_as(&device_var) == 1);
static_assert(get_as(&constant_var) == 4);
static_assert(get_as((int *)&constant_var) == 0);
static_assert(get_template_as(&device_var) == 1);
static_assert(get_template_as(&constant_var) == 4);
static_assert(get_template_as(&const_device_var) == 4);
static_assert(__builtin_pointee_address_space(device_array) == 1);
static_assert(__builtin_pointee_address_space(device_ptr) == 0);
static_assert(
    AddressSpaceSpecialization<
        __builtin_pointee_address_space(&device_var)>::value == 1);
static_assert(
    AddressSpaceSpecialization<
        __builtin_pointee_address_space(&constant_var)>::value == 4);
static_assert(
    AddressSpaceSpecialization<get_template_as(&constant_var)>::value == 4);

extern "C" __device__ int test_generic_pointer(int *p) {
  return __builtin_pointee_address_space(p);
}

// CHECK-LABEL: define{{.*}} i32 @test_generic_pointer(
// CHECK: ret i32 0

extern "C" __device__ int test_shared_local() {
  __shared__ int shared_var;
  static_assert(__builtin_pointee_address_space(&shared_var) == 3);
  static_assert(
      AddressSpaceSpecialization<
          __builtin_pointee_address_space(&shared_var)>::value == 3);
  return __builtin_pointee_address_space(&shared_var);
}

// CHECK-LABEL: define{{.*}} i32 @test_shared_local(
// CHECK: ret i32 3

extern "C" __device__ int test_device_var() {
  return __builtin_pointee_address_space(&device_var);
}

// CHECK-LABEL: define{{.*}} i32 @test_device_var(
// CHECK: ret i32 1

extern "C" __device__ int test_device_array() {
  return __builtin_pointee_address_space(device_array);
}

// CHECK-LABEL: define{{.*}} i32 @test_device_array(
// CHECK: ret i32 1

extern "C" __device__ int test_device_pointer_value() {
  return __builtin_pointee_address_space(device_ptr);
}

// CHECK-LABEL: define{{.*}} i32 @test_device_pointer_value(
// CHECK: ret i32 0

extern "C" __device__ int test_constant_var() {
  return __builtin_pointee_address_space(&constant_var);
}

// CHECK-LABEL: define{{.*}} i32 @test_constant_var(
// CHECK: ret i32 4

extern "C" __device__ int test_const_device_var() {
  return __builtin_pointee_address_space(&const_device_var);
}

// CHECK-LABEL: define{{.*}} i32 @test_const_device_var(
// CHECK: ret i32 4

extern "C" __device__ int test_explicit_cast() {
  return __builtin_pointee_address_space((int *)&constant_var);
}

// CHECK-LABEL: define{{.*}} i32 @test_explicit_cast(
// CHECK: ret i32 0

extern "C" __device__ int
test_target_address_space_3(int __attribute__((address_space(3))) *p) {
  return __builtin_pointee_address_space(p);
}

// CHECK-LABEL: define{{.*}} i32 @test_target_address_space_3(
// CHECK: ret i32 3

extern "C" __device__ int
test_target_address_space_4(int __attribute__((address_space(4))) *p) {
  return __builtin_pointee_address_space(p);
}

// CHECK-LABEL: define{{.*}} i32 @test_target_address_space_4(
// CHECK: ret i32 4

#endif
