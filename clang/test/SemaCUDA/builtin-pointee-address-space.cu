// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fsyntax-only -verify=noaux %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple amdgcn-amd-amdhsa -x hip -fsyntax-only -verify=aux %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip -fsyntax-only -verify=device %s

// aux-no-diagnostics
// device-no-diagnostics

#include "Inputs/cuda.h"

__device__ int device_var;
__device__ int device_array[4];
__device__ int *device_ptr;
__constant__ int constant_var;
__device__ const int const_device_var = 1;

void host_queries() {
  (void)__builtin_pointee_address_space(&device_var);
  // noaux-warning@-1 {{cannot determine CUDA/HIP device address space without auxiliary target information; returning default address space}}
  (void)__builtin_pointee_address_space(device_array);
  // noaux-warning@-1 {{cannot determine CUDA/HIP device address space without auxiliary target information; returning default address space}}
  (void)__builtin_pointee_address_space(&constant_var);
  // noaux-warning@-1 {{cannot determine CUDA/HIP device address space without auxiliary target information; returning default address space}}
  (void)__builtin_pointee_address_space(&const_device_var);
  // noaux-warning@-1 {{cannot determine CUDA/HIP device address space without auxiliary target information; returning default address space}}

  (void)__builtin_pointee_address_space((int *)&constant_var);
  (void)__builtin_pointee_address_space(device_ptr);
}
