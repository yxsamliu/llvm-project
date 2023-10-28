// RUN: %clang_cc1 -isystem %S/Inputs  -fsyntax-only %s

void foo();
void bar();

#include <cuda.h>
#include <forced-host-device.h>

void foo();
void bar();

void host_fun() {
  foo();
  bar();
}

__device__ void device_fun() {
  foo();
  bar();
}
