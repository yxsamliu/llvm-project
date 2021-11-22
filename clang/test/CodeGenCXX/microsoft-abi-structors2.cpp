// RUN: %clang_cc1 -emit-llvm -fno-rtti %s -std=c++11 -o - -mconstructor-aliases -triple=i386-pc-win32 -fno-rtti > %t
// RUN: FileCheck %s < %t

struct D {
  ~D();
};

D::~D() {
 static int dtor_static;
 // CHECK that the static in the dtor gets mangled correctly:
 // CHECK: @"?dtor_static@?1???1D@basic@@QAE@XZ@4HA"
}
