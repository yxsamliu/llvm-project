// RUN: %clang_cc1 -std=c++14 %s -emit-llvm -o - -triple nvptx \
// RUN:   -fcuda-is-device | FileCheck --check-prefixes=COM,CXX14 %s
// RUN: %clang_cc1 -std=c++17 %s -emit-llvm -o - -triple nvptx \
// RUN:   -fcuda-is-device | FileCheck --check-prefixes=COM,CXX17 %s

#include "Inputs/cuda.h"

// COM: @_ZL1a = internal {{.*}}constant i32 7
constexpr int a = 7;
__constant__ const int &use_a = a;

namespace B {
 // COM: @_ZN1BL1bE = internal {{.*}}constant i32 9
  constexpr int b = 9;
}
__constant__ const int &use_B_b = B::b;

struct Q {
  // CXX14: @_ZN1Q1kE = available_externally {{.*}}constant i32 5
  // CXX17: @_ZN1Q1kE = linkonce_odr {{.*}}constant i32 5
  static constexpr int k = 5;
};
__constant__ const int &use_Q_k = Q::k;

template<typename T> struct X {
  // CXX14: @_ZN1XIiE1aE = available_externally {{.*}}constant i32 123
  // CXX17: @_ZN1XIiE1aE = linkonce_odr {{.*}}constant i32 123
  static constexpr int a = 123;
};
__constant__ const int &use_X_a = X<int>::a;
