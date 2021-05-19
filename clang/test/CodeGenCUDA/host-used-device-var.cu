// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -std=c++17 -O3 -mllvm -amdgpu-internalize-symbols -emit-llvm -o - \
// RUN:   | FileCheck -check-prefix=DEV %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip %s \
// RUN:   -std=c++17 -O3 -emit-llvm -o - | FileCheck -check-prefix=HOST %s

// Negative tests.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -std=c++17 -O3 -mllvm -amdgpu-internalize-symbols -emit-llvm -o - \
// RUN:   | FileCheck -check-prefix=DEV-NEG %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip %s \
// RUN:   -std=c++17 -O3 -emit-llvm -o - | FileCheck -check-prefix=HOST-NEG %s

#include "Inputs/cuda.h"

// Check device variables used by neither host nor device functioins are not kept.

// DEV-NEG-NOT: @v1
__device__ int v1;

// DEV-NEG-NOT: @v2
__constant__ int v2;

// DEV-NEG-NOT: @_ZL2v3
static __device__ int v3;

// Check device variables used by host functions are kept.

// DEV-DAG: @u1
__device__ int u1;

// DEV-DAG: @u2
__constant__ int u2;

// Check host-used static device var is in llvm.compiler.used.
// DEV-DAG: @_ZL2u3
static __device__ int u3;

// Check device-used static device var is emitted but is not in llvm.compiler.used.
// DEV-DAG: @_ZL2u4
static __device__ int u4;

// Check device variables with used attribute are always kept.
// DEV-DAG: @u5
__device__ __attribute__((used)) int u5;

// Test external device variable ODR-used by host code is not emitted or registered.
// DEV-NEG-NOT: @ext_var
extern __device__ int ext_var;

// DEV-DAG: @inline_var = linkonce_odr addrspace(1) externally_initialized global i32 0
__device__ inline int inline_var;

template<typename T>
using func_t = T (*) (T, T);

template <typename T>
__device__ T add_func (T x, T y)
{
  return x + y;
}

// DEV-DAG: @_Z10p_add_funcIiE = linkonce_odr addrspace(1) externally_initialized global i32 (i32, i32)* @_Z8add_funcIiET_S0_S0_
template <typename T>
__device__ func_t<T> p_add_func = add_func<T>;

// Check non-constant constexpr variables ODR-used by host code only is not emitted.
// DEV-NEG-NOT: constexpr_var1a
// DEV-NEG-NOT: constexpr_var1b
constexpr int constexpr_var1a = 1;
inline constexpr int constexpr_var1b = 1;

// Check constant constexpr variables ODR-used by host code only.
// Non-inline constexpr variable has internal linkage, therefore it is not accessible by host and not kept.
// Inline constexpr variable has linkonce_ord linkage, therefore it can be accessed by host and kept.
// DEV-NEG-NOT: constexpr_var2a
// DEV-DAG: @constexpr_var2b = linkonce_odr addrspace(4) externally_initialized constant i32 2
__constant__ constexpr int constexpr_var2a = 2;
inline __constant__ constexpr int constexpr_var2b = 2;

void use(func_t<int> p);
void use(const int *p);

void fun1() {
  use(&u1);
  use(&u2);
  use(&u3);
  use(&ext_var);
  use(&inline_var);
  use(p_add_func<int>);
  use(&constexpr_var1a);
  use(&constexpr_var1b);
  use(&constexpr_var2a);
  use(&constexpr_var2b);
}

__global__ void kern1(int **x) {
  *x = &u4;
}

// Check implicit constant variable ODR-used by host code is not emitted.
// DEV-NEG-NOT: _ZN16TestConstexprVar1oE
namespace TestConstexprVar {
char o;
class ou {
public:
  ou(char) { __builtin_strlen(&o); }
};
template < typename ao > struct aw { static constexpr ao c; };
class x {
protected:
  typedef ou (*y)(const x *);
  constexpr x(y ag) : ah(ag) {}
  template < bool * > struct ak;
  template < typename > struct al {
    static bool am;
    static ak< &am > an;
  };
  template < typename ao > static x ap() { (void)aw< ao >::c; return x(nullptr); }
  y ah;
};
template < typename ao > bool x::al< ao >::am(&ap< ao >);
class ar : x {
public:
  constexpr ar() : x(as) {}
  static ou as(const x *) { return 0; }
  al< ar > av;
};
}

// Check the exact list of variables to ensure @_ZL2u4 is not among them.
// DEV: @llvm.compiler.used = {{[^@]*}} @_Z10p_add_funcIiE {{[^@]*}} @_ZL2u3 {{[^@]*}} @constexpr_var2b {{[^@]*}} @inline_var {{[^@]*}} @u1 {{[^@]*}} @u2 {{[^@]*}} @u5

// HOST-DAG: hipRegisterVar{{.*}}@u1
// HOST-DAG: hipRegisterVar{{.*}}@u2
// HOST-DAG: hipRegisterVar{{.*}}@_ZL2u3
// HOST-DAG: hipRegisterVar{{.*}}@constexpr_var2b
// HOST-DAG: hipRegisterVar{{.*}}@u5
// HOST-DAG: hipRegisterVar{{.*}}@inline_var
// HOST-DAG: hipRegisterVar{{.*}}@_Z10p_add_funcIiE
// HOST-NEG-NOT: hipRegisterVar{{.*}}@ext_var
// HOST-NEG-NOT: hipRegisterVar{{.*}}@_ZL2u4
// HOST-NEG-NOT: hipRegisterVar{{.*}}@constexpr_var1a
// HOST-NEG-NOT: hipRegisterVar{{.*}}@constexpr_var1b
// HOST-NEG-NOT: hipRegisterVar{{.*}}@constexpr_var2a
