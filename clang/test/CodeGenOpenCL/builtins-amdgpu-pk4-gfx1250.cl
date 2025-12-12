// RUN: not %clang_cc1 -cl-std=CL2.0 -O0 -triple amdgcn-unknown-unknown -target-cpu gfx1250 -emit-llvm -o - %s 2>&1 | FileCheck %s

// REQUIRES: amdgpu-registered-target

typedef __bf16 __attribute__((ext_vector_type(4))) bfloat4;
typedef half __attribute__((ext_vector_type(4))) half4;

//CHECK: error: '__builtin_amdgcn_pk4_fma_f16' needs target feature pk4-insts
//CHECK:  error: '__builtin_amdgcn_pk4_mul_f16' needs target feature pk4-insts
//CHECK:  error: '__builtin_amdgcn_pk4_max3_num_f16' needs target feature pk4-insts
//CHECK:  error: '__builtin_amdgcn_pk4_min3_num_f16' needs target feature pk4-insts
//CHECK:  error: '__builtin_amdgcn_pk4_maximum3_f16' needs target feature pk4-insts
//CHECK:  error: '__builtin_amdgcn_pk4_minimum3_f16' needs target feature pk4-insts
//CHECK:  error: '__builtin_amdgcn_pk4_add_f16' needs target feature pk4-insts
//CHECK:  error: '__builtin_amdgcn_pk4_max_num_f16' needs target feature pk4-insts
//CHECK:  error: '__builtin_amdgcn_pk4_min_num_f16' needs target feature pk4-insts
//CHECK:  error: '__builtin_amdgcn_pk4_fma_bf16' needs target feature pk4-insts
//CHECK:  error: '__builtin_amdgcn_pk4_add_bf16' needs target feature pk4-insts
//CHECK:  error: '__builtin_amdgcn_pk4_mul_bf16' needs target feature pk4-insts
//CHECK:  error: '__builtin_amdgcn_pk4_max_num_bf16' needs target feature pk4-insts
//CHECK:  error: '__builtin_amdgcn_pk4_min_num_bf16' needs target feature pk4-insts

void test_pk4(half4 a, half4 b, half4 c, bfloat4 ba, bfloat4 bb, bfloat4 bc, half4 *outf[9], bfloat4 *outbf[5]) {
  *outf[0] = __builtin_amdgcn_pk4_fma_f16(a, b, c);
  *outf[1] = __builtin_amdgcn_pk4_mul_f16(a, b);
  *outf[2] = __builtin_amdgcn_pk4_max3_num_f16(a, b, c);
  *outf[3] = __builtin_amdgcn_pk4_min3_num_f16(a, b, c);
  *outf[4] = __builtin_amdgcn_pk4_maximum3_f16(a, b, c);
  *outf[5] = __builtin_amdgcn_pk4_minimum3_f16(a, b, c);
  *outf[6] = __builtin_amdgcn_pk4_add_f16(a, b);
  *outf[7] = __builtin_amdgcn_pk4_max_num_f16(a, b);
  *outf[8] = __builtin_amdgcn_pk4_min_num_f16(a, b);

  *outbf[0] = __builtin_amdgcn_pk4_fma_bf16(ba, bb, bc);
  *outbf[1] = __builtin_amdgcn_pk4_add_bf16(ba, bb);
  *outbf[2] = __builtin_amdgcn_pk4_mul_bf16(ba, bb);
  *outbf[3] = __builtin_amdgcn_pk4_max_num_bf16(ba, bb);
  *outbf[4] = __builtin_amdgcn_pk4_min_num_bf16(ba, bb);
}
