// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1250 -verify -emit-llvm -o - %s

// REQUIRES: amdgpu-registered-target

typedef unsigned int uint;

typedef __bf16 __attribute__((ext_vector_type(4))) bfloat4;
typedef half __attribute__((ext_vector_type(4))) half4;
typedef unsigned short __attribute__((ext_vector_type(8))) ushort8;
typedef unsigned short __attribute__((ext_vector_type(32))) ushort32;

void test_unsupported(half4 a, half4 b,
              half4 c, bfloat4 ba, 
              bfloat4 bb, bfloat4 bc,
              ushort8 inpa, ushort32 inpb,
              half4 *outf[9], bfloat4 *outbf[5],
              ushort8 *outa, ushort32 *outb,
              void *sem) {
  *outf[0] = __builtin_amdgcn_pk4_fma_f16(a, b, c); // expected-error {{'__builtin_amdgcn_pk4_fma_f16' needs target feature pk4-insts}}
  *outf[1] = __builtin_amdgcn_pk4_mul_f16(a, b); // expected-error {{'__builtin_amdgcn_pk4_mul_f16' needs target feature pk4-insts}}
  *outf[2] = __builtin_amdgcn_pk4_max3_num_f16(a, b, c); // expected-error {{'__builtin_amdgcn_pk4_max3_num_f16' needs target feature pk4-insts}}
  *outf[3] = __builtin_amdgcn_pk4_min3_num_f16(a, b, c); // expected-error {{'__builtin_amdgcn_pk4_min3_num_f16' needs target feature pk4-insts}}
  *outf[4] = __builtin_amdgcn_pk4_maximum3_f16(a, b, c); // expected-error {{'__builtin_amdgcn_pk4_maximum3_f16' needs target feature pk4-insts}}
  *outf[5] = __builtin_amdgcn_pk4_minimum3_f16(a, b, c); // expected-error {{'__builtin_amdgcn_pk4_minimum3_f16' needs target feature pk4-insts}}
  *outf[6] = __builtin_amdgcn_pk4_add_f16(a, b); // expected-error {{'__builtin_amdgcn_pk4_add_f16' needs target feature pk4-insts}}
  *outf[7] = __builtin_amdgcn_pk4_max_num_f16(a, b); // expected-error {{'__builtin_amdgcn_pk4_max_num_f16' needs target feature pk4-insts}}
  *outf[8] = __builtin_amdgcn_pk4_min_num_f16(a, b); // expected-error {{'__builtin_amdgcn_pk4_min_num_f16' needs target feature pk4-insts}}

  *outbf[0] = __builtin_amdgcn_pk4_fma_bf16(ba, bb, bc); // expected-error {{'__builtin_amdgcn_pk4_fma_bf16' needs target feature pk4-insts}}
  *outbf[1] = __builtin_amdgcn_pk4_add_bf16(ba, bb); // expected-error {{'__builtin_amdgcn_pk4_add_bf16' needs target feature pk4-insts}}
  *outbf[2] = __builtin_amdgcn_pk4_mul_bf16(ba, bb); // expected-error {{'__builtin_amdgcn_pk4_mul_bf16' needs target feature pk4-insts}}
  *outbf[3] = __builtin_amdgcn_pk4_max_num_bf16(ba, bb); // expected-error {{'__builtin_amdgcn_pk4_max_num_bf16' needs target feature pk4-insts}}
  *outbf[4] = __builtin_amdgcn_pk4_min_num_bf16(ba, bb); // expected-error {{'__builtin_amdgcn_pk4_min_num_bf16' needs target feature pk4-insts}}

  __builtin_amdgcn_s_sema_set_state(sem, 1); // expected-error {{'__builtin_amdgcn_s_sema_set_state' needs target feature semaphores}}
  __builtin_amdgcn_s_sema_set_limit(sem, 1); // expected-error {{'__builtin_amdgcn_s_sema_set_limit' needs target feature semaphores}}
  __builtin_amdgcn_s_sema_signal(sem, 0); // expected-error {{'__builtin_amdgcn_s_sema_signal' needs target feature semaphores}}
  __builtin_amdgcn_s_sema_wait(sem); // expected-error {{'__builtin_amdgcn_s_sema_wait' needs target feature semaphores}}

  *outa = __builtin_amdgcn_wmma_tr_16x16_b16(inpa); // expected-error {{'__builtin_amdgcn_wmma_tr_16x16_b16' needs target feature wmma-transpose-insts}}
  *outb = __builtin_amdgcn_wmma_tr_32x32_b16(inpb); // expected-error {{'__builtin_amdgcn_wmma_tr_32x32_b16' needs target feature wmma-transpose-insts}}
}
