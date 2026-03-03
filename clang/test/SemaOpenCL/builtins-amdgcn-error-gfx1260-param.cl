// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-- -target-cpu gfx1260 -verify -S -o - %s

typedef int v2i __attribute__((ext_vector_type(2)));

void test_amdgcn_ds_block_load_mcast(__attribute__((address_space(10))) int* outptr, v2i vsrc, short offset)
{
  __builtin_amdgcn_ds_block_load_mcast_b128(outptr, vsrc, offset); // expected-error {{'__builtin_amdgcn_ds_block_load_mcast_b128' must be a constant integer}}
  __builtin_amdgcn_ds_block_load_mcast_b256(outptr, vsrc, offset); // expected-error {{'__builtin_amdgcn_ds_block_load_mcast_b256' must be a constant integer}}
  __builtin_amdgcn_ds_block_load_mcast_b512(outptr, vsrc, offset); // expected-error {{'__builtin_amdgcn_ds_block_load_mcast_b512' must be a constant integer}}
  __builtin_amdgcn_ds_block_load_mcast_b1024(outptr, vsrc, offset); // expected-error {{'__builtin_amdgcn_ds_block_load_mcast_b1024' must be a constant integer}}
}
