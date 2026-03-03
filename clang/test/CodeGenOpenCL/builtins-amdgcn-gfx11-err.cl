// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1100 -verify -emit-llvm -o - %s

void builtin_test_unsupported(int a, int b, void *sem) {
  __builtin_amdgcn_s_sleep_var(a); // expected-error {{'__builtin_amdgcn_s_sleep_var' needs target feature gfx12-insts}}
  b = __builtin_amdgcn_ds_bpermute_fi_b32(a, b); // expected-error {{'__builtin_amdgcn_ds_bpermute_fi_b32' needs target feature gfx12-insts}}

  __builtin_amdgcn_s_sema_set_state(sem, 1); // expected-error {{'__builtin_amdgcn_s_sema_set_state' needs target feature semaphores}}
  __builtin_amdgcn_s_sema_set_limit(sem, 1); // expected-error {{'__builtin_amdgcn_s_sema_set_limit' needs target feature semaphores}}
  __builtin_amdgcn_s_sema_signal(sem, 0); // expected-error {{'__builtin_amdgcn_s_sema_signal' needs target feature semaphores}}
  __builtin_amdgcn_s_sema_wait(sem); // expected-error {{'__builtin_amdgcn_s_sema_wait' needs target feature semaphores}}
}
