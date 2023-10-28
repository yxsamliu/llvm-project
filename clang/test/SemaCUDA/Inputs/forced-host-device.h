void bar();
#pragma clang force_cuda_host_device begin
void foo();
void bar();
#pragma clang force_cuda_host_device end
void foo() {}
