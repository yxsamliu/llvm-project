// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -O0 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple r600-unknown-unknown -O0 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefix=R600

typedef struct { int x; } ndrange_t;
typedef int int8 __attribute__((ext_vector_type(8)));

kernel void parent(global int *out, int value) {
  queue_t queue;
  ndrange_t range;

  enqueue_kernel(queue, 0, range, ^(void) { out[0] = value; });
  enqueue_kernel(queue, 0, range,
                 ^(local void *scratch) { ((local int *)scratch)[0] = value; },
                 4);
  enqueue_kernel(queue, 0, range, 0, 0, 0,
                 ^(void) { out[1] = value; });
  enqueue_kernel(queue, 0, range, 0, 0, 0,
                 ^(local void *scratch) { ((local int *)scratch)[0] = value; },
                 4);
  enqueue_kernel(queue, 0, range, ^(void) {});
}

kernel void aligned(global int *out, int8 vector) {
  queue_t queue;
  ndrange_t range;
  enqueue_kernel(queue, 0, range, ^(void) { out[0] = vector.s0; });
}

kernel void empty_local(void) {
  queue_t queue;
  ndrange_t range;
  enqueue_kernel(queue, 0, range,
                 ^(local void *scratch) { ((local int *)scratch)[0] = 1; },
                 4);
}

// Each enqueue form uses the runtime entry point that skips the block header.
// CHECK: call i32 @__enqueue_kernel_basic_captures(
// CHECK: call i32 @__enqueue_kernel_varargs_captures(
// CHECK: call i32 @__enqueue_kernel_basic_events_captures(
// CHECK: call i32 @__enqueue_kernel_events_varargs_captures(
// CHECK: call i32 @__enqueue_kernel_basic_captures(
// CHECK-LABEL: define dso_local amdgpu_kernel void @empty_local(
// CHECK: call i32 @__enqueue_kernel_varargs_captures(

// The captured block passes only its payload and restores it after the header.
// CHECK-LABEL: define internal amdgpu_kernel void @__parent_block_invoke_kernel(
// CHECK-SAME: <{ ptr addrspace(1), i32 }> %{{.*}})
// CHECK: getelementptr inbounds {{.*}}, ptr addrspace(5) %{{.*}}, i32 0, i32 3
// CHECK: store <{ ptr addrspace(1), i32 }> %{{.*}}, ptr addrspace(5) %{{.*}}, align 1
// CHECK: call void @__parent_block_invoke(

// A block without captures needs no explicit kernel argument.
// CHECK-LABEL: define internal amdgpu_kernel void @__parent_block_invoke_5_kernel()
// CHECK: call void @__parent_block_invoke_5(

// An over-aligned capture requires the original block alignment in the wrapper.
// CHECK-LABEL: define internal amdgpu_kernel void @__aligned_block_invoke_kernel(
// CHECK: alloca {{.*}}, align 32, addrspace(5)
// CHECK: call void @__aligned_block_invoke(

// A capture-free block with a local argument starts with that argument.
// CHECK-LABEL: define internal amdgpu_kernel void @__empty_local_block_invoke_kernel(ptr addrspace(3) %{{.*}})
// CHECK: call void @__empty_local_block_invoke(

// r600 retains the full-block ABI and the original runtime entry points.
// R600: call i32 @__enqueue_kernel_basic(
// R600: call i32 @__enqueue_kernel_varargs(
// R600: call i32 @__enqueue_kernel_basic_events(
// R600: call i32 @__enqueue_kernel_events_varargs(
// R600-LABEL: define internal amdgpu_kernel void @__parent_block_invoke_kernel(
// R600-SAME: <{ i32, i32, ptr,
// R600: store <{ i32, i32, ptr,
