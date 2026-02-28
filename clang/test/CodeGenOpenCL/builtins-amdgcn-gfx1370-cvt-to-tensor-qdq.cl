// RUN: %clang_cc1 -cl-std=CL2.0 -O0 -triple amdgcn-unknown-unknown -target-cpu gfx1370 -emit-llvm -o - %s | FileCheck %s
// REQUIRES: amdgpu-registered-target

typedef int v2i __attribute__((ext_vector_type(2)));
typedef short v2s __attribute__((ext_vector_type(2)));
typedef short v4s __attribute__((ext_vector_type(4)));
typedef unsigned short v2us __attribute__((ext_vector_type(2)));
typedef unsigned short v4us __attribute__((ext_vector_type(4)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef half v4h __attribute__((ext_vector_type(4)));
typedef half v8h __attribute__((ext_vector_type(8)));

#define PIXEL_SHAPE_8X4X8  (0 << 0)
#define PIXEL_SHAPE_4X4X8  (1 << 0)
#define PIXEL_SHAPE_4X4X16 (2 << 0)
#define PIXEL_SHAPE_4X2X16 (3 << 0)

// CHECK-LABEL: @test_cvt_to_tensor_i16_f32_4x2x16
// CHECK: call { <2 x i16>, <2 x i16> } @llvm.amdgcn.cvt.to.tensor.i16.f32.scatter2.v4f32(<4 x float> %{{.*}}, i8 %{{.*}}, i32 3, i1 true)
kernel void test_cvt_to_tensor_i16_f32_4x2x16(global v2s *out0, global v2s *out1, v4f acc_in, char scale) {
  __builtin_amdgcn_cvt_to_tensor_i16_f32_4x2x16(out0, out1, acc_in, scale, PIXEL_SHAPE_4X2X16, true);
}

// CHECK-LABEL: @test_cvt_to_tensor_u16_f32_4x2x16
// CHECK: call { <2 x i16>, <2 x i16> } @llvm.amdgcn.cvt.to.tensor.u16.f32.scatter2.v4f32(<4 x float> %{{.*}}, i8 %{{.*}}, i32 3, i1 false)
kernel void test_cvt_to_tensor_u16_f32_4x2x16(global v2us *out0, global v2us *out1, v4f acc_in, char scale) {
  __builtin_amdgcn_cvt_to_tensor_u16_f32_4x2x16(out0, out1, acc_in, scale, PIXEL_SHAPE_4X2X16, false);
}

// CHECK-LABEL: @test_cvt_to_tensor_i16_f16_8x4x8
// CHECK: call { <2 x i32>, <2 x i32> } @llvm.amdgcn.cvt.to.tensor.i16.f16.scatter2.double.v8f16(<8 x half> %{{.*}}, i8 %{{.*}}, i32 0, i1 true)
kernel void test_cvt_to_tensor_i16_f16_8x4x8(global v2i *out0, global v2i *out1, v8h acc_in, char scale) {
  __builtin_amdgcn_cvt_to_tensor_i16_f16_8x4x8(out0, out1, acc_in, scale, PIXEL_SHAPE_8X4X8, true);
}

// CHECK-LABEL: @test_cvt_to_tensor_i16_f16_4x4x8
// CHECK: call { <2 x i16>, <2 x i16> } @llvm.amdgcn.cvt.to.tensor.i16.f16.scatter2.v4f16(<4 x half> %{{.*}}, i8 %{{.*}}, i32 1, i1 false)
kernel void test_cvt_to_tensor_i16_f16_4x4x8(global v2s *out0, global v2s *out1, v4h acc_in, char scale) {
  __builtin_amdgcn_cvt_to_tensor_i16_f16_4x4x8_4x2x16(out0, out1, acc_in, scale, PIXEL_SHAPE_4X4X8, false);
}

// CHECK-LABEL: @test_cvt_to_tensor_i16_f16_4x2x16
// CHECK: call { <2 x i16>, <2 x i16> } @llvm.amdgcn.cvt.to.tensor.i16.f16.scatter2.v4f16(<4 x half> %{{.*}}, i8 %{{.*}}, i32 3, i1 true)
kernel void test_cvt_to_tensor_i16_f16_4x2x16(global v2s *out0, global v2s *out1, v4h acc_in, char scale) {
  __builtin_amdgcn_cvt_to_tensor_i16_f16_4x4x8_4x2x16(out0, out1, acc_in, scale, PIXEL_SHAPE_4X2X16, true);
}

// CHECK-LABEL: @test_cvt_to_tensor_i16_f16_4x4x16
// CHECK: call { <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16> } @llvm.amdgcn.cvt.to.tensor.i16.f16.scatter4.v8f16(<8 x half> %{{.*}}, i8 %{{.*}}, i32 2, i1 false)
kernel void test_cvt_to_tensor_i16_f16_4x4x16(global v2s *out0, global v2s *out1, global v2s *out2, global v2s *out3, v8h acc_in, char scale) {
  __builtin_amdgcn_cvt_to_tensor_i16_f16_4x4x16(out0, out1, out2, out3, acc_in, scale, PIXEL_SHAPE_4X4X16, false);
}

// CHECK-LABEL: @test_cvt_to_tensor_u16_f16_8x4x8
// CHECK: call { <2 x i32>, <2 x i32> } @llvm.amdgcn.cvt.to.tensor.u16.f16.scatter2.double.v8f16(<8 x half> %{{.*}}, i8 %{{.*}}, i32 0, i1 false)
kernel void test_cvt_to_tensor_u16_f16_8x4x8(global v2i *out0, global v2i *out1, v8h acc_in, char scale) {
  __builtin_amdgcn_cvt_to_tensor_u16_f16_8x4x8(out0, out1, acc_in, scale, PIXEL_SHAPE_8X4X8, false);
}

// CHECK-LABEL: @test_cvt_to_tensor_u16_f16_4x4x8
// CHECK: call { <2 x i16>, <2 x i16> } @llvm.amdgcn.cvt.to.tensor.u16.f16.scatter2.v4f16(<4 x half> %{{.*}}, i8 %{{.*}}, i32 1, i1 true)
kernel void test_cvt_to_tensor_u16_f16_4x4x8(global v2us *out0, global v2us *out1, v4h acc_in, char scale) {
  __builtin_amdgcn_cvt_to_tensor_u16_f16_4x4x8_4x2x16(out0, out1, acc_in, scale, PIXEL_SHAPE_4X4X8, true);
}

// CHECK-LABEL: @test_cvt_to_tensor_u16_f16_4x2x16
// CHECK: call { <2 x i16>, <2 x i16> } @llvm.amdgcn.cvt.to.tensor.u16.f16.scatter2.v4f16(<4 x half> %{{.*}}, i8 %{{.*}}, i32 3, i1 false)
kernel void test_cvt_to_tensor_u16_f16_4x2x16(global v2us *out0, global v2us *out1, v4h acc_in, char scale) {
  __builtin_amdgcn_cvt_to_tensor_u16_f16_4x4x8_4x2x16(out0, out1, acc_in, scale, PIXEL_SHAPE_4X2X16, false);
}

// CHECK-LABEL: @test_cvt_to_tensor_u16_f16_4x4x16
// CHECK: call { <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16> } @llvm.amdgcn.cvt.to.tensor.u16.f16.scatter4.v8f16(<8 x half> %{{.*}}, i8 %{{.*}}, i32 2, i1 true)
kernel void test_cvt_to_tensor_u16_f16_4x4x16(global v2us *out0, global v2us *out1, global v2us *out2, global v2us *out3, v8h acc_in, char scale) {
  __builtin_amdgcn_cvt_to_tensor_u16_f16_4x4x16(out0, out1, out2, out3, acc_in, scale, PIXEL_SHAPE_4X4X16, true);
}
