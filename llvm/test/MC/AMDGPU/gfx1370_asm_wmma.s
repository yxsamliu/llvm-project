// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1370 -show-encoding %s | %extract-encodings | llvm-mc -triple=amdgcn -mcpu=gfx1370 -disassemble -show-encoding | FileCheck --check-prefix=GFX1370 %s

v_wmma_f32_16x16_iu8 v[32:39], v[0:15], v[16:31], v[32:39] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_f32_16x16_iu8 v[32:39], v[0:15], v[16:31], v[32:39] k:128 matrix_a_signed matrix_b_signed clamp ; encoding: [0x20,0x50,0x1f,0xdd,0x20,0x04,0x80,0x10,0x10,0x90,0x00,0x00]

v_wmma_f32i32_16x16_iu8 v[32:39], v[0:15], v[16:31], v[32:39] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_f32i32_16x16_iu8 v[32:39], v[0:15], v[16:31], v[32:39] k:128 matrix_a_signed matrix_b_signed clamp ; encoding: [0x20,0x10,0x20,0xdd,0x20,0x04,0x80,0x10,0x10,0x90,0x00,0x00]

v_wmma_i32_16x16_iu8 v[32:39], v[0:15], v[16:31], v[32:39] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_i32_16x16_iu8 v[32:39], v[0:15], v[16:31], v[32:39] k:128 matrix_a_signed matrix_b_signed clamp ; encoding: [0x20,0x10,0x1f,0xdd,0x20,0x04,0x80,0x10,0x10,0x90,0x00,0x00]

v_wmma_f32_16x16_fp8_fp8 v[32:39], v[0:15], v[16:31], v[32:39] clamp
// GFX1370: v_wmma_f32_16x16_fp8_fp8 v[32:39], v[0:15], v[16:31], v[32:39] k:128 clamp ; encoding: [0x20,0x10,0x1d,0xdd,0x20,0x04,0x80,0x10,0x10,0x00,0x00,0x00]

v_wmma_f32_16x16_fp8_bf8 v[32:39], v[0:15], v[16:31], v[32:39] clamp
// GFX1370: v_wmma_f32_16x16_fp8_bf8 v[32:39], v[0:15], v[16:31], v[32:39] k:128 clamp ; encoding: [0x20,0x50,0x1d,0xdd,0x20,0x04,0x80,0x10,0x10,0x00,0x00,0x00]

v_wmma_f32_16x16_bf8_fp8 v[32:39], v[0:15], v[16:31], v[32:39] clamp
// GFX1370: v_wmma_f32_16x16_bf8_fp8 v[32:39], v[0:15], v[16:31], v[32:39] k:128 clamp ; encoding: [0x20,0x90,0x1d,0xdd,0x20,0x04,0x80,0x10,0x10,0x00,0x00,0x00]

v_wmma_f32_16x16_bf8_bf8 v[32:39], v[0:15], v[16:31], v[32:39] clamp
// GFX1370: v_wmma_f32_16x16_bf8_bf8 v[32:39], v[0:15], v[16:31], v[32:39] k:128 clamp ; encoding: [0x20,0xd0,0x1d,0xdd,0x20,0x04,0x80,0x10,0x10,0x00,0x00,0x00]

v_wmma_f16_16x16_fp8_fp8 v[32:35], v[0:15], v[16:31], v[32:35] clamp
// GFX1370: v_wmma_f16_16x16_fp8_fp8 v[32:35], v[0:15], v[16:31], v[32:35] k:128 clamp ; encoding: [0x20,0x10,0x1e,0xdd,0x20,0x04,0x80,0x10,0x10,0x00,0x00,0x00]

v_wmma_f16_16x16_fp8_bf8 v[32:35], v[0:15], v[16:31], v[32:35] clamp
// GFX1370: v_wmma_f16_16x16_fp8_bf8 v[32:35], v[0:15], v[16:31], v[32:35] k:128 clamp ; encoding: [0x20,0x50,0x1e,0xdd,0x20,0x04,0x80,0x10,0x10,0x00,0x00,0x00]

v_wmma_f16_16x16_bf8_fp8 v[32:35], v[0:15], v[16:31], v[32:35] clamp
// GFX1370: v_wmma_f16_16x16_bf8_fp8 v[32:35], v[0:15], v[16:31], v[32:35] k:128 clamp ; encoding: [0x20,0x90,0x1e,0xdd,0x20,0x04,0x80,0x10,0x10,0x00,0x00,0x00]

v_wmma_f16_16x16_bf8_bf8 v[32:35], v[0:15], v[16:31], v[32:35] clamp
// GFX1370: v_wmma_f16_16x16_bf8_bf8 v[32:35], v[0:15], v[16:31], v[32:35] k:128 clamp ; encoding: [0x20,0xd0,0x1e,0xdd,0x20,0x04,0x80,0x10,0x10,0x00,0x00,0x00]

v_wmma_f32_16x16_f8f6f4 v[32:39], v[0:15], v[16:31], v[32:39], v40, v41 k:64 matrix_a_fmt:MATRIX_FMT_FP6 matrix_b_fmt:MATRIX_FMT_FP6 matrix_a_scale:MATRIX_SCALE_LO_EVEN matrix_b_scale:MATRIX_SCALE_LO_EVEN clamp
// GFX1370: v_wmma_f32_16x16_f8f6f4 v[32:39], v[0:15], v[16:31], v[32:39], v40, v41 k:128 matrix_a_fmt:MATRIX_FMT_FP6 matrix_b_fmt:MATRIX_FMT_FP6 matrix_a_scale:MATRIX_SCALE_LO_EVEN matrix_b_scale:MATRIX_SCALE_LO_EVEN clamp ; encoding: [0x20,0x90,0x20,0xde,0x20,0x04,0x80,0x04,0x10,0x20,0x01,0x00,0x28,0x90,0x02,0x00]

v_wmma_f32_16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23] clamp
// GFX1370: v_wmma_f32_16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23] k:32 clamp ; encoding: [0x10,0x10,0x1c,0xdd,0x10,0x04,0x80,0x04,0x08,0x00,0x00,0x00]

v_wmma_f16_16x16_f16 v[16:19], v[0:7], v[8:15], v[16:19] clamp
// GFX1370: v_wmma_f16_16x16_f16 v[16:19], v[0:7], v[8:15], v[16:19] k:32 clamp ; encoding: [0x10,0x50,0x1c,0xdd,0x10,0x04,0x80,0x04,0x08,0x00,0x00,0x00]

v_wmma_f32_16x16_bf16 v[16:23], v[0:7], v[8:15], v[16:23] clamp
// GFX1370: v_wmma_f32_16x16_bf16 v[16:23], v[0:7], v[8:15], v[16:23] k:32 clamp ; encoding: [0x10,0x90,0x1c,0xdd,0x10,0x04,0x80,0x04,0x08,0x00,0x00,0x00]

v_wmma_bf16_16x16_bf16 v[16:19], v[0:7], v[8:15], v[16:19] clamp
// GFX1370: v_wmma_bf16_16x16_bf16 v[16:19], v[0:7], v[8:15], v[16:19] k:32 clamp ; encoding: [0x10,0xd0,0x1c,0xdd,0x10,0x04,0x80,0x04,0x08,0x00,0x00,0x00]

v_wmma_f32_16x16_iu16 v[8:15], v[0:3], v[4:7], v[8:15] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_f32_16x16_iu16 v[8:15], v[0:3], v[4:7], v[8:15] k:16 matrix_a_signed matrix_b_signed clamp ; encoding: [0x08,0xd0,0x21,0xdd,0x08,0x04,0x80,0x00,0x04,0x90,0x00,0x00]

v_wmma_f32i32_16x16_iu16 v[8:15], v[0:3], v[4:7], v[8:15] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_f32i32_16x16_iu16 v[8:15], v[0:3], v[4:7], v[8:15] k:16 matrix_a_signed matrix_b_signed clamp ; encoding: [0x08,0x10,0x22,0xdd,0x08,0x04,0x80,0x00,0x04,0x90,0x00,0x00]

v_wmma_i32_16x16_iu16 v[8:15], v[0:3], v[4:7], v[8:15] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_i32_16x16_iu16 v[8:15], v[0:3], v[4:7], v[8:15] k:16 matrix_a_signed matrix_b_signed clamp ; encoding: [0x08,0x90,0x21,0xdd,0x08,0x04,0x80,0x00,0x04,0x90,0x00,0x00]

v_wmma_f32_16x16_iu16 v[16:23], v[0:7], v[8:15], v[16:23] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_f32_16x16_iu16 v[16:23], v[0:7], v[8:15], v[16:23] k:32 matrix_a_signed matrix_b_signed clamp ; encoding: [0x10,0xd0,0x21,0xdd,0x10,0x04,0x80,0x04,0x08,0x90,0x00,0x00]

v_wmma_f32i32_16x16_iu16 v[16:23], v[0:7], v[8:15], v[16:23] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_f32i32_16x16_iu16 v[16:23], v[0:7], v[8:15], v[16:23] k:32 matrix_a_signed matrix_b_signed clamp ; encoding: [0x10,0x10,0x22,0xdd,0x10,0x04,0x80,0x04,0x08,0x90,0x00,0x00]

v_wmma_i32_16x16_iu16 v[16:23], v[0:7], v[8:15], v[16:23] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_i32_16x16_iu16 v[16:23], v[0:7], v[8:15], v[16:23] k:32 matrix_a_signed matrix_b_signed clamp ; encoding: [0x10,0x90,0x21,0xdd,0x10,0x04,0x80,0x04,0x08,0x90,0x00,0x00]

v_wmma_f32_16x16_iu16 v[32:39], v[0:15], v[16:31], v[32:39] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_f32_16x16_iu16 v[32:39], v[0:15], v[16:31], v[32:39] k:64 matrix_a_signed matrix_b_signed clamp ; encoding: [0x20,0xd0,0x21,0xdd,0x20,0x04,0x80,0x0c,0x10,0x90,0x00,0x00]

v_wmma_f32i32_16x16_iu16 v[32:39], v[0:15], v[16:31], v[32:39] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_f32i32_16x16_iu16 v[32:39], v[0:15], v[16:31], v[32:39] k:64 matrix_a_signed matrix_b_signed clamp ; encoding: [0x20,0x10,0x22,0xdd,0x20,0x04,0x80,0x0c,0x10,0x90,0x00,0x00]

v_wmma_i32_16x16_iu16 v[32:39], v[0:15], v[16:31], v[32:39] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_i32_16x16_iu16 v[32:39], v[0:15], v[16:31], v[32:39] k:64 matrix_a_signed matrix_b_signed clamp ; encoding: [0x20,0x90,0x21,0xdd,0x20,0x04,0x80,0x0c,0x10,0x90,0x00,0x00]

v_wmma_f32_16x16_iu8_iu16 v[6:13], v[0:1], v[2:5], v[6:13] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_f32_16x16_iu8_iu16 v[6:13], v[0:1], v[2:5], v[6:13] k:16 matrix_a_signed matrix_b_signed clamp ; encoding: [0x06,0x90,0x22,0xdd,0x06,0x04,0x80,0x00,0x02,0x90,0x00,0x00]

v_wmma_f32i32_16x16_iu8_iu16 v[6:13], v[0:1], v[2:5], v[6:13] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_f32i32_16x16_iu8_iu16 v[6:13], v[0:1], v[2:5], v[6:13] k:16 matrix_a_signed matrix_b_signed clamp ; encoding: [0x06,0xd0,0x22,0xdd,0x06,0x04,0x80,0x00,0x02,0x90,0x00,0x00]

v_wmma_i32_16x16_iu8_iu16 v[6:13], v[0:1], v[2:5], v[6:13] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_i32_16x16_iu8_iu16 v[6:13], v[0:1], v[2:5], v[6:13] k:16 matrix_a_signed matrix_b_signed clamp ; encoding: [0x06,0x50,0x22,0xdd,0x06,0x04,0x80,0x00,0x02,0x90,0x00,0x00]

v_wmma_f32_16x16_iu8_iu16 v[12:19], v[0:3], v[4:11], v[12:19] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_f32_16x16_iu8_iu16 v[12:19], v[0:3], v[4:11], v[12:19] k:32 matrix_a_signed matrix_b_signed clamp ; encoding: [0x0c,0x90,0x22,0xdd,0x0c,0x04,0x80,0x04,0x04,0x90,0x00,0x00]

v_wmma_f32i32_16x16_iu8_iu16 v[12:19], v[0:3], v[4:11], v[12:19] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_f32i32_16x16_iu8_iu16 v[12:19], v[0:3], v[4:11], v[12:19] k:32 matrix_a_signed matrix_b_signed clamp ; encoding: [0x0c,0xd0,0x22,0xdd,0x0c,0x04,0x80,0x04,0x04,0x90,0x00,0x00]

v_wmma_i32_16x16_iu8_iu16 v[12:19], v[0:3], v[4:11], v[12:19] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_i32_16x16_iu8_iu16 v[12:19], v[0:3], v[4:11], v[12:19] k:32 matrix_a_signed matrix_b_signed clamp ; encoding: [0x0c,0x50,0x22,0xdd,0x0c,0x04,0x80,0x04,0x04,0x90,0x00,0x00]

v_wmma_f32_16x16_iu8_iu16 v[24:31], v[0:7], v[8:23], v[24:31] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_f32_16x16_iu8_iu16 v[24:31], v[0:7], v[8:23], v[24:31] k:64 matrix_a_signed matrix_b_signed clamp ; encoding: [0x18,0x90,0x22,0xdd,0x18,0x04,0x80,0x0c,0x08,0x90,0x00,0x00]

v_wmma_f32i32_16x16_iu8_iu16 v[24:31], v[0:7], v[8:23], v[24:31] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_f32i32_16x16_iu8_iu16 v[24:31], v[0:7], v[8:23], v[24:31] k:64 matrix_a_signed matrix_b_signed clamp ; encoding: [0x18,0xd0,0x22,0xdd,0x18,0x04,0x80,0x0c,0x08,0x90,0x00,0x00]

v_wmma_i32_16x16_iu8_iu16 v[24:31], v[0:7], v[8:23], v[24:31] matrix_a_signed matrix_b_signed clamp
// GFX1370: v_wmma_i32_16x16_iu8_iu16 v[24:31], v[0:7], v[8:23], v[24:31] k:64 matrix_a_signed matrix_b_signed clamp ; encoding: [0x18,0x50,0x22,0xdd,0x18,0x04,0x80,0x0c,0x08,0x90,0x00,0x00]
