// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1370 -show-encoding %s | FileCheck --check-prefix=GFX1370 %s
// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1370 -show-encoding %s | %extract-encodings | llvm-mc -triple=amdgcn -mcpu=gfx1370 -disassemble -show-encoding | FileCheck --check-prefix=GFX1370 %s

v_wmma_f32_16x16_iu8 v[32:39], v[0:15], v[16:31], v[32:39] signed_a signed_b clamp
// GFX1370: v_wmma_f32_16x16_iu8 v[32:39], v[0:15], v[16:31], v[32:39] signed_a signed_b clamp ; encoding: [0x20,0x50,0x1f,0xdd,0x20,0x04,0x80,0x10,0x10,0x90,0x00,0x00]

v_wmma_f32i32_16x16_iu8 v[32:39], v[0:15], v[16:31], v[32:39] signed_a signed_b clamp
// GFX1370: v_wmma_f32i32_16x16_iu8 v[32:39], v[0:15], v[16:31], v[32:39] signed_a signed_b clamp ; encoding: [0x20,0x10,0x20,0xdd,0x20,0x04,0x80,0x10,0x10,0x90,0x00,0x00]

v_wmma_i32_16x16_iu8 v[32:39], v[0:15], v[16:31], v[32:39] signed_a signed_b clamp
// GFX1370: v_wmma_i32_16x16_iu8 v[32:39], v[0:15], v[16:31], v[32:39] signed_a signed_b clamp ; encoding: [0x20,0x10,0x1f,0xdd,0x20,0x04,0x80,0x10,0x10,0x90,0x00,0x00]

v_wmma_f32_16x16_fp8_fp8 v[32:39], v[0:15], v[16:31], v[32:39] clamp
// GFX1370: v_wmma_f32_16x16_fp8_fp8 v[32:39], v[0:15], v[16:31], v[32:39] clamp ; encoding: [0x20,0x10,0x1d,0xdd,0x20,0x04,0x80,0x10,0x10,0x00,0x00,0x00]

v_wmma_f32_16x16_fp8_bf8 v[32:39], v[0:15], v[16:31], v[32:39] clamp
// GFX1370: v_wmma_f32_16x16_fp8_bf8 v[32:39], v[0:15], v[16:31], v[32:39] clamp ; encoding: [0x20,0x50,0x1d,0xdd,0x20,0x04,0x80,0x10,0x10,0x00,0x00,0x00]

v_wmma_f32_16x16_bf8_fp8 v[32:39], v[0:15], v[16:31], v[32:39] clamp
// GFX1370: v_wmma_f32_16x16_bf8_fp8 v[32:39], v[0:15], v[16:31], v[32:39] clamp ; encoding: [0x20,0x90,0x1d,0xdd,0x20,0x04,0x80,0x10,0x10,0x00,0x00,0x00]

v_wmma_f32_16x16_bf8_bf8 v[32:39], v[0:15], v[16:31], v[32:39] clamp
// GFX1370: v_wmma_f32_16x16_bf8_bf8 v[32:39], v[0:15], v[16:31], v[32:39] clamp ; encoding: [0x20,0xd0,0x1d,0xdd,0x20,0x04,0x80,0x10,0x10,0x00,0x00,0x00]

v_wmma_f16_16x16_fp8_fp8 v[32:35], v[0:15], v[16:31], v[32:35] clamp
// GFX1370: v_wmma_f16_16x16_fp8_fp8 v[32:35], v[0:15], v[16:31], v[32:35] clamp ; encoding: [0x20,0x10,0x1e,0xdd,0x20,0x04,0x80,0x10,0x10,0x00,0x00,0x00]

v_wmma_f16_16x16_fp8_bf8 v[32:35], v[0:15], v[16:31], v[32:35] clamp
// GFX1370: v_wmma_f16_16x16_fp8_bf8 v[32:35], v[0:15], v[16:31], v[32:35] clamp ; encoding: [0x20,0x50,0x1e,0xdd,0x20,0x04,0x80,0x10,0x10,0x00,0x00,0x00]

v_wmma_f16_16x16_bf8_fp8 v[32:35], v[0:15], v[16:31], v[32:35] clamp
// GFX1370: v_wmma_f16_16x16_bf8_fp8 v[32:35], v[0:15], v[16:31], v[32:35] clamp ; encoding: [0x20,0x90,0x1e,0xdd,0x20,0x04,0x80,0x10,0x10,0x00,0x00,0x00]

v_wmma_f16_16x16_bf8_bf8 v[32:35], v[0:15], v[16:31], v[32:35] clamp
// GFX1370: v_wmma_f16_16x16_bf8_bf8 v[32:35], v[0:15], v[16:31], v[32:35] clamp ; encoding: [0x20,0xd0,0x1e,0xdd,0x20,0x04,0x80,0x10,0x10,0x00,0x00,0x00]
