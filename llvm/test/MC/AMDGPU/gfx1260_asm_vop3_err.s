// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1260 -mattr=+wavefrontsize32 %s 2>&1 | FileCheck --check-prefix=W32 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1260 -mattr=+wavefrontsize64,-wavefrontsize32 %s 2>&1 | FileCheck --check-prefix=W64 --implicit-check-not=error: %s

v_cvt_e5m3scale_pk8_fp4_f32 v5, v[0:7], v2
// W64: :[[@LINE-1]]:1: error: instruction requires wavesize=32

v_cvt_e5m3scale_pk8_fp4_f32 v5, v[0:7], v2 clamp
// W32: :[[@LINE-1]]:44: error: invalid operand for instruction
// W64: :[[@LINE-2]]:1: error: instruction requires wavesize=32

v_cvt_e5m3scale_pk8_fp4_f32 v5, v[0:7], v2 mul:2
// W32: :[[@LINE-1]]:44: error: not a valid operand.
// W64: :[[@LINE-2]]:1: error: instruction requires wavesize=32

v_cvt_e5m3scale_pk8_fp4_f32_dpp v5, v3, v1 row_share:1
// W32: :[[@LINE-1]]:1: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:1: error: instruction not supported on this GPU

v_cvt_e5m3scale_pk8_fp4_f32_dpp v5, v3, v1 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:1: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:1: error: instruction not supported on this GPU

v_cvt_e5m3scale_pk8_fp4_f32_e64_dpp v5, v3, v1 row_share:1
// W32: :[[@LINE-1]]:1: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:1: error: instruction not supported on this GPU
