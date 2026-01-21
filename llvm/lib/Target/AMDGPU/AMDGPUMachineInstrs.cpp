//===-- AMDGPUMachineInstrs.cpp -*- C++ -*---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Convenience wrappers and helpers for AMDGPU-specific machine instructions.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMachineInstrs.h"

using namespace llvm;
using namespace AMDGPUMI;

unsigned VLoadStoreIdxInst::getBitWidth() const {
  switch (getOpcode()) {
  case AMDGPU::V_LOAD_IDX_BITS:
  case AMDGPU::V_LOAD_IDX_BITS_D16:
    report_fatal_error("V_LOAD_IDX_BITS has no well defined bit width");
  case AMDGPU::V_LOAD_IDX_B32:
    return 32;
  case AMDGPU::V_LOAD_IDX_B64:
    return 64;
  case AMDGPU::V_LOAD_IDX_B96:
    return 96;
  case AMDGPU::V_LOAD_IDX_B128:
    return 128;
  case AMDGPU::V_LOAD_IDX_B160:
    return 160;
  case AMDGPU::V_LOAD_IDX_B192:
    return 192;
  case AMDGPU::V_LOAD_IDX_B224:
    return 224;
  case AMDGPU::V_LOAD_IDX_B256:
    return 256;
  case AMDGPU::V_LOAD_IDX_B288:
    return 288;
  case AMDGPU::V_LOAD_IDX_B320:
    return 320;
  case AMDGPU::V_LOAD_IDX_B352:
    return 352;
  case AMDGPU::V_LOAD_IDX_B384:
    return 384;
  case AMDGPU::V_LOAD_IDX_B512:
    return 512;
  case AMDGPU::V_LOAD_IDX_B576:
    return 576;
  case AMDGPU::V_LOAD_IDX_B1024:
    return 1024;
  case AMDGPU::V_STORE_IDX_BITS:
  case AMDGPU::V_STORE_IDX_BITS_D16:
    report_fatal_error("V_STORE_IDX_BITS has no well defined bit width");
  case AMDGPU::V_STORE_IDX_B32:
    return 32;
  case AMDGPU::V_STORE_IDX_B64:
    return 64;
  case AMDGPU::V_STORE_IDX_B96:
    return 96;
  case AMDGPU::V_STORE_IDX_B128:
    return 128;
  case AMDGPU::V_STORE_IDX_B160:
    return 160;
  case AMDGPU::V_STORE_IDX_B192:
    return 192;
  case AMDGPU::V_STORE_IDX_B224:
    return 224;
  case AMDGPU::V_STORE_IDX_B256:
    return 256;
  case AMDGPU::V_STORE_IDX_B288:
    return 288;
  case AMDGPU::V_STORE_IDX_B320:
    return 320;
  case AMDGPU::V_STORE_IDX_B352:
    return 352;
  case AMDGPU::V_STORE_IDX_B384:
    return 384;
  case AMDGPU::V_STORE_IDX_B512:
    return 512;
  case AMDGPU::V_STORE_IDX_B576:
    return 576;
  case AMDGPU::V_STORE_IDX_B1024:
    return 1024;
  default:
    llvm_unreachable("unsupported V_LOAD/STORE_IDX opcode");
  }
}

unsigned VLoadIdxInst::getOpcodeForBitWidth(unsigned Bits) {
  switch (Bits) {
  case 8:
  case 16:
    report_fatal_error("V_LOAD_IDX_BITS has no well defined bit width");
  case 32:
    return AMDGPU::V_LOAD_IDX_B32;
  case 64:
    return AMDGPU::V_LOAD_IDX_B64;
  case 96:
    return AMDGPU::V_LOAD_IDX_B96;
  case 128:
    return AMDGPU::V_LOAD_IDX_B128;
  case 160:
    return AMDGPU::V_LOAD_IDX_B160;
  case 192:
    return AMDGPU::V_LOAD_IDX_B192;
  case 224:
    return AMDGPU::V_LOAD_IDX_B224;
  case 256:
    return AMDGPU::V_LOAD_IDX_B256;
  case 288:
    return AMDGPU::V_LOAD_IDX_B288;
  case 320:
    return AMDGPU::V_LOAD_IDX_B320;
  case 352:
    return AMDGPU::V_LOAD_IDX_B352;
  case 384:
    return AMDGPU::V_LOAD_IDX_B384;
  case 512:
    return AMDGPU::V_LOAD_IDX_B512;
  case 576:
    return AMDGPU::V_LOAD_IDX_B576;
  case 1024:
    return AMDGPU::V_LOAD_IDX_B1024;
  default:
    llvm_unreachable("unsupported V_LOAD_IDX size");
  }
}

unsigned VStoreIdxInst::getOpcodeForBitWidth(unsigned Bits) {
  switch (Bits) {
  case 8:
  case 16:
    report_fatal_error("V_STORE_IDX_BITS has no well defined bit width");
  case 32:
    return AMDGPU::V_STORE_IDX_B32;
  case 64:
    return AMDGPU::V_STORE_IDX_B64;
  case 96:
    return AMDGPU::V_STORE_IDX_B96;
  case 128:
    return AMDGPU::V_STORE_IDX_B128;
  case 160:
    return AMDGPU::V_STORE_IDX_B160;
  case 192:
    return AMDGPU::V_STORE_IDX_B192;
  case 224:
    return AMDGPU::V_STORE_IDX_B224;
  case 256:
    return AMDGPU::V_STORE_IDX_B256;
  case 288:
    return AMDGPU::V_STORE_IDX_B288;
  case 320:
    return AMDGPU::V_STORE_IDX_B320;
  case 352:
    return AMDGPU::V_STORE_IDX_B352;
  case 384:
    return AMDGPU::V_STORE_IDX_B384;
  case 512:
    return AMDGPU::V_STORE_IDX_B512;
  case 576:
    return AMDGPU::V_STORE_IDX_B576;
  case 1024:
    return AMDGPU::V_STORE_IDX_B1024;
  default:
    llvm_unreachable("unsupported V_STORE_IDX size");
  }
}
