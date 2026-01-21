//===-- AMDGPUMachineInstrs.h -*- C++ -*-----------------------------------===//
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

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUMACHINEINSTRS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUMACHINEINSTRS_H

#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineInstr.h"

namespace llvm {
namespace AMDGPUMI {

class VLoadStoreIdxInst : public MachineInstr {
public:
  MachineOperand &getDataOp() { return getOperand(0); }
  MachineOperand &getIdxOp() { return getOperand(1); }
  MachineOperand &getOffsetOp() { return getOperand(2); }
  const MachineOperand &getDataOp() const { return getOperand(0); }
  const MachineOperand &getIdxOp() const { return getOperand(1); }
  const MachineOperand &getOffsetOp() const { return getOperand(2); }

  unsigned getBitWidth() const;

  static bool classof(const MachineInstr *MI) {
    switch (MI->getOpcode()) {
    case AMDGPU::V_LOAD_IDX_BITS:
    case AMDGPU::V_LOAD_IDX_BITS_D16:
    case AMDGPU::V_LOAD_IDX_B32:
    case AMDGPU::V_LOAD_IDX_B64:
    case AMDGPU::V_LOAD_IDX_B96:
    case AMDGPU::V_LOAD_IDX_B128:
    case AMDGPU::V_LOAD_IDX_B160:
    case AMDGPU::V_LOAD_IDX_B192:
    case AMDGPU::V_LOAD_IDX_B224:
    case AMDGPU::V_LOAD_IDX_B256:
    case AMDGPU::V_LOAD_IDX_B288:
    case AMDGPU::V_LOAD_IDX_B320:
    case AMDGPU::V_LOAD_IDX_B352:
    case AMDGPU::V_LOAD_IDX_B384:
    case AMDGPU::V_LOAD_IDX_B512:
    case AMDGPU::V_LOAD_IDX_B576:
    case AMDGPU::V_LOAD_IDX_B1024:
    case AMDGPU::V_STORE_IDX_BITS:
    case AMDGPU::V_STORE_IDX_BITS_D16:
    case AMDGPU::V_STORE_IDX_B32:
    case AMDGPU::V_STORE_IDX_B64:
    case AMDGPU::V_STORE_IDX_B96:
    case AMDGPU::V_STORE_IDX_B128:
    case AMDGPU::V_STORE_IDX_B160:
    case AMDGPU::V_STORE_IDX_B192:
    case AMDGPU::V_STORE_IDX_B224:
    case AMDGPU::V_STORE_IDX_B256:
    case AMDGPU::V_STORE_IDX_B288:
    case AMDGPU::V_STORE_IDX_B320:
    case AMDGPU::V_STORE_IDX_B352:
    case AMDGPU::V_STORE_IDX_B384:
    case AMDGPU::V_STORE_IDX_B512:
    case AMDGPU::V_STORE_IDX_B576:
    case AMDGPU::V_STORE_IDX_B1024:
      return true;
    default:
      return false;
    }
  }
};

class VLoadIdxInst : public VLoadStoreIdxInst {
public:
  static unsigned getOpcodeForBitWidth(unsigned Bits);

  static bool classof(const MachineInstr *MI) {
    switch (MI->getOpcode()) {
    case AMDGPU::V_LOAD_IDX_BITS:
    case AMDGPU::V_LOAD_IDX_BITS_D16:
    case AMDGPU::V_LOAD_IDX_B32:
    case AMDGPU::V_LOAD_IDX_B64:
    case AMDGPU::V_LOAD_IDX_B96:
    case AMDGPU::V_LOAD_IDX_B128:
    case AMDGPU::V_LOAD_IDX_B160:
    case AMDGPU::V_LOAD_IDX_B192:
    case AMDGPU::V_LOAD_IDX_B224:
    case AMDGPU::V_LOAD_IDX_B256:
    case AMDGPU::V_LOAD_IDX_B288:
    case AMDGPU::V_LOAD_IDX_B320:
    case AMDGPU::V_LOAD_IDX_B352:
    case AMDGPU::V_LOAD_IDX_B384:
    case AMDGPU::V_LOAD_IDX_B512:
    case AMDGPU::V_LOAD_IDX_B576:
    case AMDGPU::V_LOAD_IDX_B1024:
      return true;
    default:
      return false;
    }
  }
};

class VStoreIdxInst : public VLoadStoreIdxInst {
public:
  static unsigned getOpcodeForBitWidth(unsigned Bits);

  static bool classof(const MachineInstr *MI) {
    switch (MI->getOpcode()) {
    case AMDGPU::V_STORE_IDX_BITS:
    case AMDGPU::V_STORE_IDX_BITS_D16:
    case AMDGPU::V_STORE_IDX_B32:
    case AMDGPU::V_STORE_IDX_B64:
    case AMDGPU::V_STORE_IDX_B96:
    case AMDGPU::V_STORE_IDX_B128:
    case AMDGPU::V_STORE_IDX_B160:
    case AMDGPU::V_STORE_IDX_B192:
    case AMDGPU::V_STORE_IDX_B224:
    case AMDGPU::V_STORE_IDX_B256:
    case AMDGPU::V_STORE_IDX_B288:
    case AMDGPU::V_STORE_IDX_B320:
    case AMDGPU::V_STORE_IDX_B352:
    case AMDGPU::V_STORE_IDX_B384:
    case AMDGPU::V_STORE_IDX_B512:
    case AMDGPU::V_STORE_IDX_B576:
    case AMDGPU::V_STORE_IDX_B1024:
      return true;
    default:
      return false;
    }
  }
};

} // end namespace AMDGPUMI
} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUMACHINEINSTRS_H
