//===- BlockUniformityProfile.h - Block uniformity from PGO -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provide per-(Machine)basic-block uniformity information from PGO profiles.
//
// The source of truth is IR metadata attached during PGO use:
//   - Metadata on the function means uniformity profile is available.
//   - Metadata on a terminator means the block is uniform.
//   - Missing metadata on a terminator falls back to static control-flow
//     divergence analysis.
//   - Metadata name: "block.uniformity.profile".
// An opt-in spill-placement experiment can suppress a static divergent-branch
// fallback when direct profile votes observed only unanimous active-lane
// decisions. This changes a cost heuristic, not the IR uniformity proof.
//
// This is intentionally target-agnostic: any backend that produces
// uniformity bits in the profile can attach the same metadata and reuse this
// proxy in codegen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_BLOCKUNIFORMITYPROFILE_H
#define LLVM_CODEGEN_BLOCKUNIFORMITYPROFILE_H

#include "llvm/ADT/BitVector.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class MachineBasicBlock;
class MachineFunction;
class raw_ostream;

class BlockUniformityProfile {
public:
  LLVM_ABI void compute(const MachineFunction &MF, const UniformityInfo &UI);

  bool hasProfile() const { return HasProfile; }

  // Returns true if the block lacks a profiled-uniform annotation and static
  // analysis says it may be reached through divergent control flow.
  LLVM_ABI bool isDivergent(const MachineBasicBlock &MBB) const;

  LLVM_ABI void print(raw_ostream &OS, const MachineFunction &MF) const;

private:
  bool HasProfile = false;
  unsigned NumBlockIDs = 0;
  BitVector DivergentBlocks;
};

class BlockUniformityProfileProxy
    : public AnalysisInfoMixin<BlockUniformityProfileProxy> {
  friend AnalysisInfoMixin<BlockUniformityProfileProxy>;
  static AnalysisKey Key;

public:
  using Result = BlockUniformityProfile;
  LLVM_ABI Result run(MachineFunction &MF,
                      MachineFunctionAnalysisManager &MFAM);
};

class BlockUniformityProfilePrinterPass
    : public RequiredPassInfoMixin<BlockUniformityProfilePrinterPass> {
  raw_ostream &OS;

public:
  explicit BlockUniformityProfilePrinterPass(raw_ostream &OS) : OS(OS) {}
  LLVM_ABI PreservedAnalyses run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_BLOCKUNIFORMITYPROFILE_H
