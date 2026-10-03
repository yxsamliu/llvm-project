//===- WaveRegionCostTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodeGenTestBase.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/SpillPlacement.h"
#include "llvm/Config/Targets.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/TargetSelect.h"
#include <tuple>

using namespace llvm;

namespace {
class WaveRegionCostTest : public CodeGenTestBase {
public:
  static void SetUpTestCase() {
#if LLVM_HAS_AMDGPU_TARGET
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
#endif
  }
  void SetUp() override {
    setUpImpl("amdgcn-amd-amdhsa", "gfx900", "");
    if (!TM)
      return;
    ASSERT_TRUE(parseMIR(R"MIR(
--- |
  target triple = "amdgcn-amd-amdhsa"
  define void @test(i1 %c) {
  entry:
    br label %once
  once:
    br i1 %c, label %a, label %b
  a:
    br i1 %c, label %b, label %exit
  b:
    br i1 %c, label %a, label %exit
  exit:
    ret void
  }
...
---
name: test
body: |
  bb.0.entry:
    successors: %bb.1
    S_BRANCH %bb.1
  bb.1.once:
    successors: %bb.2, %bb.3
    S_CBRANCH_SCC1 %bb.2, implicit $scc
    S_BRANCH %bb.3
  bb.2.a:
    successors: %bb.3, %bb.4
    S_CBRANCH_SCC1 %bb.3, implicit $scc
    S_BRANCH %bb.4
  bb.3.b:
    successors: %bb.2, %bb.4
    S_CBRANCH_SCC1 %bb.2, implicit $scc
    S_BRANCH %bb.4
  bb.4.exit:
    S_ENDPGM 0
...
)MIR"));
    BitVector HasCounts(5);
    HasCounts.set(0).set(2).set(4);
    setBlockWaveCounts(*Mod->getFunction("test"), {100, 0, 1000, 0, 0},
                       HasCounts);
  }
  SpillPlacement &placement() {
    return MFAM.getResult<SpillPlacementAnalysis>(getMF("test"));
  }
};

TEST_F(WaveRegionCostTest, AcyclicUnknownBound) {
  auto &SP = placement();
  ASSERT_TRUE(SP.hasMeasuredWaveBlocks());
  EXPECT_TRUE(SP.isWaveCostDifferencePositive({0, -2, 1, 0, 0}));
  EXPECT_FALSE(SP.isWaveCostDifferencePositive({0, -11, 1, 0, 0}));
  EXPECT_FALSE(SP.isWaveCostDifferencePositive({1, -1, 0, 0, 0}));
}

TEST_F(WaveRegionCostTest, IrreducibleUnknownBound) {
  auto &SP = placement();
  EXPECT_FALSE(SP.isWaveCostDifferencePositive({0, 0, 1, -1, 0}));
  EXPECT_TRUE(SP.isWaveCostDifferencePositive({0, 0, 1, 1, 0}));
  EXPECT_FALSE(SP.isWaveCostDifferencePositive({0, 0, 0, 1, 0}));
}

TEST_F(WaveRegionCostTest, MeasuredZeroKeepsFloor) {
  auto &SP = placement();
  EXPECT_EQ(SP.getBlockFrequency(4).getFrequency(), 1u);
  // A positive placement floor is not evidence of observed execution.
  EXPECT_FALSE(SP.isWaveCostDifferencePositive({0, 0, 0, 0, 1}));
  EXPECT_FALSE(SP.isWaveCostDifferencePositive({0, 0, 0, 0, -1}));
}

TEST_F(WaveRegionCostTest, CancellationAndOverflow) {
  auto &SP = placement();
  EXPECT_FALSE(SP.isWaveCostDifferencePositive({0, 0, 0, 0, 0}));
  EXPECT_FALSE(SP.isWaveCostDifferencePositive({0, 0, INT64_MAX, 0, 0}));
  EXPECT_FALSE(SP.isWaveCostDifferencePositive({0, INT64_MIN, 1, 0, 0}));
}

TEST_F(WaveRegionCostTest, MissingProfile) {
  clearBlockWaveCounts(*Mod->getFunction("test"));
  EXPECT_FALSE(placement().hasMeasuredWaveBlocks());
  EXPECT_FALSE(placement().isWaveCostDifferencePositive({0, -1, 1, 0, 0}));
}

// A function-level uniformity marker supplies no execution frequency for
// unmeasured blocks, including those in an irreducible cycle.
TEST_F(WaveRegionCostTest, UniformityMarkerKeepsUnmeasuredFrequencies) {
  auto &MF = getMF("test");
  MF.getFunction().setMetadata(LLVMContext::MD_uniformity_profile,
                               MDNode::get(Mod->getContext(), {}));
  auto &MBFI = MFAM.getResult<MachineBlockFrequencyAnalysis>(MF);
  auto &SP = placement();
  for (unsigned Number : {1u, 3u})
    EXPECT_EQ(SP.getBlockFrequency(Number),
              MBFI.getBlockFreq(MF.getBlockNumbered(Number)));
  EXPECT_EQ(SP.getBlockFrequency(4).getFrequency(), 1u);
}

TEST_F(WaveRegionCostTest, UniformityMarkerWithoutWaveCounts) {
  auto &MF = getMF("test");
  clearBlockWaveCounts(MF.getFunction());
  MF.getFunction().setMetadata(LLVMContext::MD_uniformity_profile,
                               MDNode::get(Mod->getContext(), {}));
  auto &MBFI = MFAM.getResult<MachineBlockFrequencyAnalysis>(MF);
  auto &SP = placement();
  EXPECT_FALSE(SP.hasMeasuredWaveBlocks());
  for (auto &MBB : MF)
    EXPECT_EQ(SP.getBlockFrequency(MBB.getNumber()), MBFI.getBlockFreq(&MBB));
}

TEST_F(WaveRegionCostTest, InvalidEntry) {
  setBlockWaveCounts(*Mod->getFunction("test"), {0, 0, 1000, 0, 0});
  EXPECT_FALSE(placement().hasMeasuredWaveBlocks());
  EXPECT_FALSE(placement().isWaveCostDifferencePositive({0, -1, 1, 0, 0}));
}

TEST_F(WaveRegionCostTest, SaturatedCount) {
  setBlockWaveCounts(*Mod->getFunction("test"), {1, 0, UINT64_MAX, 0, 0});
  EXPECT_FALSE(placement().isWaveCostDifferencePositive({0, 0, 1, 0, 0}));
}

class WaveSpillOptionsTest
    : public WaveRegionCostTest,
      public testing::WithParamInterface<std::tuple<bool, bool>> {};

TEST_P(WaveSpillOptionsTest, IndependentConsumers) {
  auto [UsePlacement, UseCosts] = GetParam();
  auto &Options = cl::getRegisteredOptions();
  auto *Placement = static_cast<cl::opt<bool> *>(
      Options.lookup("wave-guided-spill-placement"));
  auto *Costs =
      static_cast<cl::opt<bool> *>(Options.lookup("wave-guided-spill-costs"));
  ASSERT_NE(Placement, nullptr);
  ASSERT_NE(Costs, nullptr);
  SaveAndRestore<bool> RestorePlacement(Placement->getValue(), UsePlacement);
  SaveAndRestore<bool> RestoreCosts(Costs->getValue(), UseCosts);

  auto &MF = getMF("test");
  auto &MBFI = MFAM.getResult<MachineBlockFrequencyAnalysis>(MF);
  auto &SP = placement();
  EXPECT_EQ(SP.hasMeasuredWaveBlocks(), UseCosts);
  // A positive raw-count bound survives disabling frequency reweighting.
  EXPECT_EQ(SP.isWaveCostDifferencePositive({0, -2, 1, 0, 0}), UseCosts);
  // Placement still uses the measured-zero floor with cost bounds disabled.
  BlockFrequency Expected = UsePlacement
                                ? BlockFrequency(1)
                                : MBFI.getBlockFreq(MF.getBlockNumbered(4));
  EXPECT_EQ(SP.getBlockFrequency(4), Expected);
}

INSTANTIATE_TEST_SUITE_P(IndependentOptions, WaveSpillOptionsTest,
                         testing::Combine(testing::Bool(), testing::Bool()));
} // namespace
