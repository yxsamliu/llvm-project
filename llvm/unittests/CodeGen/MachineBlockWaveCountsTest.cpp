//===- MachineBlockWaveCountsTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineBlockWaveCounts.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/MC/TargetRegistry.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
#include "MFCommon.inc"

class MachineBlockWaveCountsTest : public testing::Test {
protected:
  LLVMContext Ctx;
  Module M{"test", Ctx};
  std::unique_ptr<MachineFunction> MF = createMachineFunction(Ctx, M);
  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", &MF->getFunction());
  BasicBlock *Hot = BasicBlock::Create(Ctx, "hot", &MF->getFunction());
  BasicBlock *Cold = BasicBlock::Create(Ctx, "cold", &MF->getFunction());
  BasicBlock *Exit = BasicBlock::Create(Ctx, "exit", &MF->getFunction());
  MachineBasicBlock *MEntry;
  MachineBasicBlock *MHot;
  MachineBasicBlock *MCold;
  MachineBasicBlock *MExit;
  MachineBlockWaveCounts Counts;

  MachineBlockWaveCountsTest() {
    IRBuilder<> B(Entry);
    FunctionCallee Condition =
        M.getOrInsertFunction("condition", B.getInt1Ty());
    B.CreateCondBr(B.CreateCall(Condition), Hot, Cold);
    UncondBrInst::Create(Exit, Hot);
    UncondBrInst::Create(Exit, Cold);
    ReturnInst::Create(Ctx, Exit);
    MEntry = MF->CreateMachineBasicBlock(Entry);
    MHot = MF->CreateMachineBasicBlock(Hot);
    MCold = MF->CreateMachineBasicBlock(Cold);
    MExit = MF->CreateMachineBasicBlock(Exit);
    for (MachineBasicBlock *MBB : {MEntry, MHot, MCold, MExit})
      MF->push_back(MBB);
    MEntry->addSuccessor(MHot);
    MEntry->addSuccessor(MCold);
    MHot->addSuccessor(MExit);
    MCold->addSuccessor(MExit);
  }

  void setCounts() { setBlockWaveCounts(MF->getFunction(), {100, 90, 0, 100}); }
};

TEST_F(MachineBlockWaveCountsTest, MeasuredZeroAndSparseData) {
  setCounts();
  ASSERT_TRUE(extractMappedBlockWaveCounts(*MF, Counts));
  EXPECT_EQ(Counts.EntryCount, 100u);
  EXPECT_EQ(Counts.Counts.size(), 4u);
  EXPECT_EQ(Counts.Counts.at(MHot), 90u);
  EXPECT_EQ(Counts.Counts.at(MCold), 0u);
  EXPECT_EQ(Counts.RejectedBlocks, 0u);

  BitVector Present(4, true);
  Present.reset(2);
  setBlockWaveCounts(MF->getFunction(), {100, 90, 0, 100}, Present);
  ASSERT_TRUE(extractMappedBlockWaveCounts(*MF, Counts));
  EXPECT_EQ(Counts.Counts.size(), 3u);
  EXPECT_FALSE(Counts.Counts.contains(MCold));
  EXPECT_EQ(Counts.RejectedBlocks, 0u);
}

TEST_F(MachineBlockWaveCountsTest, MissingProfileClearsSnapshot) {
  setCounts();
  ASSERT_TRUE(extractMappedBlockWaveCounts(*MF, Counts));
  clearBlockWaveCounts(MF->getFunction());
  EXPECT_FALSE(extractMappedBlockWaveCounts(*MF, Counts));
  EXPECT_TRUE(Counts.Counts.empty());
  EXPECT_EQ(Counts.EntryCount, 0u);
  EXPECT_EQ(Counts.RejectedBlocks, 0u);
}

TEST_F(MachineBlockWaveCountsTest, DuplicateMappingRejectsNeighbors) {
  setCounts();
  MF->push_back(MF->CreateMachineBasicBlock(Hot));
  ASSERT_TRUE(extractMappedBlockWaveCounts(*MF, Counts));
  EXPECT_EQ(Counts.Counts.size(), 1u);
  EXPECT_TRUE(Counts.Counts.contains(MCold));
  EXPECT_EQ(Counts.RejectedBlocks, 3u);
}

TEST_F(MachineBlockWaveCountsTest, MachinePredecessorAndSuccessorMismatch) {
  setCounts();
  MEntry->addSuccessor(MExit);
  ASSERT_TRUE(extractMappedBlockWaveCounts(*MF, Counts));
  EXPECT_FALSE(Counts.Counts.contains(MEntry));
  EXPECT_FALSE(Counts.Counts.contains(MExit));
  EXPECT_TRUE(Counts.Counts.contains(MHot));
  EXPECT_TRUE(Counts.Counts.contains(MCold));
  EXPECT_EQ(Counts.RejectedBlocks, 2u);
  MEntry->removeSuccessor(MExit);
  ASSERT_TRUE(extractMappedBlockWaveCounts(*MF, Counts));
  EXPECT_EQ(Counts.Counts.size(), 4u);
  EXPECT_EQ(Counts.RejectedBlocks, 0u);
}

TEST_F(MachineBlockWaveCountsTest, UnmappedMachineNeighbor) {
  setCounts();
  MachineBasicBlock *New = MF->CreateMachineBasicBlock();
  MF->push_back(New);
  MEntry->removeSuccessor(MHot);
  MEntry->addSuccessor(New);
  New->addSuccessor(MHot);
  ASSERT_TRUE(extractMappedBlockWaveCounts(*MF, Counts));
  EXPECT_FALSE(Counts.Counts.contains(New));
  EXPECT_FALSE(Counts.Counts.contains(MEntry));
  EXPECT_FALSE(Counts.Counts.contains(MHot));
  EXPECT_EQ(Counts.RejectedBlocks, 2u);
}

TEST_F(MachineBlockWaveCountsTest, MissingMachineBlock) {
  setCounts();
  MEntry->removeSuccessor(MHot);
  MHot->removeSuccessor(MExit);
  MHot->eraseFromParent();
  ASSERT_TRUE(extractMappedBlockWaveCounts(*MF, Counts));
  EXPECT_EQ(Counts.Counts.size(), 1u);
  EXPECT_TRUE(Counts.Counts.contains(MCold));
  EXPECT_EQ(Counts.RejectedBlocks, 3u);
}

TEST_F(MachineBlockWaveCountsTest, StaleIRMappingIsNotRevived) {
  setCounts();
  cast<UncondBrInst>(Cold->getTerminator())->setSuccessor(0, Hot);
  MCold->removeSuccessor(MExit);
  MCold->addSuccessor(MHot);
  ASSERT_TRUE(extractMappedBlockWaveCounts(*MF, Counts));
  EXPECT_FALSE(Counts.Counts.contains(MCold));
}

TEST_F(MachineBlockWaveCountsTest, OriginalEntryNormalizationSurvivesNewEntry) {
  setCounts();
  BasicBlock *New = BasicBlock::Create(Ctx, "new", &MF->getFunction(), Entry);
  UncondBrInst::Create(Entry, New);
  MachineBasicBlock *MNew = MF->CreateMachineBasicBlock(New);
  MF->push_front(MNew);
  MNew->addSuccessor(MEntry);
  ASSERT_TRUE(extractMappedBlockWaveCounts(*MF, Counts));
  EXPECT_EQ(Counts.EntryCount, 100u);
  EXPECT_FALSE(Counts.Counts.contains(MNew));
  EXPECT_EQ(Counts.Counts.at(MHot), 90u);
}

TEST_F(MachineBlockWaveCountsTest, RawSaturatedCountsAreNotCostPolicy) {
  setBlockWaveCounts(MF->getFunction(), {100, UINT64_MAX, 0, 100});
  ASSERT_TRUE(extractMappedBlockWaveCounts(*MF, Counts));
  EXPECT_EQ(Counts.Counts.at(MHot), UINT64_MAX);
  setBlockWaveCounts(MF->getFunction(), {UINT64_MAX, 90, 0, 100});
  ASSERT_TRUE(extractMappedBlockWaveCounts(*MF, Counts));
  EXPECT_EQ(Counts.EntryCount, UINT64_MAX);
}

TEST(MachineBlockWaveFrequencyTest, NormalizationAndPositiveFloor) {
  EXPECT_EQ(scaleWaveCount(0, 100, BlockFrequency(1024)).getFrequency(), 1u);
  EXPECT_EQ(scaleWaveCount(25, 100, BlockFrequency(1024)).getFrequency(), 256u);
  EXPECT_EQ(scaleWaveCount(100, 100, BlockFrequency(1024)).getFrequency(),
            1024u);
  EXPECT_EQ(scaleWaveCount(1000, 100, BlockFrequency(1024)).getFrequency(),
            10240u);
  EXPECT_EQ(scaleWaveCount(1, UINT64_MAX, BlockFrequency(1024)).getFrequency(),
            1u);
}

} // namespace
