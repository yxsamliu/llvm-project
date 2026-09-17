//===- ProfDataUtilsTest.cpp - Profiling metadata tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ProfDataUtils.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
#include <initializer_list>

using namespace llvm;

namespace {
static BitVector makeBitVector(unsigned Size,
                               std::initializer_list<unsigned> SetBits) {
  BitVector Result(Size);
  for (unsigned Index : SetBits)
    Result.set(Index);
  return Result;
}

class WaveProfileTest : public testing::Test {
protected:
  LLVMContext Context;
  std::unique_ptr<Module> M;

  void SetUp() override {
    SMDiagnostic Error;
    M = parseAssemblyString(R"(
      define void @diamond(i1 %condition) {
      entry:
        br i1 %condition, label %left, label %right
      left:
        br label %exit
      right:
        br label %exit
      exit:
        ret void
      })",
                            Error, Context);
    ASSERT_TRUE(M);
  }
};

TEST_F(WaveProfileTest, RoundTripAndReplacement) {
  Function &F = *M->getFunction("diamond");
  F.setEntryCount(6400);
  SmallVector<uint64_t> Counts{99};
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
  EXPECT_TRUE(Counts.empty());
  setBlockWaveCounts(F, {100, 100, 100, 100});
  EXPECT_TRUE(extractBlockWaveCounts(F, Counts));
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 100, 100, 100}));
  EXPECT_FALSE(verifyModule(*M, &errs()));

  std::string Text;
  raw_string_ostream OS(Text);
  M->print(OS, nullptr);
  SMDiagnostic Error;
  std::unique_ptr<Module> Reloaded = parseAssemblyString(Text, Error, Context);
  ASSERT_TRUE(Reloaded);
  EXPECT_TRUE(
      extractBlockWaveCounts(*Reloaded->getFunction("diamond"), Counts));
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 100, 100, 100}));

  setBlockWaveCounts(F, {200, 0, 200, 200});
  EXPECT_TRUE(extractBlockWaveCounts(F, Counts));
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{200, 0, 200, 200}));
}

TEST_F(WaveProfileTest, PreserveAcrossInstructionChanges) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 100, 100, 100});
  cast<CondBrInst>(F.getEntryBlock().getTerminator())
      ->setCondition(ConstantInt::getTrue(Context));
  SmallVector<uint64_t> Counts{99};
  EXPECT_TRUE(extractBlockWaveCounts(F, Counts));
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 100, 100, 100}));
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(WaveProfileTest, PreserveAcrossConditionalSuccessorSwap) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  cast<CondBrInst>(F.getEntryBlock().getTerminator())->swapSuccessors();
  SmallVector<uint64_t> Counts;
  EXPECT_TRUE(extractBlockWaveCounts(F, Counts));
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 10, 90, 100}));
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(WaveProfileTest, PreserveIdentityAcrossBlockReordering) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  Left->moveAfter(Left->getNextNode());
  SmallVector<uint64_t> Counts;
  EXPECT_TRUE(extractBlockWaveCounts(F, Counts));
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 90, 10, 100}));
}

TEST_F(WaveProfileTest, PreserveAcrossEntryCountChangeButRejectRename) {
  Function &F = *M->getFunction("diamond");
  F.setEntryCount(6400);
  setBlockWaveCounts(F, {100, 100, 100, 100});
  F.setEntryCount(3200);
  SmallVector<uint64_t> Counts;
  EXPECT_TRUE(extractBlockWaveCounts(F, Counts));
  F.setName("specialized");
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
}

TEST_F(WaveProfileTest, RejectRedirectedEdge) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 100, 100, 100});
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  BasicBlock *Right = Left->getNextNode();
  cast<UncondBrInst>(Left->getTerminator())->setSuccessor(Right);
  SmallVector<uint64_t> Counts;
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
}

TEST_F(WaveProfileTest, RejectDuplicateBlockIdentity) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 100, 100, 100});
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  BasicBlock *Right = Left->getNextNode();
  Right->getTerminator()->setMetadata(
      LLVMContext::MD_wave_profile_block,
      Left->getTerminator()->getMetadata(LLVMContext::MD_wave_profile_block));
  SmallVector<uint64_t> Counts;
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
}

TEST_F(WaveProfileTest, RejectSplitBlock) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 100, 100, 100});
  BasicBlock &Entry = F.getEntryBlock();
  Entry.splitBasicBlock(Entry.begin(), "split");
  SmallVector<uint64_t> Counts;
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(WaveProfileTest, MapCountsAfterRemovingOriginalBlock) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  BasicBlock *Exit = &F.back();
  Left->replaceAllUsesWith(Exit);
  Left->eraseFromParent();

  SmallVector<uint64_t> Counts;
  BitVector HasCounts;
  uint64_t EntryCount = 0;
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts, &HasCounts));
  EXPECT_TRUE(
      extractMappedBlockWaveCounts(F, Counts, HasCounts, EntryCount));
  EXPECT_EQ(EntryCount, 100u);
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 90, 100}));
  EXPECT_EQ(HasCounts, makeBitVector(3, {1}));
}

TEST_F(WaveProfileTest, MapCountsAfterRemovingOriginalEntry) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  F.getEntryBlock().eraseFromParent();

  SmallVector<uint64_t> Counts;
  BitVector HasCounts;
  uint64_t EntryCount = 0;
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts, &HasCounts));
  EXPECT_TRUE(
      extractMappedBlockWaveCounts(F, Counts, HasCounts, EntryCount));
  EXPECT_EQ(EntryCount, 100u);
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{10, 90, 100}));
  EXPECT_EQ(HasCounts, makeBitVector(3, {0, 1, 2}));
}

TEST_F(WaveProfileTest, MapCountsAroundNewUnmeasuredBlock) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  BasicBlock *Exit = &F.back();
  BasicBlock *Inserted = BasicBlock::Create(Context, "inserted", &F, Exit);
  UncondBrInst::Create(Exit, Inserted);
  cast<UncondBrInst>(Left->getTerminator())->setSuccessor(Inserted);

  SmallVector<uint64_t> Counts;
  BitVector HasCounts;
  uint64_t EntryCount = 0;
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts, &HasCounts));
  EXPECT_TRUE(
      extractMappedBlockWaveCounts(F, Counts, HasCounts, EntryCount));
  EXPECT_EQ(EntryCount, 100u);
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 10, 90, 0, 100}));
  EXPECT_EQ(HasCounts, makeBitVector(5, {0, 2}));
}

TEST_F(WaveProfileTest, MapCountsAroundDuplicateBlockIdentity) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  BasicBlock *Right = Left->getNextNode();
  Right->getTerminator()->setMetadata(
      LLVMContext::MD_wave_profile_block,
      Left->getTerminator()->getMetadata(LLVMContext::MD_wave_profile_block));

  SmallVector<uint64_t> Counts;
  BitVector HasCounts;
  uint64_t EntryCount = 0;
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts, &HasCounts));
  EXPECT_TRUE(
      extractMappedBlockWaveCounts(F, Counts, HasCounts, EntryCount));
  EXPECT_EQ(EntryCount, 100u);
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 0, 0, 100}));
  EXPECT_EQ(HasCounts, makeBitVector(4, {}));
}

TEST_F(WaveProfileTest, MapCountsAroundRedirectedEdge) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  BasicBlock *Right = Left->getNextNode();
  cast<UncondBrInst>(Left->getTerminator())->setSuccessor(Right);

  SmallVector<uint64_t> Counts;
  BitVector HasCounts;
  uint64_t EntryCount = 0;
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts, &HasCounts));
  EXPECT_TRUE(
      extractMappedBlockWaveCounts(F, Counts, HasCounts, EntryCount));
  EXPECT_EQ(EntryCount, 100u);
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 10, 90, 100}));
  EXPECT_EQ(HasCounts, makeBitVector(4, {0}));
}

TEST_F(WaveProfileTest, RepresentExplicitlyUnmeasuredBlocks) {
  Function &F = *M->getFunction("diamond");
  BasicBlock &Entry = F.getEntryBlock();
  Entry.splitBasicBlock(Entry.begin(), "synthetic");

  SmallVector<uint64_t> ExpectedCounts{100, 0, 100, 100, 100};
  BitVector ExpectedHasCounts(ExpectedCounts.size(), true);
  ExpectedHasCounts.reset(1);
  setBlockWaveCounts(F, ExpectedCounts, ExpectedHasCounts);

  SmallVector<uint64_t> Counts;
  BitVector HasCounts;
  EXPECT_TRUE(extractBlockWaveCounts(F, Counts, &HasCounts));
  EXPECT_EQ(Counts, ExpectedCounts);
  EXPECT_EQ(HasCounts, ExpectedHasCounts);
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
  EXPECT_TRUE(Counts.empty());
}

TEST_F(WaveProfileTest, RejectUnmeasuredOriginalEntry) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  Instruction *EntryTerminator = F.getEntryBlock().getTerminator();
  MDNode *EntryMD =
      EntryTerminator->getMetadata(LLVMContext::MD_wave_profile_block);
  SmallVector<Metadata *> Ops(EntryMD->op_begin(), EntryMD->op_end());
  Ops[3] =
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt64Ty(Context), 0));
  EntryTerminator->setMetadata(LLVMContext::MD_wave_profile_block,
                               MDNode::get(Context, Ops));

  SmallVector<uint64_t> Counts;
  BitVector HasCounts;
  uint64_t EntryCount = 0;
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts, &HasCounts));
  EXPECT_FALSE(
      extractMappedBlockWaveCounts(F, Counts, HasCounts, EntryCount));
  EXPECT_EQ(EntryCount, 0u);
  EXPECT_TRUE(Counts.empty());
  EXPECT_TRUE(HasCounts.empty());
}

TEST_F(WaveProfileTest, RejectUnsupportedOrMalformedMetadata) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 100, 100, 100});
  MDNode *Valid = F.getMetadata(LLVMContext::MD_wave_profile);
  SmallVector<Metadata *> Ops(Valid->op_begin(), Valid->op_end());
  SmallVector<uint64_t> Counts{99};

  Ops[0] =
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt64Ty(Context), 3));
  F.setMetadata(LLVMContext::MD_wave_profile, MDNode::get(Context, Ops));
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
  EXPECT_TRUE(Counts.empty());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  Ops[0] = Valid->getOperand(0);
  Ops[2] =
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Context), 100));
  F.setMetadata(LLVMContext::MD_wave_profile, MDNode::get(Context, Ops));
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
  EXPECT_TRUE(Counts.empty());

  Ops[2] = nullptr;
  F.setMetadata(LLVMContext::MD_wave_profile, MDNode::get(Context, Ops));
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
  EXPECT_TRUE(Counts.empty());

  Ops[2] = Valid->getOperand(2);
  Ops.pop_back();
  F.setMetadata(LLVMContext::MD_wave_profile, MDNode::get(Context, Ops));
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
  EXPECT_TRUE(Counts.empty());
  EXPECT_FALSE(verifyModule(*M, &errs()));
}
} // namespace
