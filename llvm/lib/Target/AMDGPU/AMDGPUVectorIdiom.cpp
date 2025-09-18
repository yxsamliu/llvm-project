//===- AMDGPUVectorIdiom.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AMDGPU-specific vector idiom canonicalizations to unblock SROA and
// subsequent scalarization/vectorization.
//
// Motivation:
// - HIP vector types are often modeled as structs and copied with memcpy.
//   Address-level selects on such copies block SROA. Converting to value-level
//   operations or splitting the CFG enables SROA to break aggregates, which
//   unlocks scalarization/vectorization on AMDGPU.
//
// Example pattern:
//   %src = select i1 %c, ptr %A, ptr %B
//   call void @llvm.memcpy(ptr %dst, ptr %src, i32 16, i1 false)
//
// Objectives:
// - Canonicalize small memcpy patterns where source or destination is a select
// of pointers.
// - Prefer value-level selects (on loaded values) over address-level selects
// when safe.
// - When speculation is unsafe, split the CFG to isolate each arm.
//
// Assumptions:
// - Only handles non-volatile memcpy with constant length N where 0 < N <=
// MaxBytes (default 32).
// - Source and destination must be in the same address space.
// - Speculative loads are allowed only if a conservative alignment check
// passes.
// - No speculative stores are introduced.
//
// Transformations:
// - Source-select memcpy: attempt speculative loads -> value select -> single
// store.
//   Fallback is CFG split with two memcpy calls.
// - Destination-select memcpy: always CFG split to avoid speculative stores.
//
// Run this pass early, before SROA.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUVectorIdiom.h"
#include "AMDGPU.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/Operator.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace std;
using namespace PatternMatch;

#define DEBUG_TYPE "amdgpu-vector-idiom"

namespace {

static cl::opt<bool>
    AMDGPUVectorIdiomEnable("amdgpu-vector-idiom-enable",
                            cl::desc("Enable pass AMDGPUVectorIdiom"),
                            cl::init(true));

// Selects an integer or integer-vector element type matching NBytes, using the
// minimum proven alignment to decide the widest safe element width.
// Assumptions:
// - Pointee types are opaque; the element choice is based solely on size and
// alignment.
// - Falls back to <N x i8> if wider lanes are not safe/aligned.
static Type *getIntOrVecTypeForSize(uint64_t NBytes, LLVMContext &Ctx,
                                    Align MinProvenAlign = Align(1)) {
  auto CanUseI64 = [&]() { return MinProvenAlign >= Align(8); };
  auto CanUseI32 = [&]() { return MinProvenAlign >= Align(4); };
  auto CanUseI16 = [&]() { return MinProvenAlign >= Align(2); };

  if (NBytes == 32 && CanUseI64())
    return FixedVectorType::get(Type::getInt64Ty(Ctx), 4);

  if ((NBytes % 4) == 0 && CanUseI32())
    return FixedVectorType::get(Type::getInt32Ty(Ctx), NBytes / 4);

  if ((NBytes % 2) == 0 && CanUseI16())
    return FixedVectorType::get(Type::getInt16Ty(Ctx), NBytes / 2);

  return FixedVectorType::get(Type::getInt8Ty(Ctx), NBytes);
}

static Align minAlign(Align A, Align B) { return A < B ? A : B; }

// Checks if both pointer operands can be speculatively loaded for N bytes and
// computes the minimum alignment to use.
// Notes:
// - Intentionally conservative: relies on isDereferenceablePointer and
//   getOrEnforceKnownAlignment.
// - AA/TLI are not used for deeper reasoning here.
// Emits verbose LLVM_DEBUG logs explaining why speculation is disallowed.
// Return false reasons include: either arm not dereferenceable or computed
// known alignment < 1.
static bool bothArmsSafeToSpeculateLoads(Value *A, Value *B, uint64_t Size,
                                         Align &OutAlign, const DataLayout &DL,
                                         AssumptionCache *AC,
                                         const DominatorTree *DT,
                                         Instruction *CtxI) {
  APInt SizeAPInt(DL.getIndexTypeSizeInBits(A->getType()), Size);
  if (!isDereferenceableAndAlignedPointer(B, Align(1), SizeAPInt, DL, CtxI, AC,
                                          DT, nullptr)) {
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Not speculating loads: false arm "
                      << "(B) not dereferenceable for " << Size
                      << " bytes at align(1)\n");
    LLVM_DEBUG(dbgs() << "    false arm (B) value: " << *B << '\n');
    return false;
  }

  Align AlignB =
      llvm::getOrEnforceKnownAlignment(B, Align(1), DL, nullptr, AC, DT);

  if (AlignB < Align(1)) {
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Not speculating loads: known "
                      << "alignment of false arm (B) < 1: " << AlignB.value()
                      << '\n');
    return false;
  }

  if (!isDereferenceableAndAlignedPointer(A, Align(1), SizeAPInt, DL, CtxI, AC,
                                          DT, nullptr)) {
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Not speculating loads: true arm "
                      << "(A) not dereferenceable for " << Size
                      << " bytes at align(1)\n");
    LLVM_DEBUG(dbgs() << "    true arm (A) value: " << *A << '\n');
    return false;
  }

  Align AlignA =
      llvm::getOrEnforceKnownAlignment(A, Align(1), DL, nullptr, AC, DT);

  if (AlignA < Align(1)) {
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Not speculating loads: known "
                      << "alignment of true arm (A) < 1: " << AlignA.value()
                      << '\n');
    return false;
  }

  OutAlign = minAlign(AlignA, AlignB);
  LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Speculative loads allowed: "
                    << "minAlign=" << OutAlign.value() << '\n');
  return true;
}

struct AMDGPUVectorIdiomImpl {
  const unsigned MaxBytes;
  bool CFGChanged = false;

  AMDGPUVectorIdiomImpl(unsigned MaxBytes) : MaxBytes(MaxBytes) {}

  // Rewrites memcpy when the source is a select of pointers. Prefers a
  // value-level select (two loads + select + one store) if speculative loads
  // are safe. Otherwise, falls back to a guarded CFG split with two memcpy
  //
  // New: Wrapper entry point to run the original memcpy/select-based
  // transformations over a function (extracted from pass driver).
  bool runMemcpySelectTransforms(Function &F, const DataLayout &DL,
                                 const DominatorTree &DT,
                                 AssumptionCache &AC);
  // calls. Assumptions:
  // - Non-volatile, constant length, within MaxBytes.
  // - Source and destination in the same address space.
  bool transformSelectMemcpySource(MemCpyInst &MT, SelectInst &Sel,
                                   const DataLayout &DL,
                                   const DominatorTree *DT,
                                   AssumptionCache *AC) {
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Considering memcpy(select-src): "
                      << MT << '\n');
    IRBuilder<> B(&MT);
    Value *Dst = MT.getRawDest();
    Value *A = Sel.getTrueValue();
    Value *Bv = Sel.getFalseValue();

    ConstantInt *LenCI = cast<ConstantInt>(MT.getLength());
    uint64_t N = LenCI->getLimitedValue();

    if (Sel.isVolatile()) {
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Not rewriting: Select marked "
                        << "volatile (unexpected) in memcpy source\n");
      return false;
    }

    // This is a null check - always use CFG split
    Value *Cond = Sel.getCondition();
    ICmpInst *ICmp = dyn_cast<ICmpInst>(Cond);
    if (ICmp && ICmp->isEquality() &&
        (isa<ConstantPointerNull>(ICmp->getOperand(0)) ||
         isa<ConstantPointerNull>(ICmp->getOperand(1)))) {
      splitCFGForMemcpy(MT, Sel.getCondition(), A, Bv, true);
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Null check pattern - "
                           "using CFG split\n");
      return true;
    }

    Align DstAlign = MaybeAlign(MT.getDestAlign()).valueOrOne();
    Align AlignAB;
    bool CanSpeculate = false;

    const CallBase &CB = MT;
    const unsigned SrcArgIdx = 1;
    uint64_t DerefBytes = CB.getParamDereferenceableBytes(SrcArgIdx);
    bool HasDerefOrNull =
        CB.paramHasAttr(SrcArgIdx, Attribute::DereferenceableOrNull);
    bool HasNonNull = CB.paramHasAttr(SrcArgIdx, Attribute::NonNull);
    MaybeAlign SrcParamAlign = CB.getParamAlign(SrcArgIdx);
    Align ProvenSrcAlign =
        SrcParamAlign.value_or(MaybeAlign(MT.getSourceAlign()).valueOrOne());

    if (DerefBytes > 0) {
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] memcpy source param attrs: "
                        << "dereferenceable(" << DerefBytes << ")"
                        << (HasDerefOrNull ? " (or null)" : "")
                        << (HasNonNull ? ", nonnull" : "") << ", align "
                        << ProvenSrcAlign.value() << '\n');
      if (DerefBytes >= N && (!HasDerefOrNull || HasNonNull)) {
        LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Using memcpy source operand "
                          << "attributes at this use; accepting speculation\n");
        CanSpeculate = true;
        AlignAB = ProvenSrcAlign;
      } else {
        LLVM_DEBUG(
            dbgs() << "[AMDGPUVectorIdiom] Source param attrs not strong "
                   << "enough for speculation: need dereferenceable(" << N
                   << ") and nonnull; got dereferenceable(" << DerefBytes << ")"
                   << (HasDerefOrNull ? " (or null)" : "")
                   << (HasNonNull ? ", nonnull" : "") << '\n');
      }
    } else {
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] memcpy source param has no "
                        << "dereferenceable bytes attribute; align "
                        << ProvenSrcAlign.value() << '\n');
    }
    if (!CanSpeculate)
      CanSpeculate =
          bothArmsSafeToSpeculateLoads(A, Bv, N, AlignAB, DL, AC, DT, &MT);

    if (CanSpeculate) {
      Align MinAlign = std::min(AlignAB, DstAlign);
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Rewriting memcpy(select-src) "
                        << "with value-level select; N=" << N
                        << " minAlign=" << MinAlign.value() << '\n');

      Type *Ty = getIntOrVecTypeForSize(N, B.getContext(), MinAlign);

      LoadInst *LA = B.CreateAlignedLoad(Ty, A, MinAlign);
      LoadInst *LB = B.CreateAlignedLoad(Ty, Bv, MinAlign);
      Value *V = B.CreateSelect(Sel.getCondition(), LA, LB);

      (void)B.CreateAlignedStore(V, Dst, DstAlign);

      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Rewrote memcpy(select-src) to "
                           "value-select loads/stores: "
                        << MT << '\n');
      MT.eraseFromParent();
      return true;
    }

    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Falling back to CFG split for "
                      << "memcpy(select-src); speculation unsafe\n");
    splitCFGForMemcpy(MT, Sel.getCondition(), A, Bv, true);
    LLVM_DEBUG(
        dbgs()
        << "[AMDGPUVectorIdiom] Rewrote memcpy(select-src) by CFG split\n");
    return true;
  }

  // Rewrites memcpy when the destination is a select of pointers. To avoid
  // speculative stores, always splits the CFG and emits a memcpy per branch.
  // Assumptions mirror the source case.
  bool transformSelectMemcpyDest(MemCpyInst &MT, SelectInst &Sel) {
    Value *DA = Sel.getTrueValue();
    Value *DB = Sel.getFalseValue();
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Rewriting memcpy(select-dst) via "
                      << "CFG split to avoid speculative stores: " << MT
                      << '\n');

    splitCFGForMemcpy(MT, Sel.getCondition(), DA, DB, false);
    LLVM_DEBUG(
        dbgs()
        << "[AMDGPUVectorIdiom] Rewrote memcpy(select-dst) by CFG split\n");
    return true;
  }

  // Splits the CFG around a memcpy whose source or destination depends on a
  // condition. Clones memcpy in then/else using TruePtr/FalsePtr and rejoins.
  // Assumptions:
  // - MT has constant length and is non-volatile.
  // - TruePtr/FalsePtr are correct replacements for the selected operand.
  void splitCFGForMemcpy(MemCpyInst &MT, Value *Cond, Value *TruePtr,
                         Value *FalsePtr, bool IsSource) {
    CFGChanged = true;

    Function *F = MT.getFunction();
    BasicBlock *Cur = MT.getParent();
    BasicBlock *ThenBB = BasicBlock::Create(F->getContext(), "memcpy.then", F);
    BasicBlock *ElseBB = BasicBlock::Create(F->getContext(), "memcpy.else", F);
    BasicBlock *JoinBB =
        Cur->splitBasicBlock(BasicBlock::iterator(&MT), "memcpy.join");

    Cur->getTerminator()->eraseFromParent();
    IRBuilder<> B(Cur);
    B.CreateCondBr(Cond, ThenBB, ElseBB);

    ConstantInt *LenCI = cast<ConstantInt>(MT.getLength());

    IRBuilder<> BT(ThenBB);
    if (IsSource) {
      (void)BT.CreateMemCpy(MT.getRawDest(), MT.getDestAlign(), TruePtr,
                            MT.getSourceAlign(), LenCI, MT.isVolatile());
    } else {
      (void)BT.CreateMemCpy(TruePtr, MT.getDestAlign(), MT.getRawSource(),
                            MT.getSourceAlign(), LenCI, MT.isVolatile());
    }
    BT.CreateBr(JoinBB);

    IRBuilder<> BE(ElseBB);
    if (IsSource) {
      (void)BE.CreateMemCpy(MT.getRawDest(), MT.getDestAlign(), FalsePtr,
                            MT.getSourceAlign(), LenCI, MT.isVolatile());
    } else {
      (void)BE.CreateMemCpy(FalsePtr, MT.getDestAlign(), MT.getRawSource(),
                            MT.getSourceAlign(), LenCI, MT.isVolatile());
    }
    BE.CreateBr(JoinBB);

    MT.eraseFromParent();
  }
};

} // end anonymous namespace

AMDGPUVectorIdiomCombinePass::AMDGPUVectorIdiomCombinePass(unsigned MaxBytes)
    : MaxBytes(MaxBytes) {}

// Pass driver that locates small, constant-size, non-volatile memcpy calls
// where source or destination is a select in the same address space. Applies
// the source/destination transforms described above. Intended to run early to
// maximize SROA and subsequent optimizations.
PreservedAnalyses
AMDGPUVectorIdiomCombinePass::run(Function &F, FunctionAnalysisManager &FAM) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  auto &AC = FAM.getResult<AssumptionAnalysis>(F);

  if (!AMDGPUVectorIdiomEnable)
    return PreservedAnalyses::all();

  bool Changed = false;
  AMDGPUVectorIdiomImpl Impl(MaxBytes);

  // 1) Original memcpy/select-based transformation (refactored).
  Changed |= Impl.runMemcpySelectTransforms(F, DL, DT, AC);

  // 2) New: recover vector loads/stores from scalarized struct access.
  // Class defined at file scope.
  extern bool AMDGPUVectorRecoverRun(Function &F, const DataLayout &DL);
  Changed |= AMDGPUVectorRecoverRun(F, DL);

  if (!Changed)
    return PreservedAnalyses::all();

  // Be conservative: preserve only analyses we know remain valid.
  PreservedAnalyses PA;
  PA.preserve<AssumptionAnalysis>();
  PA.preserve<TargetLibraryAnalysis>();
  PA.preserve<TargetIRAnalysis>();

  // If we didn't change the CFG, we can keep DT/LI/PostDT.
  if (!Impl.CFGChanged) {
    PA.preserve<DominatorTreeAnalysis>();
    PA.preserve<LoopAnalysis>();
    PA.preserve<PostDominatorTreeAnalysis>();
  }

  return PA;
}

// New member function: run original memcpy/select transforms extracted from the
// pass driver.
bool AMDGPUVectorIdiomImpl::runMemcpySelectTransforms(Function &F,
                                                      const DataLayout &DL,
                                                      const DominatorTree &DT,
                                                      AssumptionCache &AC) {
  SmallVector<MemCpyInst *, 8> Worklist;
  for (Instruction &I : instructions(F)) {
    if (auto *MC = dyn_cast<MemCpyInst>(&I))
      Worklist.push_back(MC);
  }

  bool Changed = false;
  for (MemCpyInst *MT : Worklist) {
    Value *Dst = MT->getRawDest();
    Value *Src = MT->getRawSource();
    if (!isa<SelectInst>(Src) && !isa<SelectInst>(Dst))
      continue;

    LLVM_DEBUG({
      Value *DstV = MT->getRawDest();
      Value *SrcV = MT->getRawSource();
      unsigned DstAS = cast<PointerType>(DstV->getType())->getAddressSpace();
      unsigned SrcAS = cast<PointerType>(SrcV->getType())->getAddressSpace();
      Value *LenV = MT->getLength();

      auto dumpPtrForms = [&](StringRef Label, Value *V) {
        dbgs() << "      " << Label << ": " << *V << '\n';
        Value *StripCasts = V->stripPointerCasts();
        if (StripCasts != V)
          dbgs() << "        - stripCasts: " << *StripCasts << '\n';
        else
          dbgs() << "        - stripCasts: (no change)\n";
        Value *Underlying = getUnderlyingObject(V);
        if (Underlying != V)
          dbgs() << "        - underlying: " << *Underlying << '\n';
        else
          dbgs() << "        - underlying: (no change)\n";
      };

      auto dumpSelect = [&](StringRef Which, Value *V) {
        if (auto *SI = dyn_cast<SelectInst>(V)) {
          dbgs() << "  - " << Which << " is Select: " << *SI << '\n';
          dbgs() << "      cond: " << *SI->getCondition() << '\n';
          Value *T = SI->getTrueValue();
          Value *Fv = SI->getFalseValue();
          dumpPtrForms("true", T);
          dumpPtrForms("false", Fv);
        }
      };

      dbgs() << "[AMDGPUVectorIdiom] Found memcpy: " << *MT << '\n'
             << "  in function: " << F.getName() << '\n'
             << "  - volatile=" << (MT->isVolatile() ? "true" : "false") << '\n'
             << "  - sameAS=" << (DstAS == SrcAS ? "true" : "false")
             << " (dstAS=" << DstAS << ", srcAS=" << SrcAS << ")\n"
             << "  - constLen=" << (isa<ConstantInt>(LenV) ? "true" : "false");
      if (auto *LCI = dyn_cast<ConstantInt>(LenV))
        dbgs() << " (N=" << LCI->getLimitedValue() << ")";
      dbgs() << '\n'
             << "  - srcIsSelect=" << (isa<SelectInst>(SrcV) ? "true" : "false")
             << '\n'
             << "  - dstIsSelect=" << (isa<SelectInst>(DstV) ? "true" : "false")
             << '\n';
      dumpSelect("src", SrcV);
      dumpSelect("dst", DstV);
    });

    if (MT->isVolatile()) {
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Skip: memcpy is volatile\n");
      continue;
    }

    ConstantInt *LenCI = dyn_cast<ConstantInt>(MT->getLength());
    if (!LenCI) {
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Skip: memcpy length is not a "
                        << "constant integer\n");
      continue;
    }

    uint64_t N = LenCI->getLimitedValue();
    if (N == 0 || N > MaxBytes) {
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Skip: memcpy size out of range "
                        << "(N=" << N << ", MaxBytes=" << MaxBytes << ")\n");
      continue;
    }

    unsigned DstAS = cast<PointerType>(Dst->getType())->getAddressSpace();
    unsigned SrcAS = cast<PointerType>(Src->getType())->getAddressSpace();
    if (DstAS != SrcAS) {
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Skip: address space mismatch "
                        << "(dstAS=" << DstAS << ", srcAS=" << SrcAS << ")\n");
      continue;
    }

    if (auto *Sel = dyn_cast<SelectInst>(Src)) {
      Changed |= transformSelectMemcpySource(*MT, *Sel, DL, &DT, &AC);
      continue;
    }
    if (auto *Sel = dyn_cast<SelectInst>(Dst)) {
      Changed |= transformSelectMemcpyDest(*MT, *Sel);
      continue;
    }

    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Skip: neither source nor "
                      << "destination is a select of pointers\n");
  }
  return Changed;
}

// =========================
// Patch: lane fast-path vectorization and robust HIP vector detection
// =========================

namespace {

// Debug helpers
static void dbgPrintOneLine(const Value *V) {
  V->print(dbgs()); dbgs() << '\n';
}

// Classify HIP vector struct names.
static bool isHIPVectorStructName(StructType *ST) {
  return ST && ST->hasName() &&
         (ST->getName().starts_with("struct.HIP_vector_type") ||
          ST->getName().starts_with("struct.HIP_vector_base"));
}

// Forward-declare helper used later to synthesize a vector type by unwrapping
// HIP wrappers or homogeneous packed scalar structs.
static FixedVectorType *
synthesizeVectorTypeFromHIPPayload(Type *Cur, const DataLayout &DL);

static bool isHomogeneousPackedScalarStruct(StructType *ST,
                                            const DataLayout &DL,
                                            Type *&ElemTy, unsigned &NumElts);

// Struct layout validation for homogeneous packed {T,T} or {T,T,T,T}.
static bool isPackedHIPVectorLikeStruct(StructType *ST, const DataLayout &DL,
                                        Type *&ElemTy, unsigned &NumElts) {
  if (!ST || ST->isOpaque())
    return false;
  NumElts = ST->getNumElements();
  if (!(NumElts == 2 || NumElts == 4))
    return false;
  auto *T0 = ST->getElementType(0);
  auto IsAllowed = [](Type *T) {
    return T->isIntegerTy(8) || T->isIntegerTy(16) || T->isIntegerTy(32) ||
           T->isIntegerTy(64) || T->isHalfTy() || T->isFloatTy() ||
           T->isDoubleTy();
  };
  if (!IsAllowed(T0))
    return false;
  for (unsigned i = 1; i < NumElts; ++i)
    if (ST->getElementType(i) != T0)
      return false;
  const StructLayout *SL = DL.getStructLayout(ST);
  uint64_t ESize = DL.getTypeStoreSize(T0);
  for (unsigned i = 0; i < NumElts; ++i)
    if (SL->getElementOffset(i) != i * ESize)
      return false;
  ElemTy = T0;
  return true;
}

// Chase up from the load pointer to find the vector-base pointer (e.g., %f),
// and collect base + constant offset to enable StructLayout mapping.
struct HIPLaneAccess {
  Value *VecBase = nullptr;   // Pointer to start of vector payload
  Value *LaneIdx = nullptr;   // Variable or constant lane index (scalar GEP)
  Value *RawBase = nullptr;   // Alloca/global/arg or GEP when variable-dim arrays
  uint64_t ConstByteOff = 0;  // Constant byte offset from RawBase to VecBase
};

// Helper to synthesize a vector type by unwrapping HIP wrappers or
// homogeneous packed scalar structs, and to follow one-element wrapper
// structs or array layers.
static FixedVectorType *
synthesizeVectorTypeFromHIPPayload(Type *Cur, const DataLayout &DL) {
  for (unsigned depth = 0; depth < 8 && Cur; ++depth) {
    if (auto *VT = dyn_cast<FixedVectorType>(Cur))
      return VT;
    if (auto *ST = dyn_cast<StructType>(Cur)) {
      Type *ElemTy = nullptr; unsigned N = 0;
      if (isHomogeneousPackedScalarStruct(ST, DL, ElemTy, N))
        return FixedVectorType::get(ElemTy, N);
      if (isHIPVectorStructName(ST) || ST->getNumElements() == 1) {
        Cur = ST->getNumElements() ? ST->getElementType(0) : nullptr;
        continue;
      }
      // Fallthrough: try to find a vector-like payload in any member.
      for (Type *Elt : ST->elements()) {
        if (auto *VT = dyn_cast<FixedVectorType>(Elt)) return VT;
        if (auto *EST = dyn_cast<StructType>(Elt)) {
          Type *E2 = nullptr; unsigned N2 = 0;
          if (isHomogeneousPackedScalarStruct(EST, DL, E2, N2))
            return FixedVectorType::get(E2, N2);
        }
      }
      return nullptr;
    }
    if (auto *AT = dyn_cast<ArrayType>(Cur)) {
      Cur = AT->getElementType();
      continue;
    }
    break;
  }
  return nullptr;
}

static Value *stripPtrCasts(Value *V) {
  while (auto *Op = dyn_cast<Operator>(V)) {
    if (isa<BitCastOperator>(Op) || isa<AddrSpaceCastOperator>(Op)) {
      V = Op->getOperand(0);
      continue;
    }
    break;
  }
  return V;
}

static bool chaseGEPChainToVectorBase(Value *Ptr, const DataLayout &DL,
                                      HIPLaneAccess &Out, std::string &Reason) {
  LLVM_DEBUG({
    dbgs() << "[AMDGPUVectorRecover][Lane] start chase from: ";
    dbgPrintOneLine(Ptr);
  });
  Value *Cur = Ptr;
  Value *FinalArrayIdx = nullptr;

  // If the final step is "gep <scalar>, %base, %idx", peel one index and keep %idx.
  if (auto *G = dyn_cast<GEPOperator>(Cur)) {
    Type *SrcTy = G->getSourceElementType();
    if ((SrcTy->isIntegerTy() || SrcTy->isFloatingPointTy()) &&
        G->getNumIndices() == 1) {
      FinalArrayIdx = G->getOperand(G->getNumOperands() - 1);
      Cur = stripPtrCasts(G->getPointerOperand());
      LLVM_DEBUG({
        dbgs() << "[AMDGPUVectorRecover][Lane] peeled scalar GEP index: ";
        dbgPrintOneLine(FinalArrayIdx);
        dbgs() << "  vector base candidate now: ";
        dbgPrintOneLine(Cur);
      });
    } else if (auto *ST = dyn_cast<StructType>(SrcTy)) {
      // New: If final step indexes a struct field and that field is scalar,
      // treat the field number as a lane index and peel back to the struct base.
      Value *IdxV = G->getOperand(G->getNumOperands() - 1);
      if (auto *CI = dyn_cast<ConstantInt>(IdxV)) {
        unsigned Field = CI->getZExtValue();
        if (!ST->isOpaque() && Field < ST->getNumElements()) {
          Type *FTy = ST->getElementType(Field);
          if (FTy->isIntegerTy() || FTy->isFloatingPointTy()) {
            FinalArrayIdx = CI;
            Cur = stripPtrCasts(G->getPointerOperand());
            LLVM_DEBUG({
              dbgs() << "[AMDGPUVectorRecover][Lane] peeled struct-field lane index: ";
              dbgPrintOneLine(FinalArrayIdx);
              dbgs() << "  vector base candidate now: ";
              dbgPrintOneLine(Cur);
            });
          }
        }
      }
    }
  }

  int64_t OffS = 0;
  Value *Base = GetPointerBaseWithConstantOffset(Cur, OffS, DL);
  if (!Base || OffS < 0) {
    Reason = "failed to recover base/const offset";
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorRecover][Lane] " << Reason << "\n");
    return false;
  }

  Out.VecBase = Cur;
  Out.LaneIdx = FinalArrayIdx;
  Out.RawBase = Base;
  Out.ConstByteOff = static_cast<uint64_t>(OffS);
  LLVM_DEBUG({
    dbgs() << "[AMDGPUVectorRecover][Lane] base: "; dbgPrintOneLine(Base);
    dbgs() << "  const-byte-off: " << Out.ConstByteOff << "\n";
    dbgs() << "  vec-base: "; dbgPrintOneLine(Out.VecBase);
  });
  return true;
}

// Return true if ST is a homogeneous, tightly packed scalar struct of 2 or 4 elements.
static bool isHomogeneousPackedScalarStruct(StructType *ST,
                                            const DataLayout &DL,
                                            Type *&ElemTy, unsigned &NumElts) {
  if (!ST || ST->isOpaque())
    return false;
  NumElts = ST->getNumElements();
  if (!(NumElts == 2 || NumElts == 4))
    return false;
  ElemTy = ST->getElementType(0);
  auto IsAllowed = [](Type *T) {
    return T->isIntegerTy(8) || T->isIntegerTy(16) || T->isIntegerTy(32) ||
           T->isIntegerTy(64) || T->isHalfTy() || T->isFloatTy() ||
           T->isDoubleTy();
  };
  if (!IsAllowed(ElemTy))
    return false;
  for (unsigned i = 1; i < NumElts; ++i)
    if (ST->getElementType(i) != ElemTy)
      return false;
  const StructLayout *SL = DL.getStructLayout(ST);
  uint64_t ESize = DL.getTypeStoreSize(ElemTy);
  for (unsigned i = 0; i < NumElts; ++i)
    if (SL->getElementOffset(i) != i * ESize)
      return false;
  return true;
}


// Classify a GEP as pointing to the start of a HIP vector struct instance.
struct HIPVecAtStart {
  bool AtStart = false;
  StructType *VecST = nullptr;
};

static HIPVecAtStart classifyAtHIPVectorStart(const GEPOperator *GEP) {
  HIPVecAtStart R;
  if (!GEP) return R;

  Type *Ty = GEP->getSourceElementType();
  for (auto TI = gep_type_begin(GEP), TE = gep_type_end(GEP); TI != TE; ++TI) {
    if (auto *AT = dyn_cast<ArrayType>(Ty)) {
      Ty = AT->getElementType();
      continue;
    }
    if (auto *ST = dyn_cast<StructType>(Ty)) {
      if (isHIPVectorStructName(ST)) {
        R.AtStart = true;
        R.VecST = ST;
        return R;
      }
      Value *IdxV = TI.getOperand();
      auto *CI = dyn_cast<ConstantInt>(IdxV);
      if (!CI) return R;
      unsigned Field = CI->getZExtValue();
      if (ST->isOpaque() || Field >= ST->getNumElements()) return R;
      Ty = ST->getElementType(Field);
      continue;
    }
    break;
  }
  // New: After consuming all indices, also check the final element type Ty.
  // This enables recognizing arrays-of-struct where the last index lands
  // directly on a HIP vector struct without a subsequent struct-field step.
  if (auto *ST = dyn_cast<StructType>(Ty)) {
    if (isHIPVectorStructName(ST)) {
      R.AtStart = true;
      R.VecST = ST;
      return R;
    }
  }
  return R;
}

// Map RawBase+ConstByteOff to a field that carries a vector payload and
// derive the vector type and a conservative alignment.
static bool isHIPVectorFieldAtOffset(Value *RawBase, uint64_t Off,
                                     const DataLayout &DL,
                                     FixedVectorType *&VecTy, Align &VecAlign) {
  StructType *OwnerST = nullptr;
  Align BaseAlign(1);

  if (auto *AI = dyn_cast<AllocaInst>(RawBase)) {
    OwnerST = dyn_cast<StructType>(AI->getAllocatedType());
    BaseAlign = AI->getAlign();
  } else if (auto *GV = dyn_cast<GlobalVariable>(RawBase)) {
    OwnerST = dyn_cast<StructType>(GV->getValueType());
    if (GV->getAlign())
      BaseAlign = *GV->getAlign();
  } else {
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorRecover][Lane] base is not alloca/global\n");
    return false;
  }
  if (!OwnerST || OwnerST->isOpaque())
    return false;

  const StructLayout *SL = DL.getStructLayout(OwnerST);
  if (Off >= SL->getSizeInBytes())
    return false;

  unsigned Field = SL->getElementContainingOffset(Off);
  if (SL->getElementOffset(Field) != Off) {
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorRecover][Lane] not at field boundary\n");
    return false;
  }

  Type *FieldTy = OwnerST->getElementType(Field);
  Type *Cur = FieldTy;
  FixedVectorType *FoundVT = nullptr;

  // Try several layers of wrappers (HIP_vector_type -> HIP_vector_base -> union/struct 

  for (unsigned depth = 0; depth < 5 && Cur; ++depth) {
    if (auto *VT = dyn_cast<FixedVectorType>(Cur)) {
      FoundVT = VT;
      break;
    }
    if (auto *ST = dyn_cast<StructType>(Cur)) {
      // Accept homogeneous packed scalar structs as vector-like.
      Type *ElemTy = nullptr; unsigned N = 0;
      if (isHomogeneousPackedScalarStruct(ST, DL, ElemTy, N)) {
        FoundVT = FixedVectorType::get(ElemTy, N);
        break;
      }
      if (ST->getNumElements() == 1) {
        Cur = ST->getElementType(0);
        continue;
      }
      bool Advanced = false;
      if (isHIPVectorStructName(ST)) {
        Cur = ST->getNumElements() ? ST->getElementType(0) : nullptr;
        Advanced = true;
      } else {
        for (Type *Elt : ST->elements()) {
          if (auto *VT = dyn_cast<FixedVectorType>(Elt)) {
            FoundVT = VT; Advanced = true; break;
          }
          if (auto *EST = dyn_cast<StructType>(Elt)) {
            Type *E2 = nullptr; unsigned N2 = 0;
            if (isHomogeneousPackedScalarStruct(EST, DL, E2, N2)) {
              FoundVT = FixedVectorType::get(E2, N2); Advanced = true; break;
            }
          }
        }
      }
      if (FoundVT) break;
      if (!Advanced) break;
    } else {
      break;
    }
  }

  if (!FoundVT)
    return false;

  VecTy = FoundVT;
  VecAlign = BaseAlign;
  LLVM_DEBUG({
    dbgs() << "[AMDGPUVectorRecover][Lane] HIP vector payload at offset "
           << Off << " vec-ty="; VecTy->print(dbgs()); dbgs() << " base-align="
           << VecAlign.value() << "\n";
  });
  return true;
}

// Try to vectorize a scalar lane load: replace "gep scalar + load" with
// "vector load + extractelement (var idx)" when the pointer refers to a HIP vector field.
static bool tryVectorizeHIPVectorLaneLoad(LoadInst *LI, const DataLayout &DL) {
  if (!LI || LI->isVolatile() || LI->isAtomic())
    return false;
  auto *GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
  if (!GEP || !GEP->isInBounds())
    return false;

  LLVM_DEBUG({
    dbgs() << "[AMDGPUVectorRecover][Lane] Consider scalar lane load:\n  ";
    dbgPrintOneLine(LI);
    dbgs() << "  ptr: "; dbgPrintOneLine(GEP);
  });

  HIPLaneAccess A;
  std::string Reason;
  if (!chaseGEPChainToVectorBase(GEP, DL, A, Reason))
    return false;

  FixedVectorType *VecTy = nullptr;
  Align VecAlign(1);
  if (!isHIPVectorFieldAtOffset(A.RawBase, A.ConstByteOff, DL, VecTy, VecAlign)) {
    // Fallback: if RawBase is not alloca/global (e.g., GEP with variable indices),
    // check if VecBase itself points to the start of a HIP vector struct instance.
    auto *VecBaseAsGEP = dyn_cast<GEPOperator>(A.VecBase);
    HIPVecAtStart C = classifyAtHIPVectorStart(VecBaseAsGEP);
    if (!C.AtStart) {
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorRecover][Lane] Not at HIP vector start; bail\n");
      return false;
    }
    // NEW: Synthesize vector type from nested payload (vector or packed scalar struct).
    VecTy = synthesizeVectorTypeFromHIPPayload(C.VecST, DL);
    if (!VecTy) {
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorRecover][Lane] Could not synthesize vector type\n");
      return false;
    }
    // Alignment: conservatively pick min(original scalar load align, underlying-object align if available).
    Align UnderAlign(1);
    if (Value *UO = getUnderlyingObject(A.VecBase)) {
      if (auto *GV = dyn_cast<GlobalVariable>(UO)) {
        if (GV->getAlign()) UnderAlign = *GV->getAlign();
      } else if (auto *AI = dyn_cast<AllocaInst>(UO)) {
        UnderAlign = AI->getAlign();
      }
    }
    VecAlign = UnderAlign;
    LLVM_DEBUG({
      dbgs() << "[AMDGPUVectorRecover][Lane] At HIP vector start via GEP; vec-ty=";
      VecTy->print(dbgs()); dbgs() << " align=" << VecAlign.value() << "\n";
    });
  } else {
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorRecover][Lane] Recognized HIP vector via base+offset\n");
  }

  // The scalar load result type must match the vector element type. If it does
  // not, we cannot safely RAUW the scalar load with the extracted element.
  Type *LaneTy = LI->getType();
  if (LaneTy != VecTy->getElementType()) {
    LLVM_DEBUG({
      dbgs() << "[AMDGPUVectorRecover][Lane] Bail: lane type mismatch. "
             << "load-ty="; LaneTy->print(dbgs());
      dbgs() << " vec-elt-ty="; VecTy->getElementType()->print(dbgs()); dbgs() << "\n";
    });
    return false;
  }
  // Emit vector load + extractelement(%lane)
  IRBuilder<> B(LI);
  unsigned AS = cast<PointerType>(A.VecBase->getType())->getAddressSpace();
  Function *Fn = LI->getFunction();
  Type *VecPtrTy = PointerType::get(VecTy, AS);
  Value *VecPtr = B.CreateBitCast(A.VecBase, VecPtrTy);

  Align UseAlign = LI->getAlign();
  if (VecAlign < UseAlign) UseAlign = VecAlign;

  LoadInst *VLoad = B.CreateAlignedLoad(VecTy, VecPtr, UseAlign);
  VLoad->copyMetadata(*LI);

  Value *LaneIdx = A.LaneIdx ? A.LaneIdx : B.getInt32(0);
  // extractelement requires an i32 index. Normalize any integer index to i32.
  if (LaneIdx->getType()->isIntegerTy() && LaneIdx->getType() != B.getInt32Ty()) {
    unsigned BW = LaneIdx->getType()->getIntegerBitWidth();
    if (BW > 32)
      LaneIdx = B.CreateTrunc(LaneIdx, B.getInt32Ty());
    else if (BW < 32)
      LaneIdx = B.CreateZExt(LaneIdx, B.getInt32Ty());
  }
  // If somehow non-integer, bail defensively.
  if (!LaneIdx->getType()->isIntegerTy(32)) {
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorRecover][Lane] Bail: non-i32 lane index\n");
    return false;
  }
  Value *NewScalar = B.CreateExtractElement(VLoad, LaneIdx);

  LLVM_DEBUG({
    dbgs() << "[AMDGPUVectorRecover][Lane] Vectorized lane load via: ";
    VLoad->print(dbgs()); dbgs() << "\n  extract idx: "; LaneIdx->print(dbgs()); dbgs() << "\n";
  });

  // Update debug uses (dbg.value) that reference the old load before erasing it.
  // DominatorTree is not required for simple direct replacement; pass nullptr.
  replaceDbgUsesWithUndef(LI);

  LI->replaceAllUsesWith(NewScalar);
  LI->eraseFromParent();
  if (GEP->use_empty())
    GEP->eraseFromParent();
  return true;
}

} // end anonymous namespace

// =========================
// New: Scalarized Vector Recovery (class defined at file scope)
// =========================

namespace {
static cl::opt<bool> AMDGPUVectorRecoverEnable(
    "amdgpu-vector-recover-enable",
    cl::desc("Enable recovery of vector loads/stores from scalarized struct access"),
    cl::init(true));

// Small helpers for concise debug printing.
static void printOneLine(const Value *V) {
  V->print(dbgs());
  dbgs() << '\n';
}

static void dumpPtrForms(StringRef Label, Value *Ptr) {
  dbgs() << "    " << Label << ": "; printOneLine(Ptr);
  Value *Strip = Ptr->stripPointerCasts();
  dbgs() << "      - stripCasts: "; printOneLine(Strip);
  dbgs() << "      - underlying: "; printOneLine(getUnderlyingObject(Ptr));
}

// Dump a struct type definition, including its layout and recursively dumping
// any inner struct-typed elements. Intended for LLVM_DEBUG use.
static void dumpStructDetails(StructType *ST, const DataLayout &DL,
                              StringRef Prefix = "struct") {
  if (!ST)
    return;
  LLVM_DEBUG({
    dbgs() << "[AMDGPUVectorRecover] " << Prefix << " type: ";
    if (ST->hasName())
      dbgs() << ST->getName() << " = ";
    ST->print(dbgs());
    dbgs() << "\n";
    if (ST->isOpaque())
      return;
    const StructLayout *SL = DL.getStructLayout(ST);
    dbgs() << "  - isPacked=" << (ST->isPacked() ? "true" : "false")
           << " size=" << SL->getSizeInBytes()
           << " numElems=" << ST->getNumElements() << "\n";
    for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i) {
      Type *EltTy = ST->getElementType(i);
      dbgs() << "    - elem[" << i << "] off=" << SL->getElementOffset(i)
             << " ty="; EltTy->print(dbgs()); dbgs() << "\n";
      if (auto *InnerST = dyn_cast<StructType>(EltTy))
        dumpStructDetails(InnerST, DL, "inner-struct");
    }
  });
}

class AMDGPUVectorRecover {
  // Optional concise printer for instructions.
  static void printOneLine(const Instruction *I) {
    I->print(dbgs()); dbgs() << '\n';
  }
  // Returns true if the struct name suggests a HIP vector wrapper and we should
  // be aggressive about forming vector loads/stores.
  static bool isHIPVectorStructName(StructType *ST) {
    if (!ST || !ST->hasName())
      return false;
    StringRef N = ST->getName();
    return N.contains("HIP_vector") || N.contains("__half2") ||
           N.contains("half2") || N.contains("float2") || N.contains("float4") ||
           N.contains("double2") || N.contains("double4") ||
           N.contains("int2") || N.contains("int4") ||
           N.contains("uint2") || N.contains("uint4") ||
           N.contains("short2") || N.contains("short4") ||
           N.contains("ushort2") || N.contains("ushort4") ||
           N.contains("char2") || N.contains("char4") ||
           N.contains("uchar2") || N.contains("uchar4");
  }
  // Check small, homogeneous, tightly packed struct matching {T,T} or {T,T,T,T},
  // with T in HIP-supported vector element set: i8, i16, i32, i64, f16, f32, f64.
  static bool isPackedHIPVectorLikeStruct(StructType *ST, const DataLayout &DL,
                                          Type *&ElemTy, unsigned &NumElts) {
    if (!ST || ST->isOpaque())
      return false;
    NumElts = ST->getNumElements();
    if (!(NumElts == 2 || NumElts == 4))
      return false;
    ElemTy = ST->getElementType(0);
    auto IsAllowedElem = [](Type *T) {
      return T->isIntegerTy(8) || T->isIntegerTy(16) || T->isIntegerTy(32) ||
             T->isIntegerTy(64) || T->isHalfTy() || T->isFloatTy() ||
             T->isDoubleTy();
    };
    if (!IsAllowedElem(ElemTy))
      return false;
    for (unsigned i = 1; i < NumElts; ++i)
      if (ST->getElementType(i) != ElemTy)
        return false;
    const StructLayout *SL = DL.getStructLayout(ST);
    uint64_t ESize = DL.getTypeStoreSize(ElemTy);
    for (unsigned i = 0; i < NumElts; ++i)
      if (SL->getElementOffset(i) != i * ESize)
        return false;
    return true;
  }
  // Match a pointer to a struct field by recovering base object and constant byte offset.
  // Populates:
  //  - BasePtr: canonical base object (e.g., alloca/global) for clustering
  //  - ST:      struct type allocated/held by the base object
  //  - FieldIdx: index of the field within ST corresponding to the constant byte offset
  // Returns false and fills Reason on failure.
static bool matchStructFieldPtr(Value *Ptr, const DataLayout &DL,
                                Value *&BasePtr, StructType *&ST,
                                unsigned &FieldIdx, std::string &Reason) {
  LLVM_DEBUG({
    dbgs() << "[AMDGPUVectorRecover][matchStructFieldPtr] ptr=";
    Ptr->print(dbgs()); dbgs() << '\n';
  });

  int64_t OffS = 0;
  Value *BaseObj = GetPointerBaseWithConstantOffset(Ptr, OffS, DL);
  if (!BaseObj) {
    Reason = "no base object (null returned)";
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorRecover][matchStructFieldPtr] " << Reason << "\n");
    return false;
  }
    LLVM_DEBUG(dumpStructDetails(ST, DL, "base-struct"));
  if (OffS < 0) {
    Reason = (Twine("negative offset not supported: ") + Twine(OffS)).str();
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorRecover][matchStructFieldPtr] " << Reason << "\n");
    return false;
  }

  BasePtr = BaseObj;
  const uint64_t OffU = static_cast<uint64_t>(OffS);

  // Try to derive a struct type from a known struct holder.
  ST = nullptr;
  if (auto *AI = dyn_cast<AllocaInst>(BasePtr)) {
    ST = dyn_cast<StructType>(AI->getAllocatedType());
  } else if (auto *GV = dyn_cast<GlobalVariable>(BasePtr)) {
    ST = dyn_cast<StructType>(GV->getValueType());
  } else {
    // Not a recognized struct holder (e.g., implicitarg.ptr, malloc, arg, etc.)
    Reason = "base is not a recognizable struct holder (alloca/global)";
    LLVM_DEBUG({
      dbgs() << "[AMDGPUVectorRecover][matchStructFieldPtr] " << Reason << "\n";
      dbgs() << "  base="; BasePtr->print(dbgs());
      dbgs() << " offset=" << OffU << "\n";
    });
    return false;
  }

  if (!ST) {
    Reason = "base is not a struct type";
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorRecover][matchStructFieldPtr] " << Reason << "\n");
    return false;
  }
  if (ST->isOpaque()) {
    Reason = "struct type is opaque";
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorRecover][matchStructFieldPtr] " << Reason << "\n");
    return false;
  }

  // Optional debug dump, now safe.
  LLVM_DEBUG({
    dbgs() << "[AMDGPUVectorRecover] base-struct type: ";
    if (ST->hasName()) dbgs() << ST->getName() << " = ";
    ST->print(dbgs()); dbgs() << "\n";
  });

  // Check homogeneous packed {T,T} or {T,T,T,T}.
  Type *ElemTy = nullptr;
  unsigned NumElts = 0;
    StructType *UseST = ST;

    // Allow nested single-element wrapper structs used by HIP vector types.
    // Unwrap layers like { %inner } if %inner is a homogeneous packed struct.
    // Dump each candidate type for debugging.
    while (!isPackedHIPVectorLikeStruct(UseST, DL, ElemTy, NumElts)) {
      if (UseST->isOpaque() || UseST->getNumElements() != 1)
        break;
      Type *OnlyEltTy = UseST->getElementType(0);
      auto *InnerST = dyn_cast<StructType>(OnlyEltTy);
      if (!InnerST)
        break;
      LLVM_DEBUG(dumpStructDetails(InnerST, DL, "candidate-inner-struct"));
      // Try inner struct as the vector-like struct.
      UseST = InnerST;
    }

    if (!isPackedHIPVectorLikeStruct(UseST, DL, ElemTy, NumElts)) {
      Reason = "struct is not homogeneous packed {T,T}/{T,T,T,T}";
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorRecover][matchStructFieldPtr] " << Reason << "\n");
      return false;
    }

    // Map offset to a field index using the chosen struct (UseST). This assumes
    // any unwrapped inner struct starts at offset 0 within the wrapper.
    const StructLayout *SL = DL.getStructLayout(UseST);
    uint64_t OffBytes = OffU;
    for (unsigned i = 0; i < NumElts; ++i) {
      if (SL->getElementOffset(i) == OffBytes) {
        FieldIdx = i;
        LLVM_DEBUG(dbgs() << "[AMDGPUVectorRecover][matchStructFieldPtr] success: field index=" << FieldIdx << "\n");
        // Return the struct we actually matched against.
        ST = UseST;
        return true;
      }
    }

  Reason = (Twine("offset ") + Twine(OffU) + " does not match any field boundary").str();
  LLVM_DEBUG(dbgs() << "[AMDGPUVectorRecover][matchStructFieldPtr] " << Reason << "\n");
  return false;
}
  struct LaneCluster {
    SmallVector<Instruction *, 4> Ops; // LoadInst* or StoreInst*
    Value *BasePtr = nullptr;
    StructType *ST = nullptr;
    Type *ElemTy = nullptr;
    unsigned NumElts = 0;
    Align CommonAlign = Align(1);
    bool Aggressive = false;
  };
  // Returns true if a full load cluster was formed; otherwise sets Reason.
  static bool collectFullLoadCluster(LoadInst *Seed, const DataLayout &DL,
                                     LaneCluster &Out, std::string &Reason) {
    if (!Seed)
      return Reason = "null load seed", false;
    if (Seed->isVolatile())
      return Reason = "volatile load", false;
    if (Seed->isAtomic())
      return Reason = "atomic load", false;
    StructType *ST = nullptr;
    unsigned Field = 0;
    Value *Base = nullptr;
    if (!matchStructFieldPtr(Seed->getPointerOperand(), DL, Base, ST, Field, Reason))
      return false;
    Type *ElemTy = nullptr;
    unsigned N = 0;
    if (!isPackedHIPVectorLikeStruct(ST, DL, ElemTy, N))
      return Reason = "struct not packed homogeneous {T,T}/{T,T,T,T}", false;
    if (Field >= N)
      return Reason = "GEP field index out of range", false;

    SmallVector<LoadInst *, 4> Lanes(N, nullptr);
    Lanes[Field] = Seed;
    BasicBlock *BB = Seed->getParent();
    for (Instruction &I : *BB) {
      if (&I == Seed)
        continue;
      auto *Ld = dyn_cast<LoadInst>(&I);
      if (!Ld || Ld->isVolatile() || Ld->isAtomic())
        continue;
      StructType *STi = nullptr;
      unsigned Fi = 0;
      Value *BaseI = nullptr;
      std::string MReason;
      if (!matchStructFieldPtr(Ld->getPointerOperand(), DL, BaseI, STi, Fi, MReason))
        continue;
      if (STi != ST)
        continue;
      // Compare bases (base objects) to tolerate different casts/indexing.
      if (BaseI != Base)
        continue;
      if (Fi >= N)
        continue;
      if (Lanes[Fi] && Lanes[Fi] != Ld)
        continue;
      Lanes[Fi] = Ld;
    }
    for (unsigned i = 0; i < N; ++i) {
      if (!Lanes[i]) {
        Reason = (Twine("missing lane ") + Twine(i) + " in block").str();
        return false;
      }
    }

    Align A = Seed->getAlign();
    for (unsigned i = 0; i < N; ++i)
      A = std::min(A, Lanes[i]->getAlign());

    Out.Ops.assign(Lanes.begin(), Lanes.end());
    Out.BasePtr = Base;
    Out.ST = ST;
    Out.NumElts = N;
    Out.CommonAlign = A;
    Out.Aggressive = isHIPVectorStructName(ST);
    (void)ElemTy; // Will be recomputed from lane types below.
    // Choose vector element type from lane load types to avoid RAUW type mismatches.
    // Require all lane load types to match and have the same store size as the
    // struct field element type.
    Type *StructElemTy = nullptr;
    unsigned NumEltsTmp = 0;
    (void)isPackedHIPVectorLikeStruct(ST, DL, StructElemTy, NumEltsTmp); // pre-checked
    Type *LaneTy0 = cast<LoadInst>(Out.Ops[0])->getType();
    uint64_t LaneStoreSize = DL.getTypeStoreSize(LaneTy0);
    uint64_t StructElemStoreSize = DL.getTypeStoreSize(StructElemTy);
    if (LaneStoreSize != StructElemStoreSize) {
      Reason = "lane load type size mismatch with struct element size";
      return false;
    }
    for (unsigned i = 1; i < N; ++i) {
      Type *LiTy = cast<LoadInst>(Out.Ops[i])->getType();
      if (LiTy != LaneTy0) {
        Reason = "heterogeneous lane load types";
        return false;
      }
      if (DL.getTypeStoreSize(LiTy) != LaneStoreSize) {
        Reason = "inconsistent lane load type sizes";
        return false;
      }
    }
    Out.ElemTy = LaneTy0;
    LLVM_DEBUG({
      dbgs() << "[AMDGPUVectorRecover] Full load cluster formed: lanes=" << N
             << " align=" << A.value() << " elem="; Out.ElemTy->print(dbgs());
      dbgs() << "\n  base: "; Out.BasePtr->print(dbgs()); dbgs() << '\n';
      for (unsigned i = 0; i < N; ++i) {
        dbgs() << "  lane[" << i << "]: "; printOneLine(Out.Ops[i]);
      }
    });
    return true;
  }
  // Returns true if a full store cluster was formed; otherwise sets Reason.
  static bool collectFullStoreCluster(StoreInst *Seed, const DataLayout &DL,
                                      LaneCluster &Out, std::string &Reason) {
    if (!Seed)
      return Reason = "null store seed", false;
    if (Seed->isVolatile())
      return Reason = "volatile store", false;
    if (Seed->isAtomic())
      return Reason = "atomic store", false;
    StructType *ST = nullptr;
    unsigned Field = 0;
    Value *Base = nullptr;
    if (!matchStructFieldPtr(Seed->getPointerOperand(), DL, Base, ST, Field, Reason))
      return false;
    Type *ElemTy = nullptr;
    unsigned N = 0;
    if (!isPackedHIPVectorLikeStruct(ST, DL, ElemTy, N))
      return false;
    if (Field >= N)
      return false;

    SmallVector<StoreInst *, 4> Lanes(N, nullptr);
    Lanes[Field] = Seed;
    BasicBlock *BB = Seed->getParent();
    for (Instruction &I : *BB) {
      if (&I == Seed)
        continue;
      auto *St = dyn_cast<StoreInst>(&I);
      if (!St || St->isVolatile() || St->isAtomic())
        continue;
      StructType *STi = nullptr;
      unsigned Fi = 0;
      Value *BaseI = nullptr;
      std::string MReason;
      if (!matchStructFieldPtr(St->getPointerOperand(), DL, BaseI, STi, Fi, MReason))
        continue;
      if (STi != ST)
        continue;
      // Compare bases (base objects) to tolerate different casts/indexing.
      if (BaseI != Base)
        continue;
      if (Fi >= N)
        continue;
      if (Lanes[Fi] && Lanes[Fi] != St)
        continue;
      Lanes[Fi] = St;
    }
    for (unsigned i = 0; i < N; ++i) {
      if (!Lanes[i]) {
        Reason = (Twine("missing lane ") + Twine(i) + " in block").str();
        return false;
      }
    }

    Align A = Seed->getAlign();
    for (unsigned i = 0; i < N; ++i)
      A = std::min(A, Lanes[i]->getAlign());

    Out.Ops.assign(Lanes.begin(), Lanes.end());
    Out.BasePtr = Base;
    Out.ST = ST;
    Out.NumElts = N;
    Out.CommonAlign = A;
    Out.Aggressive = isHIPVectorStructName(ST);
    (void)ElemTy; // Will be recomputed from lane value types below.
    // Choose vector element type from lane store value types to avoid RAUW/type issues.
    // Require all lane value types to match and have the same store size as the
    // struct field element type.
    Type *StructElemTy = nullptr;
    unsigned NumEltsTmp = 0;
    (void)isPackedHIPVectorLikeStruct(ST, DL, StructElemTy, NumEltsTmp); // pre-checked
    Type *ValTy0 = cast<StoreInst>(Out.Ops[0])->getValueOperand()->getType();
    uint64_t LaneStoreSize = DL.getTypeStoreSize(ValTy0);
    uint64_t StructElemStoreSize = DL.getTypeStoreSize(StructElemTy);
    if (LaneStoreSize != StructElemStoreSize) {
      Reason = "lane store value type size mismatch with struct element size";
      return false;
    }
    for (unsigned i = 1; i < N; ++i) {
      auto *Si = cast<StoreInst>(Out.Ops[i]);
      Type *ViTy = Si->getValueOperand()->getType();
      if (ViTy != ValTy0) {
        Reason = "heterogeneous lane store value types";
        return false;
      }
      if (DL.getTypeStoreSize(ViTy) != LaneStoreSize) {
        Reason = "inconsistent lane store value type sizes";
        return false;
      }
    }
    Out.ElemTy = ValTy0;
    LLVM_DEBUG({
      dbgs() << "[AMDGPUVectorRecover] Full store cluster formed: lanes=" << N
             << " align=" << A.value() << " elem="; ElemTy->print(dbgs());
      dbgs() << "\n  base: "; Out.BasePtr->print(dbgs()); dbgs() << '\n';
      for (unsigned i = 0; i < N; ++i) {
        dbgs() << "  lane[" << i << "]: ";
        if (Out.Ops[i]) printOneLine(Out.Ops[i]); else dbgs() << "(null)\n";
      }
    });
    return true;
  }
  // Opaque-pointer-friendly helper to get a pointer to <N x T> with same AS.
  static Value *bitcastToVectorPointer(IRBuilder<> &B, Value *BasePtr, FixedVectorType *VecTy) {
    unsigned AS = cast<PointerType>(BasePtr->getType())->getAddressSpace();
    Type *VecPtrTy = PointerType::get(VecTy, AS);
    return B.CreateBitCast(BasePtr, VecPtrTy);
  }
  static bool vectorizeLoadCluster(LaneCluster &C) {
    auto *Ld0 = cast<LoadInst>(C.Ops[0]);
    IRBuilder<> B(Ld0);
    auto *VecTy = FixedVectorType::get(C.ElemTy, C.NumElts);
    LLVM_DEBUG({
      dbgs() << "[AMDGPUVectorRecover] Vectorizing load cluster: <"
             << C.NumElts << " x "; C.ElemTy->print(dbgs()); dbgs() << ">, align="
             << C.CommonAlign.value() << "\n  base: ";
      C.BasePtr->print(dbgs()); dbgs() << '\n';
    });
    Value *VecPtr = bitcastToVectorPointer(B, C.BasePtr, VecTy);
    LLVM_DEBUG({
      dbgs() << "[AMDGPUVectorRecover]   - inserted cast: ";
      VecPtr->print(dbgs()); dbgs() << '\n';
    });
    LoadInst *VLoad = B.CreateAlignedLoad(VecTy, VecPtr, C.CommonAlign);
    VLoad->copyMetadata(*Ld0);
    LLVM_DEBUG({
      dbgs() << "[AMDGPUVectorRecover]   - replaced with: ";
      VLoad->print(dbgs()); dbgs() << '\n';
    });
    for (unsigned i = 0; i < C.NumElts; ++i) {
      auto *Li = cast<LoadInst>(C.Ops[i]);
      Value *Lane = B.CreateExtractElement(VLoad, B.getInt32(i));
      // Update dbg.value users that referenced the old lane load.
      replaceDbgUsesWithUndef(Li);
      Li->replaceAllUsesWith(Lane);
    }
    for (Instruction *I : C.Ops)
      I->eraseFromParent();
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorRecover] Load cluster vectorized successfully\n");
    return true;
  }
  static bool vectorizeStoreCluster(LaneCluster &C) {
    auto *St0 = cast<StoreInst>(C.Ops[0]);
    IRBuilder<> B(St0);
    auto *VecTy = FixedVectorType::get(C.ElemTy, C.NumElts);
    LLVM_DEBUG({
      dbgs() << "[AMDGPUVectorRecover] Vectorizing store cluster: <"
             << C.NumElts << " x "; C.ElemTy->print(dbgs()); dbgs() << ">, align="
             << C.CommonAlign.value() << "\n  base: ";
      C.BasePtr->print(dbgs()); dbgs() << '\n';
    });
    Value *VecVal = PoisonValue::get(VecTy);
    for (unsigned i = 0; i < C.NumElts; ++i) {
      auto *Si = cast<StoreInst>(C.Ops[i]);
      VecVal = B.CreateInsertElement(VecVal, Si->getValueOperand(), B.getInt32(i));
    }
    Value *VecPtr = bitcastToVectorPointer(B, C.BasePtr, VecTy);
    LLVM_DEBUG({
      dbgs() << "[AMDGPUVectorRecover]   - inserted cast: ";
      VecPtr->print(dbgs()); dbgs() << '\n';
    });
    StoreInst *VStore = B.CreateAlignedStore(VecVal, VecPtr, C.CommonAlign);
    VStore->copyMetadata(*St0);
    LLVM_DEBUG({
      dbgs() << "[AMDGPUVectorRecover]   - replaced with: ";
      VStore->print(dbgs()); dbgs() << '\n';
    });
    // Stores rarely appear as dbg.value "values", but be defensive and drop any
    // dbg.uses that may reference them before erasing.
    for (Instruction *I : C.Ops) {
      // There is no natural replacement value for a store; if any dbg.uses exist,
      // they will be dropped by replacing with poison (no DT needed).
      replaceDbgUsesWithUndef(I);
      I->eraseFromParent();
    }

    LLVM_DEBUG(dbgs() << "[AMDGPUVectorRecover] Store cluster vectorized successfully\n");
    return true;
  }

public:
  static bool run(Function &F, const DataLayout &DL) {
    if (!AMDGPUVectorRecoverEnable)
      return false;
    bool Changed = false;
    for (BasicBlock &BB : F) {
      SmallVector<Instruction *, 32> Insts;
      for (Instruction &I : BB)
        Insts.push_back(&I);
      for (Instruction *I : Insts) {
        if (!I->getParent())
          continue;
        if (auto *Ld = dyn_cast<LoadInst>(I)) {
          // Skip if already vector-typed
          if (Ld->getType()->isVectorTy()) {
            LLVM_DEBUG(dbgs() << "[AMDGPUVectorRecover] Skip vector load\n");
            continue;
          }
          // Fast-path: scalar lane load from HIP vector via cast+GEP with variable index.
          // Example:
          //   %f = gep i8, %xx, 16            ; start of HIP vector field
          //   %p = gep float, %f, %i          ; lane i
          //   %v = load float, %p
          //
          // Rewrite to:
          //   %v4 = load <N x T>, (bitcast %f)
          //   %v  = extractelement %v4, %i
          LLVM_DEBUG({
            dbgs() << "[AMDGPUVectorRecover] Consider load for vectorization in @"
                   << I->getFunction()->getName() << ":\n";
            dbgs() << "  inst: "; printOneLine(Ld);
            dbgs() << "  pointer operands:\n";
            dumpPtrForms("load.ptr", Ld->getPointerOperand());
          });
          if (tryVectorizeHIPVectorLaneLoad(Ld, DL)) {
            LLVM_DEBUG(dbgs() << "[AMDGPUVectorRecover] Lane-load vectorized (fast-path)\n");
            Changed = true;
            continue;
          }




          // 2) Fall back: full-lane cluster recovery (existing logic)

          LaneCluster C;
          std::string Reason;
          if (collectFullLoadCluster(Ld, DL, C, Reason)) {
            Changed |= vectorizeLoadCluster(C);
          } else {
            LLVM_DEBUG({
              dbgs() << "[AMDGPUVectorRecover] Skip load: " << Reason << "\n";
            });
          }
        } else if (auto *St = dyn_cast<StoreInst>(I)) {
          LLVM_DEBUG({
            dbgs() << "[AMDGPUVectorRecover] Consider store for vectorization in @"
                   << I->getFunction()->getName() << ":\n";
            dbgs() << "  inst: "; printOneLine(St);
            dbgs() << "  pointer operands:\n";
            dumpPtrForms("store.ptr", St->getPointerOperand());
          });
          LaneCluster C;
          std::string Reason;
          if (collectFullStoreCluster(St, DL, C, Reason)) {
            Changed |= vectorizeStoreCluster(C);
          } else {
            LLVM_DEBUG({
              dbgs() << "[AMDGPUVectorRecover] Skip store: " << Reason << "\n";
            });
          }
        }
      }
    }
    return Changed;
  }
};
} // end anonymous namespace

// Thin C-style wrapper referenced by the pass driver.
namespace llvm {
bool AMDGPUVectorRecoverRun(Function &F, const DataLayout &DL) {
  return AMDGPUVectorRecover::run(F, DL);
}
} // namespace llvm