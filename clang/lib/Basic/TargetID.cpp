//===--- TargetID.cpp - Utilities for parsing target ID -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/TargetID.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/TargetParser.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

namespace clang {

static const llvm::SmallVector<llvm::StringRef, 4>
getAllPossibleAMDGPUTargetIDFeatures(const llvm::Triple &T,
                                     llvm::StringRef Proc) {
  // Entries in returned vector should be in alphabetical order.
  llvm::SmallVector<llvm::StringRef, 4> Ret;
  auto ProcKind = T.isAMDGCN() ? llvm::AMDGPU::parseArchAMDGCN(Proc)
                               : llvm::AMDGPU::parseArchR600(Proc);
  if (ProcKind == llvm::AMDGPU::GK_NONE)
    return Ret;
  auto Features = T.isAMDGCN() ? llvm::AMDGPU::getArchAttrAMDGCN(ProcKind)
                               : llvm::AMDGPU::getArchAttrR600(ProcKind);
  if (Features & llvm::AMDGPU::FEATURE_SRAM_ECC)
    Ret.push_back("sram-ecc");
  if (Features & llvm::AMDGPU::FEATURE_XNACK)
    Ret.push_back("xnack");
  return Ret;
}

const llvm::SmallVector<llvm::StringRef, 4>
getAllPossibleTargetIDFeatures(const llvm::Triple &T,
                               llvm::StringRef Processor) {
  llvm::SmallVector<llvm::StringRef, 4> Ret;
  if (T.isAMDGPU())
    return getAllPossibleAMDGPUTargetIDFeatures(T, Processor);
  return Ret;
}

/// Returns canonical processor name or empty string if \p Processor is invalid.
static llvm::StringRef getCanonicalProcessorName(const llvm::Triple &T,
                                                 llvm::StringRef Processor) {
  if (T.isAMDGPU())
    return llvm::AMDGPU::getCanonicalArchName(T, Processor);
  return Processor;
}

llvm::StringRef parseTargetID(const llvm::Triple &T,
                              llvm::StringRef OffloadArch,
                              llvm::StringMap<bool> *FeatureMap,
                              bool *IsValid) {
  llvm::StringRef ArchStr;
  auto SetValid = [&](bool Valid) {
    if (IsValid)
      *IsValid = Valid;
    return ArchStr;
  };

  auto Split = OffloadArch.split(':');
  ArchStr = getCanonicalProcessorName(T, Split.first);
  if (ArchStr.empty())
    return SetValid(false);
  if (!FeatureMap && !IsValid)
    return ArchStr;

  llvm::SmallSet<llvm::StringRef, 2> AllFeatures;
  for (auto F : getAllPossibleTargetIDFeatures(T, ArchStr))
    AllFeatures.insert(F);

  auto Features = Split.second;
  if (Features.empty())
    return SetValid(true);

  llvm::StringMap<bool> LocalFeatureMap;
  if (!FeatureMap)
    FeatureMap = &LocalFeatureMap;

  while (!Features.empty()) {
    auto Splits = Features.split(':');
    auto Sign = Splits.first.back();
    auto Feature = Splits.first.drop_back();
    if (Sign != '+' && Sign != '-')
      return SetValid(false);
    bool IsOn = Sign == '+';
    if (AllFeatures.count(Feature)) {
      auto Loc = FeatureMap->find(Feature);
      // Each feature can only show up at most once in target ID.
      if (Loc != FeatureMap->end())
        return SetValid(false);
      (*FeatureMap)[Feature] = IsOn;
    } else
      return SetValid(false);
    Features = Splits.second;
  }
  return SetValid(true);
};

std::string getCanonicalTargetID(llvm::StringRef Processor,
                                 const llvm::StringMap<bool> &Features) {
  std::string TargetID = Processor.str();
  std::map<const llvm::StringRef, bool> OrderedMap;
  for (const auto &F : Features)
    OrderedMap[F.first()] = F.second;
  for (auto F : OrderedMap)
    TargetID = TargetID + ':' + F.first.str() + (F.second ? "+" : "-");
  return TargetID;
}

/// Parse canonical target ID, assuming it is valid.
static llvm::StringRef
parseCanonicalTargetIDWithoutCheck(llvm::StringRef OffloadArch,
                                   llvm::StringMap<bool> *FeatureMap) {
  llvm::StringRef ArchStr;
  auto Split = OffloadArch.split(':');
  ArchStr = Split.first;
  assert(!ArchStr.empty());
  if (!FeatureMap)
    return ArchStr;

  auto Features = Split.second;
  if (Features.empty())
    return ArchStr;

  while (!Features.empty()) {
    auto Splits = Features.split(':');
    auto Sign = Splits.first.back();
    auto Feature = Splits.first.drop_back();
    assert(Sign == '+' || Sign == '-');
    bool IsOn = Sign == '+';
    auto Loc = FeatureMap->find(Feature);
    // Each feature can only show up at most once in target ID.
    assert(Loc == FeatureMap->end());
    (*FeatureMap)[Feature] = IsOn;
    Features = Splits.second;
  }
  return ArchStr;
};

// For a specific processor, a feature either shows up in all target IDs, or
// does not show up in any target IDs. Otherwise the target ID combination
// is invalid.
bool isValidTargetIDCombination(
    const std::set<llvm::StringRef> &TargetIDs,
    llvm::SmallVector<llvm::StringRef, 2> *ConflictingTIDs) {
  struct Info {
    llvm::StringRef TargetID;
    llvm::StringMap<bool> Features;
  };
  llvm::StringMap<Info> FeatureMap;
  for (auto &ID : TargetIDs) {
    llvm::StringMap<bool> Features;
    llvm::StringRef Proc = parseCanonicalTargetIDWithoutCheck(ID, &Features);
    auto Loc = FeatureMap.find(Proc);
    if (Loc == FeatureMap.end())
      FeatureMap[Proc] = Info{ID, Features};
    else {
      auto ExistingFeatures = Loc->second.Features;
      for (auto &F : Features) {
        if (ExistingFeatures.find(F.first()) == ExistingFeatures.end()) {
          if (ConflictingTIDs) {
            ConflictingTIDs->push_back(Loc->second.TargetID);
            ConflictingTIDs->push_back(ID);
          }
          return false;
        }
      }
    }
  }
  return true;
}

} // namespace clang
