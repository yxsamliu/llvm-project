//===-- AMDGPUFeatures.td - AMDGPU Feature Definitions -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def FeatureFP64 : SubtargetFeature<"fp64",
  "HasFP64",
  "true",
  "Enable double precision operations"
>;

def FeatureFMA : SubtargetFeature<"fmaf",
  "HasFMA",
  "true",
  "Enable single precision FMA (not as fast as mul+add, but fused)"
>;

// Starting with GFX13, memory is shared between LDS and the Vector Cache.
// The LDS size can be configured to multiple values based on the split ratio.
// As a result, different artificial limitations may apply.
class SubtargetFeatureLocalMemorySizeLimit <int Value> : SubtargetFeature<
  "local-memory-size-limit-"#Value,
  "LocalMemorySizeLimit",
  !cast<string>(Value),
  "Artificial limit on local memory (LDS), specified in bytes"
>;

// Limit max LDS for shaders to 64KiB.
def FeatureLocalMemorySizeLimit65536 : SubtargetFeatureLocalMemorySizeLimit<65536>;

// By default, shaders have the LDS size set to 128 KiB to
// maintain compatibility with older GFX architectures
def FeatureLocalMemorySizeLimit131072 : SubtargetFeatureLocalMemorySizeLimit<131072>;

// Addressable local memory size is the maximum number of bytes of LDS that can
// be allocated to a single workgroup.
class SubtargetFeatureAddressableLocalMemorySize <int Value> : SubtargetFeature<
  "addressablelocalmemorysize"#Value,
  "AddressableLocalMemorySize",
  !cast<string>(Value),
  "The size of local memory in bytes"
>;

def FeatureAddressableLocalMemorySize32768 : SubtargetFeatureAddressableLocalMemorySize<32768>;
def FeatureAddressableLocalMemorySize65536 : SubtargetFeatureAddressableLocalMemorySize<65536>;
def FeatureAddressableLocalMemorySize163840 : SubtargetFeatureAddressableLocalMemorySize<163840>;
def FeatureAddressableLocalMemorySize196608 : SubtargetFeatureAddressableLocalMemorySize<196608>;
def FeatureAddressableLocalMemorySize327680 : SubtargetFeatureAddressableLocalMemorySize<327680>;
def FeatureAddressableLocalMemorySize675840 : SubtargetFeatureAddressableLocalMemorySize<675840>;

class SubtargetFeatureWavefrontSize <int ValueLog2> : SubtargetFeature<
  "wavefrontsize"#!shl(1, ValueLog2),
  "WavefrontSizeLog2",
  !cast<string>(ValueLog2),
  "The number of threads per wavefront"
>;

def FeatureWavefrontSize16 : SubtargetFeatureWavefrontSize<4>;
def FeatureWavefrontSize32 : SubtargetFeatureWavefrontSize<5>;
def FeatureWavefrontSize64 : SubtargetFeatureWavefrontSize<6>;

class SubtargetFeatureGeneration <string Value, string FeatureName,
                                 string Subtarget,
                                  list<SubtargetFeature> Implies> :
        SubtargetFeature <FeatureName, "Gen", Subtarget#"::"#Value,
                          Value#" GPU generation", Implies>;

def FeaturePromoteAlloca : SubtargetFeature <"promote-alloca",
  "EnablePromoteAlloca",
  "true",
  "Enable promote alloca pass"
>;

