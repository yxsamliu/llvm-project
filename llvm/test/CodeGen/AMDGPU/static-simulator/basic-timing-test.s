--- |
  ; ModuleID = '../llvm/test/CodeGen/AMDGPU/static-simulator/basic-timing-test.mir'
  source_filename = "../llvm/test/CodeGen/AMDGPU/static-simulator/basic-timing-test.mir"
  target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
  target triple = "amdgcn"
  
  define void @vopd_single_cycle_occupancy() #0 {
  entry:
    unreachable
  }
  
  define void @trans_half_rate_occupancy() #0 {
  entry:
    unreachable
  }
  
  define void @valu_single_cycle_occupancy() #0 {
  entry:
    unreachable
  }
  
  define void @spill_reload_test() #0 {
  entry:
    unreachable
  }
  
  define void @bank_conflict_test() #0 {
  entry:
    unreachable
  }
  
  define void @tuple_bank_conflict_test() #0 {
  entry:
    unreachable
  }
  
  define void @salu_bank_conflict_test() #0 {
  entry:
    unreachable
  }
  
  define void @wmma_no_bank_conflict_test() #0 {
  entry:
    unreachable
  }
  
  define void @source_cache_test() #0 {
  entry:
    unreachable
  }
  
  define void @tuple_cache_test() #0 {
  entry:
    unreachable
  }
  
  define void @cache_eviction_test() #0 {
  entry:
    unreachable
  }
  
  define void @pk8_valu_occupancy_test() #0 {
  entry:
    unreachable
  }
  
  define void @pk16_valu_occupancy_test() #0 {
  entry:
    unreachable
  }
  
  define void @trans_holds_valu_in_wmma_window() #0 {
  entry:
    unreachable
  }
  
  define void @trans_holds_trans_in_wmma_window() #0 {
  entry:
    unreachable
  }
  
  define void @trans_no_hold_outside_wmma() #0 {
  entry:
    unreachable
  }
  
  define void @regbank_in_wmma_window_no_stall() #0 {
  entry:
    unreachable
  }
  
  define void @regbank_outside_wmma_window_does_stall() #0 {
  entry:
    unreachable
  }
  
  define void @regbank_wmma_summary_check() #0 {
  entry:
    unreachable
  }
  
  define void @longlatvalu_waits_for_wmma_window() #0 {
  entry:
    unreachable
  }
  
  define void @longlatvalu_holds_valu_after_wmma() #0 {
  entry:
    unreachable
  }
  
  define void @wmma_cache_hit_on_reuse() #0 {
  entry:
    unreachable
  }
  
  define void @wmma_dest_invalidated() #0 {
  entry:
    unreachable
  }
  
  define void @wmma_then_valu_cache_hit() #0 {
  entry:
    unreachable
  }
  
  attributes #0 = { "target-cpu"="gfx1250" }
...
---
name:            vopd_single_cycle_occupancy
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $sgpr0, $vgpr0, $vgpr1, $vgpr2, $vgpr3
  
    $vgpr10, $vgpr11 = V_DUAL_MUL_F32_e32_X_MUL_F32_e32_gfx1250 $sgpr0, $vgpr0, $sgpr0, $vgpr1, implicit $mode, implicit $exec, implicit $mode, implicit $exec, implicit $mode, implicit $exec
    $vgpr12, $vgpr13 = V_DUAL_MUL_F32_e32_X_MUL_F32_e32_gfx1250 $sgpr0, $vgpr2, $sgpr0, $vgpr3, implicit $mode, implicit $exec, implicit $mode, implicit $exec, implicit $mode, implicit $exec
    $vgpr14 = V_MOV_B32_e32 42, implicit $exec
    S_ENDPGM 0, implicit $vgpr10, implicit $vgpr11, implicit $vgpr12, implicit $vgpr13, implicit $vgpr14
...
---
name:            trans_half_rate_occupancy
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0, $vgpr1, $vgpr2
  
    $vgpr10 = V_EXP_F32_e32 $vgpr0, implicit $exec, implicit $mode
    $vgpr11 = V_EXP_F32_e32 $vgpr1, implicit $exec, implicit $mode
    $vgpr12 = V_EXP_F32_e32 $vgpr2, implicit $exec, implicit $mode
    S_ENDPGM 0, implicit $vgpr10, implicit $vgpr11, implicit $vgpr12
...
---
name:            valu_single_cycle_occupancy
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    $vgpr10 = V_MOV_B32_e32 1, implicit $exec
    $vgpr11 = V_MOV_B32_e32 2, implicit $exec
    $vgpr12 = V_MOV_B32_e32 3, implicit $exec
    $vgpr13 = V_MOV_B32_e32 4, implicit $exec
    S_ENDPGM 0, implicit $vgpr10, implicit $vgpr11, implicit $vgpr12, implicit $vgpr13
...
---
name:            spill_reload_test
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0, $vgpr1, $vgpr2, $sgpr0, $sgpr1, $sgpr2
  
    SCRATCH_STORE_DWORD_SADDR $vgpr0, $sgpr0, 0, 0, implicit $exec, implicit $flat_scr
    SCRATCH_STORE_DWORD_SADDR $vgpr1, $sgpr0, 4, 0, implicit $exec, implicit $flat_scr
    $vgpr10 = SCRATCH_LOAD_DWORD_SADDR $sgpr0, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr11 = SCRATCH_LOAD_DWORD_SADDR $sgpr0, 4, 0, implicit $exec, implicit $flat_scr
    $vgpr2 = V_WRITELANE_B32 $sgpr1, 0, $vgpr2, implicit $exec
    $vgpr2 = V_WRITELANE_B32 $sgpr2, 1, $vgpr2, implicit $exec
    $sgpr10 = V_READLANE_B32 $vgpr2, 0, implicit $exec
    $sgpr11 = V_READLANE_B32 $vgpr2, 1, implicit $exec
    S_ENDPGM 0, implicit $vgpr10, implicit $vgpr11, implicit $sgpr10, implicit $sgpr11
...
---
name:            bank_conflict_test
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0, $vgpr1, $vgpr8, $vgpr16
  
    $vgpr20 = V_ADD_F32_e32 $vgpr0, $vgpr8, implicit $exec, implicit $mode
    $vgpr21 = V_FMA_F32_e64 0, $vgpr0, 0, $vgpr8, 0, $vgpr16, 0, 0, implicit $exec, implicit $mode
    $vgpr22 = V_ADD_F32_e32 $vgpr0, $vgpr1, implicit $exec, implicit $mode
    S_ENDPGM 0, implicit $vgpr20, implicit $vgpr21, implicit $vgpr22
...
---
name:            tuple_bank_conflict_test
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $vgpr8_vgpr9
  
    $vgpr20_vgpr21 = V_PK_ADD_F32 8, $vgpr0_vgpr1, 8, $vgpr8_vgpr9, 0, 0, 0, 0, 0, implicit $mode, implicit $exec
    S_ENDPGM 0, implicit $vgpr20_vgpr21
...
---
name:            salu_bank_conflict_test
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $sgpr0, $sgpr1, $sgpr4, $sgpr8
  
    $sgpr20 = S_ADD_I32 $sgpr0, $sgpr4, implicit-def $scc
    $sgpr21 = S_ADD_I32 $sgpr0, $sgpr1, implicit-def $scc
    S_ENDPGM 0, implicit $sgpr20, implicit $sgpr21
...
---
name:            wmma_no_bank_conflict_test
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23
  
    early-clobber $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23 = V_WMMA_F32_16X16X32_BF16_w32_twoaddr 8, $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, 8, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, 8, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, 0, 0, 0, 0, implicit $exec
    S_ENDPGM 0, implicit $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23
...
---
name:            source_cache_test
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0, $vgpr1, $vgpr8
  
    $vgpr20 = V_ADD_F32_e32 $vgpr0, $vgpr8, implicit $exec, implicit $mode
    $vgpr21 = V_ADD_F32_e32 $vgpr0, $vgpr8, implicit $exec, implicit $mode
    $vgpr22 = V_ADD_F32_e32 $vgpr0, $vgpr1, implicit $exec, implicit $mode
    $vgpr0 = V_MOV_B32_e32 42, implicit $exec
    $vgpr23 = V_ADD_F32_e32 $vgpr0, $vgpr8, implicit $exec, implicit $mode
    S_ENDPGM 0, implicit $vgpr20, implicit $vgpr21, implicit $vgpr22, implicit $vgpr23
...
---
name:            tuple_cache_test
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $vgpr8_vgpr9
  
    $vgpr20_vgpr21 = V_PK_ADD_F32 8, $vgpr0_vgpr1, 8, $vgpr8_vgpr9, 0, 0, 0, 0, 0, implicit $mode, implicit $exec
    $vgpr22_vgpr23 = V_PK_ADD_F32 8, $vgpr0_vgpr1, 8, $vgpr8_vgpr9, 0, 0, 0, 0, 0, implicit $mode, implicit $exec
    $vgpr0_vgpr1 = V_PK_ADD_F32 8, $vgpr20_vgpr21, 8, $vgpr22_vgpr23, 0, 0, 0, 0, 0, implicit $mode, implicit $exec
    $vgpr24_vgpr25 = V_PK_ADD_F32 8, $vgpr0_vgpr1, 8, $vgpr8_vgpr9, 0, 0, 0, 0, 0, implicit $mode, implicit $exec
    S_ENDPGM 0, implicit $vgpr24_vgpr25
...
---
name:            cache_eviction_test
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0, $vgpr8, $vgpr16, $vgpr24, $vgpr32, $vgpr1
  
    $vgpr40 = V_ADD_F32_e32 $vgpr0, $vgpr1, implicit $exec, implicit $mode
    $vgpr41 = V_ADD_F32_e32 $vgpr8, $vgpr1, implicit $exec, implicit $mode
    $vgpr42 = V_ADD_F32_e32 $vgpr16, $vgpr1, implicit $exec, implicit $mode
    $vgpr43 = V_ADD_F32_e32 $vgpr24, $vgpr1, implicit $exec, implicit $mode
    $vgpr44 = V_ADD_F32_e32 $vgpr32, $vgpr1, implicit $exec, implicit $mode
    $vgpr45 = V_ADD_F32_e32 $vgpr0, $vgpr1, implicit $exec, implicit $mode
    $vgpr46 = V_ADD_F32_e32 $vgpr8, $vgpr1, implicit $exec, implicit $mode
    S_ENDPGM 0, implicit $vgpr46
...
---
name:            pk8_valu_occupancy_test
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $vgpr2
  
    early-clobber $vgpr8_vgpr9_vgpr10_vgpr11 = V_CVT_SCALE_PK8_F16_FP8_e64 $vgpr0_vgpr1, $vgpr2, 0, implicit $mode, implicit $exec
    $vgpr20 = V_MOV_B32_e32 1, implicit $exec
    $vgpr21 = V_MOV_B32_e32 2, implicit $exec
    $vgpr22 = V_MOV_B32_e32 3, implicit $exec
    $vgpr23 = V_MOV_B32_e32 4, implicit $exec
    S_ENDPGM 0, implicit $vgpr23
...
---
name:            pk16_valu_occupancy_test
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1_vgpr2, $vgpr3
  
    early-clobber $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15 = V_CVT_SCALE_PK16_F16_FP6_e64 $vgpr0_vgpr1_vgpr2, $vgpr3, 0, implicit $mode, implicit $exec
    $vgpr20 = V_MOV_B32_e32 1, implicit $exec
    $vgpr21 = V_MOV_B32_e32 2, implicit $exec
    S_ENDPGM 0, implicit $vgpr21
...
---
name:            trans_holds_valu_in_wmma_window
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, $vgpr24
  
    early-clobber $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23 = V_WMMA_F32_16X16X32_BF16_w32_twoaddr 8, $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, 8, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, 8, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, 0, 0, 0, 0, implicit $exec
    $sgpr10 = S_MOV_B32 1
    $vgpr30 = V_EXP_F32_e32 $vgpr24, implicit $exec, implicit $mode
    $vgpr31 = V_MOV_B32_e32 42, implicit $exec
    S_ENDPGM 0, implicit $vgpr31
...
---
name:            trans_holds_trans_in_wmma_window
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, $vgpr24, $vgpr25
  
    early-clobber $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23 = V_WMMA_F32_16X16X32_BF16_w32_twoaddr 8, $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, 8, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, 8, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, 0, 0, 0, 0, implicit $exec
    $sgpr10 = S_MOV_B32 1
    $vgpr30 = V_EXP_F32_e32 $vgpr24, implicit $exec, implicit $mode
    $vgpr31 = V_EXP_F32_e32 $vgpr25, implicit $exec, implicit $mode
    S_ENDPGM 0, implicit $vgpr31
...
---
name:            trans_no_hold_outside_wmma
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0, $vgpr1
  
    $vgpr10 = V_EXP_F32_e32 $vgpr0, implicit $exec, implicit $mode
    $vgpr11 = V_MOV_B32_e32 42, implicit $exec
    $vgpr12 = V_ADD_F32_e32 $vgpr0, $vgpr1, implicit $exec, implicit $mode
    S_ENDPGM 0, implicit $vgpr12
...
---
name:            regbank_in_wmma_window_no_stall
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, $vgpr0, $vgpr8
  
    early-clobber $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23 = V_WMMA_F32_16X16X32_BF16_w32_twoaddr 8, $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, 8, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, 8, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, 0, 0, 0, 0, implicit $exec
    $vgpr40 = V_ADD_F32_e32 $vgpr0, $vgpr8, implicit $exec, implicit $mode
    S_ENDPGM 0, implicit $vgpr40
...
---
name:            regbank_outside_wmma_window_does_stall
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0, $vgpr8
  
    $vgpr40 = V_ADD_F32_e32 $vgpr0, $vgpr8, implicit $exec, implicit $mode
    S_ENDPGM 0, implicit $vgpr40
...
---
name:            regbank_wmma_summary_check
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, $vgpr0, $vgpr8, $vgpr16
  
    early-clobber $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23 = V_WMMA_F32_16X16X32_BF16_w32_twoaddr 8, $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, 8, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, 8, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, 0, 0, 0, 0, implicit $exec
    $vgpr40 = V_ADD_F32_e32 $vgpr0, $vgpr8, implicit $exec, implicit $mode
    $vgpr41 = V_FMA_F32_e64 0, $vgpr0, 0, $vgpr8, 0, $vgpr16, 0, 0, implicit $exec, implicit $mode
    S_ENDPGM 0, implicit $vgpr41
...
---
name:            longlatvalu_waits_for_wmma_window
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, $vgpr24_vgpr25, $vgpr26, $vgpr27
  
    early-clobber $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23 = V_WMMA_F32_16X16X32_BF16_w32_twoaddr 8, $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, 8, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, 8, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, 0, 0, 0, 0, implicit $exec
    $vgpr30 = V_EXP_F32_e32 $vgpr27, implicit $exec, implicit $mode
    early-clobber $vgpr32_vgpr33_vgpr34_vgpr35 = V_CVT_SCALE_PK8_F16_FP8_e64 $vgpr24_vgpr25, $vgpr26, 0, implicit $mode, implicit $exec
    S_ENDPGM 0, implicit $vgpr35
...
---
name:            longlatvalu_holds_valu_after_wmma
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, $vgpr24_vgpr25, $vgpr26, $vgpr27
  
    early-clobber $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23 = V_WMMA_F32_16X16X32_BF16_w32_twoaddr 8, $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, 8, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, 8, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, 0, 0, 0, 0, implicit $exec
    $sgpr10 = S_MOV_B32 1
    $sgpr11 = S_MOV_B32 2
    $sgpr12 = S_MOV_B32 3
    $sgpr13 = S_MOV_B32 4
    $sgpr14 = S_MOV_B32 5
    $sgpr15 = S_MOV_B32 6
    $sgpr16 = S_MOV_B32 7
    $sgpr17 = S_MOV_B32 8
    early-clobber $vgpr32_vgpr33_vgpr34_vgpr35 = V_CVT_SCALE_PK8_F16_FP8_e64 $vgpr24_vgpr25, $vgpr26, 0, implicit $mode, implicit $exec
    $vgpr40 = V_EXP_F32_e32 $vgpr27, implicit $exec, implicit $mode
    S_ENDPGM 0, implicit $vgpr40
...
---
name:            wmma_cache_hit_on_reuse
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, $vgpr24_vgpr25_vgpr26_vgpr27_vgpr28_vgpr29_vgpr30_vgpr31
  
    early-clobber $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23 = V_WMMA_F32_16X16X32_BF16_w32_twoaddr 8, $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, 8, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, 8, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, 0, 0, 0, 0, implicit $exec
    early-clobber $vgpr24_vgpr25_vgpr26_vgpr27_vgpr28_vgpr29_vgpr30_vgpr31 = V_WMMA_F32_16X16X32_BF16_w32_twoaddr 8, $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, 8, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, 8, $vgpr24_vgpr25_vgpr26_vgpr27_vgpr28_vgpr29_vgpr30_vgpr31, 0, 0, 0, 0, implicit $exec
    S_ENDPGM 0, implicit $vgpr31
...
---
name:            wmma_dest_invalidated
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23
  
    early-clobber $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23 = V_WMMA_F32_16X16X32_BF16_w32_twoaddr 8, $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, 8, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, 8, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, 0, 0, 0, 0, implicit $exec
    $vgpr40 = V_ADD_F32_e32 $vgpr16, $vgpr17, implicit $exec, implicit $mode
    S_ENDPGM 0, implicit $vgpr40
...
---
name:            wmma_then_valu_cache_hit
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:       []
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  laneSharedVGPRSize: 0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, $vgpr1
  
    early-clobber $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23 = V_WMMA_F32_16X16X32_BF16_w32_twoaddr 8, $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, 8, $vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15, 8, $vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23, 0, 0, 0, 0, implicit $exec
    $vgpr40 = V_ADD_F32_e32 $vgpr0, $vgpr1, implicit $exec, implicit $mode
    S_ENDPGM 0, implicit $vgpr40
...
