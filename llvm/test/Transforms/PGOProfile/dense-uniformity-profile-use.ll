; RUN: split-file %s %t
; RUN: opt -passes=pgo-instr-gen -pgo-instrument-entry -S %t/input.ll | FileCheck %s --check-prefix=GEN
; RUN: opt -passes=pgo-instr-gen -pgo-instrument-entry -S %t/input.ll -o %t/gen-off.ll
; RUN: opt -passes=pgo-instr-gen -pgo-instrument-entry -pgo-dense-uniformity-metadata -S %t/input.ll -o %t/gen-on.ll
; RUN: diff %t/gen-off.ll %t/gen-on.ll
; RUN: %python %t/raw.py > %t/full.raw
; RUN: llvm-profdata merge %t/full.raw -o %t/full.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/full.profdata -S %t/input.ll -o %t/default.ll
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/full.profdata -pgo-dense-uniformity-metadata=false -S %t/input.ll -o %t/off.ll
; RUN: diff %t/default.ll %t/off.ll
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/full.profdata -pgo-dense-uniformity-metadata -S %t/input.ll -o %t/on.ll
; RUN: FileCheck %s --check-prefix=FULL < %t/on.ll
; RUN: FileCheck %s --check-prefix=NONE < %t/off.ll
; RUN: %python %t/compare.py %t/on.ll %t/off.ll
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/full.profdata -pgo-dense-uniformity-metadata -pgo-uniformity-metadata=false -S %t/input.ll | FileCheck %s --check-prefix=COUNTS --implicit-check-not=uniformity.profile
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/full.profdata -pgo-dense-uniformity-metadata -pgo-wave-metadata=false -S %t/input.ll | FileCheck %s --check-prefix=FULL --implicit-check-not=wave.count
; RUN: sed 's/+wavefrontsize64/+wavefrontsize32/' %t/input.ll > %t/wave32.ll
; RUN: sed 's/"target-features"="+wavefrontsize64"//' %t/input.ll > %t/unknown.ll
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/full.profdata -pgo-dense-uniformity-metadata -S %t/unknown.ll | FileCheck %s --check-prefix=NONE
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/full.profdata -pgo-dense-uniformity-metadata -S %t/off.ll | FileCheck %s --check-prefix=NONE
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/full.profdata -pgo-dense-uniformity-metadata -mtriple=amdgpu9.42-amd-amdhsa -S %t/unknown.ll | FileCheck %s --check-prefix=FULL
; RUN: sed 's/"target-features"="+wavefrontsize64"/"target-cpu"="gfx942"/' %t/input.ll > %t/cpu.ll
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/full.profdata -pgo-dense-uniformity-metadata -S %t/cpu.ll | FileCheck %s --check-prefix=FULL
; RUN: sed 's/+wavefrontsize64/+wavefrontsize32,+wavefrontsize64/' %t/input.ll > %t/ambiguous.ll
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/full.profdata -pgo-dense-uniformity-metadata -S %t/ambiguous.ll | FileCheck %s --check-prefix=NONE
; RUN: %python %t/raw.py partial > %t/partial.raw
; RUN: llvm-profdata merge %t/partial.raw -o %t/partial.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/partial.profdata -pgo-dense-uniformity-metadata -S %t/input.ll | FileCheck %s --check-prefix=NONE
; RUN: %python %t/raw.py zero > %t/zero.raw
; RUN: llvm-profdata merge %t/zero.raw -o %t/zero.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/zero.profdata -pgo-dense-uniformity-metadata -S %t/input.ll | FileCheck %s --check-prefix=ZERO
; RUN: %python %t/raw.py overflow > %t/overflow.raw
; RUN: not llvm-profdata merge %t/overflow.raw -o %t/overflow.profdata 2>&1 | FileCheck %s --check-prefix=CORRUPT
; RUN: %python %t/raw.py wave32 > %t/wave32.raw
; RUN: llvm-profdata merge %t/wave32.raw -o %t/wave32.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/wave32.profdata -pgo-dense-uniformity-metadata -S %t/wave32.ll | FileCheck %s --check-prefix=FULL
; RUN: %python %t/raw.py sparse > %t/sparse.raw
; RUN: llvm-profdata merge %t/sparse.raw -o %t/sparse.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/sparse.profdata -pgo-dense-uniformity-metadata -S %t/input.ll | FileCheck %s --check-prefix=NONE
; RUN: %python %t/raw.py short > %t/short.raw
; RUN: llvm-profdata merge %t/short.raw -o %t/short.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/short.profdata -pgo-dense-uniformity-metadata -S %t/input.ll | FileCheck %s --check-prefix=NONE
; RUN: %python %t/raw.py missing > %t/missing.raw
; RUN: llvm-profdata merge %t/missing.raw -o %t/missing.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/missing.profdata -pgo-dense-uniformity-metadata -S %t/input.ll | FileCheck %s --check-prefix=NONE

; The real generator fixes the prefix and suffix identities used by raw.py.
; GEN: call void @llvm.instrprof.increment({{.*}}i64 942389667449461396, i32 5, i32 0)
; GEN: a:
; GEN: call void @llvm.instrprof.increment.step({{.*}}i32 5, i32 3, i64 0)
; GEN: b:
; GEN: call void @llvm.instrprof.increment({{.*}}i32 5, i32 1)
; GEN: exit:
; GEN: call void @llvm.instrprof.increment.step({{.*}}i32 5, i32 4, i64 0)
;
; Previously unmeasured a and exit are full-wave only in the exact case.
; The zero-step counters have zero lane/uniform counts in every fixture; their
; positive 0/0 uniformity bits cannot be used to classify the blocks.
; FULL: a:
; FULL: br label %exit, !block.uniformity.profile
; FULL: exit:
; FULL: ret void, !block.uniformity.profile
; NONE: a:
; NONE: br label %exit{{(, !wave.profile.block ![0-9]+)?$}}
; NONE: exit:
; NONE: ret void{{(, !wave.profile.block ![0-9]+)?$}}
; ZERO: a:
; ZERO: br label %exit{{(, !wave.profile.block ![0-9]+)?$}}
; ZERO: exit:
; ZERO: ret void, !block.uniformity.profile
; CORRUPT: excessively large counter value
; COUNTS: !{!"function_entry_count", i64 6400}

;--- compare.py
import re
import sys
from pathlib import Path

def strip_blocks(path):
    return re.sub(r", !block\.uniformity\.profile ![0-9]+", "", Path(path).read_text())
# The same profile and all other IR/metadata, including branch hints, wave
# counts, select weights, function attributes and profile summary, are retained.
assert strip_blocks(sys.argv[1]) == strip_blocks(sys.argv[2])

;--- input.ll
source_filename = "dense-uniformity-profile-use.ll"
target triple = "amdgcn-amd-amdhsa"
define void @diamond(i1 %cond, i1 %select_cond, ptr %p) "target-features"="+wavefrontsize64" {
entry:
  br i1 %cond, label %a, label %b
a:
  store volatile i32 1, ptr %p
  br label %exit
b:
  store volatile i32 2, ptr %p
  br label %exit
exit:
  %value = select i1 %select_cond, i32 1, i32 2
  store volatile i32 %value, ptr %p
  ret void
}


;--- raw.py
import hashlib
import struct
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "uniform"
name = b"diamond"
name_ref = int.from_bytes(hashlib.md5(name).digest()[:8], "little")
# Raw v12, IR entry-first layout; hash checked against pgo-instr-gen above.
func_hash = 942389667449461396
lanes = [6400, 3200, 1600, 0, 0]
waves = [100, 50, 100, 50, 100]
if mode == "partial":
    waves = [110, 55, 110, 55, 110]
if mode == "zero":
    lanes[1] = 6400
    waves[1] = 100
    waves[3] = 0
if mode == "overflow":
    waves[3:] = [(1 << 58) + 50, (1 << 58) + 100]
if mode == "wave32":
    waves = [w * 2 for w in waves]
if mode == "short":
    waves.pop()
if mode == "missing":
    waves = []
version = 12 | (1 << 56) | (1 << 58) | (1 << 54)
if mode == "sparse":
    lanes = lanes[:3]
    waves = waves[:3]
    version &= ~(1 << 54)
uniform = [0] * len(lanes) if mode == "partial" else lanes[:]
num_counters = len(lanes) + len(waves)
counter_delta = 80
uniform_delta = counter_delta + num_counters * 8
names_delta = uniform_delta + len(uniform) * 8
names = bytes([len(name), 0]) + name
header = [0xff6c70726f667281, version, 0, 1, 0, num_counters, 0,
          0, 0, len(uniform), 0, uniform_delta, len(names), counter_delta,
          uniform_delta, names_delta, 0, 0, 2]
record = struct.pack("<7QI4HII4x", name_ref, func_hash, counter_delta,
                     uniform_delta, 0, 0, 0, num_counters, 0, 0, 0, 64, 0,
                     len(waves))
counts = lanes + waves + uniform
sys.stdout.buffer.write(struct.pack("<19Q", *header) + record +
                        struct.pack("<" + "Q" * len(counts), *counts) +
                        names + bytes((-len(names)) % 8))
