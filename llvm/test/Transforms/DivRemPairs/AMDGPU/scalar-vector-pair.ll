; RUN: opt -S -passes='div-rem-pairs,verify' %s | FileCheck %s
; REQUIRES: amdgpu-registered-target

target triple = "amdgcn-amd-amdhsa"

; Vectorization of the remainder must not hide a reusable scalar quotient.
; CHECK-LABEL: define <2 x i64> @pair(
; CHECK: %q = udiv i64 %x, %d
; CHECK: [[M:%.*]] = mul i64 %q, %d
; CHECK: %r.scalar.decomposed = sub i64 %x, [[M]]
; CHECK: urem i64
; CHECK-NOT: urem <2 x i64>
; CHECK: ret <2 x i64>
define <2 x i64> @pair(i64 noundef %x, <2 x i32> noundef %y) {
  %y0 = extractelement <2 x i32> %y, i32 0
  %d = zext i32 %y0 to i64
  %q = udiv i64 %x, %d
  %xs0 = insertelement <2 x i64> poison, i64 %x, i32 0
  %xs = insertelement <2 x i64> %xs0, i64 %q, i32 1
  %ys = zext <2 x i32> %y to <2 x i64>
  %r = urem <2 x i64> %xs, %ys
  ret <2 x i64> %r
}

; Reuse also applies to signed pairs. The exact flag cannot poison the remainder.
; CHECK-LABEL: define <2 x i32> @signed_exact(
; CHECK: %q = sdiv i32 %x, %y
; CHECK: [[M:%.*]] = mul i32 %q, %y
; CHECK: %r.scalar.decomposed = sub i32 %x, [[M]]
; CHECK-NOT: srem <2 x i32>
; CHECK: ret <2 x i32>
define <2 x i32> @signed_exact(i32 noundef %x, i32 noundef %y) {
  %q = sdiv exact i32 %x, %y
  %xs0 = insertelement <2 x i32> poison, i32 %x, i32 0
  %xs = insertelement <2 x i32> %xs0, i32 %q, i32 1
  %ys0 = insertelement <2 x i32> poison, i32 %y, i32 0
  %ys = insertelement <2 x i32> %ys0, i32 7, i32 1
  %r = srem <2 x i32> %xs, %ys
  ret <2 x i32> %r
}

; The existing undef handling must still correlate the quotient and product.
; CHECK-LABEL: define <2 x i32> @freeze_pair(
; CHECK: %x.frozen = freeze i32 %x
; CHECK: %y.frozen = freeze i32 %y
; CHECK: %q = udiv i32 %x.frozen, %y.frozen
; CHECK: [[M:%.*]] = mul i32 %q, %y.frozen
; CHECK: %r.scalar.decomposed = sub i32 %x.frozen, [[M]]
define <2 x i32> @freeze_pair(i32 %x, i32 %y) {
  %q = udiv i32 %x, %y
  %xs0 = insertelement <2 x i32> poison, i32 %x, i32 0
  %xs = insertelement <2 x i32> %xs0, i32 %q, i32 1
  %ys0 = insertelement <2 x i32> poison, i32 %y, i32 0
  %ys = insertelement <2 x i32> %ys0, i32 7, i32 1
  %r = urem <2 x i32> %xs, %ys
  ret <2 x i32> %r
}

; A scalar cast with a stronger poison contract is not an equivalent lane.
; CHECK-LABEL: define <2 x i64> @poison_cast(
; CHECK: %r = urem <2 x i64> %xs, %ys
; CHECK-NOT: scalar.decomposed
define <2 x i64> @poison_cast(i64 noundef %x, <2 x i32> noundef %y) {
  %y0 = extractelement <2 x i32> %y, i32 0
  %d = zext nneg i32 %y0 to i64
  %q = udiv i64 %x, %d
  %xs0 = insertelement <2 x i64> poison, i64 %x, i32 0
  %xs = insertelement <2 x i64> %xs0, i64 %q, i32 1
  %ys = zext <2 x i32> %y to <2 x i64>
  %r = urem <2 x i64> %xs, %ys
  ret <2 x i64> %r
}

; CHECK-LABEL: define <2 x i64> @different_divisor(
; CHECK: %r = urem <2 x i64> %xs, %ys
; CHECK-NOT: scalar.decomposed
define <2 x i64> @different_divisor(i64 noundef %x, <2 x i32> noundef %y) {
  %y1 = extractelement <2 x i32> %y, i32 1
  %d = zext i32 %y1 to i64
  %q = udiv i64 %x, %d
  %xs0 = insertelement <2 x i64> poison, i64 %x, i32 0
  %xs = insertelement <2 x i64> %xs0, i64 %q, i32 1
  %ys = zext <2 x i32> %y to <2 x i64>
  %r = urem <2 x i64> %xs, %ys
  ret <2 x i64> %r
}

; Do not move a division before a vector remainder that originally preceded it.
; CHECK-LABEL: define <2 x i32> @late_division(
; CHECK: %r = urem <2 x i32> %xs, %ys
; CHECK: %q = udiv i32 %x, %y
; CHECK-NOT: scalar.decomposed
define <2 x i32> @late_division(i32 noundef %x, i32 noundef %y) {
  %xs0 = insertelement <2 x i32> poison, i32 %x, i32 0
  %xs = insertelement <2 x i32> %xs0, i32 9, i32 1
  %ys0 = insertelement <2 x i32> poison, i32 %y, i32 0
  %ys = insertelement <2 x i32> %ys0, i32 7, i32 1
  %r = urem <2 x i32> %xs, %ys
  %q = udiv i32 %x, %y
  %ret = insertelement <2 x i32> %r, i32 %q, i32 1
  ret <2 x i32> %ret
}
