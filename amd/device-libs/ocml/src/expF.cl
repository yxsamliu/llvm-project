/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(exp)(float x) {
    float r = BUILTIN_EXP_F32(x);
    if (DAZ_OPT() || UNSAFE_MATH_OPT())
        return r;

    r = x < -0x1.66d3e8p+5f ? 0.0f : r;
    r = x > 0x1.344136p+5f ? PINF_F32 : r;
    return r;
}
