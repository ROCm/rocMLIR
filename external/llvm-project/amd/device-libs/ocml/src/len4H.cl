/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

REQUIRES_16BIT_INSTS CONSTATTR half
MATH_MANGLE(len4)(half x, half y, half z, half w)
{
    float fx = (float)x;
    float fy = (float)y;
    float fz = (float)z;
    float fw = (float)w;

    float d2 = MATH_MAD(fx, fx, MATH_MAD(fy, fy, MATH_MAD(fz, fz, fw*fw)));

    half ret = (half)BUILTIN_SQRT_F32(d2);

    if (!FINITE_ONLY_OPT()) {
        ret = (BUILTIN_ISINF_F16(x) |
               BUILTIN_ISINF_F16(y) |
               BUILTIN_ISINF_F16(z) |
               BUILTIN_ISINF_F16(w)) ? PINF_F16 : ret;
    }

    return ret;
}

