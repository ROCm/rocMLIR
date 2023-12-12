/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR half
MATH_MANGLE(rlen3)(half x, half y, half z)
{
    float fx = (float)x;
    float fy = (float)y;
    float fz = (float)z;

    float d2 = BUILTIN_MAD_F32(fx, fx, BUILTIN_MAD_F32(fy, fy, fz*fz));

    half ret = (half)BUILTIN_AMDGPU_RSQRT_F32(d2);

    if (!FINITE_ONLY_OPT()) {
        ret = (BUILTIN_ISINF_F16(x) |
               BUILTIN_ISINF_F16(y) |
               BUILTIN_ISINF_F16(z)) ? 0.0h : ret;
    }

    return ret;
}

