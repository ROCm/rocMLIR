/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(rsqrt)(float x)
{
    if (DAZ_OPT()) {
        return BUILTIN_AMDGPU_RSQRT_F32(x);
    } else {
        bool s = x < 0x1.0p-100f;
        return BUILTIN_AMDGPU_RSQRT_F32(x * (s ? 0x1.0p+100f : 1.0f)) * (s ? 0x1.0p+50f : 1.0f);
    }
}

