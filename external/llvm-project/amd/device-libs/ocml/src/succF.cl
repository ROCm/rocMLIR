/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(succ)(float x)
{
    int ix = AS_INT(x);
    int mx = SIGNBIT_SP32 - ix;
    mx = ix < 0 ? mx : ix;
    int t = mx + (x != PINF_F32 && !BUILTIN_ISNAN_F32(x));
    int r = SIGNBIT_SP32 - t;
    r = t < 0 ? r : t;
    r = mx == -1 ? SIGNBIT_SP32 : r;
    return AS_FLOAT(r);
}

