/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

float
MATH_MANGLE(tan)(float x)
{
    float ax = BUILTIN_ABS_F32(x);

    struct redret r = MATH_PRIVATE(trigred)(AS_FLOAT(ax));

#if defined EXTRA_PRECISION
    float t = MATH_PRIVATE(tanred)(r.hi + r.lo, r.i & 1);
#else
    float t = MATH_PRIVATE(tanred)(r.hi, r.i & 1);
#endif

    t = AS_FLOAT(AS_INT(t) ^ (AS_INT(x) ^ AS_INT(ax)));

    if (!FINITE_ONLY_OPT()) {
        t = BUILTIN_ISFINITE_F32(ax) ? t : QNAN_F32;
    }

    return t;
}

