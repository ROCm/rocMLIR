/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

extern CONSTATTR double2 MATH_PRIVATE(epexpep)(double2 x);

CONSTATTR double
MATH_MANGLE(tanh)(double x)
{
    double y = BUILTIN_ABS_F64(x);
    double2 e = MATH_PRIVATE(epexpep)(con(y, 0.0));
    double2 ei = rcp(e);
    double2 t = fdiv(fsub(e, ei), fadd(e, ei));
    double z = t.hi;

    z = y > 19.0625 ? 1.0 : z;
    z = y < 0x1.0p-27 ? y : z;

    return BUILTIN_COPYSIGN_F64(z, x);
}

