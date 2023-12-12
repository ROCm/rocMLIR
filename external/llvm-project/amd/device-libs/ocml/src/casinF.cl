/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float2
MATH_MANGLE(casin)(float2 z)
{
    float2 a = MATH_MANGLE(casinh)((float2)(-z.y, z.x));
    return (float2)(a.y, -a.x);
}

