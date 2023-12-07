/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(ncdfinv)

CONSTATTR half
MATH_MANGLE(ncdfinv)(half x)
{
    return (half)MATH_UPMANGLE(ncdfinv)((float)x);
}

