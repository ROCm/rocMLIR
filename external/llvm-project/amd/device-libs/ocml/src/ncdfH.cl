/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(ncdf)

CONSTATTR half
MATH_MANGLE(ncdf)(half x)
{
    return (half)MATH_UPMANGLE(ncdf)((float)x);
}

