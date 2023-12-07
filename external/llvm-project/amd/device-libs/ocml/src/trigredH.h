/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

struct redret {
    half hi;
    short i;
};

struct scret {
    half s;
    half c;
};

extern CONSTATTR struct redret  MATH_PRIVATE(trigred)(half x);
extern CONSTATTR struct scret  MATH_PRIVATE(sincosred)(half x);
extern CONSTATTR half MATH_PRIVATE(tanred)(half x, short i);

