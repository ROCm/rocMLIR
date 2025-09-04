//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>

_CLC_DEF _CLC_OVERLOAD float __clc_fmin(float x, float y) {
  return __builtin_fminf(x, y);
}
<<<<<<< HEAD:external/llvm-project/libclc/clc/lib/r600/math/clc_fmin.cl

#define __FLOAT_ONLY
#define FUNCTION __clc_fmin
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef FUNCTION
=======
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a:external/llvm-project/libclc/clc/lib/spirv/math/clc_fmin.cl

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
_CLC_DEF _CLC_OVERLOAD double __clc_fmin(double x, double y) {
  return __builtin_fmin(x, y);
}
<<<<<<< HEAD:external/llvm-project/libclc/clc/lib/r600/math/clc_fmin.cl

#define __DOUBLE_ONLY
#define FUNCTION __clc_fmin
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef FUNCTION

=======
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a:external/llvm-project/libclc/clc/lib/spirv/math/clc_fmin.cl
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
_CLC_DEF _CLC_OVERLOAD half __clc_fmin(half x, half y) {
  return __builtin_fminf16(x, y);
}
#endif

#define __CLC_FUNCTION __clc_fmin
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
