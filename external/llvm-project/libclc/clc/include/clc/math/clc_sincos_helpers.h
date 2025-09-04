//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_MATH_CLC_SINCOS_HELPERS_H__
#define __CLC_MATH_CLC_SINCOS_HELPERS_H__

#define __CLC_FLOAT_ONLY
#define __CLC_BODY <clc/math/clc_sincos_helpers.inc>

#include <clc/math/gentype.inc>

<<<<<<< HEAD
#define __DOUBLE_ONLY
=======
#define __CLC_DOUBLE_ONLY
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
#define __CLC_BODY <clc/math/clc_sincos_helpers_fp64.inc>

#include <clc/math/gentype.inc>

#endif // __CLC_MATH_CLC_SINCOS_HELPERS_H__
