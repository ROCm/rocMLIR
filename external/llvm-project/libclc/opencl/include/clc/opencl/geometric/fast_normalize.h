//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

<<<<<<< HEAD
#define __FLOAT_ONLY
#define FUNCTION fast_normalize
=======
#ifndef __CLC_OPENCL_GEOMETRIC_FAST_NORMALIZE_H__
#define __CLC_OPENCL_GEOMETRIC_FAST_NORMALIZE_H__

#define __CLC_FLOAT_ONLY
#define __CLC_FUNCTION fast_normalize
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
#define __CLC_GEOMETRIC_RET_GENTYPE
#define __CLC_BODY <clc/geometric/unary_decl.inc>

#include <clc/math/gentype.inc>

#undef FUNCTION
#undef __CLC_GEOMETRIC_RET_GENTYPE
<<<<<<< HEAD
=======

#endif // __CLC_OPENCL_GEOMETRIC_FAST_NORMALIZE_H__
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
