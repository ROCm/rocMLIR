//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

<<<<<<< HEAD
#define FUNCTION remquo
=======
#ifndef __CLC_OPENCL_MATH_REMQUO_H__
#define __CLC_OPENCL_MATH_REMQUO_H__

#define __CLC_FUNCTION remquo
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

#define __CLC_BODY <clc/math/remquo_decl.inc>
#include <clc/math/gentype.inc>

#if _CLC_GENERIC_AS_SUPPORTED
#define __CLC_BODY <clc/math/remquo_decl.inc>
#define __CLC_ADDRESS_SPACE generic
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE
#endif

<<<<<<< HEAD
#undef FUNCTION
=======
#undef __CLC_FUNCTION

#endif // __CLC_OPENCL_MATH_REMQUO_H__
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
