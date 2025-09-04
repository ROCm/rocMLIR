//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_INTEGER_CTZ_H__
#define __CLC_OPENCL_INTEGER_CTZ_H__

#if __OPENCL_C_VERSION__ >= CL_VERSION_2_0

#include <clc/opencl/opencl-base.h>

<<<<<<< HEAD
#define FUNCTION ctz
=======
#define __CLC_FUNCTION ctz
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
#define __CLC_BODY <clc/shared/unary_decl.inc>

#include <clc/integer/gentype.inc>

#undef FUNCTION

#endif // __OPENCL_C_VERSION__ >= CL_VERSION_2_0

#endif // __CLC_OPENCL_INTEGER_CTZ_H__
