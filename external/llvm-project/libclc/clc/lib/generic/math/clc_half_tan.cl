//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_tan.h>

<<<<<<< HEAD
#define __FLOAT_ONLY
#define FUNCTION __clc_half_tan
#define __IMPL_FUNCTION(x) __clc_tan
=======
#define __CLC_FLOAT_ONLY
#define __CLC_FUNCTION __clc_half_tan
#define __CLC_IMPL_FUNCTION(x) __clc_tan
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
#define __CLC_BODY <clc/shared/unary_def.inc>

#include <clc/math/gentype.inc>
