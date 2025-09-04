//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>

<<<<<<< HEAD
#define FUNCTION __clc_add_sat
#define __IMPL_FUNCTION(x) __builtin_elementwise_add_sat
=======
#define __CLC_FUNCTION __clc_add_sat
#define __CLC_IMPL_FUNCTION(x) __builtin_elementwise_add_sat
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
#define __CLC_BODY <clc/shared/binary_def.inc>

#include <clc/integer/gentype.inc>
