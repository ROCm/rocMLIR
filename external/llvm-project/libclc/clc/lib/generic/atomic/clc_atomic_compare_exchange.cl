//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/atomic/clc_atomic_compare_exchange.h>

<<<<<<<< HEAD:external/llvm-project/libclc/clc/include/clc/shared/binary_def_with_int_second_arg.inc
#ifndef __IMPL_FUNCTION
#define __IMPL_FUNCTION(x) __CLC_CONCAT(__clc_, x)
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE FUNCTION(__CLC_GENTYPE x, __CLC_INTN y) {
  return __IMPL_FUNCTION(FUNCTION)(x, y);
}
========
#define __CLC_BODY <clc_atomic_compare_exchange.inc>
#include <clc/integer/gentype.inc>

#define __CLC_BODY <clc_atomic_compare_exchange.inc>
#include <clc/math/gentype.inc>
>>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a:external/llvm-project/libclc/clc/lib/generic/atomic/clc_atomic_compare_exchange.cl
