//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

<<<<<<< HEAD:external/llvm-project/libclc/clc/include/clc/math/unary_def_via_fp32.inc
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE FUNCTION(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(FUNCTION(__CLC_CONVERT_FLOATN(x)));
=======
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_clz(__CLC_GENTYPE x) {
  return __builtin_elementwise_ctlz(x, (__CLC_GENTYPE)__CLC_GENSIZE);
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a:external/llvm-project/libclc/clc/lib/generic/integer/clc_clz.inc
}
