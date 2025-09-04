//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

<<<<<<< HEAD:external/llvm-project/libclc/clc/include/clc/shared/binary_decl_with_scalar_second_arg.inc
_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE FUNCTION(__CLC_GENTYPE x,
                                               __CLC_GENTYPE y);

#ifndef __CLC_SCALAR
_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE FUNCTION(__CLC_GENTYPE x,
                                               __CLC_SCALAR_GENTYPE y);
#endif
=======
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE bitfield_insert(__CLC_GENTYPE base,
                                                     __CLC_GENTYPE insert,
                                                     uint offset, uint count) {
  return __clc_bitfield_insert(base, insert, offset, count);
}
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a:external/llvm-project/libclc/opencl/lib/generic/integer/bitfield_insert.inc
