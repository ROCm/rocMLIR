//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

<<<<<<< HEAD
_CLC_OVERLOAD _CLC_DECL __CLC_INTN FUNCTION(__CLC_FLOATN x);
=======
#if __CLC_VECSIZE_OR_1 == 1
#define __CLC_RETTYPE __CLC_INTN
#else
#define __CLC_RETTYPE __CLC_BIT_INTN
#endif

_CLC_OVERLOAD _CLC_CONST _CLC_DECL __CLC_RETTYPE
__CLC_FUNCTION(__CLC_GENTYPE x);

#undef __CLC_RETTYPE
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
