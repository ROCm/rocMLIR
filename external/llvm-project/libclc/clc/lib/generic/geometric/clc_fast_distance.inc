//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

<<<<<<<< HEAD:external/llvm-project/libclc/opencl/lib/generic/geometric/fast_normalize.inc
// Geometric functions are only defined for scalar, vec2, vec3 and vec4
// Only handle vector implementations
#if (__CLC_VECSIZE_OR_1 == 2 || __CLC_VECSIZE_OR_1 == 3 ||                     \
     __CLC_VECSIZE_OR_1 == 4)

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE fast_normalize(__CLC_GENTYPE p) {
  __CLC_SCALAR_GENTYPE l2 = dot(p, p);
  return l2 == 0.0f ? p : p * half_rsqrt(l2);
========
#if (__CLC_VECSIZE_OR_1 == 1 || __CLC_VECSIZE_OR_1 == 2 ||                     \
     __CLC_VECSIZE_OR_1 == 3 || __CLC_VECSIZE_OR_1 == 4)

_CLC_OVERLOAD _CLC_DEF __CLC_SCALAR_GENTYPE
__clc_fast_distance(__CLC_GENTYPE p0, __CLC_GENTYPE p1) {
  return __clc_fast_length(p0 - p1);
>>>>>>>> 65badf35006dbef6b7f3103aabcb23a348f44c59:external/llvm-project/libclc/clc/lib/generic/geometric/clc_fast_distance.inc
}

#endif
