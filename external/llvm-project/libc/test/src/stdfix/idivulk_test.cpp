//===-- Unittests for idivulk ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

<<<<<<<< HEAD:external/llvm-project/libclc/opencl/lib/generic/geometric/fast_normalize.cl
#include <clc/opencl/clc.h>

_CLC_OVERLOAD _CLC_DEF float fast_normalize(float p) { return normalize(p); }

#define __CLC_BODY <fast_normalize.inc>
#define __FLOAT_ONLY
#include <clc/math/gentype.inc>
#undef __FLOAT_ONLY
========
#include "IdivTest.h"

#include "llvm-libc-macros/stdfix-macros.h" // unsigned long accum
#include "src/stdfix/idivulk.h"

LIST_IDIV_TESTS(ulk, unsigned long accum, unsigned long int,
                LIBC_NAMESPACE::idivulk);
>>>>>>>> 1b7ebbdeb3e02c5a7a49551c244ad1835a6f9c60:external/llvm-project/libc/test/src/stdfix/idivulk_test.cpp
