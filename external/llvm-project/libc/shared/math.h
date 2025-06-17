<<<<<<<< HEAD:external/llvm-project/clang/include/clang/Basic/BuiltinsSPIRVCL.td
//===--- BuiltinsSPIRVCL.td - SPIRV Builtin function database ---*- C++ -*-===//
========
//===-- Floating point math functions ---------------------------*- C++ -*-===//
>>>>>>>> 65badf35006dbef6b7f3103aabcb23a348f44c59:external/llvm-project/libc/shared/math.h
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

<<<<<<<< HEAD:external/llvm-project/clang/include/clang/Basic/BuiltinsSPIRVCL.td
include "clang/Basic/BuiltinsSPIRVBase.td"

def generic_cast_to_ptr_explicit
    : SPIRVBuiltin<"void*(void*, int)", [NoThrow, Const, CustomTypeChecking]>;
========
#ifndef LLVM_LIBC_SHARED_MATH_H
#define LLVM_LIBC_SHARED_MATH_H

#include "libc_common.h"

#include "math/expf.h"

#endif // LLVM_LIBC_SHARED_MATH_H
>>>>>>>> 65badf35006dbef6b7f3103aabcb23a348f44c59:external/llvm-project/libc/shared/math.h
