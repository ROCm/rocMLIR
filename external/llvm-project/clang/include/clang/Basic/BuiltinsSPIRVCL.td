<<<<<<<< HEAD:external/llvm-project/clang/include/clang/Basic/BuiltinsSPIRVVK.td
//===--- BuiltinsSPIRVVK.td - SPIRV Builtin function database ---*- C++ -*-===//
========
//===--- BuiltinsSPIRVCL.td - SPIRV Builtin function database ---*- C++ -*-===//
>>>>>>>> 65badf35006dbef6b7f3103aabcb23a348f44c59:external/llvm-project/clang/include/clang/Basic/BuiltinsSPIRVCL.td
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsSPIRVBase.td"

<<<<<<<< HEAD:external/llvm-project/clang/include/clang/Basic/BuiltinsSPIRVVK.td

def reflect : SPIRVBuiltin<"void(...)", [NoThrow, Const]>;
def faceforward : SPIRVBuiltin<"void(...)", [NoThrow, Const, CustomTypeChecking]>;
========
def generic_cast_to_ptr_explicit
    : SPIRVBuiltin<"void*(void*, int)", [NoThrow, Const, CustomTypeChecking]>;
>>>>>>>> 65badf35006dbef6b7f3103aabcb23a348f44c59:external/llvm-project/clang/include/clang/Basic/BuiltinsSPIRVCL.td
