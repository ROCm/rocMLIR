<<<<<<<< HEAD:external/llvm-project/mlir/lib/Support/StateStack.cpp
//===- StateStack.cpp - Utility for storing a stack of state --------------===//
========
//===----------------------------------------------------------------------===//
>>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a:external/llvm-project/libclc/clc/lib/generic/integer/clc_bitfield_insert.cl
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

<<<<<<<< HEAD:external/llvm-project/mlir/lib/Support/StateStack.cpp
#include "mlir/Support/StateStack.h"

namespace mlir {

void StateStackFrame::anchor() {}

} // namespace mlir
========
#include <clc/integer/clc_bitfield_insert.h>

#define __CLC_BODY <clc_bitfield_insert.inc>
#include <clc/integer/gentype.inc>
>>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a:external/llvm-project/libclc/clc/lib/generic/integer/clc_bitfield_insert.cl
