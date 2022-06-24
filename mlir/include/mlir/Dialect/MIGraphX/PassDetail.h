//===- PassDetail.h - MIGraphX Pass details -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_MIGRAPHX_TRANSFORMS_PASSDETAIL_H_
#define MLIR_MIGRAPHX_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace func {
class FuncDialect;
} // namespace func

namespace migraphx {
class MIGraphXDialect;
} // namespace migraphx

#define GEN_PASS_CLASSES
#include "mlir/Dialect/MIGraphX/Passes.h.inc"

} // namespace mlir

#endif // MLIR_MIGRAPHX_TRANSFORMS_PASSDETAIL_H_
