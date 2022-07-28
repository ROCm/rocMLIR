//===-- Passes.td - MIGraphX pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIGRAPHX_PASSES
#define MLIR_DIALECT_MIGRAPHX_PASSES

include "mlir/Pass/PassBase.td"

def MIGraphXTransformPass : Pass<"migraphx-transform", "func::FuncOp"> {
  let summary = "apply migraphx operation optimization transform";
  let constructor = "mlir::migraphx::createMIGraphXTransformPass()";
}

#endif // MLIR_DIALECT_MIGRAPHX_PASSES
