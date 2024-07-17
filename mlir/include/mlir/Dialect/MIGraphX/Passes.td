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
  let dependentDialects = ["migraphx::MIGraphXDialect"];
}

def MIGraphXIREEGen : Pass<"migraphx-iree-gen", "::mlir::ModuleOp"> {
  let summary = "Use IREE as the kernel generator";
  let dependentDialects = ["mlir::gpu::GPUDialect", "mlir::ROCDL::ROCDLDialect"];
  let options = [
    Option<"arch", "arch", "std::string",
           /*default=*/"\"gfx908\"",
           "Target arch.">,
    Option<"features", "features", "std::string",
           /*default=*/"\"+sramecc,-xnack\"",
           "Target features.">
  ];
}

#endif // MLIR_DIALECT_MIGRAPHX_PASSES
