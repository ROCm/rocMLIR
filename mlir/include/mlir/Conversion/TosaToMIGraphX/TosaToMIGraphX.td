//===-- conversion.td - Tosa to MIGraphX conversion pattern definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//#ifndef MLIR_DIALECT_MIGRAPHX_PASSES
//#define MLIR_DIALECT_MIGRAPHX_PASSES
include "mlir/Pass/PassBase.td"
//include "mlir/IR/OpBase.td"
include "mlir/Dialect/Tosa/IR/TosaOps.td"
include "mlir/Dialect/MIGraphX/MIGraphXOps.td"

//def : Pat<(mlir::tosa::Tosa_AddOp $input1, $input2), (mlir::migraphx::MIGraphX_AddOp $input1, $input2)>;
def : Pat<(Tosa_AddOp $input1, $input2), (MIGraphX_AddOp $input1, $input2)>;

//#endif // MLIR_DIALECT_MIGRAPHX_PASSES
