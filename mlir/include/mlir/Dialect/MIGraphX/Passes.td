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

def MIGraphXRealizeInt4Pass : Pass<"migraphx-realize-int4", "func::FuncOp"> {
  let summary = "Rewrite migraphx.[un]pack into int4 tensors";
  let description = [{
    MIGraphX's native type system doesn't allow int4 tensors.

    We want to have int4 tensors at the edges of our functions so that we can
    use MLIR's narrow type emulation to rewrite a buffer of linear i4 to
    a buffer of linear i8 late in the computation. This pass begins
    by rewriting
    ```
    %y = migraphx.unpack %x {axis = 1} : <4x2xi8, 2x1> -> <4x4xi8, 4x1>
    ```
    to
    ```
    %x' = migraphx.unpack %x {axis = 1} : <4x2xi8, 2x1> -> <4x2x2xi4, 4x2x1>
    %cast = migraphx.convert zeroExtend %x' : <4x2x2xi4, 4x2x1> -> <4x2x2xi8, 4x2x1>
    %y = migraphx.reshape %cast : <4x2x2xi8, 4x2x1> -> <4x4xi8, 4x1>
    ```

    Then, we rewrite that int4-producing `unpack` until we hit a funciton boundary,
    traversing reshapes and transposes, at which point we replace the i8 argument with
    the i4 one.
  }];
}

#endif // MLIR_DIALECT_MIGRAPHX_PASSES
