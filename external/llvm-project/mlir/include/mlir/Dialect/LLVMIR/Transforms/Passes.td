//===-- Passes.td - LLVM pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_TRANSFORMS_PASSES
#define MLIR_DIALECT_LLVMIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def LLVMLegalizeForExport : Pass<"llvm-legalize-for-export"> {
  let summary = "Legalize LLVM dialect to be convertible to LLVM IR";
  let constructor = "::mlir::LLVM::createLegalizeForExportPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def LLVMRequestCWrappers
    : Pass<"llvm-request-c-wrappers", "::mlir::func::FuncOp"> {
  let summary = "Request C wrapper emission for all functions";
  let description = [{
    Annotate every builtin function in the module with the LLVM dialect
    attribute that instructs the conversion to LLVM to emit the C wrapper for
    the function. This pass is expected to be applied immediately before the
    conversion of builtin functions to LLVM to avoid the attribute being
    dropped by other passes.
  }];
  let constructor = "::mlir::LLVM::createRequestCWrappersPass()";
}

def NVVMOptimizeForTarget : Pass<"llvm-optimize-for-nvvm-target"> {
  let summary = "Optimize NVVM IR";
  let constructor = "::mlir::NVVM::createOptimizeForTargetPass()";
}

def SoftwareBF16 : Pass<"llvm-software-bf16"> {
  let summary = "Convert BF16 to I16 in LLVM IR";
  let description = [{
    This pass erases the BF16 type from LLVM IR.

    Some LLVM targets do not support LLVM's `bfloat` type, or only support it
    incompletely. To allow using the `bf16` type on such targets, this pass
    replaces all of its uses by `i16` and then replaces operations on `bf16` by
    extending the 16-bit values into `f32`, then computes the floating-point
    operation on the extended value, and then truncates the results.
  }];
  let constructor = "mlir::LLVM::createSoftwareBF16Pass()";
}

#endif // MLIR_DIALECT_LLVMIR_TRANSFORMS_PASSES
