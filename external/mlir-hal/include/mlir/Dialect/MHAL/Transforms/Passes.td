//===-- Passes.td - MHAL pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MHAL_PASSES
#define MLIR_DIALECT_MHAL_PASSES

include "mlir/Pass/PassBase.td"

def MHALTargetKernelsPass : Pass<"mhal-target-kernels", "ModuleOp"> {
  let summary = "clone all kernel funcs into MHAL module";
  let options = [
    ListOption<"targets", "targets", "std::string", "List of target architectures">
  ];
}

def MHALPackageTargetsPass : Pass<"mhal-package-targets", "ModuleOp"> {
  let summary = "apply target implementation to host kernel funcs";
}

def MHALInferGraphPass : Pass<"mhal-infer-graph", "func::FuncOp"> {
  let summary = "convert kernel func call ops to mhal.launch ops with dependencies";
  let dependentDialects = ["mhal::MHALDialect"];
}

def MHALBufferizePass : Pass<"mhal-bufferize", "func::FuncOp"> {
  let summary = "Bufferize the mhal dialect.";
  let dependentDialects = ["bufferization::BufferizationDialect",
                           "memref::MemRefDialect"];
}

def MHALSelectTargetsPass : Pass<"mhal-select-targets", "func::FuncOp"> {
  let summary = "Select runtime target for execution";
  let options = [
    ListOption<"targetTypes", "target-types", "std::string", "List of target types">,
    ListOption<"targetArchs", "archs", "std::string", "List of chip architectures">
  ];
}

#endif // MLIR_DIALECT_MHAL_PASSES
