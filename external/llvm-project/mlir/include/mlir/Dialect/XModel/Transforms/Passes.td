//===-- Passes.td - XModel pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XMODEL_PASSES
#define MLIR_DIALECT_XMODEL_PASSES

include "mlir/Pass/PassBase.td"

def XModelTargetKernelsPass : Pass<"xmodel-target-kernels", "ModuleOp"> {
  let summary = "clone all kernel funcs into xmodel module";
  let options = [
    ListOption<"targets", "targets", "std::string", "List of target architectures">
  ];
}

def XModelPackageTargetsPass : Pass<"xmodel-package-targets", "ModuleOp"> {
  let summary = "apply target implementation to host kernel funcs";
}

def XModelAsyncGraphPass : Pass<"xmodel-async-graph", "func::FuncOp"> {
  let summary = "convert kernel func call ops to async.launch ops with dependencies";
  let dependentDialects = ["xmodel::XModelDialect", "async::AsyncDialect"];
}

def XModelSelectTargetsPass : Pass<"xmodel-select-targets", "func::FuncOp"> {
  let summary = "Select runtime target for execution";
  let options = [
    ListOption<"targetTypes", "target-types", "std::string", "List of target types">,
    ListOption<"targetChips", "chips", "std::string", "List of chip architectures">
  ];
}

#endif // MLIR_DIALECT_XMODEL_PASSES
