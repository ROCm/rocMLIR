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

def MHALAnnotateAccessKindsPass : Pass<"mhal-annotate-access-kinds", "func::FuncOp"> {
  let summary = "Annotate arguments for read and write access";
  let description = [{
    Annotate arguments and results of functions to preserve read-only/write-only/read-write
    information post-bufferization.
  }];
  let dependentDialects = ["mhal::MHALDialect"];
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

def MHalEmulateNarrowTypePass : Pass<"mhal-emulate-narrow-type", "func::FuncOp"> {
    let summary = "Emulate memrefs of 4-bit integers (or 2-bit, maybe)";
  let description = [{
    Coordinates the rewrite patterns present in upstream MLIR for supporting
    i4 memrefs by rewriting them to linear i8 ones. Also rewrite `mhal.launch`
    just like `func.call` gets rewritten.

    Note that this pass is meant to target the host/test code, and so doesn't handle
    vector loads/stores because they won't have been produced by the host runner.
  }];

  let dependentDialects = ["affine::AffineDialect", "arith::ArithDialect"];
}

def MHALSelectTargetsPass : Pass<"mhal-select-targets", "func::FuncOp"> {
  let summary = "Select runtime target for execution";
  let options = [
    ListOption<"targetTypes", "target-types", "std::string", "List of target types">,
    ListOption<"targetArchs", "archs", "std::string", "List of chip architectures">
  ];
}

def MHALPrefillPass : Pass<"mhal-prefill", "func::FuncOp"> {
  let summary = "Prefill all mem. buffers before a kernel launch.";
  let dependentDialects = ["mhal::MHALDialect"];
}

def MHALDropBinaryMetadataPass : Pass<"mhal-drop-binary-metadata", "ModuleOp"> {
  let summary = "drops all metadata stored in GPU binaries";
}

#endif // MLIR_DIALECT_MHAL_PASSES
