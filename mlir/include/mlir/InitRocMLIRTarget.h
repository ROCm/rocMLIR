//===- InitRocMLIRTarget.h - MLIR rock target registration ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for attaching the rock target interface for
// the `#rocdl.target` attribute.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INITROCMLIRTARGET_H
#define MLIR_INITROCMLIRTARGET_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
/// Registers the `TargetAttrInterface` for the `#rocdl.target` attribute in the
/// given registry.
void registerRocTarget(DialectRegistry &registry);

/// Registers the `TargetAttrInterface` for the `#rocdl.target` attribute in the
/// registry associated with the given context.
void registerRocTarget(MLIRContext &context);
} // namespace mlir

#endif // MLIR_INITROCMLIRTARGET_H
