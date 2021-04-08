//===- GridwiseConvCppOutputHelper.h - MLIR convolution cpp output helper
//--===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MLIR cpp convolution output helper
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIOPEN_GRIDWISECONVCPPOUTPUTHELPER_H
#define MLIR_DIALECT_MIOPEN_GRIDWISECONVCPPOUTPUTHELPER_H

#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

static constexpr int kConv2DTensorDimension = 5;
static constexpr StringLiteral kVarName[3] = {"weight", "input", "output"};

static inline void
EmitLayoutString(llvm::raw_ostream &output,
                 llvm::ArrayRef<mlir::Attribute> &layoutArrayAttr,
                 llvm::StringRef prefix, llvm::StringRef suffix,
                 llvm::StringRef delimiter = "") {
  for (int i = 0; i < kConv2DTensorDimension; ++i) {
    auto attr = layoutArrayAttr[i];
    if (auto strAttr = attr.dyn_cast<StringAttr>()) {
      output << prefix << strAttr.getValue() << suffix;
    }
    if (i < kConv2DTensorDimension - 1) {
      output << delimiter;
    }
  }
}

#endif // MLIR_DIALECT_MIOPEN_GRIDWISECONVCPPOUTPUTHELPER_H
