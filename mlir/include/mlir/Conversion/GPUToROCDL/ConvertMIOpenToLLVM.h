//===- ConvertMIOpenToLLVM.h - Utils to convert from the miopen dialect ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_MIOPENTOLLVM_CONVERTVECTORTOLLVM_H_
#define MLIR_CONVERSION_MIOPENTOLLVM_CONVERTVECTORTOLLVM_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class LLVMTypeConverter;
class ModuleOp;
template <typename T> class OperationPass;

/// Collect a set of patterns to convert from the Vector dialect to LLVM.
void populateMIOpenToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                            OwningRewritePatternList &patterns);
} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVM_H_
