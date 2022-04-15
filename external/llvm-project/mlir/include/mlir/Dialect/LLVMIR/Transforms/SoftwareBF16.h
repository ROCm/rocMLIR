//===- LegalizeForExport.h - Prepare for translation to LLVM IR -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_TRANSFORMS_SOFTWAREBF16_H
#define MLIR_DIALECT_LLVMIR_TRANSFORMS_SOFTWAREBF16_H

#include <memory>

namespace mlir {
class Operation;
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

namespace LLVM {

/// Since AMD GPU targets don't natively support bfloat as a type, even though
/// LLVM does, populate patterns and type conversions to convert uses of bf16 to
/// i16. To be removed when/if the backend has been taught about bfloat.
void populateBF16ToLLVMConversionPatterns(LLVMTypeConverter &,
                                          RewritePatternSet &);
/// Creates a pass that converts BF16 type
std::unique_ptr<Pass> createSoftwareBF16Pass();

} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_TRANSFORMS_SOFTWAREBF16_H
