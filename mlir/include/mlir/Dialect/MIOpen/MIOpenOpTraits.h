//===- MIOpenOpTraits.h - MLIR MIOpen dialect operation traits --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares C++ classes for some of operation traits in the MIOpen
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIOPEN_OPTRAITS_H_
#define MLIR_DIALECT_MIOPEN_OPTRAITS_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {
namespace miopen {

/// An operation trait for convolution ops.
template <typename ConcreteType>
class ConvolutionOp : public TraitBase<ConcreteType, ConvolutionOp> {};

} // namespace miopen
} // namespace OpTrait
} // namespace mlir

#endif // MLIR_DIALECT_MIOPEN_OPTRAITS_H_
