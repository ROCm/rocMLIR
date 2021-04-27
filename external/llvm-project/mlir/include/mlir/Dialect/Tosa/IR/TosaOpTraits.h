//===- SPIRVOps.h - MLIR SPIR-V operation traits ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares C++ classes for operation traits in the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TOSA_IR_TOSAOPTRAITS_H_
#define MLIR_DIALECT_TOSA_IR_TOSAOPTRAITS_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {
namespace tosa {

// A trait for ops that are elementwise in the abstract, and thus can
// be fused.  The existing Elementwise trait requires identical shapes
// for any non-scalar operands and results, but this one doesn't.
template <typename ConcreteType>
struct AbstractElementwise
    : public TraitBase<ConcreteType, AbstractElementwise> {};

} // namespace tosa
} // namespace OpTrait
} // namespace mlir

#endif // MLIR_DIALECT_TOSA_IR_TOSAOPTRAITS_H_
