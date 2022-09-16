//===- InitRockDialects.h - MLIR Rock Dialects Registration -----------*-
// C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all our custom
// dialects and passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INITROCKDIALECTS_H_
#define MLIR_INITROCKDIALECTS_H_

#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"
#include "mlir/Dialect/Rock/Rock.h"

#include "mlir/IR/Dialect.h"

namespace mlir {

// Add all the MLIR dialects to the provided registry.
inline void registerRockDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<rock::RockDialect,
                  migraphx::MIGraphXDialect>();
  // clang-format on
}

} // namespace mlir

#endif // MLIR_INITROCKDIALECTS_H_