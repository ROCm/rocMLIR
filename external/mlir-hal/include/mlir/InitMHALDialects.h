//===- InitMHALDialects.h - MHAL Dialects Registration ----------*- C++ -*-===//
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

#ifndef MLIR_INITMHALDIALECTS_H_
#define MLIR_INITMHALDIALECTS_H_

// MHAL includes
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MHAL/Transforms/BufferizableOpInterfaceImpl.h"

namespace mlir {

// Add all the MLIR dialects to the provided registry.
inline void registerMHALDialects(DialectRegistry &registry) {
  // Register MHAL specific dialects
  registry.insert<mhal::MHALDialect>();

  // Register bufferization hooks for mhal interfaces
  mhal::registerBufferizableOpInterfaceExternalModels(registry);
}

} // namespace mlir

#endif // MLIR_INITMHALDIALECTS_H_
