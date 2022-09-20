//===- InitRocMLIRPasses.h - rocMLIR Passes Registration --------*- C++ -*-===//
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

#ifndef MLIR_INITROCMLIRPASSES_H_
#define MLIR_INITROCMLIRPASSES_H_

#include "mlir/Conversion/RocMLIRPasses.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"

#include "mlir/Transforms/Passes.h"

#include <cstdlib>

namespace mlir {

// This function may be called to register the rocMLIR passes with the
// global registry.
// If you're building a compiler, you likely don't need this: you would build a
// pipeline programmatically without the need to register with the global
// registry, since it would already be calling the creation routine of the
// individual passes.
// The global registry is interesting to interact with the command-line tools.
inline void registerRocMLIRPasses() {
  registerRocMLIRConversionPasses();
  rock::registerPasses();

  rock::registerPipelines();
}

} // namespace mlir

#endif // MLIR_INITROCMLIRPASSES_H_
