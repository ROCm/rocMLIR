//===- InitMIOpenPasses.h - MLIR MIOpen Passes Registration -----------*- C++
//-*-===//
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

#ifndef MLIR_INITMIOPENPASSES_H_
#define MLIR_INITMIOPENPASSES_H_

#include "mlir/Conversion/MIOpenPasses.h"
#include "mlir/Dialect/MIGraphX/Passes.h"
#include "mlir/Dialect/MIOpen/Passes.h"

#include "mlir/Transforms/Passes.h"

#include <cstdlib>

namespace mlir {

// This function may be called to register the MLIR MIOpen passes with the
// global registry.
// If you're building a compiler, you likely don't need this: you would build a
// pipeline programmatically without the need to register with the global
// registry, since it would already be calling the creation routine of the
// individual passes.
// The global registry is interesting to interact with the command-line tools.
inline void registerMIOpenPasses() {
  registerMIOpenConversionPasses();
  miopen::registerPasses();
  migraphx::registerPasses();
}

} // namespace mlir

#endif // MLIR_INITMIOPEN_PASSES_H_