//===- Passes.h - rocMLIR passes not tied to dialects -------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2023 ADvanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ROCMLIRTRANSFORMS_PASSES_H_
#define MLIR_ROCMLIRTRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace rocmlir {
#define GEN_PASS_DECL_COMPILEGPUWITHCOMGRPASS
#define GEN_PASS_REGISTRATION
#include "mlir/RocmlirTransforms/Passes.h.inc"
} // namespace rocmlir
} // namespace mlir

#endif // MLIR_ROCMLIRTRANSFORMS_PASSES_H
