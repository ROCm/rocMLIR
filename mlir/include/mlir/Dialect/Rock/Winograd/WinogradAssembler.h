//===-- WinogradAssembler.h - Assemble Winograd kernels ----------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//
//
// Assembles Winograd convolution kernels from pre-written assembly sources
// into HSACO (HSA Code Object) binaries suitable for GPU execution.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_WINOGRAD_WINOGRAD_ASSEMBLER_H
#define MLIR_DIALECT_ROCK_WINOGRAD_WINOGRAD_ASSEMBLER_H

#include "mlir/Dialect/Rock/Winograd/WinogradSolver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace mlir {
namespace rock {
namespace winograd {

/// Assemble a Winograd kernel into an HSACO binary.
///
/// Takes the resolved kernel selection (which identifies the assembly source
/// file and compilation options) along with the target GPU parameters, and
/// produces a ready-to-load HSACO code object.
///
/// \param selection The resolved kernel selection from WinogradSolver
/// \param chip      Target GPU chip (e.g. "gfx942")
/// \param triple    Target triple (e.g. "amdgcn-amd-amdhsa")
/// \param features  Target features string (e.g. "+sramecc,+xnack")
/// \returns The assembled HSACO binary, or std::nullopt on failure
std::optional<llvm::SmallVector<char, 0>>
assembleWinogradKernel(const WinogradKernelSelection &selection,
                       llvm::StringRef chip, llvm::StringRef triple,
                       llvm::StringRef features);

} // namespace winograd
} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_WINOGRAD_WINOGRAD_ASSEMBLER_H
