//===- RockWriterOpInterface.cpp -  -------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockWriterOpInterface, which abstracts rock operations
// that write into memory
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/RockWriterOpInterface.h"

namespace mlir {
namespace rock {
#include "mlir/Dialect/Rock/IR/RockWriterOpInterface.cpp.inc"
} // namespace rock
} // namespace mlir
