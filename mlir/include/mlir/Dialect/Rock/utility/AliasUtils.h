//===- AliasUtils.h - Utility functions for alias analysis -----*- C++ -*-===//
//
// Copyright 2025 Advanced Micro Devices.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Utility functions to work with alias analysis attributes
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_UTILITY_ALIASUTILS_H
#define MLIR_DIALECT_ROCK_UTILITY_ALIASUTILS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir {
namespace rock {
/// Add the direct-to-LDS load alias scope to the given operation.
/// This marks the operation as being part of the direct-to-LDS load scope.
/// It also marks the operation as not aliasing with LDS loads.
void addDirectToLDSLoadAliasScope(LLVM::AliasAnalysisOpInterface op);

/// Add the LDS load no alias scope to the given operation.
/// This marks the operation as being part of the LDS load scope and does not
/// alias with direct-to-LDS loads. It also adds the operation to the different
/// scope as ops without any scope alias with everything.
void addLDSLoadNoAliasScope(LLVM::AliasAnalysisOpInterface op);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_ALIASUTILS_H
