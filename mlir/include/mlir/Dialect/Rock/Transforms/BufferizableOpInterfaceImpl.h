//===- BufferizableOpInterfaceImpl.h - Impl. of BufferizableOpInterface ---===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H
#define MLIR_DIALECT_ROCK_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {

class DialectRegistry;

namespace rock {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H
