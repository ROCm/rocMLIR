//===- DialectRegistration.cpp - Register MIOpen dialect ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpenOps/MIOpenOps.h"
using namespace mlir;

// Static initialization for MIOpen dialect registration.
static DialectRegistration<miopen::MIOpenOpsDialect> MIOpenOps;
