//===- RockToGPU.h - Conversion from Rock to GPU ----------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ROCKTOGPU_ROCKTOGPU_H_
#define MLIR_CONVERSION_ROCKTOGPU_ROCKTOGPU_H_

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class LLVMTypeConverter;
class TypeConverter;
class Pass;

template <typename T> class OperationPass;

#define GEN_PASS_DECL_CONVERTROCKTOGPUPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_ROCKTOGPU_ROCKTOGPU_H_
