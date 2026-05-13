//===-- MIGraphXAttentionToRock.h -------------------------------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2026 Advanced Micro Devices
//
// Pass declaration for lowering migraphx.attention to rock.attention.
// See MIGraphXAttentionToRock.cpp for the polarity contract with the
// host-side AttentionDecompose pattern.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MIGRAPHXATTENTIONTOROCK_H
#define MLIR_CONVERSION_MIGRAPHXATTENTIONTOROCK_H

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL_MIGRAPHXATTENTIONTOROCKPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_MIGRAPHXATTENTIONTOROCK_H
