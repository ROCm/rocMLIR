//===- AttentionUtils.h - Shared rules for migraphx.attention ----- C++ -===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2026 Advanced Micro Devices
//
//===----------------------------------------------------------------------===//
//
// Small inline helpers that encode contracts shared by several pieces of
// migraphx.attention's lowering chain. Keeping them here means the
// verifier, the host AttentionDecompose, the GPU MIGraphXAttentionToRock
// lowering, and rocmlir-gen all derive the same answers from the same code.
//
// Anything that's only used in one place, or that requires
// path-specific inputs (e.g. expectedQKShape, which the verifier
// computes from pre-splitKV operands while the host decompose computes
// from post-splitKV-reshaped types), should stay local to that pass.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_MIGRAPHX_IR_ATTENTIONUTILS_H_
#define MLIR_MIGRAPHX_IR_ATTENTIONUTILS_H_

#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace migraphx {

/// The element type that the first GEMM (Q*K) of a migraphx.attention
/// produces, given Q's element type. For float Q the QK output stays in
/// Q's type; for integer Q the first GEMM is a quantized matmul whose
/// output is i32 (the body is then expected to dequantize that i32 to a
/// float type). Used by AttentionOp::verify, MIGraphXTransform's host
/// AttentionDecompose, and rocmlir-gen so all three derive the same QK
/// type for the same Q.
inline Type computeAttentionQKElemType(Type qElemType, MLIRContext *ctx) {
  if (isa<FloatType>(qElemType))
    return qElemType;
  return IntegerType::get(ctx, 32);
}

/// Returns true if `op` is in the closed set of migraphx ops that
/// MIGraphXAttentionToRock::lowerMIGraphXElementwiseToScalar can lower to
/// a scalar arith / math equivalent inside a linalg.generic body. The
/// AttentionOp verifier consults this so the verifier never accepts a
/// preSoftmaxBody that the lowering would later reject; the lowering
/// itself uses the same membership rule (encoded as a dispatch table) to
/// decide what to emit.
///
/// IMPORTANT: this list and
/// MIGraphXAttentionToRock::lowerMIGraphXElementwiseToScalar must stay in
/// lock-step. Adding a new body op is a one-line change in two coupled
/// places (this function plus the lowering's dispatch table).
inline bool isAllowedInPreSoftmaxBody(Operation &op) {
  return isa<migraphx::AddOp, migraphx::SubOp, migraphx::MulOp, migraphx::DivOp,
             migraphx::PowOp, migraphx::NegOp, migraphx::AbsOp,
             migraphx::CeilOp, migraphx::FloorOp, migraphx::ExpOp,
             migraphx::LogOp, migraphx::SqrtOp, migraphx::TanhOp,
             migraphx::ErfOp, migraphx::RecipOp, migraphx::ReluOp,
             migraphx::SigmoidOp, migraphx::WhereOp, migraphx::ConvertOp,
             migraphx::DeQuantizeLinearOp>(op);
}

} // namespace migraphx
} // namespace mlir

#endif // MLIR_MIGRAPHX_IR_ATTENTIONUTILS_H_
