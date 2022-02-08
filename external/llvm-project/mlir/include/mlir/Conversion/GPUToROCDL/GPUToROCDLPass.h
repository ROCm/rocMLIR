//===- GPUToROCDLPass.h - Convert GPU kernel to ROCDL dialect ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUTOROCDL_GPUTOROCDLPASS_H_
#define MLIR_CONVERSION_GPUTOROCDL_GPUTOROCDLPASS_H_

#include "mlir/Conversion/GPUToROCDL/Runtimes.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
class ConversionTarget;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

template <typename OpT>
class OperationPass;

namespace gpu {
class GPUModuleOp;
} // namespace gpu

/// Since AMD GPU targets don't natively support bfloat as a type, even though
/// LLVM does, populate patterns and type conversions to convert uses of bf16 to
/// i16. To be removed when/if the backend has been taught about bfloat
void populateBF16ToROCDLConversionPatterns(LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns);

/// Collect a set of patterns to convert from the GPU dialect to ROCDL.
/// If `runtime` is Unknown, gpu.printf will not be lowered
void populateGpuToROCDLConversionPatterns(LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns,
                                          gpu::amd::Runtime runtime);

/// Configure target to convert from the GPU dialect to ROCDL.
void configureGpuToROCDLConversionLegality(ConversionTarget &target);

/// Creates a pass that lowers GPU dialect operations to ROCDL counterparts. The
/// index bitwidth used for the lowering of the device side index computations
/// is configurable.
std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
createLowerGpuOpsToROCDLOpsPass(
    unsigned indexBitwidth = kDeriveIndexBitwidthFromDataLayout,
    gpu::amd::Runtime runtime = gpu::amd::Runtime::Unknown);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOROCDL_GPUTOROCDLPASS_H_
