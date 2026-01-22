//===- compileUtils.cpp - Rock compile utility functions -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/compileUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <optional>
using namespace mlir;
using namespace mlir::rock;

#define DEBUG_TYPE "rock-compile-utils"

namespace mlir {
namespace rock {
FailureOr<std::pair<gpu::ObjectAttr, DenseMap<StringRef, size_t>>> createGpuBinary(
    OpBuilder builder, ModuleOp moduleOp, SmallVectorImpl<KernelInfo> &kernels) {
  // Get the HSACO binary from the triton.hsaco attribute
  auto hsacoAttr = moduleOp->getAttrOfType<StringAttr>("triton.hsaco");
  if (!hsacoAttr) {
    return failure();
  }

  // Build a map from kernel names to their info
  DenseMap<StringRef, size_t> kernelMap;
  for (size_t i = 0; i < kernels.size(); ++i) {
    kernelMap[kernels[i].name] = i;
  }

  // Create kernel metadata for the gpu.binary
  MLIRContext *ctx = builder.getContext();
  SmallVector<gpu::KernelMetadataAttr> kernelMetadata;
  auto ptrType = LLVM::LLVMPointerType::get(ctx);

  for (const KernelInfo &kernel : kernels) {
    // Create a function type with 5 pointer arguments (matching HSACO metadata)
    // GEMM kernels typically expect: A, B, C, workspace1, workspace2
    SmallVector<Type> argTypes(5, ptrType);
    auto kernelFuncType = FunctionType::get(ctx, argTypes, {});

    // Create metadata for this kernel
    // KernelMetadataAttr::get(StringAttr name, Type functionType, ...)
    auto metadata =
        gpu::KernelMetadataAttr::get(builder.getStringAttr(kernel.name),
                                     /*functionType=*/kernelFuncType,
                                     /*argAttrs=*/nullptr,
                                     /*metadata=*/nullptr);
    kernelMetadata.push_back(metadata);
  }

  // Create the kernel table
  auto kernelTable = gpu::KernelTableAttr::get(ctx, kernelMetadata);

  // Create the ROCDL target attribute
  // ROCDLTargetAttr::get(ctx, optLevel, triple, chip, features, abiVersion,
  // ...)
  auto rocdlTarget = ROCDL::ROCDLTargetAttr::get(
      ctx,
      /*optLevel=*/2,
      /*triple=*/"amdgcn-amd-amdhsa",
      /*chip=*/"gfx1100", // TODO: get from module attributes
      /*features=*/"",
      /*abiVersion=*/"400");

  // Create the object attribute with the HSACO
  // ObjectAttr::get(Attribute target, CompilationTarget format, StringAttr
  // object, ...)
  auto objectAttr = gpu::ObjectAttr::get(
      rocdlTarget,
      gpu::CompilationTarget::Binary, // format enum directly
      hsacoAttr,
      /*properties=*/nullptr, kernelTable);
  return std::make_pair(objectAttr, kernelMap);
}

} // namespace rock
} // namespace mlir
