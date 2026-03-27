//===- MIGraphX.cpp - C Interface for MIGraphX dialect
//------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/MIGraphX.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/MIGraphX/Pipeline/Pipeline.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/ExecutionEngine/RocmDeviceName.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/TargetSelect.h"

namespace {
static MlirNamedAttribute makeI32NamedAttr(MlirContext ctx, const char *name,
                                           int32_t value) {
  MlirType i32Type = mlirIntegerTypeGet(ctx, 32);
  MlirAttribute attr = mlirIntegerAttrGet(i32Type, value);
  return mlirNamedAttributeGet(
      mlirIdentifierGet(ctx, mlirStringRefCreateFromCString(name)), attr);
}

static MlirAttribute
makeSegmentSizesAttr(MlirContext ctx, const int32_t *segments, intptr_t count) {
  return mlirDenseI32ArrayGet(ctx, count, segments);
}
} // namespace

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(MIGraphX, migraphx,
                                      mlir::migraphx::MIGraphXDialect)

MlirTypeID rocmlirMIXRShapedTypeGetTypeId() {
  return wrap(mlir::migraphx::MIXRShapedType::getTypeID());
}

bool rocmlirIsAMIXRShapedType(MlirType type) {
  return llvm::isa<mlir::migraphx::MIXRShapedType>(unwrap(type));
}

MlirType rocmlirMIXRShapedTypeGet(intptr_t rank, const int64_t *shape,
                                  const int64_t *strides,
                                  MlirType elementType) {
  return wrap(mlir::migraphx::MIXRShapedType::get(
      llvm::ArrayRef(shape, static_cast<size_t>(rank)),
      llvm::ArrayRef(strides, static_cast<size_t>(rank)), unwrap(elementType)));
}

MlirType rocmlirMIXRShapedTypeAsTensor(MlirType type) {
  return wrap(
      llvm::cast<mlir::migraphx::MIXRShapedType>(unwrap(type)).asTensor());
}

// Returns block_size, grid_size and cluster_size as uint32_t[3]
MLIR_CAPI_EXPORTED void mlirGetKernelAttrs(MlirModule module, uint32_t *attrs) {
  auto mod = unwrap(module);
  size_t count = 0;
  mod.walk([&](mlir::gpu::BinaryOp binary) {
    mlir::gpu::KernelTableAttr metadata =
        mlir::cast<mlir::gpu::ObjectAttr>(binary.getObjects()[0]).getKernels();
    for (auto kernel : metadata) {
      auto block = kernel.getAttr<mlir::IntegerAttr>("block_size");
      auto grid = kernel.getAttr<mlir::IntegerAttr>("grid_size");
      auto cluster = kernel.getAttr<mlir::IntegerAttr>("cluster_size");
      if (!block || !grid || !cluster)
        continue;
      attrs[0] = block.getInt();
      attrs[1] = grid.getInt();
      attrs[2] = cluster.getInt();
      ++count;
    }
  });
  assert(count == 1 && "invalid number of kernels");
}

// Returns the size of compiled binary if called with null ptr
// and return the compiled binary when buffer is provided
MLIR_CAPI_EXPORTED bool mlirGetBinary(MlirModule module, size_t *size,
                                      char *bin) {
  bool success = false;
  auto mod = unwrap(module);
  if (bin == nullptr && size == nullptr)
    return success;
  mod.walk([&](mlir::gpu::BinaryOp binary) {
    auto object = llvm::cast<mlir::gpu::ObjectAttr>(binary.getObjects()[0]);
    if (bin != nullptr) { // return binary regardless the presence of *size
      llvm::StringRef hsaco = object.getObject().getValue();
      std::copy(hsaco.begin(), hsaco.end(), bin);
      success = true;
    } else {
      *size = object.getObject().getValue().size();
      success = true;
    }
  });
  return success;
}

// pipelines

MLIR_CAPI_EXPORTED
void mlirMIGraphXAddHighLevelPipeline(MlirPassManager pm) {
  auto passMan = unwrap(pm);
  if (failed(applyPassManagerCLOptions(*passMan)))
    llvm::errs() << "Failed to apply command-line options.\n";
  passMan->setNesting(mlir::PassManager::Nesting::Implicit);
  mlir::migraphx::addHighLevelPipeline(*passMan);
  mlir::rock::buildBufferizePipeline(*passMan);
}

MLIR_CAPI_EXPORTED bool
mlirMIGraphXAddBackendPipeline(MlirPassManager pm,
                               const MlirMIGraphXBackendOptions *opts) {
  if (!opts) {
    llvm::errs() << "opts is null\n";
    return false;
  }
  if (!opts->arch) {
    llvm::errs() << "opts->arch must not be null\n";
    return false;
  }
  // opts->perfConfig is accepted for API parity with rocmlirTriton but unused
  // in rocMLIR; callers may pass NULL.
  if (opts->optLevel < 0 || opts->optLevel > 3) {
    llvm::errs() << "opts->optLevel must be 0, 1, 2, or 3; got "
                 << opts->optLevel << "\n";
    return false;
  }
  auto *passMan = unwrap(pm);
  if (failed(applyPassManagerCLOptions(*passMan)))
    return false;
  passMan->setNesting(mlir::PassManager::Nesting::Implicit);
  mlir::rock::KernelOptions kOpts;
  kOpts.applicabilityMode = mlir::rock::ApplicabilityMode::Full;
  kOpts.tuningFallback = false;
  mlir::rock::buildKernelPipeline(*passMan, kOpts);
  llvm::StringRef archStr(opts->arch);
  mlir::RocmDeviceName devName;
  if (archStr.empty() || mlir::failed(devName.parse(archStr))) {
    llvm::errs() << "Invalid architecture: " << archStr << "\n";
    return false;
  }
  mlir::rock::BackendOptions backendOpts;
  backendOpts.triple = devName.getTriple().str();
  backendOpts.chip = devName.getChip().str();
  backendOpts.features = devName.getFeaturesForBackend();
  backendOpts.optLevel = opts->optLevel;
  mlir::rock::buildBackendPipeline(*passMan, backendOpts);

  return true;
}

// Op creation helpers

MLIR_CAPI_EXPORTED MlirOperation rocmlirMIGraphXAttentionCreate(
    MlirLocation location, MlirValue queries, MlirValue keys, MlirValue values,
    intptr_t numPreSoftmaxInputs, const MlirValue *preSoftmaxElemWiseInputs,
    MlirType resultType, MlirType lseType, MlirType softmaxType,
    MlirRegion preSoftmaxBody, uint32_t features, MlirValue currentSeqLen,
    MlirValue prefixOffset, int32_t splitKV, int32_t slidingWindowSize) {
  MlirContext ctx = mlirLocationGetContext(location);
  MlirOperationState state = mlirOperationStateGet(
      mlirStringRefCreateFromCString("migraphx.attention"), location);

  // Operands: queries, keys, values, variadic preSoftmaxElemWiseInputs,
  //           optional currentSeqLen, optional prefixOffset
  llvm::SmallVector<MlirValue, 8> operands;
  operands.push_back(queries);
  operands.push_back(keys);
  operands.push_back(values);
  for (intptr_t i = 0; i < numPreSoftmaxInputs; ++i)
    operands.push_back(preSoftmaxElemWiseInputs[i]);
  bool hasCurrentSeqLen = !mlirValueIsNull(currentSeqLen);
  bool hasPrefixOffset = !mlirValueIsNull(prefixOffset);
  if (hasCurrentSeqLen)
    operands.push_back(currentSeqLen);
  if (hasPrefixOffset)
    operands.push_back(prefixOffset);
  mlirOperationStateAddOperands(&state, operands.size(), operands.data());

  // Results: always resultType, optionally lseType
  llvm::SmallVector<MlirType, 2> results;
  results.push_back(resultType);
  if (!mlirTypeIsNull(lseType))
    results.push_back(lseType);
  mlirOperationStateAddResults(&state, results.size(), results.data());

  MlirType i32Type = mlirIntegerTypeGet(ctx, 32);

  // operandSegmentSizes: [1(Q), 1(K), 1(V), numPreSoftmax, hasSeqLen,
  // hasPrefix]
  int32_t segSizes[] = {1,
                        1,
                        1,
                        static_cast<int32_t>(numPreSoftmaxInputs),
                        hasCurrentSeqLen ? 1 : 0,
                        hasPrefixOffset ? 1 : 0};
  MlirNamedAttribute segNamedAttr = mlirNamedAttributeGet(
      mlirIdentifierGet(ctx,
                        mlirStringRefCreateFromCString("operandSegmentSizes")),
      makeSegmentSizesAttr(ctx, segSizes, 6));
  mlirOperationStateAddAttributes(&state, 1, &segNamedAttr);

  // resultSegmentSizes
  int32_t resSizes[] = {1, mlirTypeIsNull(lseType) ? 0 : 1};
  int64_t resDim = 2;
  MlirAttribute resSegAttr = mlirDenseElementsAttrInt32Get(
      mlirRankedTensorTypeGet(1, &resDim, i32Type, mlirAttributeGetNull()), 2,
      resSizes);
  MlirNamedAttribute resSegNamedAttr = mlirNamedAttributeGet(
      mlirIdentifierGet(ctx,
                        mlirStringRefCreateFromCString("resultSegmentSizes")),
      resSegAttr);
  mlirOperationStateAddAttributes(&state, 1, &resSegNamedAttr);

  // Optional softmaxType attribute
  if (!mlirTypeIsNull(softmaxType)) {
    MlirAttribute typeAttr = mlirTypeAttrGet(softmaxType);
    MlirNamedAttribute namedAttr = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("softmaxType")),
        typeAttr);
    mlirOperationStateAddAttributes(&state, 1, &namedAttr);
  }

  // Features attribute
  if (features != 0) {
    MlirNamedAttribute namedAttr =
        makeI32NamedAttr(ctx, "features", static_cast<int32_t>(features));
    mlirOperationStateAddAttributes(&state, 1, &namedAttr);
  }

  // splitKV attribute
  if (splitKV > 1) {
    MlirNamedAttribute namedAttr = makeI32NamedAttr(ctx, "splitKV", splitKV);
    mlirOperationStateAddAttributes(&state, 1, &namedAttr);
  }

  // slidingWindowSize attribute
  if (slidingWindowSize > 0) {
    MlirNamedAttribute namedAttr =
        makeI32NamedAttr(ctx, "slidingWindowSize", slidingWindowSize);
    mlirOperationStateAddAttributes(&state, 1, &namedAttr);
  }

  // preSoftmaxBody region
  mlirOperationStateAddOwnedRegions(&state, 1, &preSoftmaxBody);

  return mlirOperationCreate(&state);
}
