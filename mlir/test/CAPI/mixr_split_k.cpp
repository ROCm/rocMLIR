//===- tosa_miir.cpp - Simple test of C and MIIR APIs ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: mlir-mixr-split-k-test 2>&1 | FileCheck %s
 *  */
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/GPU.h"
#include "mlir-c/Dialect/MIGraphX.h"
#include "mlir-c/Dialect/Rock.h"
#include "mlir-c/Dialect/Tosa.h"
#include "mlir-c/IR.h"
#include "mlir-c/IntegerSet.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir-c/RegisterRocMLIR.h"

#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/MIGraphX/Pipeline/Pipeline.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitRocMLIRDialects.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

void printToString(MlirStringRef str, void *userData) {
  std::string *strref = static_cast<std::string *>(userData);
  strref->append(str.data, str.length);
}

template <typename T, size_t N>
struct ShapedTensorDescr {
  ~ShapedTensorDescr() = default;

  using IndexType = int64_t;
  inline constexpr size_t getNumDims() { return N; }

  static ShapedTensorDescr<T, N> get(std::vector<IndexType> &&dims,
                                     T initValue) {
    assert(dims.size() == N && "initialization vector must match tensor rank");
    ShapedTensorDescr<T, N> description;
    std::copy_n(dims.begin(), N, description.dims);
    std::exclusive_scan(dims.rbegin(), dims.rend(), description.strides, 1,
                        std::multiplies<>{});
    std::reverse(description.strides, description.strides + N);

    description.size =
        std::accumulate(dims.cbegin(), dims.cend(), 1, std::multiplies<>());

    return description;
  }

  IndexType dims[N];
  IndexType strides[N];
  IndexType size = 0;

private:
  ShapedTensorDescr(){};
};

template <typename T, size_t N>
std::ostream &operator<<(std::ostream &stream, ShapedTensorDescr<T, N> &descr) {
  using IndexType = typename ShapedTensorDescr<T, N>::IndexType;

  std::string typeStr = "?";
  if constexpr (std::is_same_v<T, float>) {
    typeStr = "f32";
  }

  stream << "!migraphx.shaped<";
  std::for_each(descr.dims, descr.dims + N,
                [&stream](IndexType dim) { stream << dim << 'x'; });
  stream << typeStr << ", ";

  size_t counter = 0;
  std::for_each(descr.strides, descr.strides + N,
                [&stream, &counter](IndexType stride) {
                  const char *separator = (counter < N - 1) ? "x" : "";
                  stream << stride << separator;
                  ++counter;
                });

  stream << ">";
  return stream;
}

template <typename T>
struct CRAIIWrapper {
  CRAIIWrapper(T entity) : entity(entity){};
  CRAIIWrapper() = delete;
  ~CRAIIWrapper() {
    if constexpr (std::is_same_v<T, MlirContext>) {
      mlirContextDestroy(entity);
    } else if constexpr (std::is_same_v<T, MlirModule>) {
      mlirModuleDestroy(entity);
    } else if constexpr (std::is_same_v<T, MlirPassManager>) {
      mlirPassManagerDestroy(entity);
    }
  }

  T &get() { return entity; }

private:
  T entity;
};

MlirModule makeAndDumpMIXR(MlirContext ctx, MlirLocation location) {
  MlirModule moduleOp = mlirModuleCreateEmpty(location);
  MlirBlock moduleBody = mlirModuleGetBody(moduleOp);

  MlirType f32Type = mlirF32TypeGet(ctx);

  auto aDescr = ShapedTensorDescr<float, 3>::get({1, 64, 1024}, 1.0f);
  MlirType matAType = rocmlirMIXRShapedTypeGet(aDescr.getNumDims(), aDescr.dims,
                                               aDescr.strides, f32Type);

  auto bDescr = ShapedTensorDescr<float, 3>::get({1, 1024, 64}, 1.0f);
  MlirType matBType = rocmlirMIXRShapedTypeGet(bDescr.getNumDims(), bDescr.dims,
                                               bDescr.strides, f32Type);

  MlirType funcBodyArgTypes[] = {matAType, matBType};
  MlirLocation funcBodyLocations[] = {location, location};
  MlirRegion funcBodyRegion = mlirRegionCreate();
  MlirBlock funcBody =
      mlirBlockCreate(sizeof(funcBodyArgTypes) / sizeof(MlirType),
                      funcBodyArgTypes, funcBodyLocations);
  mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody);

  auto cDescr = ShapedTensorDescr<float, 3>::get({1, 64, 64}, 1.0f);
  MlirType matCType = rocmlirMIXRShapedTypeGet(cDescr.getNumDims(), cDescr.dims,
                                               cDescr.strides, f32Type);

  //-------------- func op

  // Set func attributes
  std::stringstream stream;
  stream << "(" << aDescr << ", " << bDescr << ") -> (" << cDescr << ")";
  MlirAttribute funcTypeAttr = mlirAttributeParseGet(
      ctx, mlirStringRefCreateFromCString(stream.str().c_str()));
  MlirAttribute funcNameAttr =
      mlirAttributeParseGet(ctx, mlirStringRefCreateFromCString("\"main\""));
  MlirNamedAttribute funcAttrs[] = {
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx,
                            mlirStringRefCreateFromCString("function_type")),
          funcTypeAttr),
      mlirNamedAttributeGet(
          mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("sym_name")),
          funcNameAttr)};

  // Set func op
  MlirOperationState funcState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("func.func"), location);
  mlirOperationStateAddAttributes(&funcState, 2, funcAttrs);
  mlirOperationStateAddOwnedRegions(&funcState, 1, &funcBodyRegion);
  MlirOperation func = mlirOperationCreate(&funcState);
  mlirBlockInsertOwnedOperation(moduleBody, 0, func);

  // set additional attributes
  mlirOperationSetAttributeByName(
      func, mlirStringRefCreateFromCString("kernel"), mlirUnitAttrGet(ctx));
  mlirOperationSetAttributeByName(
      func, mlirStringRefCreateFromCString("arch"),
      mlirStringAttrGet(
          ctx, mlirStringRefCreateFromCString("gfx908:sramecc+:xnack-")));

  //-------------- dot = migraphx.dot
  MlirValue funcArg0 = mlirBlockGetArgument(funcBody, 0);
  MlirValue funcArg1 = mlirBlockGetArgument(funcBody, 1);
  MlirValue dotOperands[] = {funcArg0, funcArg1};

  MlirOperationState dotOpState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("migraphx.dot"), location);
  mlirOperationStateAddResults(&dotOpState, 1, &matCType);
  mlirOperationStateAddOperands(&dotOpState, 2, dotOperands);
  MlirOperation dotOp = mlirOperationCreate(&dotOpState);
  mlirBlockAppendOwnedOperation(funcBody, dotOp);
  MlirValue dotValue = mlirOperationGetResult(dotOp, 0);

  MlirValue retOperands[] = {dotValue};
  MlirOperationState retState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("func.return"), location);
  mlirOperationStateAddOperands(&retState, 1, retOperands);
  MlirOperation ret = mlirOperationCreate(&retState);
  mlirBlockAppendOwnedOperation(funcBody, ret);

  return moduleOp;
}

static bool constructAndTraverseIr(MlirContext ctx) {
  // generate initial migraphx mlir code
  MlirLocation location = mlirLocationUnknownGet(ctx);
  auto moduleOp1 = CRAIIWrapper<MlirModule>(makeAndDumpMIXR(ctx, location));

  // print the initial migraphx mlir code
  MlirOperation moduleMO = mlirModuleGetOperation(moduleOp1.get());
  mlirOperationDump(moduleMO);

  // run high level pipeline
  auto pm0 = CRAIIWrapper<MlirPassManager>(mlirPassManagerCreate(ctx));
  mlirMIGraphXAddHighLevelPipeline(pm0.get());
  MlirLogicalResult status0 = mlirPassManagerRunOnOp(pm0.get(), moduleMO);
  if (mlirLogicalResultIsFailure(status0)) {
    std::cerr << "Highlevel Pipeline failed" << std::endl;
    return false;
  }
  mlirOperationDump(moduleMO);

  // set perf config string
  MlirStringRef perfStr =
      mlirStringRefCreateFromCString("v2:64,64,16,32,32,4,4,1,1\0");
  mlirRockTuningSetFromStr(moduleOp1.get(), perfStr);

  // run applicability pipeline
  auto pm1 = CRAIIWrapper<MlirPassManager>(mlirPassManagerCreate(ctx));
  mlirMIGraphXAddApplicabilityPipeline(pm1.get());
  MlirLogicalResult status1 = mlirPassManagerRunOnOp(pm1.get(), moduleMO);
  if (mlirLogicalResultIsFailure(status1)) {
    std::cerr << "Applicability Pipeline failed" << std::endl;
    return false;
  }
  mlirOperationDump(moduleMO);
  return true;
}

int main() {
  auto ctx = CRAIIWrapper<MlirContext>(mlirContextCreate());
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirRegisterRocMLIRDialects(registry);
  mlirRegisterRocMLIRPasses();
  mlirContextAppendDialectRegistry(ctx.get(), registry);
  mlirContextLoadAllAvailableDialects(ctx.get());
  mlirDialectRegistryDestroy(registry);

  bool status = constructAndTraverseIr(ctx.get());
  if (!status) {
    printf("FAILED!\n");
    return 1;
  }

  return 0;
}
