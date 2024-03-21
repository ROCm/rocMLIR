//===- tosa_miir.cpp - Simple test of C and MIIR APIs ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: mlir-mixr-split-k-test -t=DataType::F32 -m 1024 -n 1024 -k 64   2>&1 |
// FileCheck %s --check-prefix=M1024_N1024_K64

// RUN: mlir-mixr-split-k-test -t=DataType::F32 -m 8192 -n 8192 -k 64   2>&1 |
// FileCheck %s --check-prefix=M8192_N8192_K64

// RUN: mlir-mixr-split-k-test -t=DataType::F32 -m 64 -n 64 -k 1024   2>&1 |
// FileCheck %s --check-prefix=M64_N64_K1024

// RUN: mlir-mixr-split-k-test -t=DataType::F32 -split-k 1 -ew-type=none  2>&1 |
// FileCheck %s --check-prefix=F32_EW_NONE_SK1

// RUN: mlir-mixr-split-k-test -t=DataType::F32 -split-k 4 -ew-type=none  2>&1 |
// FileCheck %s --check-prefix=F32_EW_NONE_SK4

// RUN: mlir-mixr-split-k-test -t=DataType::F32 -split-k 4
// -ew-type=ProgramOptions::dependant  2>&1 | FileCheck %s
// --check-prefix=F32_EW_DEPENDANT_SK4

// RUN: mlir-mixr-split-k-test -t=DataType::F32 -split-k 1
// -ew-type=ProgramOptions::dependant  2>&1 | FileCheck %s
// --check-prefix=F32_EW_DEPENDANT_SK1

// RUN: mlir-mixr-split-k-test -t=DataType::F32 -split-k 4
// -ew-type=ProgramOptions::dependant  2>&1 | FileCheck %s
// --check-prefix=F32_EW_INDEPENDANT_SK4

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
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitRocMLIRDialects.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/Support/CommandLine.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <iostream>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

enum class DataType : uint32_t { F32, F16 };

template <DataType ElementType, size_t N>
struct ShapedTensorDescr {
  ~ShapedTensorDescr() = default;

  using IndexType = int64_t;
  inline constexpr size_t getNumDims() { return N; }

  static ShapedTensorDescr<ElementType, N> get(std::vector<IndexType> &&dims) {
    assert(dims.size() == N && "initialization vector must match tensor rank");
    ShapedTensorDescr<ElementType, N> description;
    std::copy_n(dims.begin(), N, description.dims);
    std::exclusive_scan(dims.rbegin(), dims.rend(), description.strides, 1,
                        std::multiplies<>{});
    std::reverse(description.strides, description.strides + N);

    description.size =
        std::accumulate(dims.cbegin(), dims.cend(), 1, std::multiplies<>());

    return description;
  }

  constexpr inline const char *getTypeAsStr() {
    if constexpr (ElementType == DataType::F32) {
      return "f32";
    } else if constexpr (ElementType == DataType::F16) {
      return "f16";
    }
  }

  IndexType dims[N];
  IndexType strides[N];
  IndexType size = 0;

private:
  ShapedTensorDescr(){};
};

template <DataType ElementType, size_t N>
std::ostream &operator<<(std::ostream &stream,
                         ShapedTensorDescr<ElementType, N> &descr) {
  using IndexType = typename ShapedTensorDescr<ElementType, N>::IndexType;
  const std::string typeStr = descr.getTypeAsStr();

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

struct ProgramOptions {
  int64_t M;
  int64_t N;
  int64_t K;
  int64_t splitKFactor;
  std::string targetArch;
  RocmlirTuningParamSetKind tuningLevel;
  int32_t verbosityLevel;

  enum ElementwiseOpType {
    none,
    dependant,
    independant,
  } elementwiseOpType;
};

std::ostream &operator<<(std::ostream &stream,
                         RocmlirSplitKSelectionLikelihood likelihood) {
  switch (likelihood) {
  case RocmlirSplitKSelectionLikelihood::always: {
    stream << "always";
    break;
  }
  case RocmlirSplitKSelectionLikelihood::maybe: {
    stream << "maybe";
    break;
  }
  case RocmlirSplitKSelectionLikelihood::never: {
    stream << "never";
    break;
  }
  }
  return stream;
}

template <typename T>
std::ostream &operator<<(std::ostream &stream, const std::vector<T> &vec) {
  for (size_t i = 0; i < vec.size(); ++i) {
    const char *separator = i == (vec.size() - 1) ? "" : ", ";

    if constexpr (std::is_same_v<T, MlirAttribute>) {
      if (mlirAttributeIsAInteger(vec[i])) {
        stream << mlirIntegerAttrGetValueInt(vec[i]) << separator;
      } else if (mlirAttributeIsAFloat(vec[i])) {
        stream << mlirFloatAttrGetValueDouble(vec[i]) << separator;
      }
    } else {
      stream << vec[i] << separator;
    }
  }
  return stream;
}

template <DataType ElementType>
MlirModule makeAndDumpMIXR(MlirContext ctx, MlirLocation location,
                           const ProgramOptions &options) {
  MlirModule moduleOp = mlirModuleCreateEmpty(location);
  MlirBlock moduleBody = mlirModuleGetBody(moduleOp);

  MlirType mlirFPType;
  if constexpr (ElementType == DataType::F32) {
    mlirFPType = mlirF32TypeGet(ctx);
  } else {
    mlirFPType = mlirF16TypeGet(ctx);
  }

  auto aDescr =
      ShapedTensorDescr<ElementType, 3>::get({1, options.M, options.K});
  MlirType matAType = rocmlirMIXRShapedTypeGet(aDescr.getNumDims(), aDescr.dims,
                                               aDescr.strides, mlirFPType);

  auto bDescr =
      ShapedTensorDescr<ElementType, 3>::get({1, options.K, options.N});
  MlirType matBType = rocmlirMIXRShapedTypeGet(bDescr.getNumDims(), bDescr.dims,
                                               bDescr.strides, mlirFPType);

  MlirType funcBodyArgTypes[] = {matAType, matBType};
  MlirLocation funcBodyLocations[] = {location, location};
  MlirRegion funcBodyRegion = mlirRegionCreate();
  MlirBlock funcBody =
      mlirBlockCreate(sizeof(funcBodyArgTypes) / sizeof(MlirType),
                      funcBodyArgTypes, funcBodyLocations);
  mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody);

  auto cDescr =
      ShapedTensorDescr<ElementType, 3>::get({1, options.M, options.N});
  MlirType matCType = rocmlirMIXRShapedTypeGet(cDescr.getNumDims(), cDescr.dims,
                                               cDescr.strides, mlirFPType);

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
          ctx, mlirStringRefCreateFromCString(options.targetArch.c_str())));

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

  MlirValue returnValue = dotValue;
  if (options.elementwiseOpType != ProgramOptions::ElementwiseOpType::none) {
    //-------------- rely = migraphx.relu
    MlirValue reluOperands[] = {dotValue};
    if (options.elementwiseOpType ==
        ProgramOptions::ElementwiseOpType::independant) {
      reluOperands[0] = funcArg0;
    }
    MlirType reluType = rocmlirMIXRShapedTypeGet(
        cDescr.getNumDims(), cDescr.dims, cDescr.strides, mlirFPType);
    MlirOperationState reluState = mlirOperationStateGet(
        mlirStringRefCreateFromCString("migraphx.relu"), location);
    mlirOperationStateAddResults(&reluState, 1, &reluType);
    mlirOperationStateAddOperands(&reluState, 1, reluOperands);

    MlirOperation reluOp = mlirOperationCreate(&reluState);
    mlirBlockAppendOwnedOperation(funcBody, reluOp);
    MlirValue reulResult = mlirOperationGetResult(reluOp, 0);
    if (options.elementwiseOpType ==
        ProgramOptions::ElementwiseOpType::dependant) {
      returnValue = reulResult;
    }
  }

  //-------------- func.return
  MlirValue retOperands[] = {returnValue};
  MlirOperationState retState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("func.return"), location);
  mlirOperationStateAddOperands(&retState, 1, retOperands);
  MlirOperation ret = mlirOperationCreate(&retState);
  mlirBlockAppendOwnedOperation(funcBody, ret);

  return moduleOp;
}

template <DataType ElementType>
static bool constructAndTraverseIr(MlirContext ctx,
                                   const ProgramOptions &options) {
  // generate initial migraphx mlir code
  MlirLocation location = mlirLocationUnknownGet(ctx);

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.setf(std::ios::boolalpha);
  std::cout.precision(2);

  // descr -> splitK hint using only high-level information
  const int64_t numCUs =
      mlir::rock::lookupArchInfo(options.targetArch).minNumCU;
  const int64_t numGroups = 1;
  RocmlirSplitKSelectionLikelihood likelihood = mlirIsSplitKFaster(
      numGroups, options.M, options.N, options.K, numCUs, options.tuningLevel);
  std::cout << "splitk selection likelihood: " << likelihood << std::endl;
  // M1024_N1024_K64: splitk selection likelihood: maybe
  // M8192_N8192_K64: splitk selection likelihood: never
  // M64_N64_K1024: splitk selection likelihood: always

  auto moduleOp = CRAIIWrapper<MlirModule>(
      makeAndDumpMIXR<ElementType>(ctx, location, options));

  // print the initial migraphx mlir code
  MlirOperation moduleMO = mlirModuleGetOperation(moduleOp.get());
  if (options.verbosityLevel > 0)
    mlirOperationDump(moduleMO);

  // run high level pipeline
  auto pm0 = CRAIIWrapper<MlirPassManager>(mlirPassManagerCreate(ctx));
  mlirMIGraphXAddHighLevelPipeline(pm0.get());
  MlirLogicalResult status0 = mlirPassManagerRunOnOp(pm0.get(), moduleMO);
  if (mlirLogicalResultIsFailure(status0)) {
    std::cerr << "Highlevel Pipeline failed" << std::endl;
    return false;
  }
  if (options.verbosityLevel > 0)
    mlirOperationDump(moduleMO);

  std::stringstream stream;
  stream << "v2:64,64,16,32,32,4," << options.splitKFactor << ",1,1";
  std::string streamStr = stream.str() + "\0";
  MlirStringRef perfStr = mlirStringRefCreateFromCString(streamStr.data());

  // descr -> test whether the generated migraphx module is fusible
  const bool isFusible = mlirIsModuleFusible(moduleOp.get(), perfStr);
  std::cout << "is fusible: " << isFusible << std::endl;
  // F32_EW_NONE_SK1: is fusible: true
  // F32_EW_NONE_SK4: is fusible: true
  // F32_EW_DEPENDANT_SK4: is fusible: false
  // F32_EW_DEPENDANT_SK1: is fusible: true
  // F32_EW_INDEPENDANT_SK4: is fusible: true

  // set perf config string
  bool isOk = mlirRockTuningSetFromStr(moduleOp.get(), perfStr);
  if (!isOk) {
    std::cerr << "failed to set the perfConfig string" << std::endl;
    return false;
  }

  if (options.verbosityLevel > 1) {
    // run applicability pipeline
    auto pm1 = CRAIIWrapper<MlirPassManager>(mlirPassManagerCreate(ctx));
    mlirMIGraphXAddApplicabilityPipeline(pm1.get());
    MlirLogicalResult status1 = mlirPassManagerRunOnOp(pm1.get(), moduleMO);
    if (mlirLogicalResultIsFailure(status1)) {
      std::cerr << "Applicability Pipeline failed" << '\n';
      return false;
    }
    mlirOperationDump(moduleMO);
  }

  if (isFusible) {
    auto pm2 = CRAIIWrapper<MlirPassManager>(mlirPassManagerCreate(ctx));
    mlirMIGraphXAddBackendPipeline(pm2.get(), options.targetArch.c_str());
    MlirLogicalResult status2 = mlirPassManagerRunOnOp(pm2.get(), moduleMO);
    if (mlirLogicalResultIsFailure(status2)) {
      std::cerr << "Backend Pipeline failed" << '\n';
      return false;
    }
    if (options.verbosityLevel > 2)
      mlirOperationDump(moduleMO);
  }

  // descr -> prefill args logic
  size_t numPrefillArgs = mlirGetNumPrefillArgs(moduleOp.get());
  std::cout << "num prefill args: " << numPrefillArgs << '\n';
  // F32_EW_NONE_SK1" num prefill args: 0
  // F32_EW_NONE_SK4: num prefill args: 1

  std::vector<size_t> prefillArgIndices(numPrefillArgs);
  std::vector<MlirAttribute> prefillArgValues(numPrefillArgs);
  mlirGetPrefillArgsInfo(moduleOp.get(), prefillArgIndices.data(),
                         prefillArgValues.data());

  std::cout << "prefill arg indices: " << prefillArgIndices << '\n';
  // F32_EW_NONE_SK1" prefill arg indices:
  // F32_EW_NONE_SK4: prefill arg indices: 2

  std::cout << "prefill arg init values: " << prefillArgValues << '\n';
  // F32_EW_NONE_SK1: prefill arg init values:
  // F32_EW_NONE_SK4: prefill arg init values: 0.00

  // descr -> auxiliary buffers logic
  size_t numAuxBuffers = mlirGetNumAuxBuffers(moduleOp.get());
  std::cout << "num aux buffers: " << numAuxBuffers << '\n';
  // F32_EW_NONE_SK1: num aux buffers: 0
  // F32_EW_NONE_SK4: num aux buffers: 0

  std::vector<size_t> auxBuffersSizes(numAuxBuffers);
  std::vector<MlirAttribute> auxBuffersInitValues(numAuxBuffers);
  mlirGetAuxBuffersInfo(moduleOp.get(), auxBuffersSizes.data(),
                        auxBuffersInitValues.data());

  std::cout << "aux buffers sizes: " << auxBuffersSizes << '\n';
  // F32_EW_NONE_SK1: aux buffers sizes:
  // F32_EW_NONE_SK4: aux buffers sizes:

  std::cout << "aux buffers init values: " << auxBuffersInitValues << '\n';
  // F32_EW_NONE_SK1: aux buffers init values:
  // F32_EW_NONE_SK4: aux buffers init values:

  return true;
}

llvm::cl::opt<int64_t> mDim("m", llvm::cl::desc("M dimension"),
                            llvm::cl::init(64));
llvm::cl::opt<int64_t> nDim("n", llvm::cl::desc("M dimension"),
                            llvm::cl::init(64));
llvm::cl::opt<int64_t> kDim("k", llvm::cl::desc("M dimension"),
                            llvm::cl::init(1024));
llvm::cl::opt<int64_t>
    splitKFactor("split-k", llvm::cl::desc("num splits along k-th dimension"),
                 llvm::cl::init(4));

llvm::cl::opt<DataType>
    dataType("t", llvm::cl::desc("used data type"),
             llvm::cl::values(clEnumVal(DataType::F32, "f32"),
                              clEnumVal(DataType::F16, "f16")),
             llvm::cl::init(DataType::F32));

llvm::cl::opt<std::string> targetArch("target-arch",
                                      llvm::cl::desc("target architecture"),
                                      llvm::cl::init("gfx908:sramecc+:xnack-"));

llvm::cl::opt<RocmlirTuningParamSetKind> tuningLevel(
    "tuning-kind", llvm::cl::desc("tuning level"),
    llvm::cl::values(clEnumVal(RocmlirTuningParamSetKindQuick, "Quick"),
                     clEnumVal(RocmlirTuningParamSetKindFull, "Full"),
                     clEnumVal(RocmlirTuningParamSetKindExhaustive,
                               "Exhaustive")),
    llvm::cl::init(
        RocmlirTuningParamSetKind::RocmlirTuningParamSetKindExhaustive));

llvm::cl::opt<ProgramOptions::ElementwiseOpType> elementwiseOpType(
    "ew-type", llvm::cl::desc("elementwise operaion appendend to gemm/conv"),
    llvm::cl::values(clEnumVal(ProgramOptions::none, "none"),
                     clEnumVal(ProgramOptions::dependant, "dependant"),
                     clEnumVal(ProgramOptions::independant, "independant")),
    llvm::cl::init(ProgramOptions::ElementwiseOpType::none));

llvm::cl::opt<int32_t> verbosityLevel("v", llvm::cl::desc("verbosity level"),
                                      llvm::cl::init(0));

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  ProgramOptions options{
      mDim.getValue(),           nDim.getValue(),
      kDim.getValue(),           splitKFactor.getValue(),
      targetArch.getValue(),     tuningLevel.getValue(),
      verbosityLevel.getValue(), elementwiseOpType.getValue(),
  };

  auto ctx = CRAIIWrapper<MlirContext>(mlirContextCreate());
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirRegisterRocMLIRDialects(registry);
  mlirRegisterRocMLIRPasses();
  mlirContextAppendDialectRegistry(ctx.get(), registry);
  mlirContextLoadAllAvailableDialects(ctx.get());
  mlirDialectRegistryDestroy(registry);

  bool isOk = true;
  switch (dataType.getValue()) {
  case DataType::F32: {
    isOk = constructAndTraverseIr<DataType::F32>(ctx.get(), options);
    break;
  }
  case DataType::F16: {
    isOk = constructAndTraverseIr<DataType::F16>(ctx.get(), options);
    break;
  }
  }
  if (!isOk) {
    printf("FAILED!\n");
    return 1;
  }

  return 0;
}
