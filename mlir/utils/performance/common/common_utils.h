//===- common_utils.h common utility functions shared between the drivers
//--------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemAlloc.h"

#include "hip/hip_runtime.h"

enum class DataType : uint32_t { F32, F16, BF16, I8 };

struct BatchInfo {
  size_t groups;
  size_t batchStrideA;
  size_t batchStrideB;
  size_t batchStrideC;
};

struct BenchmarkArgs {
  llvm::cl::opt<uint64_t> gemmG{"g", llvm::cl::desc("G dimennsion of gemm()"),
                                llvm::cl::value_desc("positive integer"),
                                llvm::cl::init(1)};

  llvm::cl::opt<uint64_t> gemmM{"m", llvm::cl::desc("M dimennsion of gemm()"),
                                llvm::cl::value_desc("positive integer"),
                                llvm::cl::Required};

  llvm::cl::opt<uint64_t> gemmK{"k", llvm::cl::desc("K dimennsion of gemm()"),
                                llvm::cl::value_desc("positive integer"),
                                llvm::cl::Required};

  llvm::cl::opt<uint64_t> gemmN{"n", llvm::cl::desc("N dimennsion of gemm()"),
                                llvm::cl::value_desc("positive integer"),
                                llvm::cl::Required};

  llvm::cl::opt<bool> transposeA{
      "transA", llvm::cl::desc("whether matrix A is GxMxK (default) or GxKxM."),
      llvm::cl::init(false)};

  llvm::cl::opt<bool> transposeB{
      "transB", llvm::cl::desc("whether matrix B is GxKxN (default) or GxNxK."),
      llvm::cl::init(false)};

  llvm::cl::opt<int> kernelRepeats{
      "kernel-repeats",
      llvm::cl::desc("Number of times to run the rocblas kernel"),
      llvm::cl::value_desc("positive integer"), llvm::cl::init(1)};

  llvm::cl::opt<DataType> dataType{
      "t", llvm::cl::desc("Data type"),
      llvm::cl::values(clEnumValN(DataType::F32, "f32", "32-bit float"),
                       clEnumValN(DataType::F16, "f16", "16-bit float"),
                       clEnumValN(DataType::BF16, "bf16", "bfloat16"),
                       clEnumValN(DataType::I8, "i8", "8-bit integer")),
      llvm::cl::Required};

  llvm::cl::opt<std::string> operation{
      "operation", llvm::cl::desc("Operation (ignored, for MLIR compat)"),
      llvm::cl::init("gemm")};

  llvm::cl::opt<std::string> arch{"arch",
                                  llvm::cl::desc("Arch (ignored, MLIR compat)"),
                                  llvm::cl::init("gemm")};
   llvm::cl::opt<std::string>
    perfConfig("perf_config",
               llvm::cl::desc("Perf config (ignored, MLIR compat)"),
               llvm::cl::init(""));

};

#define HIP_ABORT_IF_FAIL(expr)                                                \
  do {                                                                         \
    hipError_t error = expr;                                                   \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error %s at %s:%d in %s\n",                         \
              hipGetErrorString(error), __FILE__, __LINE__, #expr);            \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static llvm::APFloat::Semantics getLlvmFltSemantics(DataType dataType) {
  switch (dataType) {
  case DataType::F32:
    return llvm::APFloat::S_IEEEsingle;
  case DataType::F16:
    return llvm::APFloat::S_IEEEhalf;
  case DataType::BF16:
    return llvm::APFloat::S_BFloat;
  case DataType::I8:
    assert(0 && "Can't have i8 floats");
  }
}

static void *allocAndFill(DataType dataType, size_t byteSize, bool isOut) {
  uint8_t *ret = reinterpret_cast<uint8_t *>(llvm::safe_malloc(byteSize));
  std::vector<llvm::APInt> intPattern;
  if (dataType != DataType::I8) {
    std::vector<llvm::APFloat> pattern = {
        llvm::APFloat(0.5), llvm::APFloat(-1.0), llvm::APFloat(0.75)};

    llvm::APFloat::Semantics sem = getLlvmFltSemantics(dataType);

    for (auto &flt : pattern) {
      bool dontCare = false;
      flt.convert(llvm::APFloat::EnumToSemantics(sem),
                  llvm::APFloat::rmNearestTiesToEven, &dontCare);
      intPattern.push_back(flt.bitcastToAPInt());
    }
  } else { // int8
    size_t bitWidth = (isOut ? 32 : 8);
    for (int64_t i : {1, -1, 2}) {
      intPattern.emplace_back(bitWidth, i);
    }
  }

  size_t bytesPerElem = intPattern[0].getBitWidth() / 8;
  size_t elems = byteSize / bytesPerElem;
  for (size_t i = 0; i < elems; ++i) {
    const llvm::APInt &elem = intPattern[i % intPattern.size()];
    for (size_t byte = 0; i < bytesPerElem; ++i) {
      uint8_t value = elem.extractBitsAsZExtValue(8, byte * 8);
      ret[byte + bytesPerElem * i] = value;
    }
  }
  return ret;
}

static size_t getByteSize(DataType dataType, size_t elems, bool isOut) {
  switch (dataType) {
  case DataType::F32:
    return elems * 4;
  case DataType::F16:
  case DataType::BF16:
    return elems * 2;
  case DataType::I8:
    return elems * (isOut ? 4 : 1);
  }
}

static void *getGpuBuffer(const void *hostMem, size_t byteSize) {
  void *gpuBuffer;
  HIP_ABORT_IF_FAIL(hipMalloc(&gpuBuffer, byteSize));
  HIP_ABORT_IF_FAIL(
      hipMemcpy(gpuBuffer, hostMem, byteSize, hipMemcpyHostToDevice));
  return gpuBuffer;
}
