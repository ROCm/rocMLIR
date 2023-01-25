//===-------- benchmarkUtils.h - common benchmark utility functions -------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_UTILS_PERFORMANCE_COMMON_BENCHMARKUTILS_H
#define MLIR_UTILS_PERFORMANCE_COMMON_BENCHMARKUTILS_H

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemAlloc.h"

#include "hip/hip_runtime.h"

enum class DataType : uint32_t { F32, F16, BF16, I8 };

// Common options to the different benchmark drivers
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
  llvm::cl::opt<std::string> perfConfig{
      "perf_config", llvm::cl::desc("Perf config (ignored, MLIR compat)"),
      llvm::cl::init("")};
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

// Map custom data types to LLVM data types
llvm::APFloat::Semantics getLlvmFltSemantics(DataType dataType);

// Allocate and fill an area of memory
void *allocAndFill(DataType dataType, size_t byteSize, bool isOut);

// Return sizeof(dataType) in bytes
size_t getByteSize(DataType dataType, size_t elems, bool isOut);

// Allocate a device buffer and copy the date from host memory
void *getGpuBuffer(const void *hostMem, size_t byteSize);

#endif // MLIR_UTILS_PERFORMANCE_COMMON_BENCHMARKUTILS_H