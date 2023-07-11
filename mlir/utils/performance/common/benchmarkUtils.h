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

#include "hip/hip_runtime.h"

// Common options to the different benchmark drivers

namespace benchmark {

enum class DataType : uint32_t { F32, F16, BF16, I8, UNKNOWN };
struct BenchmarkArgs {
  uint64_t gemmG{0};
  uint64_t gemmM{0};
  uint64_t gemmK{0};
  uint64_t gemmN{0};
  DataType dataType{DataType::UNKNOWN};

  bool transposeA{false};
  bool transposeB{false};
  int kernelRepeats{1};
};

// Parse command line arguments
BenchmarkArgs parseCommandLine(const std::string &name, int argc, char **argv);

// Display the problem we are testing (useful for debug)
void printProblem(BenchmarkArgs args);

#define HIP_ABORT_IF_FAIL(expr)                                                \
  do {                                                                         \
    hipError_t error = expr;                                                   \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error %s at %s:%d in %s\n",                         \
              hipGetErrorString(error), __FILE__, __LINE__, #expr);            \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Allocate and fill an area of memory on the host
void *allocAndFill(DataType dataType, size_t byteSize, bool isOut);

// Allocate and fill a single constant on the host
void *makeHostConstant(float val, DataType dataType);

// Return sizeof(dataType)*elems in bytes
size_t getByteSize(DataType dataType, size_t elems, bool isOut);

// Return sizeof(dataType)
size_t getBytesPerElement(DataType dataType, bool isOut);

// Allocate a device buffer and copy the date from host memory
void *getGpuBuffer(const void *hostMem, size_t byteSize);

// Get the GPU device name
std::string get_device_name();

} // namespace benchmark

#endif // MLIR_UTILS_PERFORMANCE_COMMON_BENCHMARKUTILS_H
