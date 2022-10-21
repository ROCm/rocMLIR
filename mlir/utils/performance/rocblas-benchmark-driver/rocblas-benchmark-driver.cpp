//===- rocblas-benchmark-driver.cpp - RocBLAS benchmark driver --------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

// With much credit to
// https://github.com/ROCmSoftwarePlatform/rocBLAS-Examples/blob/develop/Level-3/gemm_strided_batched/gemm_strided_batched.cpp

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemAlloc.h"

#include "hip/hip_runtime.h"
#include "rocblas/rocblas.h"
// Drop weird backcompat macro
#undef rocblas_gemm_strided_batched_ex

#include <cstdio>

#define ROCBLAS_ABORT_IF_FAIL(expr)                                            \
  do {                                                                         \
    rocblas_status status = expr;                                              \
    if (status != rocblas_status_success) {                                    \
      fprintf(stderr, "rocBLAS error %s at %s:%d in %s\n",                     \
              rocblas_status_to_string(status), __FILE__, __LINE__, #expr);    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define HIP_ABORT_IF_FAIL(expr)                                                \
  do {                                                                         \
    hipError_t error = expr;                                                   \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error %s at %s:%d in %s\n",                         \
              hipGetErrorString(error), __FILE__, __LINE__, #expr);            \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

enum class DataType : uint32_t { F32, F16, BF16, I8 };

static llvm::cl::opt<uint64_t> gemmG("g",
                                     llvm::cl::desc("G dimennsion of gemm()"),
                                     llvm::cl::value_desc("positive integer"),
                                     llvm::cl::init(1));

static llvm::cl::opt<uint64_t> gemmM("m",
                                     llvm::cl::desc("M dimennsion of gemm()"),
                                     llvm::cl::value_desc("positive integer"),
                                     llvm::cl::Required);

static llvm::cl::opt<uint64_t> gemmK("k",
                                     llvm::cl::desc("K dimennsion of gemm()"),
                                     llvm::cl::value_desc("positive integer"),
                                     llvm::cl::Required);

static llvm::cl::opt<uint64_t> gemmN("n",
                                     llvm::cl::desc("N dimennsion of gemm()"),
                                     llvm::cl::value_desc("positive integer"),
                                     llvm::cl::Required);

static llvm::cl::opt<bool>
    transposeA("transA",
               llvm::cl::desc("whether matrix A is GxMxK (default) or GxKxM."),
               llvm::cl::init(false));

static llvm::cl::opt<bool>
    transposeB("transB",
               llvm::cl::desc("whether matrix B is GxKxN (default) or GxNxK."),
               llvm::cl::init(false));

// static llvm::cl::opt<bool>
//    transposeC("transC",
//               llvm::cl::desc("whether matrix C is GxMxN (default) or
//               GxNxM."), llvm::cl::init(false));

static llvm::cl::opt<int>
    kernelRepeats("kernel-repeats",
                  llvm::cl::desc("Number of times to run the rocblas kernel"),
                  llvm::cl::value_desc("positive integer"), llvm::cl::init(1));

static llvm::cl::opt<DataType>
    dataType("t", llvm::cl::desc("Data type"),
             llvm::cl::values(clEnumValN(DataType::F32, "f32", "32-bit float"),
                              clEnumValN(DataType::F16, "f16", "16-bit float"),
                              clEnumValN(DataType::BF16, "bf16", "bfloat16"),
                              clEnumValN(DataType::I8, "i8", "8-bit integer")),
             llvm::cl::Required);

static size_t getByteSize(size_t elems, bool isOut) {
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

static llvm::APFloat::Semantics getLlvmFltSemantics() {
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

static void *allocAndFill(size_t byteSize, bool isOut) {
  uint8_t *ret = reinterpret_cast<uint8_t *>(llvm::safe_malloc(byteSize));

  std::vector<llvm::APInt> intPattern;
  if (dataType != DataType::I8) {
    std::vector<llvm::APFloat> pattern = {
        llvm::APFloat(0.5), llvm::APFloat(-1.0), llvm::APFloat(0.75)};

    llvm::APFloat::Semantics sem = getLlvmFltSemantics();

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

static void *makeHostConstant(float val) {
  llvm::APInt bytes;
  if (dataType != DataType::I8) {
    llvm::APFloat flt(val);
    llvm::APFloat::Semantics sem = getLlvmFltSemantics();
    bool dontCare;
    flt.convert(llvm::APFloat::EnumToSemantics(sem),
                llvm::APFloat::rmNearestTiesToEven, &dontCare);
    bytes = flt.bitcastToAPInt();
  } else {
    bytes = llvm::APInt(32, static_cast<int32_t>(val));
  }
  bytes = bytes.zextOrTrunc(32);
  uint32_t memVal = bytes.getZExtValue();
  uint32_t *ret = reinterpret_cast<uint32_t *>(llvm::safe_malloc(4));
  *ret = memVal;
  return ret;
}

static void *getGpuBuffer(const void *hostMem, size_t byteSize) {
  void *gpuBuffer;
  HIP_ABORT_IF_FAIL(hipMalloc(&gpuBuffer, byteSize));
  HIP_ABORT_IF_FAIL(
      hipMemcpy(gpuBuffer, hostMem, byteSize, hipMemcpyHostToDevice));
  return gpuBuffer;
}

static rocblas_datatype getRocblasType(bool isOut) {
  switch (dataType) {
  case DataType::F32:
    return rocblas_datatype_f32_r;
  case DataType::F16:
    return rocblas_datatype_f16_r;
  case DataType::BF16:
    return rocblas_datatype_bf16_r;
  case DataType::I8:
    return isOut ? rocblas_datatype_i32_r : rocblas_datatype_i8_r;
  }
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "rocMLIR rocBLAS benchmark driver");

  rocblas_operation blasTransA, blasTransB;
  size_t lda, ldb;

  if (transposeA) {
    blasTransA = rocblas_operation_transpose;
    lda = gemmK;
  } else {
    blasTransA = rocblas_operation_none;
    lda = gemmM;
  }
  if (transposeB) {
    blasTransB = rocblas_operation_transpose;
    ldb = gemmN;
  } else {
    blasTransB = rocblas_operation_none;
    ldb = gemmK;
  }
  size_t ldc = gemmM;

  size_t strideA = gemmM * gemmK, strideB = gemmK * gemmN,
         strideC = gemmM * gemmN;
  size_t aElems = strideA * gemmG, bElems = strideB * gemmG,
         cElems = strideC * gemmG;
  size_t aBytes = getByteSize(aElems, false),
         bBytes = getByteSize(bElems, false),
         cBytes = getByteSize(cElems, true);

  rocblas_datatype inType = getRocblasType(false);
  rocblas_datatype outType = getRocblasType(true);

  rocblas_initialize();
  rocblas_handle handle;
  ROCBLAS_ABORT_IF_FAIL(rocblas_create_handle(&handle));

  void *aHost = allocAndFill(aBytes, false);
  void *bHost = allocAndFill(bBytes, false);
  void *cHost = allocAndFill(cBytes, true);

  void *aDevice = getGpuBuffer(aHost, aBytes);
  void *bDevice = getGpuBuffer(bHost, bBytes);
  void *cDevice = getGpuBuffer(cHost, cBytes);

  void *alpha = makeHostConstant(1.0);
  void *beta = makeHostConstant(0.0);

  ROCBLAS_ABORT_IF_FAIL(
      rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

  for (int i = 0, e = kernelRepeats; i < e; ++i)
    ROCBLAS_ABORT_IF_FAIL(rocblas_gemm_strided_batched_ex(
        handle, blasTransA, blasTransB, gemmM, gemmN, gemmK, alpha, aDevice,
        inType, lda, strideA, bDevice, inType, ldb, strideB, beta, cDevice,
        outType, ldc, strideC, cDevice, outType, ldc, strideC, gemmG,
        /*computeType=*/outType, rocblas_gemm_algo_standard, 0,
        rocblas_gemm_flags_none));

  HIP_ABORT_IF_FAIL(hipMemcpy(cHost, cDevice, cBytes, hipMemcpyDeviceToHost));
  ROCBLAS_ABORT_IF_FAIL(rocblas_destroy_handle(handle));
  free(aHost);
  free(bHost);
  free(cHost);
  HIP_ABORT_IF_FAIL(hipFree(aDevice));
  HIP_ABORT_IF_FAIL(hipFree(bDevice));
  HIP_ABORT_IF_FAIL(hipFree(cDevice));
  free(alpha);
  free(beta);
  return 0;
}
