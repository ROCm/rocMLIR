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

#include "../common/common_utils.h"

#include "rocblas/rocblas.h"
// Drop weird backcompat macro
#undef rocblas_gemm_strided_batched_ex
#undef rocblas_gemm_ex

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

static void *makeHostConstant(float val, DataType dataType) {
  llvm::APInt bytes;
  if (dataType != DataType::I8) {
    llvm::APFloat flt(val);
    llvm::APFloat::Semantics sem = getLlvmFltSemantics(dataType);
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

static rocblas_datatype getRocblasType(bool isOut, DataType dataType) {
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
  auto args = BenchmarkArgs();
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "rocMLIR rocBLAS benchmark driver");

  rocblas_operation blasTransA, blasTransB;
  size_t lda, ldb;

  // Please note: rocMLIR is using a row-major format
  // to store matrices, while rocBLAS is using a
  // column-major format. This means that transposition concepts are
  // inverted
  if (!args.transposeA) {
    blasTransA = rocblas_operation_transpose;
    lda = args.gemmK;
  } else {
    blasTransA = rocblas_operation_none;
    lda = args.gemmM;
  }
  if (!args.transposeB) {
    blasTransB = rocblas_operation_transpose;
    ldb = args.gemmN;
  } else {
    blasTransB = rocblas_operation_none;
    ldb = args.gemmK;
  }
  size_t ldc = args.gemmM;

  size_t strideA = args.gemmM * args.gemmK, strideB = args.gemmK * args.gemmN,
         strideC = args.gemmM * args.gemmN;
  size_t aElems = strideA * args.gemmG, bElems = strideB * args.gemmG,
         cElems = strideC * args.gemmG;
  size_t aBytes = getByteSize(args.dataType, aElems, false),
         bBytes = getByteSize(args.dataType, bElems, false),
         cBytes = getByteSize(args.dataType, cElems, true);

  rocblas_datatype inType = getRocblasType(false, args.dataType);
  rocblas_datatype outType = getRocblasType(true, args.dataType);

  rocblas_initialize();
  rocblas_handle handle;
  ROCBLAS_ABORT_IF_FAIL(rocblas_create_handle(&handle));

  void *aHost = allocAndFill(args.dataType, aBytes, false);
  void *bHost = allocAndFill(args.dataType, bBytes, false);
  void *cHost = allocAndFill(args.dataType, cBytes, true);

  void *aDevice = getGpuBuffer(aHost, aBytes);
  void *bDevice = getGpuBuffer(bHost, bBytes);
  void *cDevice = getGpuBuffer(cHost, cBytes);

  void *alpha = makeHostConstant(1.0, args.dataType);
  void *beta = makeHostConstant(0.0, args.dataType);

  ROCBLAS_ABORT_IF_FAIL(
      rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

  for (int i = 0, e = args.kernelRepeats; i < e; ++i)
    if (args.gemmG == 1) {
      ROCBLAS_ABORT_IF_FAIL(rocblas_gemm_ex(
          handle, blasTransA, blasTransB, args.gemmM, args.gemmN, args.gemmK,
          alpha, aDevice, inType, lda, bDevice, inType, ldb, beta, cDevice,
          outType, ldc, cDevice, outType, ldc,
          /*computeType=*/outType, rocblas_gemm_algo_standard, 0,
          rocblas_gemm_flags_none));

    } else {
      ROCBLAS_ABORT_IF_FAIL(rocblas_gemm_strided_batched_ex(
          handle, blasTransA, blasTransB, args.gemmM, args.gemmN, args.gemmK,
          alpha, aDevice, inType, lda, strideA, bDevice, inType, ldb, strideB,
          beta, cDevice, outType, ldc, strideC, cDevice, outType, ldc, strideC,
          args.gemmG,
          /*computeType=*/outType, rocblas_gemm_algo_standard, 0,
          rocblas_gemm_flags_none));
    }

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
