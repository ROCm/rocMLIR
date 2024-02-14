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
// https://github.com/ROCm/rocBLAS-Examples/blob/develop/Level-3/gemm_strided_batched/gemm_strided_batched.cpp

// Include common utility functions
#include "../common/benchmarkUtils.h"

#include "rocblas/internal/rocblas-beta.h"
#include "rocblas/rocblas.h"
#include <hip/hip_runtime_api.h>
#include <rocblas/internal/rocblas-types.h>
// Drop weird backcompat macro
#undef rocblas_gemm_strided_batched_ex
#undef rocblas_gemm_ex

#include <cstdio>
#include <iostream>

#define ROCBLAS_ABORT_IF_FAIL(expr)                                            \
  do {                                                                         \
    rocblas_status status = expr;                                              \
    if (status != rocblas_status_success) {                                    \
      fprintf(stderr, "rocBLAS error %s at %s:%d in %s\n",                     \
              rocblas_status_to_string(status), __FILE__, __LINE__, #expr);    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static rocblas_datatype getRocblasType(benchmark::DataType dataType) {
  switch (dataType) {
  case benchmark::DataType::F32:
    return rocblas_datatype_f32_r;
  case benchmark::DataType::I32:
    return rocblas_datatype_i32_r;
  case benchmark::DataType::F16:
    return rocblas_datatype_f16_r;
  case benchmark::DataType::BF16:
    return rocblas_datatype_bf16_r;
  case benchmark::DataType::I8:
    return rocblas_datatype_i8_r;
  case benchmark::DataType::F8:
    return rocblas_datatype_f8_r;
  case benchmark::DataType::UNKNOWN:
    assert(0 && "Data type unknown");
  }
}

/// This code is taken from
/// https://github.com/ROCm/AMDMIGraphX/blob/84a8f450f20521242b74bd803824673739d6e858/src/targets/gpu/rocblas.cpp
/// and is used to set the GEMM compute type (i.e., the type of the GEMM
/// accumulation buffer)
bool get_compute_fp32_flag() {
  const auto device_name = benchmark::get_device_name();
  return (device_name.find("gfx908") != std::string::npos ||
          device_name.find("gfx90a") != std::string::npos ||
          device_name.find("gfx94") != std::string::npos);
}

benchmark::DataType getComputeType(benchmark::DataType inputType,
                                   benchmark::DataType outputType) {
  if (inputType == benchmark::DataType::F8)
    return benchmark::DataType::F32;
  if ((inputType == benchmark::DataType::F16 ||
       inputType == benchmark::DataType::BF16) &&
      get_compute_fp32_flag())
    return benchmark::DataType::F32;
  if (inputType == benchmark::DataType::I8)
    return benchmark::DataType::I32;
  return outputType;
}

int main(int argc, char **argv) {
  auto args =
      benchmark::parseCommandLine("rocblas-benchmark-driver", argc, argv);

  rocblas_operation blasTransA, blasTransB;
  size_t lda, ldb;

  // Please note: MIGraphx and MLIR are using row-major format
  // to store matrices, while rocBLAS is using a colum-major format.
  // To be compliant to rocBLAS format, MIGraphx swaps the inputs
  // and tells rocBLAS that B is nxk and A is kxm. So the result
  // will be a nxm matrix stored in colum-major order. We can simply
  // recover the original matrix C by reading the matrix in a row-major
  // way.
  if (args.transposeA) {
    blasTransA = rocblas_operation_transpose;
    lda = args.gemmM;
  } else {
    blasTransA = rocblas_operation_none;
    lda = args.gemmK;
  }
  if (args.transposeB) {
    blasTransB = rocblas_operation_transpose;
    ldb = args.gemmK;
  } else {
    blasTransB = rocblas_operation_none;
    ldb = args.gemmN;
  }
  size_t ldc = args.gemmN;

  size_t strideA = args.gemmM * args.gemmK, strideB = args.gemmK * args.gemmN,
         strideC = args.gemmM * args.gemmN;
  size_t aElems = strideA * args.gemmG, bElems = strideB * args.gemmG,
         cElems = strideC * args.gemmG;
  size_t aBytes = getByteSize(args.dataType, aElems),
         bBytes = getByteSize(args.dataType, bElems),
         cBytes = getByteSize(benchmark::DataType::F32, cElems);

  benchmark::DataType computeDataType =
      getComputeType(args.dataType, args.outDataType);

  rocblas_datatype inType = getRocblasType(args.dataType);
  rocblas_datatype outType = getRocblasType(args.outDataType);
  rocblas_datatype computeType = getRocblasType(computeDataType);

  rocblas_initialize();
  rocblas_handle handle;
  ROCBLAS_ABORT_IF_FAIL(rocblas_create_handle(&handle));

  void *aHost = benchmark::allocAndFill(args.dataType, aBytes);
  void *bHost = benchmark::allocAndFill(args.dataType, bBytes);
  void *cHost = benchmark::allocAndFill(args.outDataType, cBytes);

  void *aDevice = benchmark::getGpuBuffer(aHost, aBytes);
  void *bDevice = benchmark::getGpuBuffer(bHost, bBytes);
  void *cDevice = benchmark::getGpuBuffer(cHost, cBytes);

  void *alpha = makeHostConstant(1.0, computeDataType);
  void *beta = makeHostConstant(0.0, computeDataType);

  ROCBLAS_ABORT_IF_FAIL(
      rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

  rocblas_gemm_flags rocblas_flags = rocblas_gemm_flags_none;

  float milliseconds = 0.0;
  int warmupRuns = 1;
  for (int i = 0, e = args.kernelRepeats + warmupRuns; i < e; ++i) {
    hipEvent_t startEvent, stopEvent;
    HIP_ABORT_IF_FAIL(hipEventCreate(&startEvent));
    HIP_ABORT_IF_FAIL(hipEventCreate(&stopEvent));
    HIP_ABORT_IF_FAIL(hipEventRecord(startEvent, NULL));
    if (args.gemmG == 1 && args.dataType == benchmark::DataType::F8) {
      ROCBLAS_ABORT_IF_FAIL(
          rocblas_gemm_ex3(handle, /*trans_a=*/blasTransB,
                           /*trans_b=*/blasTransA, /*m=*/args.gemmN,
                           /*n=*/args.gemmM, /*k=*/args.gemmK, alpha,
                           /*a=*/bDevice, /*a_type=*/inType, /*lda=*/ldb,
                           /*b=*/aDevice, /*b_type=*/inType, /*ldb=*/lda, beta,
                           cDevice, outType, ldc, cDevice, outType, ldc,
                           /*computeType=*/rocblas_compute_type_f32,
                           rocblas_gemm_algo_standard, 0, rocblas_flags));
    } else if (args.gemmG == 1) {
      ROCBLAS_ABORT_IF_FAIL(
          rocblas_gemm_ex(handle, /*trans_a=*/blasTransB,
                          /*trans_b=*/blasTransA, /*m=*/args.gemmN,
                          /*n=*/args.gemmM, /*k=*/args.gemmK, alpha,
                          /*a=*/bDevice, /*a_type=*/inType, /*lda=*/ldb,
                          /*b=*/aDevice, /*b_type=*/inType, /*ldb=*/lda, beta,
                          cDevice, outType, ldc, cDevice, outType, ldc,
                          /*computeType=*/computeType,
                          rocblas_gemm_algo_standard, 0, rocblas_flags));
    } else if (args.dataType == benchmark::DataType::F8) {
      ROCBLAS_ABORT_IF_FAIL(rocblas_gemm_strided_batched_ex3(
          handle, /*trans_a=*/blasTransB, /*trans_b=*/blasTransA,
          /*m=*/args.gemmN, /*n=*/args.gemmM, /*k=*/args.gemmK, alpha,
          /*a=*/bDevice, /*a_type=*/inType, /*lda=*/ldb, /*stride_a=*/strideB,
          /*b=*/aDevice, /*b_type=*/inType, /*ldb=*/lda, /*stride_b=*/strideA,
          beta, cDevice, outType, ldc, strideC, cDevice, outType, ldc, strideC,
          args.gemmG,
          /*computeType=*/rocblas_compute_type_f32, rocblas_gemm_algo_standard,
          0, rocblas_flags));
    } else {
      ROCBLAS_ABORT_IF_FAIL(rocblas_gemm_strided_batched_ex(
          handle, /*trans_a=*/blasTransB, /*trans_b=*/blasTransA,
          /*m=*/args.gemmN, /*n=*/args.gemmM, /*k=*/args.gemmK, alpha,
          /*a=*/bDevice, /*a_type=*/inType, /*lda=*/ldb, /*stride_a=*/strideB,
          /*b=*/aDevice, /*b_type=*/inType, /*ldb=*/lda, /*stride_b=*/strideA,
          beta, cDevice, outType, ldc, strideC, cDevice, outType, ldc, strideC,
          args.gemmG,
          /*computeType=*/computeType, rocblas_gemm_algo_standard, 0,
          rocblas_flags));
    }
    float currentMilliseconds = 0.0;
    HIP_ABORT_IF_FAIL(hipEventRecord(stopEvent, NULL));
    HIP_ABORT_IF_FAIL(hipEventSynchronize(stopEvent));
    HIP_ABORT_IF_FAIL(
        hipEventElapsedTime(&currentMilliseconds, startEvent, stopEvent));
    HIP_ABORT_IF_FAIL(hipEventDestroy(stopEvent));
    HIP_ABORT_IF_FAIL(hipEventDestroy(startEvent));
    // Don't record the first run
    if (i < warmupRuns)
      continue;
    milliseconds += currentMilliseconds;
  }
  float avgTime = milliseconds / args.kernelRepeats;
  std::cout << "Best kernel time: " << avgTime << "\n";
  std::cout << "Best kernel tflops: "
            << ((2 * args.gemmG * args.gemmM * args.gemmN * args.gemmK) /
                avgTime) *
                   1e-9
            << "\n";

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
