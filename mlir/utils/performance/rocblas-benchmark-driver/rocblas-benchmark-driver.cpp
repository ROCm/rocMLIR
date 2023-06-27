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

// Include common utility functions
#include "../common/benchmarkUtils.h"

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

static rocblas_datatype getRocblasType(bool isOut,
                                       benchmark::DataType dataType) {
  switch (dataType) {
  case benchmark::DataType::F32:
    return rocblas_datatype_f32_r;
  case benchmark::DataType::F16:
    return rocblas_datatype_f16_r;
  case benchmark::DataType::BF16:
    return rocblas_datatype_bf16_r;
  case benchmark::DataType::I8:
    return isOut ? rocblas_datatype_i32_r : rocblas_datatype_i8_r;
  case benchmark::DataType::UNKNOWN:
    assert(0 && "Data type unknown");
  }
}

/// This code is taken from
/// https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/blob/84a8f450f20521242b74bd803824673739d6e858/src/targets/gpu/rocblas.cpp
/// and is used to set the GEMM compute type (i.e., the type of the GEMM
/// accumulation buffer)
bool get_compute_fp32_flag() {
  const auto device_name = benchmark::get_device_name();
  return (device_name.find("gfx908") != std::string::npos ||
          device_name.find("gfx90a") != std::string::npos);
}

/// This code is taken from
/// https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/blob/84a8f450f20521242b74bd803824673739d6e858/src/targets/gpu/rocblas.cpp
/// and is used to query if the `rocblas_gemm_flgas_pack_int8x4` is set
bool get_int8_x4_format(rocblas_handle &handle) {
  rocblas_gemm_flags flag;
  rocblas_query_int8_layout_flag(handle, &flag);
  return flag == rocblas_gemm_flags_pack_int8x4;
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
  size_t aBytes = getByteSize(args.dataType, aElems, false),
         bBytes = getByteSize(args.dataType, bElems, false),
         cBytes = getByteSize(args.dataType, cElems, true);

  rocblas_datatype inType = getRocblasType(false, args.dataType);
  rocblas_datatype outType = getRocblasType(true, args.dataType);
  rocblas_datatype computeType = outType;

  if ((inType == rocblas_datatype_f16_r) && get_compute_fp32_flag()) {
    computeType = rocblas_datatype_f32_r;
  }

  rocblas_initialize();
  rocblas_handle handle;
  ROCBLAS_ABORT_IF_FAIL(rocblas_create_handle(&handle));

  void *aHost = benchmark::allocAndFill(args.dataType, aBytes, false);
  void *bHost = benchmark::allocAndFill(args.dataType, bBytes, false);
  void *cHost = benchmark::allocAndFill(args.dataType, cBytes, true);

  void *aDevice = benchmark::getGpuBuffer(aHost, aBytes);
  void *bDevice = benchmark::getGpuBuffer(bHost, bBytes);
  void *cDevice = benchmark::getGpuBuffer(cHost, cBytes);

  void *alpha = makeHostConstant(1.0, args.dataType);
  void *beta = makeHostConstant(0.0, args.dataType);

  ROCBLAS_ABORT_IF_FAIL(
      rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

  rocblas_gemm_flags rocblas_flags = get_int8_x4_format(handle)
                                         ? rocblas_gemm_flags_pack_int8x4
                                         : rocblas_gemm_flags_none;

  for (int i = 0, e = args.kernelRepeats; i < e; ++i)
    if (args.gemmG == 1) {
      ROCBLAS_ABORT_IF_FAIL(
          rocblas_gemm_ex(handle, /*trans_a=*/blasTransB,
                          /*trans_b=*/blasTransA, /*m=*/args.gemmN,
                          /*n=*/args.gemmM, /*k=*/args.gemmK, alpha,
                          /*a=*/bDevice, /*a_type=*/inType, /*lda=*/ldb,
                          /*b=*/aDevice, /*b_type=*/inType, /*ldb=*/lda, beta,
                          cDevice, outType, ldc, cDevice, outType, ldc,
                          /*computeType=*/computeType,
                          rocblas_gemm_algo_standard, 0, rocblas_flags));

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
