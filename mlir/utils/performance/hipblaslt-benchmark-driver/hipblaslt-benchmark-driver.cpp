#include "../common/benchmarkUtils.h"

#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#define HIPBLASLT_ABORT_IF_FAIL(expr)                                          \
  do {                                                                         \
    hipblasStatus_t status = expr;                                             \
    if (status != HIPBLAS_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "HipBLASLt error %d at %s:%d in %s\n", (int)status,      \
              __FILE__, __LINE__, #expr);                                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static hipDataType toHipDataType(benchmark::DataType dataType) {
  switch (dataType) {
  case benchmark::DataType::F32:
    return HIP_R_32F;
  case benchmark::DataType::F16:
    return HIP_R_16F;
  case benchmark::DataType::BF16:
    return HIP_R_16BF;
  case benchmark::DataType::I8:
    return HIP_R_8I;
  case benchmark::DataType::F8:
    return HIP_R_8F_E4M3;
  case benchmark::DataType::I32:
    return HIP_R_32I;
  case benchmark::DataType::UNKNOWN:
    fprintf(stderr, "Unsupported data type\n");
    exit(1);
  }
  __builtin_unreachable();
}

static bool supportsF32Accumulation() {
  const auto device_name = benchmark::get_device_name();
  return (device_name.find("gfx908") != std::string::npos ||
          device_name.find("gfx90a") != std::string::npos ||
          device_name.find("gfx94") != std::string::npos ||
          device_name.find("gfx95") != std::string::npos);
}

static benchmark::DataType getComputeDataType(benchmark::DataType inputType,
                                              benchmark::DataType outputType) {
  if (inputType == benchmark::DataType::F8)
    return benchmark::DataType::F32;

  if ((inputType == benchmark::DataType::F16 ||
       inputType == benchmark::DataType::BF16) &&
      supportsF32Accumulation())
    return benchmark::DataType::F32;

  if (inputType == benchmark::DataType::I8)
    return benchmark::DataType::I32;

  return outputType;
}

static hipblasComputeType_t toHipComputeType(benchmark::DataType dataType) {
  switch (dataType) {
  case benchmark::DataType::F32:
    return HIPBLAS_COMPUTE_32F;
  case benchmark::DataType::F16:
  case benchmark::DataType::BF16:
    return HIPBLAS_COMPUTE_16F;
  case benchmark::DataType::I32:
    return HIPBLAS_COMPUTE_32I;
  default:
    fprintf(stderr, "Unexpected compute data type\n");
    exit(1);
  }
}

static void setBatchAttributes(hipblasLtMatrixLayout_t layout,
                               int64_t batch_count, int64_t stride) {
  HIPBLASLT_ABORT_IF_FAIL(hipblasLtMatrixLayoutSetAttribute(
      layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count)));
  HIPBLASLT_ABORT_IF_FAIL(hipblasLtMatrixLayoutSetAttribute(
      layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
      sizeof(stride)));
}

int main(int argc, char **argv) {
  auto args =
      benchmark::parseCommandLine("hipblaslt-benchmark-driver", argc, argv);

  const int64_t m = args.gemmM;
  const int64_t n = args.gemmN;
  const int64_t k = args.gemmK;
  const int64_t batch_count = args.gemmG > 0 ? args.gemmG : 1;

  // MIGraphx and MLIR use row-major format, while hipBLASLt uses column-major.
  // To emulate row-major, we swap A and B matrices and compute B * A instead of
  // A * B.
  const hipblasOperation_t trans_a =
      args.transposeB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  const hipblasOperation_t trans_b =
      args.transposeA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  const int64_t lda = args.transposeB ? k : n;
  const int64_t ldb = args.transposeA ? m : k;
  const int64_t ldc = n;
  const int64_t ldd = n;

  benchmark::DataType computeDataType =
      getComputeDataType(args.dataType, args.outDataType);
  hipDataType inputType = toHipDataType(args.dataType);
  hipDataType outputType = toHipDataType(args.outDataType);
  hipDataType scaleType = toHipDataType(computeDataType);
  hipblasComputeType_t computeType = toHipComputeType(computeDataType);

  void *alpha = benchmark::makeHostConstant(1.0, computeDataType);
  void *beta = benchmark::makeHostConstant(0.0, computeDataType);

  const size_t strideA = m * k, strideB = k * n, strideC = m * n;
  const size_t aBytes =
      benchmark::getBytesPerElement(args.dataType) * strideA * batch_count;
  const size_t bBytes =
      benchmark::getBytesPerElement(args.dataType) * strideB * batch_count;
  const size_t cBytes =
      benchmark::getBytesPerElement(args.outDataType) * strideC * batch_count;
  const size_t dBytes = cBytes;

  void *h_a = benchmark::allocAndFill(args.dataType, aBytes);
  void *h_b = benchmark::allocAndFill(args.dataType, bBytes);
  void *h_c = benchmark::allocAndFill(args.outDataType, cBytes);
  void *h_d = benchmark::allocAndFill(args.outDataType, dBytes);

  void *d_a = benchmark::getGpuBuffer(h_a, aBytes);
  void *d_b = benchmark::getGpuBuffer(h_b, bBytes);
  void *d_c = benchmark::getGpuBuffer(h_c, cBytes);
  void *d_d = benchmark::getGpuBuffer(h_d, dBytes);

  hipblasLtHandle_t handle;
  HIPBLASLT_ABORT_IF_FAIL(hipblasLtCreate(&handle));

  hipblasLtMatmulDesc_t matmul;
  HIPBLASLT_ABORT_IF_FAIL(
      hipblasLtMatmulDescCreate(&matmul, computeType, scaleType));

  HIPBLASLT_ABORT_IF_FAIL(hipblasLtMatmulDescSetAttribute(
      matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
  HIPBLASLT_ABORT_IF_FAIL(hipblasLtMatmulDescSetAttribute(
      matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

  void *d_scale_a = nullptr, *d_scale_b = nullptr;
  if (args.dataType == benchmark::DataType::F8) {
    float h_scale = 1.0f;
    HIP_ABORT_IF_FAIL(hipMalloc(&d_scale_a, sizeof(float)));
    HIP_ABORT_IF_FAIL(hipMalloc(&d_scale_b, sizeof(float)));
    HIP_ABORT_IF_FAIL(
        hipMemcpy(d_scale_a, &h_scale, sizeof(float), hipMemcpyHostToDevice));
    HIP_ABORT_IF_FAIL(
        hipMemcpy(d_scale_b, &h_scale, sizeof(float), hipMemcpyHostToDevice));

    hipblasLtMatmulMatrixScale_t scale_mode =
        HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    HIPBLASLT_ABORT_IF_FAIL(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode,
        sizeof(scale_mode)));
    HIPBLASLT_ABORT_IF_FAIL(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode,
        sizeof(scale_mode)));
    HIPBLASLT_ABORT_IF_FAIL(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_a,
        sizeof(void *)));
    HIPBLASLT_ABORT_IF_FAIL(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_b,
        sizeof(void *)));
  }

  hipblasLtMatrixLayout_t matA, matB, matC, matD;
  HIPBLASLT_ABORT_IF_FAIL(
      hipblasLtMatrixLayoutCreate(&matA, inputType, n, k, lda));
  HIPBLASLT_ABORT_IF_FAIL(
      hipblasLtMatrixLayoutCreate(&matB, inputType, k, m, ldb));
  HIPBLASLT_ABORT_IF_FAIL(
      hipblasLtMatrixLayoutCreate(&matC, outputType, n, m, ldc));
  HIPBLASLT_ABORT_IF_FAIL(
      hipblasLtMatrixLayoutCreate(&matD, outputType, n, m, ldd));

  if (batch_count > 1) {
    setBatchAttributes(matA, batch_count, strideA);
    setBatchAttributes(matB, batch_count, strideB);
    setBatchAttributes(matC, batch_count, strideC);
    setBatchAttributes(matD, batch_count, strideC);
  }

  std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults;
  HIPBLASLT_ABORT_IF_FAIL(hipblaslt_ext::getAllAlgos(
      handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM, trans_a, trans_b,
      inputType, inputType, outputType, outputType, computeType,
      heuristicResults));

  if (heuristicResults.empty()) {
    fprintf(stderr, "No algorithms found\n");
    exit(1);
  }

  std::vector<int> validIdx;
  for (size_t j = 0; j < heuristicResults.size(); j++) {
    size_t workspace_size = 0;
    if (hipblaslt_ext::matmulIsAlgoSupported(
            handle, matmul, alpha, matA, matB, beta, matC, matD,
            heuristicResults[j].algo,
            workspace_size) == HIPBLAS_STATUS_SUCCESS) {
      validIdx.push_back(j);
      heuristicResults[j].workspaceSize = workspace_size;
    }
  }

  if (validIdx.empty()) {
    fprintf(stderr, "No supported algorithms for this problem size\n");
    exit(1);
  }

  size_t maxWorkspaceSize = 0;
  for (int idx : validIdx) {
    maxWorkspaceSize =
        std::max(maxWorkspaceSize, heuristicResults[idx].workspaceSize);
  }

  void *workspace = nullptr;
  if (maxWorkspaceSize > 0) {
    HIP_ABORT_IF_FAIL(hipMalloc(&workspace, maxWorkspaceSize));
  }

  const int warmupRuns = 2;
  const int benchmarkRuns = args.kernelRepeats;
  float bestAvgTime = std::numeric_limits<float>::max();
  int bestAlgoIndex = -1;

  for (size_t algoTestIdx = 0; algoTestIdx < validIdx.size(); ++algoTestIdx) {
    int algoIdx = validIdx[algoTestIdx];
    float algoMilliseconds = 0.0;
    bool algoFailed = false;

    for (int i = 0, e = benchmarkRuns + warmupRuns; i < e; ++i) {
      hipEvent_t startEvent, stopEvent;
      HIP_ABORT_IF_FAIL(hipEventCreate(&startEvent));
      HIP_ABORT_IF_FAIL(hipEventCreate(&stopEvent));
      HIP_ABORT_IF_FAIL(hipEventRecord(startEvent, NULL));

      hipblasStatus_t status = hipblasLtMatmul(
          handle, matmul, alpha, d_b, matA, d_a, matB, beta, d_c, matC, d_d,
          matD, &heuristicResults[algoIdx].algo, workspace,
          heuristicResults[algoIdx].workspaceSize, 0);

      float currentMilliseconds = 0.0;
      HIP_ABORT_IF_FAIL(hipEventRecord(stopEvent, NULL));
      HIP_ABORT_IF_FAIL(hipEventSynchronize(stopEvent));
      HIP_ABORT_IF_FAIL(
          hipEventElapsedTime(&currentMilliseconds, startEvent, stopEvent));
      HIP_ABORT_IF_FAIL(hipEventDestroy(stopEvent));
      HIP_ABORT_IF_FAIL(hipEventDestroy(startEvent));

      if (status != HIPBLAS_STATUS_SUCCESS) {
        algoFailed = true;
        break;
      }

      if (i < warmupRuns)
        continue;
      algoMilliseconds += currentMilliseconds;
    }

    if (algoFailed) {
      continue;
    }

    const float algoAvgTime = algoMilliseconds / benchmarkRuns;

    if (algoAvgTime < bestAvgTime) {
      bestAvgTime = algoAvgTime;
      bestAlgoIndex = algoIdx;
    }
  }

  if (bestAlgoIndex == -1) {
    fprintf(stderr, "All algorithms failed during benchmark testing\n");
    exit(1);
  }

  const hipblasLtMatmulHeuristicResult_t bestResult =
      heuristicResults[bestAlgoIndex];
  float milliseconds = 0.0;

  for (int i = 0, e = args.kernelRepeats + warmupRuns; i < e; ++i) {
    hipEvent_t startEvent, stopEvent;
    HIP_ABORT_IF_FAIL(hipEventCreate(&startEvent));
    HIP_ABORT_IF_FAIL(hipEventCreate(&stopEvent));
    HIP_ABORT_IF_FAIL(hipEventRecord(startEvent, NULL));

    HIPBLASLT_ABORT_IF_FAIL(hipblasLtMatmul(
        handle, matmul, alpha, d_b, matA, d_a, matB, beta, d_c, matC, d_d, matD,
        &bestResult.algo, workspace, bestResult.workspaceSize, 0));

    float currentMilliseconds = 0.0;
    HIP_ABORT_IF_FAIL(hipEventRecord(stopEvent, NULL));
    HIP_ABORT_IF_FAIL(hipEventSynchronize(stopEvent));
    HIP_ABORT_IF_FAIL(
        hipEventElapsedTime(&currentMilliseconds, startEvent, stopEvent));
    HIP_ABORT_IF_FAIL(hipEventDestroy(stopEvent));
    HIP_ABORT_IF_FAIL(hipEventDestroy(startEvent));

    if (i < warmupRuns)
      continue;
    milliseconds += currentMilliseconds;
  }

  const float avgTime = milliseconds / args.kernelRepeats;
  std::cout << "Best kernel time: " << avgTime << "\n";
  std::cout << "Best kernel tflops: "
            << ((2 * batch_count * m * n * k) / avgTime) * 1e-9 << "\n";

  HIP_ABORT_IF_FAIL(hipMemcpy(h_d, d_d, dBytes, hipMemcpyDeviceToHost));

  if (workspace)
    HIP_ABORT_IF_FAIL(hipFree(workspace));
  if (d_scale_a)
    HIP_ABORT_IF_FAIL(hipFree(d_scale_a));
  if (d_scale_b)
    HIP_ABORT_IF_FAIL(hipFree(d_scale_b));

  HIPBLASLT_ABORT_IF_FAIL(hipblasLtMatmulDescDestroy(matmul));
  HIPBLASLT_ABORT_IF_FAIL(hipblasLtMatrixLayoutDestroy(matA));
  HIPBLASLT_ABORT_IF_FAIL(hipblasLtMatrixLayoutDestroy(matB));
  HIPBLASLT_ABORT_IF_FAIL(hipblasLtMatrixLayoutDestroy(matC));
  HIPBLASLT_ABORT_IF_FAIL(hipblasLtMatrixLayoutDestroy(matD));
  HIPBLASLT_ABORT_IF_FAIL(hipblasLtDestroy(handle));

  HIP_ABORT_IF_FAIL(hipFree(d_a));
  HIP_ABORT_IF_FAIL(hipFree(d_b));
  HIP_ABORT_IF_FAIL(hipFree(d_c));
  HIP_ABORT_IF_FAIL(hipFree(d_d));

  free(h_a);
  free(h_b);
  free(h_c);
  free(h_d);
  free(alpha);
  free(beta);

  return 0;
}
