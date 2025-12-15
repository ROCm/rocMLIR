#include "../common/benchmarkUtils.h"
#include "../common/hip_f8_impl.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static hipDataType getF8HipType() {
  const auto device_name = benchmark::get_device_name();
  if (device_name.find("gfx94") != std::string::npos) {
    return HIP_R_8F_E4M3_FNUZ;
  }
  return HIP_R_8F_E4M3;
}

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
    return getF8HipType();
  case benchmark::DataType::I32:
    return HIP_R_32I;
  case benchmark::DataType::F4:
  case benchmark::DataType::F8E8M0FNU:
  case benchmark::DataType::UNKNOWN:
    fprintf(stderr, "Unsupported data type\n");
    exit(1);
  }
  __builtin_unreachable();
}

static benchmark::DataType getComputeDataType(benchmark::DataType inputType,
                                              benchmark::DataType outputType) {
  if (inputType == benchmark::DataType::I8)
    return benchmark::DataType::I32;

  // All floating-point types (F8, F16, BF16, F32) use F32 compute type
  if (inputType == benchmark::DataType::F8 ||
      inputType == benchmark::DataType::F16 ||
      inputType == benchmark::DataType::BF16 ||
      inputType == benchmark::DataType::F32)
    return benchmark::DataType::F32;

  // F4 and other types are not supported by hipBLASLt
  if (inputType == benchmark::DataType::F4 ||
      inputType == benchmark::DataType::F8E8M0FNU) {
    fprintf(stderr, "Data type not supported by hipBLASLt\n");
    exit(1);
  }

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

// Convert BF16 (stored as uint16) to float
// BF16 is just the upper 16 bits of float32
static float bf16ToFloat(uint16_t bf16) {
  uint32_t f32_bits = static_cast<uint32_t>(bf16) << 16;
  float result;
  memcpy(&result, &f32_bits, sizeof(result));
  return result;
}

// Convert F16 to float using HIP's __half type
static float f16ToFloat(uint16_t f16) {
  __half h;
  memcpy(&h, &f16, sizeof(h));
  return __half2float(h);
}

// Check if device uses FNUZ (negative zero is NaN) FP8 format
static bool isF8Fnuz() {
  const auto device_name = benchmark::get_device_name();
  return device_name.find("gfx94") != std::string::npos;
}

// Convert FP8 E4M3 to float using the appropriate format for the device
static float fp8ToFloat(uint8_t fp8) {
  if (isF8Fnuz()) {
    return benchmark::cast_from_f8<3, 4, float, true>(fp8);
  }
  return benchmark::cast_from_f8<3, 4, float, false>(fp8);
}

// Get element as float from buffer
static float getElementAsFloat(const void *buf, size_t idx,
                               benchmark::DataType dtype) {
  switch (dtype) {
  case benchmark::DataType::F32:
    return static_cast<const float *>(buf)[idx];
  case benchmark::DataType::F16:
    return f16ToFloat(static_cast<const uint16_t *>(buf)[idx]);
  case benchmark::DataType::BF16:
    return bf16ToFloat(static_cast<const uint16_t *>(buf)[idx]);
  case benchmark::DataType::I32:
    return static_cast<float>(static_cast<const int32_t *>(buf)[idx]);
  case benchmark::DataType::I8:
    return static_cast<float>(static_cast<const int8_t *>(buf)[idx]);
  case benchmark::DataType::F8:
    return fp8ToFloat(static_cast<const uint8_t *>(buf)[idx]);
  default:
    return 0.0f;
  }
}

// Print tensor in format compatible with rocmlir-gen's printMemrefF32
static void printTensor(const void *data, int64_t m, int64_t n,
                        int64_t batchCount, benchmark::DataType dtype) {
  std::cout << "data = \n";

  size_t idx = 0;
  for (int64_t g = 0; g < batchCount; ++g) {
    if (batchCount > 1)
      std::cout << "[";
    for (int64_t i = 0; i < m; ++i) {
      std::cout << "[";
      for (int64_t j = 0; j < n; ++j) {
        float val = getElementAsFloat(data, idx++, dtype);
        printf("%g", val);
        if (j < n - 1)
          std::cout << ",   ";
      }
      std::cout << "]";
      if (i < m - 1)
        std::cout << ",\n";
    }
    if (batchCount > 1) {
      std::cout << "]";
      if (g < batchCount - 1)
        std::cout << ",\n";
    }
  }
  std::cout << "\n";
}

int main(int argc, char **argv) {
  auto args =
      benchmark::parseCommandLine("hipblaslt-benchmark-driver", argc, argv);

  const int64_t m = args.gemmM;
  const int64_t n = args.gemmN;
  const int64_t k = args.gemmK;
  const int64_t batch_count = args.gemmG > 0 ? args.gemmG : 1;

  // Please note: MIGraphx and MLIR are using row-major format
  // to store matrices, while hipBLASLt is using a column-major format.
  // To be compliant to hipBLASLt format, MIGraphx swaps the inputs
  // and tells hipBLASLt that B is nxk and A is kxm. So the result
  // will be a nxm matrix stored in column-major order. We can simply
  // recover the original matrix C by reading the matrix in a row-major
  // way.
  const hipblasOperation_t trans_a =
      args.transposeB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  const hipblasOperation_t trans_b =
      args.transposeA ? HIPBLAS_OP_T : HIPBLAS_OP_N;

  // Matrix layout dimensions depend on transpose flags.
  // Leading dimension equals rows for column-major contiguous matrices.
  const int64_t matA_rows = args.transposeB ? k : n;
  const int64_t matA_cols = args.transposeB ? n : k;

  const int64_t matB_rows = args.transposeA ? m : k;
  const int64_t matB_cols = args.transposeA ? k : m;

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

  void *h_a = benchmark::allocAndFill(args.dataType, aBytes);
  void *h_b = benchmark::allocAndFill(args.dataType, bBytes);
  void *h_c = benchmark::allocAndFill(args.outDataType, cBytes);

  void *d_a = benchmark::getGpuBuffer(h_a, aBytes);
  void *d_b = benchmark::getGpuBuffer(h_b, bBytes);
  // Since beta = 0, C is not read, so we use the same buffer for both C and D
  void *d_c = benchmark::getGpuBuffer(h_c, cBytes);

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
  HIPBLASLT_ABORT_IF_FAIL(hipblasLtMatrixLayoutCreate(
      &matA, inputType, matA_rows, matA_cols, matA_rows));
  HIPBLASLT_ABORT_IF_FAIL(hipblasLtMatrixLayoutCreate(
      &matB, inputType, matB_rows, matB_cols, matB_rows));
  HIPBLASLT_ABORT_IF_FAIL(
      hipblasLtMatrixLayoutCreate(&matC, outputType, n, m, n));
  HIPBLASLT_ABORT_IF_FAIL(
      hipblasLtMatrixLayoutCreate(&matD, outputType, n, m, n));

  if (batch_count > 1) {
    // Due to A/B swap: matA uses strideB, matB uses strideA
    setBatchAttributes(matA, batch_count, strideB);
    setBatchAttributes(matB, batch_count, strideA);
    setBatchAttributes(matC, batch_count, strideC);
    setBatchAttributes(matD, batch_count, strideC);
  }

  std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults;
  HIPBLASLT_ABORT_IF_FAIL(hipblaslt_ext::getAllAlgos(
      handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM, trans_a, trans_b,
      inputType, inputType, outputType, outputType, computeType,
      heuristicResults));

  // If user specified a algorithm index, use it directly
  if (args.algoIndex >= 0) {
    printf("Using user-specified algorithm index: %d\n", args.algoIndex);

    std::vector<int> requestedIndices = {args.algoIndex};
    std::vector<hipblasLtMatmulHeuristicResult_t> selectedAlgos;

    hipblasStatus_t status = hipblaslt_ext::getAlgosFromIndex(
        handle, requestedIndices, selectedAlgos);

    if (status != HIPBLAS_STATUS_SUCCESS || selectedAlgos.empty()) {
      fprintf(stderr, "Error: Algorithm index %d is not available\n",
              args.algoIndex);
      exit(1);
    }

    heuristicResults = selectedAlgos;
    printf("Successfully loaded algorithm index %d\n", args.algoIndex);
  }

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

  const int warmupRuns = args.warmupRuns;
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
          handle, matmul, alpha, d_b, matA, d_a, matB, beta, d_c, matC, d_c,
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

  int algoIndex =
      hipblaslt_ext::getIndexFromAlgo(heuristicResults[bestAlgoIndex].algo);

  if (algoIndex >= 0) {
    std::cout << "Best algorithm index: " << algoIndex << "\n";
    std::cout << "To reuse this algorithm: --algo-index " << algoIndex << "\n";
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
        handle, matmul, alpha, d_b, matA, d_a, matB, beta, d_c, matC, d_c, matD,
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

  HIP_ABORT_IF_FAIL(hipMemcpy(h_c, d_c, cBytes, hipMemcpyDeviceToHost));

  if (args.printResults) {
    printTensor(h_c, m, n, batch_count, args.outDataType);
  }

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

  free(h_a);
  free(h_b);
  free(h_c);
  free(alpha);
  free(beta);

  return 0;
}
