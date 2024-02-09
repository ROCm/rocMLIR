//===---------- ck-benchmark-driver.cpp - CK benchmark driver ------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//
//
// Performance driver to run GEMM kernels from the Composable Kernel library
//
//===----------------------------------------------------------------------===//

// Include common utility functions
#include "../common/benchmarkUtils.h"

// CK includes
#include "ck/ck.hpp"
// CK __noinline__ definition clashes with the one in libc
#undef __noinline__

#include "ck/library/tensor_operation_instance/gpu/batched_gemm.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm_add_add_fastgelu.hpp"

#include "ck/tensor_operation/gpu/device/device_batched_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

// System includes
#include <iostream>
#include <set>
#include <string>

// CK specific definitions
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AddAddFastGelu = ck::tensor_operation::element_wise::AddAddFastGelu;
using BaseArgument = ck::tensor_operation::device::BaseArgument;
using BaseInvoker = ck::tensor_operation::device::BaseInvoker;

template <typename ALayout, typename BLayout, typename DT>
using FastGeluAddAddGemmDeviceOp =
    ck::tensor_operation::device::DeviceGemmMultipleD<
        ALayout, BLayout, ck::Tuple<Row, Row>, Row, DT, DT, ck::Tuple<DT, DT>,
        DT, PassThrough, PassThrough, AddAddFastGelu>;

template <typename ALayout, typename BLayout, typename DT>
using GemmDeviceOp = ck::tensor_operation::device::DeviceGemm<
    ALayout, BLayout, Row, DT, DT, DT, PassThrough, PassThrough, PassThrough>;

template <typename ALayout, typename BLayout, typename DT>
using BatchedGemmDeviceOp = ck::tensor_operation::device::DeviceBatchedGemm<
    ALayout, BLayout, Row, DT, DT, DT, PassThrough, PassThrough, PassThrough>;

// Simple structure to wrap memory parameters together
struct GemmMemoryParameters {
  void *aDevice;
  void *bDevice;
  void *cDevice;
  void *d0Device;
  void *d1Device;

  size_t strideA;
  size_t strideB;
  size_t strideC;

  size_t batchStrideA;
  size_t batchStrideB;
  size_t batchStrideC;
};

// Main utility functions to run GEMM
template <typename ALayout, typename BLayout, typename DT>
struct GemmRunner {
  using D = GemmDeviceOp<ALayout, BLayout, DT>;
  using Dptr = std::unique_ptr<D>;

  static auto makeArg(const Dptr &op_ptr, const GemmMemoryParameters &params,
                      const benchmark::BenchmarkArgs &args) {
    return op_ptr->MakeArgumentPointer(
        params.aDevice, params.bDevice, params.cDevice, args.gemmM, args.gemmN,
        args.gemmK, params.strideA, params.strideB, params.strideC,
        PassThrough{}, PassThrough{}, PassThrough{});
  }

  static auto getInstances() {
    return ck::tensor_operation::device::instance::
        DeviceOperationInstanceFactory<D>::GetInstances();
  }
};

template <typename ALayout, typename BLayout, typename DT>
struct FastGeluAddAddGemmRunner {
  using D = FastGeluAddAddGemmDeviceOp<ALayout, BLayout, DT>;
  using Dptr = std::unique_ptr<D>;

  static auto makeArg(const Dptr &op_ptr, const GemmMemoryParameters &params,
                      const benchmark::BenchmarkArgs &args) {
    return op_ptr->MakeArgumentPointer(
        params.aDevice, params.bDevice, {params.d0Device, params.d1Device},
        params.cDevice, args.gemmM, args.gemmN, args.gemmK, params.strideA,
        params.strideB, {int(params.strideC), int(params.strideC)},
        params.strideC, PassThrough{}, PassThrough{}, AddAddFastGelu{});
  }

  static auto getInstances() {
    return ck::tensor_operation::device::instance::
        DeviceOperationInstanceFactory<D>::GetInstances();
  }
};

// Main utility functions to run batched GEMM
template <typename ALayout, typename BLayout, typename DT>
struct BatchedGemmRunner {

  using D = BatchedGemmDeviceOp<ALayout, BLayout, DT>;
  using Dptr = std::unique_ptr<D>;

  static auto makeArg(const Dptr &opPtr, const GemmMemoryParameters &params,
                      const benchmark::BenchmarkArgs &args) {
    return opPtr->MakeArgumentPointer(
        static_cast<DT *>(params.aDevice), static_cast<DT *>(params.bDevice),
        static_cast<DT *>(params.cDevice), args.gemmM, args.gemmN, args.gemmK,
        params.strideA, params.strideB, params.strideC, params.batchStrideA,
        params.batchStrideB, params.batchStrideC, args.gemmG, PassThrough{},
        PassThrough{}, PassThrough{});
  }

  static auto getInstances() {
    return ck::tensor_operation::device::instance::
        DeviceOperationInstanceFactory<D>::GetInstances();
  }
};

// Given the layout of A and B and the data type, loop over the different
// instances for a given problem size and pick the best configuration
template <typename OpRunner>
void run(const GemmMemoryParameters &params,
         const benchmark::BenchmarkArgs &args) {

  // All instances for a given A/B layout and data type
  const auto opPtrs = OpRunner::getInstances();

  float bestTflops = 0;
  float bestAveTime = 0.0;
  size_t bestKernelId = 0;
  std::string bestKernelName = "";
  bool found = false;

  // find the best config
  for (size_t i = 0; i < opPtrs.size(); ++i) {
    auto &opPtr = opPtrs[i];
    auto argumentPtr = OpRunner::makeArg(opPtr, params, args);
    auto invokerPtr = opPtr->MakeInvokerPointer();

    if (opPtr->IsSupportedArgument(argumentPtr.get())) {
      found = true;
      float aveTime =
          invokerPtr->Run(argumentPtr.get(), StreamConfig{nullptr, true});

      std::size_t flop = 2 * args.gemmM * args.gemmN * args.gemmK * args.gemmG;
      float tflops = static_cast<float>(flop) / 1.E9 / aveTime;

      if (tflops > bestTflops) {
        bestTflops = tflops;
        bestAveTime = aveTime;
        bestKernelName = opPtr->GetTypeString();
        bestKernelId = i;
      }
    }
  }
  if (!found) {
    bestAveTime = std::nan("");
  }
  std::cout << "Best kernel name: " << bestKernelName << "\n";
  std::cout << "Best kernel time: " << bestAveTime << "\n";
  std::cout << "Best kernel flops: " << bestTflops << "\n";
  std::cout << "Best kernel ID: " << bestKernelId << "\n";
}

template <typename ALayout, typename BLayout, typename DT>
void runLayout(const GemmMemoryParameters &params,
               const benchmark::BenchmarkArgs &args) {

  assert(params.strideA != 0 && "stride of A should be set");
  assert(params.strideB != 0 && "stride of B should be set");
  assert(params.strideC != 0 && "stride of C should be set");

  if (args.gemmG == 1) {
    if (args.fusion == "fastgelu_add_add") {
      run<FastGeluAddAddGemmRunner<ALayout, BLayout, DT>>(params, args);
    } else {
      assert(args.fusion.empty() && "unsupported CK fusion!");
      run<GemmRunner<ALayout, BLayout, DT>>(params, args);
    }
  } else {
    run<BatchedGemmRunner<ALayout, BLayout, DT>>(params, args);
  }
}

template <typename DT>
void runDataType(GemmMemoryParameters params,
                 const benchmark::BenchmarkArgs &args) {
  params.strideC = args.gemmN;
  if (args.transposeA && args.transposeB) {
    params.strideA = args.gemmM;
    params.strideB = args.gemmK;
    runLayout<Col, Col, DT>(params, args);
  } else if (args.transposeA && !args.transposeB) {
    params.strideA = args.gemmM;
    params.strideB = args.gemmN;
    runLayout<Col, Row, DT>(params, args);
  } else if (!args.transposeA && args.transposeB) {
    params.strideA = args.gemmK;
    params.strideB = args.gemmK;
    runLayout<Row, Col, DT>(params, args);
  } else {
    params.strideA = args.gemmK;
    params.strideB = args.gemmN;
    runLayout<Row, Row, DT>(params, args);
  }
}

int main(int argc, char **argv) {
  auto args = benchmark::parseCommandLine("ck-benchmark-driver", argc, argv);

  // Preliminary checks
  if (args.dataType == benchmark::DataType::F8 && args.gemmG != 1) {
    std::cerr << "CK does not support fp8 batched gemm!\n";
    exit(1);
  }
  if (args.dataType != args.outDataType){
    std::cerr << "CK does not support different input/output data types!\n";
    exit(1);
  }

  size_t batchStrideA = args.gemmM * args.gemmK,
         batchStrideB = args.gemmK * args.gemmN,
         batchStrideC = args.gemmM * args.gemmN;
  size_t aElems = batchStrideA * args.gemmG, bElems = batchStrideB * args.gemmG,
         cElems = batchStrideC * args.gemmG;
  size_t aBytes = benchmark::getByteSize(args.dataType, aElems),
         bBytes = benchmark::getByteSize(args.dataType, bElems),
         cBytes = benchmark::getByteSize(args.outDataType, cElems);

  void *d0Host = nullptr, *d1Host = nullptr, *d0Device = nullptr,
       *d1Device = nullptr;

  std::set<std::string> supportedFusions{"fastgelu_add_add"};
  if (!args.fusion.empty()) {
    if (supportedFusions.find(args.fusion) == supportedFusions.end()) {
      std::cerr << args.fusion
                << " is an invalid CK fusion! Supported fusions are:\n";
      for (auto s : supportedFusions)
        std::cerr << s << "\n";
      exit(1);
    }

    d0Host = benchmark::allocAndFill(args.dataType, cBytes);
    d1Host = benchmark::allocAndFill(args.outDataType, cBytes);
    d0Device = benchmark::getGpuBuffer(d0Host, cBytes);
    d1Device = benchmark::getGpuBuffer(d1Host, cBytes);
  }

  void *aHost = benchmark::allocAndFill(args.dataType, aBytes);
  void *bHost = benchmark::allocAndFill(args.dataType, bBytes);
  void *cHost = benchmark::allocAndFill(args.outDataType, cBytes);

  void *aDevice = benchmark::getGpuBuffer(aHost, aBytes);
  void *bDevice = benchmark::getGpuBuffer(bHost, bBytes);
  void *cDevice = benchmark::getGpuBuffer(cHost, cBytes);

  auto gemmParams = GemmMemoryParameters{
      aDevice, bDevice, cDevice,      d0Device,     d1Device,    0,
      0,       0,       batchStrideA, batchStrideB, batchStrideC};

  switch (args.dataType) {
  case benchmark::DataType::F32:
    runDataType<float>(gemmParams, args);
    break;
  case benchmark::DataType::F16:
    runDataType<ck::half_t>(gemmParams, args);
    break;
  case benchmark::DataType::I8:
    runDataType<int8_t>(gemmParams, args);
    break;
  case benchmark::DataType::F8:
    runDataType<ck::f8_t>(gemmParams, args);
    break;
  default:
    assert(0 && "DataType not supported");
  }

  free(aHost);
  free(bHost);
  free(cHost);
  HIP_ABORT_IF_FAIL(hipFree(aDevice));
  HIP_ABORT_IF_FAIL(hipFree(bDevice));
  HIP_ABORT_IF_FAIL(hipFree(cDevice));
  if (!args.fusion.empty()) {
    free(d0Host);
    free(d1Host);
    HIP_ABORT_IF_FAIL(hipFree(d0Device));
    HIP_ABORT_IF_FAIL(hipFree(d1Device));
  }
  return 0;
}
