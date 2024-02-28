#include "../common/benchmarkUtils.h"
#include "CLI/CLI.hpp"
#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/batched_gemm_bias_softmax_gemm_permute.hpp"
#include "ck/library/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include <algorithm>
#include <ck/utility/data_type.hpp>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <tuple>
#include <utility>

enum class EncodedType { F32, F16, I8 };

struct Options {
  ck::index_t sequenceLength{384};
  ck::index_t headDims{64};
  ck::index_t groupSize{1};
  bool hasAttnBias{false};
  bool transposeQ{false};
  bool transposeK{false};
  bool transposeV{false};
  bool transposeO{false};
  bool onlyMatmulFlops{true};
  bool verbose{false};
  benchmark::DataType elementType{benchmark::DataType::F32};
};

auto getSize(std::vector<ck::index_t> &lengths) {
  auto size = std::accumulate(lengths.begin(), lengths.end(), 0, std::plus<>{});
  return size;
}

auto getStrides(std::vector<ck::index_t> &lengths) {
  std::vector<ck::index_t> strides(lengths.size());
  std::exclusive_scan(lengths.rbegin(), lengths.rend(), strides.rbegin(), 1,
                      std::multiplies<>{});
  return strides;
}

auto transpose(std::vector<ck::index_t> &strides) {
  auto endIdx = strides.size() - 1;
  std::swap(strides[endIdx - 1], strides[endIdx]);
}

struct Data {
  Data(Options options) : options(options) {
    const ck::index_t G0{1};
    const ck::index_t G1{options.groupSize};
    const ck::index_t S{options.sequenceLength};
    const ck::index_t H{options.headDims};

    // matrix Q
    qLengths = std::vector<ck::index_t>{G0, G1, S, H};
    qStrides = getStrides(qLengths);
    if (options.transposeQ)
      transpose(qStrides);

    // matrix K
    kLengths = std::vector<ck::index_t>{G0, G1, S, H};
    kStrides = getStrides(kLengths);
    if (options.transposeK)
      transpose(kStrides);

    // matrix V
    vLengths = std::vector<ck::index_t>{G0, G1, S, H};
    vStrides = getStrides(vLengths);
    if (options.transposeV)
      transpose(vStrides);

    // matrix O
    oLengths = std::vector<ck::index_t>{G0, G1, S, H};
    oStrides = getStrides(oLengths);
    if (options.transposeO)
      transpose(oStrides);

    // matrix Bias
    biasLengths = std::vector<ck::index_t>{G0, G1, S, S};
    biasStrides = getStrides(biasLengths);

    const auto qElems = getSize(qLengths);
    const auto kElems = getSize(kLengths);
    const auto vElems = getSize(vLengths);
    const auto oElems = getSize(oLengths);
    const auto biasElems = getSize(biasLengths);

    const auto qBytes =
        benchmark::getByteSize(options.elementType, qElems, false);
    const auto kBytes =
        benchmark::getByteSize(options.elementType, kElems, false);
    const auto vBytes =
        benchmark::getByteSize(options.elementType, vElems, false);
    const auto oBytes =
        benchmark::getByteSize(options.elementType, oElems, true);
    const auto biasBytes =
        benchmark::getByteSize(options.elementType, biasElems, false);

    qHost = benchmark::allocAndFill(options.elementType, qBytes, false);
    kHost = benchmark::allocAndFill(options.elementType, kBytes, false);
    vHost = benchmark::allocAndFill(options.elementType, vBytes, false);
    oHost = benchmark::allocAndFill(options.elementType, oBytes, true);
    biasHost = benchmark::allocAndFill(options.elementType, biasBytes, false);

    qDevice = benchmark::getGpuBuffer(qHost, qBytes);
    kDevice = benchmark::getGpuBuffer(kHost, kBytes);
    vDevice = benchmark::getGpuBuffer(vHost, vBytes);
    oDevice = benchmark::getGpuBuffer(oHost, oBytes);
    biasDevice = benchmark::getGpuBuffer(biasHost, biasBytes);
  }
  ~Data() {
    free(qHost);
    free(kHost);
    free(vHost);
    free(oHost);
    free(biasHost);

    HIP_ABORT_IF_FAIL(hipFree(qDevice));
    HIP_ABORT_IF_FAIL(hipFree(kDevice));
    HIP_ABORT_IF_FAIL(hipFree(vDevice));
    HIP_ABORT_IF_FAIL(hipFree(oDevice));
    HIP_ABORT_IF_FAIL(hipFree(biasDevice));
  }

  void *qHost{nullptr};
  void *kHost{nullptr};
  void *vHost{nullptr};
  void *oHost{nullptr};
  void *biasHost{nullptr};

  void *qDevice{nullptr};
  void *kDevice{nullptr};
  void *vDevice{nullptr};
  void *oDevice{nullptr};
  void *biasDevice{nullptr};

  std::vector<ck::index_t> qLengths{};
  std::vector<ck::index_t> kLengths{};
  std::vector<ck::index_t> vLengths{};
  std::vector<ck::index_t> oLengths{};
  std::vector<ck::index_t> biasLengths{};

  std::vector<ck::index_t> qStrides{};
  std::vector<ck::index_t> kStrides{};
  std::vector<ck::index_t> vStrides{};
  std::vector<ck::index_t> oStrides{};
  std::vector<ck::index_t> biasStrides{};

private:
  Options options;
};

struct FlopCounts {
  FlopCounts(int groupSize, int sequenceLength, int headDims) {
    double g{static_cast<double>(groupSize)};
    double N{static_cast<double>(sequenceLength)};
    double D{static_cast<double>(headDims)};

    gemm1 = 2.0 * g * N * N * D;
    scale = N * N;
    gemm2 = 2.0 * g * N * N * D;
    softmax = 5.0 * N * D;
  }

  double get(bool onlyMatmulFlops = true) {
    auto gemmFlopCount{gemm1 + gemm2 + scale};
    return onlyMatmulFlops ? gemmFlopCount : gemmFlopCount + softmax;
  }

  double gemm1{};
  double scale{};
  double gemm2{};
  double softmax{};
};

constexpr static auto MaskingSpec =
    ck::tensor_operation::device::MaskingSpecialization::MaskDisabled;

// common element wise operations
using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using B0ElementOp = ck::tensor_operation::element_wise::PassThrough;
using B1ElementOp = ck::tensor_operation::element_wise::PassThrough;
using CElementOp = ck::tensor_operation::element_wise::PassThrough;

template <typename DataElementType>
struct AttentionWithoutBias {
  using ADataType = DataElementType;
  using B0DataType = DataElementType;
  using B1DataType = DataElementType;
  using CDataType = DataElementType;

  using Acc0ElementOp = ck::tensor_operation::element_wise::Scale;

  // clang-format off
  using DeviceOp =
      ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute<
          2,
          1,
          1,
          1,
          1,
          ADataType,
          B0DataType,
          B1DataType,
          CDataType,
          ck::Tuple<>,
          ck::Tuple<>,
          AElementOp,
          B0ElementOp,
          Acc0ElementOp,
          B1ElementOp,
          CElementOp,
          MaskingSpec>;
  // clang-format on
  using Dptr = std::unique_ptr<DeviceOp>;

  static auto makeArgs(const Dptr &opPtr, Data &data, const Options &options) {
    // clang-format off
    auto argumentPtr = opPtr->MakeArgumentPointer(
        data.qDevice,
        data.kDevice,
        data.vDevice,
        data.oDevice,
        {},
        {},
        data.qLengths,
        data.qStrides,
        data.kLengths,
        data.kStrides,
        data.vLengths,
        data.vStrides,
        data.oLengths,
        data.oStrides,
        {},
        {},
        {},
        {},
        AElementOp{},
        B0ElementOp{},
        Acc0ElementOp{1 / sqrtf(options.sequenceLength)},
        B1ElementOp{},
        CElementOp{});
    // clang-format on
    return argumentPtr;
  }

  static auto getInstances() {
    return ck::tensor_operation::device::instance::
        DeviceOperationInstanceFactory<DeviceOp>::GetInstances();
  }

  static double getFlopCount(const Options &options) {
    FlopCounts counts(options.groupSize, options.sequenceLength,
                      options.headDims);
    return counts.get(options.onlyMatmulFlops);
  }
};

template <typename DataElementType>
struct AttentionWithBias {
  using ADataType = DataElementType;
  using B0DataType = DataElementType;
  using B1DataType = DataElementType;
  using D0DataType = DataElementType;
  using CDataType = DataElementType;

  using Acc0ElementOp = ck::tensor_operation::element_wise::ScaleAdd;

  // clang-format off
  using DeviceOp =
      ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute<
          2,
          1,
          1,
          1,
          1,
          ADataType,
          B0DataType,
          B1DataType,
          CDataType,
          ck::Tuple<D0DataType>,
          ck::Tuple<>,
          AElementOp,
          B0ElementOp,
          Acc0ElementOp,
          B1ElementOp,
          CElementOp,
          MaskingSpec>;
  // clang-format on
  using Dptr = std::unique_ptr<DeviceOp>;

  static auto makeArgs(const Dptr &opPtr, Data &data, const Options &options) {
    // clang-format off
    auto argumentPtr = opPtr->MakeArgumentPointer(
        data.qDevice,
        data.kDevice,
        data.vDevice,
        data.oDevice,
        std::array<void*, 1>{data.biasDevice},
        {},
        data.qLengths,
        data.qStrides,
        data.kLengths,
        data.kStrides,
        data.vLengths,
        data.vStrides,
        data.oLengths,
        data.oStrides,
        std::array<std::vector<ck::index_t>, 1>{data.biasLengths},
        std::array<std::vector<ck::index_t>, 1>{data.biasStrides},
        {},
        {},
        AElementOp{},
        B0ElementOp{},
        Acc0ElementOp{1 / sqrtf(options.sequenceLength)},
        B1ElementOp{},
        CElementOp{});
    // clang-format on
    return argumentPtr;
  }

  static auto getInstances() {
    return ck::tensor_operation::device::instance::
        DeviceOperationInstanceFactory<DeviceOp>::GetInstances();
  }

  static double getFlopCount(const Options &options) {
    FlopCounts counts(options.groupSize, options.sequenceLength,
                      options.headDims);

    // scale and add are similar element-wise operations which
    // operate on the same matrix size regarding the AttentionOp
    counts.scale *= 2;
    return counts.get(options.onlyMatmulFlops);
  }
};

template <typename Operation>
void runBenchmark(const Options &options) {
  // get device op instances
  const auto opPtrs = Operation::getInstances();

  if (opPtrs.empty()) {
    throw std::runtime_error("failed to find any config for the given options");
  } else {
    std::cout << "found " << opPtrs.size() << " `AttenOp` candidates"
              << std::endl;
  }

  Data data{options};

  std::string bestKernelName{"none"};
  int bestKernelId{-1};
  float minTime{std::numeric_limits<float>::max()};

  for (size_t idx = 0; idx < opPtrs.size(); ++idx) {
    auto &opPtr = opPtrs[idx];
    auto argumentPtr = Operation::makeArgs(opPtr, data, options);
    if (opPtr->IsSupportedArgument(argumentPtr.get())) {
      std::string opName = opPtr->GetTypeString();
      auto invokerPtr = opPtr->MakeInvokerPointer();
      StreamConfig streamConfig{nullptr, true};
      float avgTime = invokerPtr->Run(argumentPtr.get(), streamConfig);

      avgTime /= static_cast<float>(streamConfig.nrepeat_);
      if (avgTime < minTime) {
        minTime = avgTime;
        bestKernelName = opPtr->GetTypeString();
        bestKernelId = static_cast<int>(idx);
      }
      if (options.verbose) {
        std::cout << idx << ") under test\n\t" << opPtr->GetTypeString()
                  << std::endl;
      }
    } else {
      if (options.verbose) {
        std::cout << idx << ") arguments are NOT supported for\n\t"
                  << opPtr->GetTypeString() << std::endl;
      }
    }
  }

  if (bestKernelId >= 0) {
    const auto flop = Operation::getFlopCount(options);
    const float tflops = static_cast<float>(flop) / 1.E9 / minTime;

    std::cout << "Best kernel report:"
              << "\nkernel time: " << minTime << "\nkernel ID: " << bestKernelId
              << "\nTFlops: " << tflops << "\nkernel name:\n\t"
              << bestKernelName << std::endl;
  } else {
    throw std::runtime_error(
        "arguments are not supported by all AttentionOp candidates");
  }
}

struct Selector {
  void dispatch(Options &options) {
    try {
      switch (options.elementType) {
      case benchmark::DataType::F32: {
        selectOperation<float>(options);
        break;
      }
      case benchmark::DataType::F16: {
        selectOperation<ck::half_t>(options);
        break;
      }
      case benchmark::DataType::I8: {
        selectOperation<int8_t>(options);
        break;
      }
      default: {
        throw std::runtime_error("provided unsopported data element type");
      }
      };
    } catch (std::runtime_error &err) {
      throw err;
    }
  }

  template <typename Type>
  void selectOperation(Options &options) {
    if (options.hasAttnBias) {
      runBenchmark<AttentionWithBias<Type>>(options);
    } else {
      runBenchmark<AttentionWithoutBias<Type>>(options);
    }
  }
};

int main(int argc, char *argv[]) {
  Options options{};
  CLI::App app{"elementwise example with multi-dim block/grid"};

  app.add_option("--seq_len", options.sequenceLength,
                 "sequence length of attention");
  app.add_option("--head_dim", options.sequenceLength,
                 "head dimension of attention");
  app.add_option("--group_dim", options.groupSize,
                 "group dimension of attention");
  app.add_flag("--with-bias", options.hasAttnBias, "has bias term");
  app.add_flag("--trans-q", options.transposeQ, "transpose Q");
  app.add_flag("--trans-k", options.transposeK, "transpose K");
  app.add_flag("--trans-v", options.transposeV, "transpose V");
  app.add_flag("--trans-o", options.transposeO, "transpose O");
  app.add_flag("--only-matmul-flops", options.onlyMatmulFlops,
               "ignore softmax flops");

  std::map<std::string, benchmark::DataType> map{
      {"f32", benchmark::DataType::F32},
      {"f16", benchmark::DataType::F16},
      {"i8", benchmark::DataType::I8}};
  app.add_option("--type", options.elementType, "data type")
      ->transform(CLI::CheckedTransformer(map, CLI::ignore_case));

  app.add_flag("-v", options.verbose, "verbose");
  CLI11_PARSE(app, argc, argv);

  try {
    Selector selector{};
    selector.dispatch(options);
  } catch (std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    return -1;
  }

  return 0;
}
