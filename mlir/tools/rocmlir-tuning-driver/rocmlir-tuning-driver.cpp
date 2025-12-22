//===- rocmlir-tuning-driver.cpp - rocMLIR tuning driver -------------===//
//
// Copyright (c) 2022 Advanced Micro Devices Inc.
//
// Part of the rocMLIR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a wrapper script that reads in a MLIR file containing a rocMLIR
// kernel and tunes it. It will run the kernel with all applicable perf configs
// and report the execution time for each perf config. It is a very intentially
// specific program designed to eliminate JIT overhead, process spawn overhead
// and the like.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/ExecutionEngine/RocmDeviceName.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/InitRocMLIRCLOptions.h"
#include "mlir/InitRocMLIRDialects.h"
#include "mlir/InitRocMLIRPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <thread>

// Utilities to allocate buffers
#include "../utils/performance/common/benchmarkUtils.h"

#include "CacheFlush.h"
#include "ConcurrentQueue.h"

#include <hip/hip_runtime.h>

#if !defined(_HIP_CLANG_ONLY__)
// GCC complains if we don't do this
template <std::size_t n, typename... Ts,
          typename std::enable_if<n == sizeof...(Ts)>::type * = nullptr>
void pArgs(const std::tuple<Ts...> &, void *) {}

template <std::size_t n, typename... Ts,
          typename std::enable_if<n != sizeof...(Ts)>::type * = nullptr>
void pArgs(const std::tuple<Ts...> &formals, void **_vargs) {
  using T = typename std::tuple_element<n, std::tuple<Ts...>>::type;

  static_assert(!std::is_reference<T>{},
                "A __global__ function cannot have a reference as one of its "
                "arguments.");
  _vargs[n] =
      const_cast<void *>(reinterpret_cast<const void *>(&std::get<n>(formals)));
  return pArgs<n + 1>(formals, _vargs);
}
#endif

// Needs to go second lest we get compiler issues
#include <hip/hip_ext.h>

using namespace mlir;
using namespace rocmlir::tuningdriver;

static llvm::cl::opt<std::string> inputFilename{
    llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-")};

static llvm::cl::opt<rock::TuningParamSetKind> tuningSpaceKind(
    "tuning-space", llvm::cl::desc("Tuning space to use for this run"),
    llvm::cl::values(
        clEnumValN(rock::TuningParamSetKind::Quick, "quick",
                   "Quick tuning space"),
        clEnumValN(rock::TuningParamSetKind::Full, "full",
                   "Full tuning space, excluding known-bad configurations"),
        clEnumValN(
            rock::TuningParamSetKind::Greedy, "greedy",
            "Tune all possible tile sizes and try NUM_RANDOM_PER_TILE_SIZE "
            "random configurations for "
            "each tile size. Then, greedily select the best tile size, and "
            "brute force tune the rest of params"),
        clEnumValN(rock::TuningParamSetKind::Exhaustive, "exhaustive",
                   "All tuning space combinations, even inapplicable ones")),
    llvm::cl::value_desc("tuning space to use"),
    llvm::cl::init(rock::TuningParamSetKind::Full));

static llvm::cl::opt<unsigned> numIterations(
    "num-iterations",
    llvm::cl::desc("Number of times to run each kernel for averaging"),
    llvm::cl::value_desc("number of runs"), llvm::cl::init(100));

static llvm::cl::opt<unsigned> warmupIterations(
    "warmup-iterations", llvm::cl::desc("Number of warmup runs"),
    llvm::cl::value_desc("number of warmup runs"), llvm::cl::init(10));

static llvm::cl::opt<bool>
    useMedian("use-median",
              llvm::cl::desc("Use median of runs instead of mean for timing "
                             "(overrides trim-percent)"),
              llvm::cl::init(false));

static llvm::cl::opt<unsigned> trimPercent(
    "trim-percent",
    llvm::cl::desc("Percentage to trim from top and bottom of results"),
    llvm::cl::value_desc("trim percentage [0, 50)"), llvm::cl::init(10));

static llvm::cl::opt<unsigned> sleepUs(
    "sleep-us",
    llvm::cl::desc("Microseconds to sleep between runs to avoid throttling"),
    llvm::cl::value_desc("microseconds to sleep"), llvm::cl::init(1000));

static llvm::cl::opt<bool> showStats(
    "show-stats",
    llvm::cl::desc(
        "Print detailed stats (min, max, median, stddev, cv) in JSON format. "
        "In case of small kernels print total_cpu_time and number of "
        "iterations."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> showAllMeasurements(
    "show-all-measurements",
    llvm::cl::desc(
        "Print all individual timing measurements in JSON format. In case of "
        "small kernels print total_cpu_time and number of iterations."),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> benchmarkConfig(
    "benchmark-config",
    llvm::cl::desc(
        "Run benchmark with specific perf config only (skip tuning)"),
    llvm::cl::value_desc("perf config string"), llvm::cl::init(""));

static llvm::cl::opt<unsigned> numCompileThreads(
    "num-compile-threads",
    llvm::cl::desc("Number of parallel compilation threads (0 = auto)"),
    llvm::cl::value_desc("thread count"), llvm::cl::init(0));

// Ripped out of JitRunner.cpp
static OwningOpRef<ModuleOp> parseMLIRInput(StringRef inputFilename,
                                            MLIRContext *context) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, context);
}

static benchmark::DataType getDataType(Type inputType) {
  if (inputType.isF32()) {
    return benchmark::DataType::F32;
  } else if (inputType.isInteger(32)) {
    return benchmark::DataType::I32;
  } else if (inputType.isF16()) {
    return benchmark::DataType::F16;
  } else if (inputType.isBF16()) {
    return benchmark::DataType::BF16;
  } else if (inputType.isInteger(8)) {
    return benchmark::DataType::I8;
  } else if (isa<Float8E4M3FNUZType, Float8E4M3FNType, Float8E5M2Type,
                 Float8E5M2FNUZType>(inputType)) {
    return benchmark::DataType::F8;
  } else if (isa<Float8E8M0FNUType>(inputType)) {
    return benchmark::DataType::F8E8M0FNU;
  } else if (isa<Float4E2M1FNType>(inputType)) {
    return benchmark::DataType::F4;
  } else {
    llvm::errs() << "Unknown data type: " << inputType << "\n";
    llvm_unreachable("Kernels only accept ints or floats");
  }
}

// intentionally leaky macro
#define HIPCHECK(expr)                                                         \
  if (hipSuccess != (expr)) {                                                  \
    return failure();                                                          \
  }

static double computeMedian(const std::vector<double> &values) {
  if (values.empty())
    return 0.0;

  assert(std::is_sorted(values.begin(), values.end()) &&
         "values must be sorted");

  size_t n = values.size();
  if (n % 2 == 0) {
    return (values[n / 2 - 1] + values[n / 2]) / 2.0;
  }
  // else
  return values[n / 2];
}

static double computeMean(const std::vector<double> &values) {
  if (values.empty())
    return 0.0;

  double sum = 0.0;
  for (double value : values) {
    sum += value;
  }

  return sum / values.size();
}

static double computeStdDev(const std::vector<double> &values, double mean) {
  if (values.size() < 2)
    return 0.0;

  double sumSquares = 0.0;
  for (float val : values) {
    double diff = val - mean;
    sumSquares += diff * diff;
  }

  return std::sqrt(sumSquares / values.size());
}

static std::vector<double> trimValues(const std::vector<double> &values,
                                      unsigned trimPct) {
  if (values.empty() || trimPct == 0)
    return values;

  if (trimPct >= 50)
    return {};

  assert(std::is_sorted(values.begin(), values.end()) &&
         "values must be sorted");

  size_t trimCount = values.size() * trimPct / 100;
  size_t startIdx = trimCount;
  size_t endIdx = values.size() - trimCount;

  return std::vector<double>(values.begin() + startIdx,
                             values.begin() + endIdx);
}

struct BenchmarkParams {
  unsigned numIterations;
  unsigned warmupIterations;
  bool useMedian;
  unsigned trimPercent;
  unsigned sleepUs;
  bool showStats;
  bool showAllMeasurements;
  rock::TuningParamSetKind tuningSpaceKind;
  const unsigned numCompileThreads;
  std::string benchmarkConfig;
};

enum class CompilationStatus {
  NotApplicable,     // Config not applicable for this kernel
  CompilationFailed, // Config applicable but compilation failed
  Success            // Successfully compiled
};

struct CompilationResult {
  SmallString<64> perfConfig;
  CompilationStatus status = CompilationStatus::NotApplicable;
  SmallVector<std::string> hipModules;
  SmallVector<uint32_t> blockSizes;
  SmallVector<uint32_t> gridSizes;
};

static LogicalResult
measureSmallKernel(unsigned iterations, hipStream_t stream,
                   const std::vector<hipFunction_t> &functions,
                   ArrayRef<uint32_t> blockSizes, ArrayRef<uint32_t> gridSizes,
                   std::vector<void *> &argPointers,
                   std::vector<double> &measurements, double &smallKernelCpuMs,
                   bool benchmarkMode) {
  // Special case for small kernels, where we measure the time for all kernels
  // at once, using CPU timers.
  auto iterationStart = std::chrono::steady_clock::now();
  for (unsigned iter = 0; iter < iterations; ++iter) {
    // Do not flush caches in benchmark mode, as we do not want to
    // time the cache flush (it's okay if we are running in tuning mode).
    if (!benchmarkMode) {
      if (failed(flushInstructionCache(stream))) {
        return failure();
      }
      if (failed(flushL2Cache(stream))) {
        return failure();
      }
    }
    for (auto [func, blockSize, gridSize] :
         llvm::zip(functions, blockSizes, gridSizes)) {
      HIPCHECK(hipExtModuleLaunchKernel(
          func, gridSize * blockSize, 1, 1, blockSize, 1, 1, 0, stream,
          argPointers.data(), nullptr, nullptr, nullptr));
    }
  }

  HIPCHECK(hipStreamSynchronize(stream));
  smallKernelCpuMs = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - iterationStart)
                         .count();
  measurements.push_back(smallKernelCpuMs / iterations);
  return success();
}

static LogicalResult
measureLargeKernel(unsigned iterations, hipStream_t stream,
                   const std::vector<hipFunction_t> &functions,
                   ArrayRef<uint32_t> blockSizes, ArrayRef<uint32_t> gridSizes,
                   std::vector<void *> &argPointers,
                   std::vector<double> &measurements) {
  // Measure runs normally.
  for (unsigned iter = 0; iter < iterations; ++iter) {
    if (failed(flushInstructionCache(stream))) {
      return failure();
    }
    if (failed(flushL2Cache(stream))) {
      return failure();
    }

    double totalMilliseconds = 0.0;

    for (auto [func, blockSize, gridSize] :
         llvm::zip(functions, blockSizes, gridSizes)) {
      hipEvent_t startEvent, stopEvent;
      HIPCHECK(hipEventCreate(&startEvent));
      HIPCHECK(hipEventCreate(&stopEvent));

      HIPCHECK(hipExtModuleLaunchKernel(
          func, gridSize * blockSize, 1, 1, blockSize, 1, 1, 0, stream,
          argPointers.data(), nullptr, startEvent, stopEvent));
      HIPCHECK(hipEventSynchronize(stopEvent));

      float currentMilliseconds = 0.0;
      HIPCHECK(
          hipEventElapsedTime(&currentMilliseconds, startEvent, stopEvent));

      HIPCHECK(hipEventDestroy(stopEvent));
      HIPCHECK(hipEventDestroy(startEvent));

      totalMilliseconds += static_cast<double>(currentMilliseconds);
    }

    measurements.push_back(totalMilliseconds);
  }

  return success();
}

// In order to match rocprof, returns time in nanoseconds
static FailureOr<double>
benchmarkKernels(ArrayRef<std::string> binaries,
                 ArrayRef<std::string> funcNames, ArrayRef<uint32_t> blockSizes,
                 ArrayRef<uint32_t> gridSizes, ArrayRef<void *> hostBuffers,
                 MutableArrayRef<void *> gpuBuffers,
                 ArrayRef<size_t> bufferSizes, const BenchmarkParams &params) {
  bool benchmarkMode = !params.benchmarkConfig.empty();
  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));
  auto streamCleanup = llvm::make_scope_exit([&]() {
    hipError_t destroyStatus = hipStreamDestroy(stream);
    if (destroyStatus != hipSuccess) {
      llvm::errs() << "HIP error in hipStreamDestroy: "
                   << hipGetErrorString(destroyStatus) << "\n";
    }
  });

  // Initialize device buffers
  for (size_t i = 0; i < bufferSizes.size(); i++) {
    HIPCHECK(hipMemcpyAsync(gpuBuffers[i], hostBuffers[i], bufferSizes[i],
                            hipMemcpyHostToDevice, stream));
  }

  // HIP wants an array of pointers to each argument
  std::vector<void *> argPointers;
  for (void *&item : gpuBuffers) {
    argPointers.push_back(reinterpret_cast<void *>(&item));
  }

  // Load all modules once to reduce overhead
  std::vector<hipModule_t> modules;
  std::vector<hipFunction_t> functions;
  auto moduleCleanup = llvm::make_scope_exit([&]() {
    for (hipModule_t mod : modules) {
      if (!mod)
        continue;
      hipError_t status = hipModuleUnload(mod);
      if (status != hipSuccess) {
        llvm::errs() << "HIP error in hipModuleUnload: "
                     << hipGetErrorString(status) << "\n";
      }
    }
  });

  for (auto [binary, funcName] : llvm::zip(binaries, funcNames)) {
    hipModule_t mod;
    HIPCHECK(hipModuleLoadData(&mod, binary.c_str()));
    modules.push_back(mod);

    hipFunction_t func;
    HIPCHECK(hipModuleGetFunction(&func, mod, funcName.c_str()));
    functions.push_back(func);
  }

  // Sleep guard to avoid GPU throttling
  auto sleepGuard = llvm::make_scope_exit([&params] {
    if (params.sleepUs > 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(params.sleepUs));
    }
  });

  bool isSmallKernel = false;
  unsigned iterations = params.numIterations;

  if (params.warmupIterations > 0) {
    // Warmup run. We measure the warmup to get an estimate of the kernel
    // runtime. We will use this estimate to determine if the kernel is small or
    // not.
    double totalMillisecondsWarmup = 0.0;
    for (unsigned iter = 0; iter < params.warmupIterations; ++iter) {
      for (auto [func, blockSize, gridSize] :
           llvm::zip(functions, blockSizes, gridSizes)) {
        hipEvent_t startEvent, stopEvent;
        HIPCHECK(hipEventCreate(&startEvent));
        HIPCHECK(hipEventCreate(&stopEvent));

        HIPCHECK(hipExtModuleLaunchKernel(
            func, gridSize * blockSize, 1, 1, blockSize, 1, 1, 0, stream,
            argPointers.data(), nullptr, startEvent, stopEvent));

        HIPCHECK(hipStreamSynchronize(stream));

        float currentMilliseconds = 0.0;
        HIPCHECK(
            hipEventElapsedTime(&currentMilliseconds, startEvent, stopEvent));

        HIPCHECK(hipEventDestroy(stopEvent));
        HIPCHECK(hipEventDestroy(startEvent));

        totalMillisecondsWarmup += static_cast<double>(currentMilliseconds);
      }
    }
    totalMillisecondsWarmup /= params.warmupIterations;
    assert(totalMillisecondsWarmup >= 0.0f &&
           "totalMillisecondsWarmup must be greater than 0");

    // We want to get at least 1ms of kernel execution time
    // (counting all iterations), so increase the number of iterations
    // if necessary.
    constexpr float minTotalMilliseconds = 1.0f;
    iterations = std::max<unsigned>(
        iterations, static_cast<unsigned>(std::ceil(minTotalMilliseconds /
                                                    totalMillisecondsWarmup)));

    // Depending on the runtime of the kernel,
    // we will use a different approach to measure the runs.
    // We consider a kernel to be small if a single iteration takes less than
    // 1ms to run.
    constexpr float smallKernelThreshold = 1.0f;
    isSmallKernel = totalMillisecondsWarmup < smallKernelThreshold;
  }

  // Measure runs
  std::vector<double> measurements;
  double smallKernelCpuMs = 0.0;

  if (isSmallKernel) {
    if (failed(measureSmallKernel(iterations, stream, functions, blockSizes,
                                  gridSizes, argPointers, measurements,
                                  smallKernelCpuMs, benchmarkMode)))
      return failure();
  } else {
    if (failed(measureLargeKernel(iterations, stream, functions, blockSizes,
                                  gridSizes, argPointers, measurements)))
      return failure();
  }

  if (params.showAllMeasurements) {
    if (isSmallKernel) {
      llvm::outs() << "{\"total_cpu_time\":" << smallKernelCpuMs
                   << ",\"iterations\":" << iterations << "}\t";
    } else {
      llvm::outs() << "[";
      for (size_t i = 0; i < measurements.size(); ++i) {
        if (i > 0)
          llvm::outs() << ",";
        llvm::outs() << measurements[i];
      }
      llvm::outs() << "]\t";
    }
  }

  std::sort(measurements.begin(), measurements.end());

  if (params.showStats) {
    // We cannot show the rest of the stats because the small kernel case uses
    // one timer only, so we cannot actually compute the min, max, etc.
    if (isSmallKernel) {
      llvm::outs() << "{\"total_cpu_time\":" << smallKernelCpuMs
                   << ",\"iterations\":" << iterations << "}\t";
    }
    if (measurements.size() > 1) {
      float median = computeMedian(measurements);
      float min = measurements.front();
      float max = measurements.back();
      float mean = computeMean(measurements);
      float stdDev = computeStdDev(measurements, mean);
      float coefficientOfVariation = (mean > 0) ? (stdDev / mean * 100) : 0;
      llvm::outs() << "{\"min\":" << min << ",\"median\":" << median
                   << ",\"max\":" << max << ",\"stddev\":" << stdDev
                   << ",\"cv\":" << coefficientOfVariation << "}\t";
    }
  }

  auto msToNs = [](double ms) { return 1e6 * ms; };
  if (params.useMedian)
    return msToNs(computeMedian(measurements));
  else
    return msToNs(computeMean(trimValues(measurements, params.trimPercent)));
}

static int toKernelOrder(Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr); intAttr)
    return intAttr.getInt();
  return -1;
}

static LogicalResult extractFuncOps(ModuleOp op,
                                    SmallVectorImpl<func::FuncOp> &kernels) {
  if (!op->hasAttr("mhal.arch")) {
    return op->emitOpError(
        "no architecture set, set mhal.arch on the input module");
  }
  op.walk([&kernels](func::FuncOp f) {
    Attribute kernel = f->getAttr("kernel");
    if (!kernel)
      return;
    kernels.push_back(f);
  });

  std::sort(kernels.begin(), kernels.end(),
            [](const func::FuncOp &a, const func::FuncOp &b) {
              int kernelA = toKernelOrder(a->getAttr("kernel"));
              int kernelB = toKernelOrder(b->getAttr("kernel"));
              return kernelA < kernelB;
            });
  return success();
}

static bool doesModuleHaveFusions(ModuleOp module) {
  WalkResult result = module.walk([](Operation *op) {
    // Check for linalg.generic or rock.reduce (standalone fusion ops)
    if (isa<linalg::GenericOp>(op) || isa<rock::ReduceOp>(op)) {
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

static LogicalResult runTuningLoop(ModuleOp source) {
  // Verify prerequisites
  SmallVector<func::FuncOp> funcs;
  if (failed(extractFuncOps(source, funcs)))
    return failure();
  // We need a copy since HIP'll want a C string
  SmallVector<std::string> kernelFuncNames;
  SmallVector<size_t> bufferLengths;
  for (func::FuncOp &funcOp : funcs) {
    kernelFuncNames.push_back(funcOp.getSymName().str());
  }
  ArrayRef<Type> argTypes = funcs[0].getArgumentTypes();
  for (Type argType : argTypes) {
    auto shapedTy = dyn_cast<ShapedType>(argType);
    if (!shapedTy) {
      return funcs[0].emitOpError("all kernel inputs must be shaped types");
    }
    if (!shapedTy.hasStaticShape()) {
      return funcs[0].emitOpError(
          "all kernel arguments must have static shape");
    }
    int64_t sizeInBits =
        shapedTy.getNumElements() * shapedTy.getElementTypeBitWidth();
    bufferLengths.push_back(llvm::divideCeil(sizeInBits, 8));
  }

  // 2. Set up compilation options (shared across all threads)
  rock::KernelOptions applicabilityOpts;
  applicabilityOpts.applicabilityMode =
      mlir::rock::ApplicabilityMode::Applicability;
  applicabilityOpts.tuningFallback = false;

  rock::KernelOptions compilationKernOpts;
  compilationKernOpts.applicabilityMode =
      mlir::rock::ApplicabilityMode::NonApplicability;
  compilationKernOpts.tuningFallback = false;

  RocmDeviceName deviceName;
  StringRef archName =
      source->getAttrOfType<StringAttr>("mhal.arch").getValue();
  if (failed(deviceName.parse(archName)))
    return source->emitOpError("could not parse arch name: " + archName);
  rock::BackendOptions backendOpts;
  backendOpts.triple = deviceName.getTriple().str();
  backendOpts.chip = deviceName.getChip().str();
  std::string backendFeatures = deviceName.getFeaturesForBackend();
  backendOpts.features = backendFeatures;
  backendOpts.optLevel = 3;
  backendOpts.suppressDiagnostic = true;

  // 3. Initialize host buffers and allocate device buffers
  std::vector<void *> hostBuffers;
  std::vector<void *> gpuBuffers;
  auto bufferCleanup = llvm::make_scope_exit([&]() {
    for (void *buffer : hostBuffers)
      free(buffer);
    for (void *buffer : gpuBuffers) {
      // hipFree does not allow nullptrs, so make sure to check for it first
      if (!buffer)
        continue;
      hipError_t status = hipFree(buffer);
      if (status != hipSuccess) {
        llvm::errs() << "HIP error in hipFree(buffer): "
                     << hipGetErrorString(status) << "\n";
      }
    }
    if (failed(cleanupCacheFlushArtifacts())) {
      llvm::errs() << "Failed to cleanup cache flush artifacts\n";
    }
  });
  assert(argTypes.size() == bufferLengths.size() &&
         "number of arguments and buffer lengths must match");
  for (auto [argType, bufferLength] : llvm::zip(argTypes, bufferLengths)) {
    benchmark::DataType type = getDataType(getElementTypeOrSelf(argType));
    void *hostBuffer = benchmark::allocAndFill(type, bufferLength);
    void *gpuBuffer = nullptr;
    hipError_t hipStatus = hipMalloc(&gpuBuffer, bufferLength);
    if (hipStatus != hipSuccess) {
      free(hostBuffer);
      llvm::errs() << "HIP error in hipMalloc(gpuBuffer): "
                   << hipGetErrorString(hipStatus) << "\n";
      return failure();
    }
    hostBuffers.push_back(hostBuffer);
    gpuBuffers.push_back(gpuBuffer);
  }

  // 4. Multi-iteration tuning loop
  SmallString<64> bestConfigOverall;
  float bestTimeOverall = std::numeric_limits<float>::max();

  // NOTE: Compilation (PassManager::run()) resets the cl opts, so we have to
  // save the values.
  const BenchmarkParams benchmarkParams = {
      numIterations,     warmupIterations, useMedian,           trimPercent,
      sleepUs,           showStats,        showAllMeasurements, tuningSpaceKind,
      numCompileThreads, benchmarkConfig};

  unsigned numTuningIterations =
      rock::getNumberOfIterations(benchmarkParams.tuningSpaceKind);
  if (!benchmarkParams.benchmarkConfig.empty() && numTuningIterations != 1) {
    llvm::errs() << "benchmarking should do a single tuning iteration\n";
    return failure();
  }

  // Main iteration loop - wraps config generation, compilation, AND
  // benchmarking
  for (unsigned iterIdx = 0; iterIdx < numTuningIterations; ++iterIdx) {
    // PHASE 1: Collect perf configs for this iteration
    std::vector<SmallString<64>> configs;

    if (!benchmarkParams.benchmarkConfig.empty()) {
      // Benchmark mode - just one config
      configs.emplace_back(benchmarkParams.benchmarkConfig);
    } else {
      // Tuning mode - get configs from tuning space
      rock::TuningParamSpaceSettings settings{iterIdx, bestConfigOverall};
      std::unique_ptr<rock::TuningParamSet> tuningSpace(
          rock::createTunableParamSpace(source, benchmarkParams.tuningSpaceKind,
                                        settings));

      if (tuningSpace->tuningRange.empty()) {
        llvm::errs() << "Tuning range is empty for iteration " << iterIdx
                     << "\n";
        return failure();
      }

      for (rock::RockTuningParamAttrInterface tuningAttr :
           tuningSpace->tuningRange) {
        SmallString<64> perfConfig;
        tuningAttr.getPerfConfigStr(perfConfig);
        configs.push_back(perfConfig);
      }
    }

    // Determine number of parallel threads
    unsigned numThreads = (benchmarkParams.numCompileThreads > 0)
                              ? benchmarkParams.numCompileThreads
                              : std::thread::hardware_concurrency();
    if (numThreads == 0)
      numThreads = 4; // fallback

    // Don't create more threads than configs to compile
    numThreads = std::min(numThreads, static_cast<unsigned>(configs.size()));

    // Serialize source module once (shared by all threads for cloning)
    std::string sourceModuleStr;
    llvm::raw_string_ostream sourceOs(sourceModuleStr);
    source->print(sourceOs);
    sourceOs.flush();

    // PHASE 2: Parallel compilation phase
    ConcurrentQueue<CompilationResult> compilationResults;
    std::mutex outputMutex; // For thread-safe console output

    auto compileConfig = [&](size_t idx) -> CompilationResult {
      CompilationResult result;
      result.perfConfig = configs[idx];
      // Each thread needs its own context and pass managers for thread-safety
      DialectRegistry threadRegistry;
      registerRocMLIRDialects(threadRegistry);
      MLIRContext threadCtx(threadRegistry);
      threadCtx.getDiagEngine().registerHandler([](Diagnostic &diag) {});

      // Parse the serialized module in this thread's context
      OwningOpRef<ModuleOp> threadSource =
          parseSourceString<ModuleOp>(sourceModuleStr, &threadCtx);
      if (!threadSource)
        return result;

      // Set up pipelines for this thread
      PassManager threadApplicability(&threadCtx,
                                      PassManager::getAnyOpAnchorName(),
                                      PassManager::Nesting::Implicit);
      PassManager threadCompilation(&threadCtx,
                                    PassManager::getAnyOpAnchorName(),
                                    PassManager::Nesting::Implicit);

      rock::buildKernelPipeline(threadApplicability, applicabilityOpts);
      rock::buildKernelPipeline(threadCompilation, compilationKernOpts);
      rock::buildBackendPipeline(threadCompilation, backendOpts);

      StringAttr perfConfigAttr =
          StringAttr::get(&threadCtx, result.perfConfig);

      // Helper to copy IR with perf config set
      auto copyIRThread = [&](ModuleOp src,
                              StringAttr attr) -> OwningOpRef<ModuleOp> {
        OwningOpRef<ModuleOp> copy = cast<ModuleOp>(src->clone());
        copy->walk([&attr](rock::RockGemmWrapperInterface op) {
          op->setAttr("perf_config", attr);
        });
        copy->walk([&attr](rock::RockGemmGemmWrapperInterface op) {
          op->setAttr("perf_config", attr);
        });
        return copy;
      };

      if (doesModuleHaveFusions(threadSource.get()) &&
          !rock::isModuleFusible(threadSource.get(), result.perfConfig)) {
        result.status = CompilationStatus::NotApplicable;
        return result;
      }

      // Applicability check
      OwningOpRef<ModuleOp> sourceCopy =
          copyIRThread(threadSource.get(), perfConfigAttr);
      if (failed(threadApplicability.run(sourceCopy.get()))) {
        result.status = CompilationStatus::NotApplicable;
        return result;
      }

      // Extract block and grid sizes
      for (auto &fnName : kernelFuncNames) {
        auto tunedFunc = sourceCopy->lookupSymbol<func::FuncOp>(fnName);
        if (!tunedFunc) {
          result.status = CompilationStatus::CompilationFailed;
          return result;
        }
        result.blockSizes.push_back(
            tunedFunc->getAttrOfType<IntegerAttr>("block_size").getInt());
        result.gridSizes.push_back(
            tunedFunc->getAttrOfType<IntegerAttr>("grid_size").getInt());
      }

      // Compilation
      if (failed(threadCompilation.run(sourceCopy.get()))) {
        std::lock_guard<std::mutex> lock(outputMutex);
        llvm::errs() << "Backend pipeline failed for config: "
                     << result.perfConfig << "\n";
        result.status = CompilationStatus::CompilationFailed;
        return result;
      }

      // Extract binaries
      for (const auto &fnName : kernelFuncNames) {
        auto binary =
            sourceCopy->lookupSymbol<gpu::BinaryOp>(fnName + "_module");
        if (!binary) {
          result.status = CompilationStatus::CompilationFailed;
          return result;
        }
        result.hipModules.push_back(
            cast<gpu::ObjectAttr>(binary.getObjects()[0])
                .getObject()
                .getValue()
                .str());
      }

      result.status = CompilationStatus::Success;
      return result;
    };

    // Launch parallel compilation tasks with dynamic work stealing
    // Note: We use atomic counter instead of static partitioning because
    // compilation times vary dramatically between configs (NotApplicable is
    // fast, full compilation is slow). Dynamic work stealing provides better
    // load balancing by allowing fast threads to pick up more work.
    std::atomic<size_t> nextIdx{0};
    std::atomic<size_t> activeThreads{numThreads};
    auto worker = [&] {
      while (true) {
        size_t idx = nextIdx.fetch_add(1, std::memory_order_relaxed);
        if (idx >= configs.size())
          break;

        if (compilationResults.isTerminated())
          break; // Avoid unnecessary work

        if (!compilationResults.push(compileConfig(idx)))
          break; // Queue terminated
      }

      if (activeThreads.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        // Last thread - signal termination
        compilationResults.terminate();
      }
    };

    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    for (unsigned i = 0; i < numThreads; ++i) {
      threads.emplace_back(worker);
    }

    auto threadCleanup = llvm::make_scope_exit([&] {
      // In case of early termination, signal all threads to stop
      compilationResults.terminate();
      for (auto &t : threads) {
        t.join();
      }
    });

    int64_t validResults = 0;
    // Sequential benchmarking phase (must be sequential for accurate timing)
    CompilationResult result;
    while (compilationResults.pop(result)) {
      llvm::outs() << result.perfConfig << "\t";

      if (result.status == CompilationStatus::CompilationFailed) {
        llvm::errs() << "Compilation failed\n";
        return failure();
      }

      if (result.status == CompilationStatus::NotApplicable) {
        llvm::outs() << "N/A\n";
        continue;
      }

      assert(result.status == CompilationStatus::Success &&
             "Unexpected compilation status in benchmarking phase");

      FailureOr<double> timing =
          benchmarkKernels(result.hipModules, kernelFuncNames,
                           result.blockSizes, result.gridSizes, hostBuffers,
                           gpuBuffers, bufferLengths, benchmarkParams);

      if (failed(timing)) {
        llvm::errs() << "Kernel execution failed\n";
        return failure();
      }
      llvm::outs() << timing.value() << "\n";

      validResults++;
      // Find best config
      if (rock::needToUpdateBest(benchmarkParams.tuningSpaceKind)) {
        if (timing.value() < bestTimeOverall) {
          bestTimeOverall = timing.value();
          bestConfigOverall = result.perfConfig;
        }
      }
    }

    if (validResults == 0) {
      llvm::errs() << "No valid configurations found\n";
      return failure();
    }
  } // End of iteration loop

  return success();
}
#undef HIPCHECK

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  mlir::registerMLIRCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "rocMLIR tuning driver");

  if (trimPercent >= 50) {
    llvm::errs() << "trim-percent must be less than 50 to avoid trimming all "
                    "measurements\n";
    return EXIT_FAILURE;
  }

  DialectRegistry registry;
  registerRocMLIRDialects(registry);
  registerRocMLIRPasses();

  MLIRContext ctx(registry);

  OwningOpRef<ModuleOp> source = parseMLIRInput(inputFilename, &ctx);
  if (!source) {
    llvm::errs() << "Could not parse input IR\n";
    return EXIT_FAILURE;
  }

  ModuleOp module;
  WalkResult findModule = source->walk([&](func::FuncOp op) -> WalkResult {
    FailureOr<StringAttr> mayBeArch = rock::getArch(op);
    if (succeeded(mayBeArch)) {
      module = op->getParentOfType<ModuleOp>();
      module->setAttr("mhal.arch", mayBeArch.value());
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!findModule.wasInterrupted()) {
    source->emitOpError(
        "no architecture set, set mhal.arch on the input module or func");
    llvm::errs() << "Tuning loop failed\n";
    return EXIT_FAILURE;
  }

  if (failed(runTuningLoop(module))) {
    llvm::errs() << "Tuning loop failed\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
