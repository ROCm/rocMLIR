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
#include <cstdlib>
#include <mutex>
#include <thread>

// Utilities to allocate buffers
#include "../utils/performance/common/benchmarkUtils.h"
#include "CacheFlush.h"

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

llvm::cl::opt<std::string> inputFilename{
    llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-")};

static llvm::cl::opt<rock::TuningParamSetKind> tuningSpaceKind(
    "tuning-space", llvm::cl::desc("Tuning space to use for this run"),
    llvm::cl::values(
        clEnumValN(rock::TuningParamSetKind::Quick, "quick",
                   "Quick tuning space"),
        clEnumValN(rock::TuningParamSetKind::Full, "full",
                   "Full tuning space, excluding known-bad configurations"),
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
    llvm::cl::desc("Show detailed statistics (min, max, median, stddev, cv)"),
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

static float computeMedian(const std::vector<float> &values) {
  if (values.empty())
    return 0.0;

  assert(std::is_sorted(values.begin(), values.end()) &&
         "values must be sorted");

  size_t n = values.size();
  if (n % 2 == 0) {
    return (values[n / 2 - 1] + values[n / 2]) / 2.0;
  } else {
    return values[n / 2];
  }
}

static float computeMean(const std::vector<float> &values) {
  if (values.empty())
    return 0.0;

  float sum = 0.0;
  for (size_t i = 0; i < values.size(); ++i) {
    sum += values[i];
  }

  return sum / values.size();
}

static float computeStdDev(const std::vector<float> &values, float mean) {
  if (values.size() < 2)
    return 0.0;

  float sumSquares = 0.0;
  for (float val : values) {
    float diff = val - mean;
    sumSquares += diff * diff;
  }

  return std::sqrt(sumSquares / values.size());
}

static std::vector<float> trimValues(const std::vector<float> &values,
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

  return std::vector<float>(values.begin() + startIdx, values.begin() + endIdx);
}

struct BenchmarkParams {
  unsigned numIterations;
  unsigned warmupIterations;
  bool useMedian;
  unsigned trimPercent;
  unsigned sleepUs;
  bool showStats;
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

// In order to match rocprof, returns time in nanoseconds
static FailureOr<double>
benchmarkKernels(ArrayRef<std::string> binaries,
                 ArrayRef<std::string> funcNames, ArrayRef<uint32_t> blockSizes,
                 ArrayRef<uint32_t> gridSizes, ArrayRef<void *> hostBuffers,
                 MutableArrayRef<void *> gpuBuffers,
                 ArrayRef<size_t> bufferSizes, const BenchmarkParams &params) {
  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

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

  // Warmup run
  for (unsigned iter = 0; iter < params.warmupIterations; ++iter) {
    for (auto [func, blockSize, gridSize] :
         llvm::zip(functions, blockSizes, gridSizes)) {
      HIPCHECK(hipExtModuleLaunchKernel(
          func, gridSize * blockSize, 1, 1, blockSize, 1, 1, 0, stream,
          argPointers.data(), nullptr, nullptr, nullptr));
    }
  }

  // Measure runs
  std::vector<float> measurements;

  for (unsigned iter = 0; iter < params.numIterations; ++iter) {
    if (failed(flushInstructionCache(stream))) {
      return failure();
    }
    if (failed(flushL2Cache(stream))) {
      return failure();
    }
    HIPCHECK(hipStreamSynchronize(stream));

    float totalMilliseconds = 0.0;

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

      totalMilliseconds += currentMilliseconds;
    }

    measurements.push_back(totalMilliseconds);
  }

  for (hipModule_t mod : modules) {
    HIPCHECK(hipModuleUnload(mod));
  }

  HIPCHECK(hipStreamDestroy(stream));

  std::sort(measurements.begin(), measurements.end());

  if (params.showStats && measurements.size() > 1) {
    float median = computeMedian(measurements);
    float min = measurements.front();
    float max = measurements.back();
    float mean = computeMean(measurements);
    float stdDev = computeStdDev(measurements, mean);
    float coefficientOfVariation = (mean > 0) ? (stdDev / mean * 100) : 0;
    llvm::outs() << "[min: " << min << ", median: " << median
                 << ", max: " << max << ", stddev: " << stdDev
                 << ", cv: " << coefficientOfVariation << "%]\t";
  }

  auto msToNs = [](float ms) { return 1e6 * static_cast<double>(ms); };
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
  assert(argTypes.size() == bufferLengths.size() &&
         "number of arguments and buffer lengths must match");
  for (auto [argType, bufferLength] : llvm::zip(argTypes, bufferLengths)) {
    benchmark::DataType type = getDataType(getElementTypeOrSelf(argType));
    void *hostBuffer = benchmark::allocAndFill(type, bufferLength);
    void *gpuBuffer = nullptr;
    HIPCHECK(hipMalloc(&gpuBuffer, bufferLength));
    hostBuffers.push_back(hostBuffer);
    gpuBuffers.push_back(gpuBuffer);
  }

  // 4. Collect perf configs to compile
  std::vector<SmallString<64>> configs;
  if (!benchmarkConfig.empty()) {
    // Benchmark mode - just one config
    configs.emplace_back(benchmarkConfig);
  } else {
    // Tuning mode - all configs from tuning space
    std::unique_ptr<rock::TuningParamSet> tuningSpace(
        rock::createTunableParamSpace(source, tuningSpaceKind));

    if (tuningSpace->tuningRange.empty()) {
      llvm::errs() << "Tuning range is empty\n";
      return failure();
    }

    for (rock::RockTuningParamAttrInterface tuningAttr :
         tuningSpace->tuningRange) {
      SmallString<64> perfConfig;
      tuningAttr.getPerfConfigStr(perfConfig);
      configs.push_back(perfConfig);
    }
  }

  // NOTE: Compilation (PassManager::run()) resets the cl opts, so we have to
  // save the values.
  const BenchmarkParams benchmarkParams = {numIterations, warmupIterations,
                                           useMedian,     trimPercent,
                                           sleepUs,       showStats};

  // Determine number of parallel threads
  unsigned numThreads = (numCompileThreads > 0)
                            ? numCompileThreads
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

  // Parallel compilation phase
  std::vector<CompilationResult> compilationResults(configs.size());
  std::mutex outputMutex; // For thread-safe console output
  std::atomic<bool> compilationFailed{
      false}; // Flag to signal early termination

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
    PassManager threadCompilation(&threadCtx, PassManager::getAnyOpAnchorName(),
                                  PassManager::Nesting::Implicit);

    rock::buildKernelPipeline(threadApplicability, applicabilityOpts);
    rock::buildKernelPipeline(threadCompilation, compilationKernOpts);
    rock::buildBackendPipeline(threadCompilation, backendOpts);

    StringAttr perfConfigAttr = StringAttr::get(&threadCtx, result.perfConfig);

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

    // Applicability check
    OwningOpRef<ModuleOp> sourceCopy =
        copyIRThread(threadSource.get(), perfConfigAttr);
    if (!rock::isModuleFusible(sourceCopy.get(), result.perfConfig)) {
      result.status = CompilationStatus::NotApplicable;
      return result;
    }

    if (failed(threadApplicability.run(sourceCopy.get()))) {
      result.status = CompilationStatus::NotApplicable;
      return result;
    }

    // Extract block and grid sizes
    for (auto &fnName : kernelFuncNames) {
      auto tunedFunc = sourceCopy->lookupSymbol<func::FuncOp>(fnName);
      if (!tunedFunc) {
        result.status = CompilationStatus::CompilationFailed;
        compilationFailed.store(true, std::memory_order_relaxed);
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
      compilationFailed.store(true, std::memory_order_relaxed);
      return result;
    }

    // Extract binaries
    for (const auto &fnName : kernelFuncNames) {
      auto binary = sourceCopy->lookupSymbol<gpu::BinaryOp>(fnName + "_module");
      if (!binary) {
        result.status = CompilationStatus::CompilationFailed;
        compilationFailed.store(true, std::memory_order_relaxed);
        return result;
      }
      result.hipModules.push_back(cast<gpu::ObjectAttr>(binary.getObjects()[0])
                                      .getObject()
                                      .getValue()
                                      .str());
    }

    result.status = CompilationStatus::Success;
    return result;
  };

  // Launch parallel compilation tasks with dynamic work stealing
  // Note: We use atomic counter instead of static partitioning because
  // compilation times vary dramatically between configs (NotApplicable is fast,
  // full compilation is slow). Dynamic work stealing provides better load
  // balancing by allowing fast threads to pick up more work.
  {
    std::atomic<size_t> nextIdx{0};

    // Thread pool with work stealing pattern
    auto worker = [&]() {
      while (true) {
        // Check if any compilation has failed (relaxed: just an optimization
        // hint)
        if (compilationFailed.load(std::memory_order_relaxed))
          break;

        size_t idx = nextIdx.fetch_add(1, std::memory_order_relaxed);
        if (idx >= configs.size())
          break;

        compilationResults[idx] = compileConfig(idx);
      }
    };

    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    for (unsigned i = 0; i < numThreads; ++i) {
      threads.emplace_back(worker);
    }

    for (auto &t : threads) {
      t.join();
    }
  }

  // Check if any compilation failed and terminate early
  if (compilationFailed.load(std::memory_order_relaxed)) {
    llvm::errs()
        << "Compilation failed for one or more configs. Terminating.\n";
    return failure();
  }

  // Sequential benchmarking phase (must be sequential for accurate timing)
  // Note: Due to early exit on compilation failures, only NotApplicable and
  // Success statuses are possible here.
  for (const auto &result : compilationResults) {
    llvm::outs() << result.perfConfig << "\t";

    if (result.status == CompilationStatus::NotApplicable) {
      llvm::outs() << "N/A\n";
      continue;
    }

    // At this point, status must be Success (we exited early on any failures)
    assert(result.status == CompilationStatus::Success &&
           "Unexpected compilation status in benchmarking phase");

    FailureOr<double> timing = benchmarkKernels(
        result.hipModules, kernelFuncNames, result.blockSizes, result.gridSizes,
        hostBuffers, gpuBuffers, bufferLengths, benchmarkParams);

    if (failed(timing)) {
      llvm::errs() << "Kernel execution failed\n";
      return failure();
    }
    llvm::outs() << timing << "\n";
  }
  for (void *buffer : hostBuffers) {
    free(buffer);
  }
  for (void *buffer : gpuBuffers) {
    HIPCHECK(hipFree(buffer));
  }
  if (failed(cleanupCacheFlushArtifacts())) {
    return failure();
  }
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
