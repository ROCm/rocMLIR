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
#include "llvm/Support/ThreadPool.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <thread>

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

/// Process exit code signalling that a perf config's GPU run exceeded the
/// per-config run-timeout budget (--gpu-run-timeout) and the kernel is
/// presumed hung. Must be non-zero and distinct from EXIT_FAILURE.
static constexpr int kExitGpuTimeout = 3;

//===----------------------------------------------------------------------===//
// Shared Resources for Multi-threaded Compilation
//===----------------------------------------------------------------------===//

/// Returns a shared dialect registry, initialized exactly once.
static DialectRegistry &getSharedDialectRegistry() {
  static std::once_flag initFlag;
  static DialectRegistry registry;
  std::call_once(initFlag, []() { registerRocMLIRDialects(registry); });
  return registry;
}

/// Returns a shared LLVM ThreadPool for all MLIR contexts.
static llvm::DefaultThreadPool &getSharedThreadPool() {
  static llvm::DefaultThreadPool pool;
  return pool;
}

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

static llvm::cl::opt<unsigned> rep(
    "rep",
    llvm::cl::desc("Target benchmark time in milliseconds. The number of "
                   "measured iterations is derived from this budget and the "
                   "estimated per-launch runtime (Triton do_bench style)."),
    llvm::cl::value_desc("benchmark milliseconds"), llvm::cl::init(100));

static llvm::cl::opt<unsigned> warmup(
    "warmup",
    llvm::cl::desc("Target warmup time in milliseconds. The number of warmup "
                   "iterations is derived from this budget and the estimated "
                   "per-launch runtime (Triton do_bench style)."),
    llvm::cl::value_desc("warmup milliseconds"), llvm::cl::init(25));

static llvm::cl::opt<bool> legacyBenchmarkMode(
    "legacy-benchmark-mode",
    llvm::cl::desc(
        "Use the legacy rocMLIR benchmarking method (fixed iteration counts "
        "with a small-vs-large-kernel split) instead of the default "
        "Triton do_bench-style time-budget measurement. Kept for "
        "apples-to-apples comparison against older rocMLIR versions. In this "
        "mode --num-iterations/--warmup-iterations control the run counts and "
        "--rep/--warmup are ignored."),
    llvm::cl::init(false));

static llvm::cl::opt<unsigned> numIterations(
    "num-iterations",
    llvm::cl::desc("Number of times to run each kernel for averaging (only "
                   "used with --legacy-benchmark-mode)"),
    llvm::cl::value_desc("number of runs"), llvm::cl::init(100));

static llvm::cl::opt<unsigned> warmupIterations(
    "warmup-iterations",
    llvm::cl::desc(
        "Number of warmup runs (only used with --legacy-benchmark-mode)"),
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
        "Print detailed stats (min, max, median, stddev, cv) in JSON format."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> showAllMeasurements(
    "show-all-measurements",
    llvm::cl::desc("Print all individual timing measurements in JSON format."),
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

static llvm::cl::opt<bool> waitForCompiles(
    "wait-for-compiles",
    llvm::cl::desc("Wait for all compilations to finish before benchmarking"),
    llvm::cl::init(false));

static llvm::cl::opt<unsigned> gpuRunTimeout(
    "gpu-run-timeout",
    llvm::cl::desc(
        "Per-perf-config GPU-run timeout in seconds. 0 (default) disables the "
        "timeout. This does not include compilation. When > 0, stream "
        "synchronization is bounded with polling; a run that exceeds this "
        "budget is presumed hung and the process exits with a distinct exit "
        "code."),
    llvm::cl::value_desc("seconds"), llvm::cl::init(0));

static llvm::cl::opt<bool> flushLastLevelCache(
    "flush-last-level-cache",
    llvm::cl::desc(
        "Size the cache-flush buffer to the architecture's last-level cache "
        "(e.g. AMD Infinity Cache) instead of the per-XCD L2 cache size "
        "reported by the HIP runtime. Defaults to the L2 cache size."),
    llvm::cl::init(false));

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

// intentionally leaky macro
#define HIPCHECK(expr)                                                         \
  if (hipSuccess != (expr)) {                                                  \
    return failure();                                                          \
  }

using SteadyTimePoint = std::chrono::steady_clock::time_point;

static std::optional<SteadyTimePoint> makeTimeoutDeadline(unsigned timeoutSec) {
  if (timeoutSec == 0)
    return std::nullopt;
  return std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);
}

static bool hasTimedOut(const std::optional<SteadyTimePoint> &deadline) {
  return deadline && std::chrono::steady_clock::now() >= *deadline;
}

static LogicalResult synchronizeStreamWithTimeout(
    hipStream_t stream, const std::optional<SteadyTimePoint> &gpuRunDeadline,
    unsigned timeoutSec, StringRef perfConfig, StringRef phase) {
  // Preserve the historical behavior when no GPU-run timeout was requested.
  if (!gpuRunDeadline) {
    HIPCHECK(hipStreamSynchronize(stream));
    return success();
  }

  // hipStreamSynchronize can block forever when a kernel wedges. Polling the
  // stream gives the driver a chance to stop the run and let tuning advance.
  while (true) {
    hipError_t status = hipStreamQuery(stream);
    if (status == hipSuccess)
      return success();

    if (status != hipErrorNotReady) {
      llvm::errs() << "HIP error while synchronizing stream during " << phase
                   << " for config: " << perfConfig << " - "
                   << hipGetErrorString(status) << "\n";
      return failure();
    }

    if (hasTimedOut(gpuRunDeadline)) {
      llvm::errs() << "GPU run timed out after " << timeoutSec << "s during "
                   << phase << " for config: " << perfConfig
                   << " (kernel presumed hung)\n";
      llvm::errs().flush();
      std::_Exit(kExitGpuTimeout);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
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
  unsigned warmupMs;
  unsigned repMs;
  bool useMedian;
  unsigned trimPercent;
  unsigned sleepUs;
  bool showStats;
  bool showAllMeasurements;
  rock::TuningParamSetKind tuningSpaceKind;
  const unsigned numCompileThreads;
  std::string benchmarkConfig;
  bool waitForCompiles;
  unsigned gpuRunTimeoutSec;
  bool flushLastLevelCache;
  // Legacy benchmarking path (fixed iteration counts + small/large kernel
  // split). When false, the default Triton do_bench-style measurement is used
  // and numIterations/warmupIterations are ignored.
  bool legacyBenchmarkMode;
  unsigned numIterations;
  unsigned warmupIterations;
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

// Thread-local resources to avoid per-config initialization overhead.
// Each worker thread gets its own context, PassManagers, and parsed module.
// Note: MLIR's MLIRContext cannot be safely shared across parallel pass
// executions - it asserts when the registry is modified during multi-threaded
// execution. Therefore, each thread needs its own context.
struct ThreadResources {
  std::unique_ptr<MLIRContext> ctx;
  std::unique_ptr<PassManager> applicabilityPM;
  std::unique_ptr<PassManager> compilationPM;
  OwningOpRef<ModuleOp> sourceModule;

  ThreadResources() = default;
  ThreadResources(ThreadResources &&) = default;
  ThreadResources &operator=(ThreadResources &&) = default;

  // Non-copyable
  ThreadResources(const ThreadResources &) = delete;
  ThreadResources &operator=(const ThreadResources &) = delete;

  // Initialize all resources for this thread
  bool initialize(const std::string &sourceModuleStr,
                  const rock::KernelOptions &applicabilityOpts,
                  const rock::KernelOptions &compilationKernOpts,
                  const rock::BackendOptions &backendOpts) {
    // Use the shared dialect registry (initialized exactly once)
    DialectRegistry &registry = getSharedDialectRegistry();
    // Create context with threading disabled internally, attach shared pool
    ctx = std::make_unique<MLIRContext>(registry,
                                        MLIRContext::Threading::DISABLED);
    ctx->setThreadPool(getSharedThreadPool());
    ctx->loadAllAvailableDialects();
    ctx->getDiagEngine().registerHandler([](Diagnostic &) {});

    // Pre-build pipelines once per thread
    applicabilityPM = std::make_unique<PassManager>(
        ctx.get(), PassManager::getAnyOpAnchorName(),
        PassManager::Nesting::Implicit);
    compilationPM = std::make_unique<PassManager>(
        ctx.get(), PassManager::getAnyOpAnchorName(),
        PassManager::Nesting::Implicit);

    rock::buildKernelPipeline(*applicabilityPM, applicabilityOpts);
    rock::buildKernelPipeline(*compilationPM, compilationKernOpts);
    rock::buildBackendPipeline(*compilationPM, backendOpts);

    // Parse source module once per thread
    sourceModule = parseSourceString<ModuleOp>(sourceModuleStr, ctx.get());
    return sourceModule && *sourceModule;
  }

  bool isValid() const { return sourceModule && *sourceModule; }
};

// Legacy small-kernel path: run all iterations back-to-back and time the whole
// batch with a single CPU timer. Only used with --legacy-benchmark-mode.
static LogicalResult
measureSmallKernel(unsigned iterations, hipStream_t stream,
                   const std::vector<hipFunction_t> &functions,
                   ArrayRef<uint32_t> blockSizes, ArrayRef<uint32_t> gridSizes,
                   std::vector<void *> &argPointers,
                   std::vector<double> &measurements, double &smallKernelCpuMs,
                   bool benchmarkMode, bool useLastLevelCacheSize,
                   const std::optional<SteadyTimePoint> &gpuRunDeadline,
                   unsigned timeoutSec, StringRef perfConfig) {
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
      if (failed(flushCache(stream, useLastLevelCacheSize))) {
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

  if (failed(synchronizeStreamWithTimeout(stream, gpuRunDeadline, timeoutSec,
                                          perfConfig, "measurement")))
    return failure();
  smallKernelCpuMs = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - iterationStart)
                         .count();
  measurements.push_back(smallKernelCpuMs / iterations);
  return success();
}

// Legacy large-kernel path: time each iteration individually with GPU events,
// synchronizing after every launch. Only used with --legacy-benchmark-mode.
static LogicalResult measureLargeKernel(
    unsigned iterations, hipStream_t stream,
    const std::vector<hipFunction_t> &functions, ArrayRef<uint32_t> blockSizes,
    ArrayRef<uint32_t> gridSizes, std::vector<void *> &argPointers,
    std::vector<double> &measurements, bool useLastLevelCacheSize,
    const std::optional<SteadyTimePoint> &gpuRunDeadline, unsigned timeoutSec,
    StringRef perfConfig) {
  // Measure runs normally.
  for (unsigned iter = 0; iter < iterations; ++iter) {
    if (failed(flushInstructionCache(stream))) {
      return failure();
    }
    if (failed(flushCache(stream, useLastLevelCacheSize))) {
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
      if (failed(synchronizeStreamWithTimeout(
              stream, gpuRunDeadline, timeoutSec, perfConfig, "measurement")))
        return failure();

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

static LogicalResult measureKernel(
    unsigned iterations, hipStream_t stream,
    const std::vector<hipFunction_t> &functions, ArrayRef<uint32_t> blockSizes,
    ArrayRef<uint32_t> gridSizes, std::vector<void *> &argPointers,
    std::vector<double> &measurements,
    const std::optional<SteadyTimePoint> &gpuRunDeadline, unsigned timeoutSec,
    StringRef perfConfig, bool useLastLevelCacheSize) {
  // Pre-allocate one event pair per iteration so we can record them all in a
  // tight loop and synchronize only once at the end. This matches Triton's
  // do_bench, which minimizes host-side overhead between launches (no
  // per-iteration synchronization).
  std::vector<hipEvent_t> startEvents(iterations, nullptr);
  std::vector<hipEvent_t> stopEvents(iterations, nullptr);
  llvm::scope_exit eventCleanup([&]() {
    for (hipEvent_t event : startEvents) {
      if (event)
        (void)hipEventDestroy(event);
    }
    for (hipEvent_t event : stopEvents) {
      if (event)
        (void)hipEventDestroy(event);
    }
  });
  for (unsigned iter = 0; iter < iterations; ++iter) {
    HIPCHECK(hipEventCreate(&startEvents[iter]));
    HIPCHECK(hipEventCreate(&stopEvents[iter]));
  }

  // Record all iterations back-to-back. Each measurement brackets the full
  // kernel chain (one start before, one stop after), matching how Triton times
  // the whole callable rather than each kernel individually.
  for (unsigned iter = 0; iter < iterations; ++iter) {
    if (failed(flushInstructionCache(stream))) {
      return failure();
    }
    if (failed(flushCache(stream, useLastLevelCacheSize))) {
      return failure();
    }

    HIPCHECK(hipEventRecord(startEvents[iter], stream));
    for (auto [func, blockSize, gridSize] :
         llvm::zip(functions, blockSizes, gridSizes)) {
      HIPCHECK(hipExtModuleLaunchKernel(
          func, gridSize * blockSize, 1, 1, blockSize, 1, 1, 0, stream,
          argPointers.data(), nullptr, nullptr, nullptr));
    }
    HIPCHECK(hipEventRecord(stopEvents[iter], stream));
  }

  // Single synchronization after all iterations have been queued.
  if (failed(synchronizeStreamWithTimeout(stream, gpuRunDeadline, timeoutSec,
                                          perfConfig, "measurement")))
    return failure();

  for (unsigned iter = 0; iter < iterations; ++iter) {
    float currentMilliseconds = 0.0;
    HIPCHECK(hipEventElapsedTime(&currentMilliseconds, startEvents[iter],
                                 stopEvents[iter]));
    measurements.push_back(static_cast<double>(currentMilliseconds));
  }

  return success();
}

// In order to match rocprof, returns time in nanoseconds
static FailureOr<double>
benchmarkKernels(ArrayRef<std::string> binaries,
                 ArrayRef<std::string> funcNames, ArrayRef<uint32_t> blockSizes,
                 ArrayRef<uint32_t> gridSizes,
                 MutableArrayRef<void *> gpuBuffers, hipStream_t stream,
                 StringRef perfConfig, const BenchmarkParams &params) {
  // HIP wants an array of pointers to each argument
  std::vector<void *> argPointers;
  for (void *&item : gpuBuffers) {
    argPointers.push_back(reinterpret_cast<void *>(&item));
  }

  // Load all modules once to reduce overhead
  std::vector<hipModule_t> modules;
  std::vector<hipFunction_t> functions;
  llvm::scope_exit moduleCleanup([&]() {
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
  llvm::scope_exit sleepGuard([&params] {
    if (params.sleepUs > 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(params.sleepUs));
    }
  });

  // Apply one user-specified GPU-run deadline across all stream
  // synchronization for this perf config. Compilation and module loading are
  // not included.
  std::optional<SteadyTimePoint> gpuRunDeadline =
      makeTimeoutDeadline(params.gpuRunTimeoutSec);

  std::vector<double> measurements;
  double smallKernelCpuMs = 0.0;
  bool isSmallKernel = false;
  unsigned smallKernelIters = 0;

  if (params.legacyBenchmarkMode) {
    // Legacy benchmarking: fixed --num-iterations/--warmup-iterations counts
    // with a small-vs-large-kernel split. Kept for apples-to-apples comparison
    // with older rocMLIR versions.
    bool benchmarkMode = !params.benchmarkConfig.empty();
    unsigned iterations = params.numIterations;

    if (params.warmupIterations > 0) {
      // Warmup run. We measure the warmup to get an estimate of the kernel
      // runtime. We will use this estimate to determine if the kernel is small
      // or not.
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

          if (failed(synchronizeStreamWithTimeout(stream, gpuRunDeadline,
                                                  params.gpuRunTimeoutSec,
                                                  perfConfig, "warmup")))
            return failure();

          float currentMilliseconds = 0.0;
          HIPCHECK(
              hipEventElapsedTime(&currentMilliseconds, startEvent, stopEvent));

          HIPCHECK(hipEventDestroy(stopEvent));
          HIPCHECK(hipEventDestroy(startEvent));

          // hipEventElapsedTime seemingly can return negative values for fast
          // kernels due to GPU clock precision issues. This is extremely
          // relevant when we have a small number of warmup iterations (e.g., 1)
          // for small kernels. Clamp to the documented resolution of ~1
          // microsecond (0.001 ms) if this is the case.
          if (currentMilliseconds < 0.0f) {
            constexpr float minMeasurableMs = 0.001f;
            currentMilliseconds = minMeasurableMs;
          }

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
          iterations, static_cast<unsigned>(std::ceil(
                          minTotalMilliseconds / totalMillisecondsWarmup)));

      // Depending on the runtime of the kernel, we will use a different
      // approach to measure the runs. We consider a kernel to be small if a
      // single iteration takes less than 1ms to run.
      constexpr float smallKernelThreshold = 1.0f;
      isSmallKernel = totalMillisecondsWarmup < smallKernelThreshold;
    }

    smallKernelIters = iterations;
    if (isSmallKernel) {
      if (failed(measureSmallKernel(iterations, stream, functions, blockSizes,
                                    gridSizes, argPointers, measurements,
                                    smallKernelCpuMs, benchmarkMode,
                                    params.flushLastLevelCache, gpuRunDeadline,
                                    params.gpuRunTimeoutSec, perfConfig)))
        return failure();
    } else {
      if (failed(measureLargeKernel(iterations, stream, functions, blockSizes,
                                    gridSizes, argPointers, measurements,
                                    params.flushLastLevelCache, gpuRunDeadline,
                                    params.gpuRunTimeoutSec, perfConfig)))
        return failure();
    }
  } else {
    // Estimate the per-launch runtime so we can size warmup/benchmark iteration
    // counts from the requested time budgets (Triton do_bench style). We time a
    // handful of launches (flushing caches between them) using a single event
    // pair.
    constexpr unsigned estimateRuns = 5;
    double estimateMs = 0.0;
    {
      hipEvent_t startEvent, stopEvent;
      HIPCHECK(hipEventCreate(&startEvent));
      HIPCHECK(hipEventCreate(&stopEvent));

      HIPCHECK(hipEventRecord(startEvent, stream));
      for (unsigned iter = 0; iter < estimateRuns; ++iter) {
        // The cache flushes are inside the timed window here (unlike the actual
        // measurement loop, which flushes before recording the start event).
        // This is intentional, to match Triton's do_bench, whose estimate loop
        // also includes the cache clear.
        if (failed(flushInstructionCache(stream)))
          return failure();
        if (failed(flushCache(stream, params.flushLastLevelCache)))
          return failure();
        for (auto [func, blockSize, gridSize] :
             llvm::zip(functions, blockSizes, gridSizes)) {
          HIPCHECK(hipExtModuleLaunchKernel(
              func, gridSize * blockSize, 1, 1, blockSize, 1, 1, 0, stream,
              argPointers.data(), nullptr, nullptr, nullptr));
        }
      }
      HIPCHECK(hipEventRecord(stopEvent, stream));
      if (failed(synchronizeStreamWithTimeout(stream, gpuRunDeadline,
                                              params.gpuRunTimeoutSec,
                                              perfConfig, "warmup")))
        return failure();

      float elapsedMs = 0.0;
      HIPCHECK(hipEventElapsedTime(&elapsedMs, startEvent, stopEvent));
      HIPCHECK(hipEventDestroy(stopEvent));
      HIPCHECK(hipEventDestroy(startEvent));

      estimateMs = static_cast<double>(elapsedMs) / estimateRuns;
      // hipEventElapsedTime can return tiny/negative values for very fast
      // kernels due to GPU clock precision. Clamp to the documented ~1
      // microsecond resolution (0.001 ms) to avoid divide-by-zero / overflow
      // below.
      constexpr double minMeasurableMs = 0.001;
      if (estimateMs < minMeasurableMs)
        estimateMs = minMeasurableMs;
    }

    // Derive iteration counts from the time budgets, like Triton's do_bench.
    unsigned nWarmup = std::max<unsigned>(
        1, static_cast<unsigned>(params.warmupMs / estimateMs));
    unsigned iterations =
        std::max<unsigned>(1, static_cast<unsigned>(params.repMs / estimateMs));

    // Warm-up (untimed): just run the kernel chain nWarmup times.
    for (unsigned iter = 0; iter < nWarmup; ++iter) {
      for (auto [func, blockSize, gridSize] :
           llvm::zip(functions, blockSizes, gridSizes)) {
        HIPCHECK(hipExtModuleLaunchKernel(
            func, gridSize * blockSize, 1, 1, blockSize, 1, 1, 0, stream,
            argPointers.data(), nullptr, nullptr, nullptr));
      }
    }
    if (failed(synchronizeStreamWithTimeout(stream, gpuRunDeadline,
                                            params.gpuRunTimeoutSec, perfConfig,
                                            "warmup")))
      return failure();

    // Measure runs
    if (failed(measureKernel(iterations, stream, functions, blockSizes,
                             gridSizes, argPointers, measurements,
                             gpuRunDeadline, params.gpuRunTimeoutSec,
                             perfConfig, params.flushLastLevelCache)))
      return failure();
  }

  if (params.showAllMeasurements) {
    if (isSmallKernel) {
      llvm::outs() << "{\"total_cpu_time\":" << smallKernelCpuMs
                   << ",\"iterations\":" << smallKernelIters << "}\t";
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
    // The legacy small-kernel path uses a single CPU timer, so only the
    // aggregate CPU time is available (no per-run min/max/median).
    if (isSmallKernel) {
      llvm::outs() << "{\"total_cpu_time\":" << smallKernelCpuMs
                   << ",\"iterations\":" << smallKernelIters << "}\t";
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
    Attribute kernel = f->getAttr("rock.kernel");
    if (!kernel)
      return;
    kernels.push_back(f);
  });

  std::sort(kernels.begin(), kernels.end(),
            [](const func::FuncOp &a, const func::FuncOp &b) {
              int kernelA = toKernelOrder(a->getAttr("rock.kernel"));
              int kernelB = toKernelOrder(b->getAttr("rock.kernel"));
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

  // 3. Create HIP stream and allocate device buffers
  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));
  llvm::scope_exit streamCleanup([&]() {
    hipError_t status = hipStreamDestroy(stream);
    if (status != hipSuccess) {
      llvm::errs() << "HIP error in hipStreamDestroy: "
                   << hipGetErrorString(status) << "\n";
    }
  });

  std::vector<void *> gpuBuffers;
  llvm::scope_exit bufferCleanup([&]() {
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
  for (size_t bufferLength : bufferLengths) {
    void *gpuBuffer = nullptr;
    hipError_t hipStatus = hipMalloc(&gpuBuffer, bufferLength);
    if (hipStatus != hipSuccess) {
      llvm::errs() << "HIP error in hipMalloc: " << hipGetErrorString(hipStatus)
                   << "\n";
      return failure();
    }
    gpuBuffers.push_back(gpuBuffer);
    // 0x3F decodes as a finite, non-zero value for all floating point types
    HIPCHECK(hipMemsetAsync(gpuBuffer, 0x3F, bufferLength, stream));
  }

  // 4. Multi-iteration tuning loop
  SmallString<64> bestConfigOverall;
  float bestTimeOverall = std::numeric_limits<float>::max();

  // NOTE: Compilation (PassManager::run()) resets the cl opts, so we have to
  // save the values.
  const BenchmarkParams benchmarkParams = {warmup,
                                           rep,
                                           useMedian,
                                           trimPercent,
                                           sleepUs,
                                           showStats,
                                           showAllMeasurements,
                                           tuningSpaceKind,
                                           numCompileThreads,
                                           benchmarkConfig,
                                           waitForCompiles,
                                           gpuRunTimeout,
                                           flushLastLevelCache,
                                           legacyBenchmarkMode,
                                           numIterations,
                                           warmupIterations};

  rock::TuningParamSetKind effectiveKind = benchmarkParams.tuningSpaceKind;
  unsigned numTuningIterations = rock::getNumberOfIterations(effectiveKind);
  if (!benchmarkParams.benchmarkConfig.empty() && numTuningIterations != 1) {
    llvm::errs() << "benchmarking should do a single tuning iteration\n";
    return failure();
  }

  // Serialize source module once (shared by all threads for parsing).
  // The source module doesn't change between greedy iterations.
  std::string sourceModuleStr;
  {
    llvm::raw_string_ostream sourceOs(sourceModuleStr);
    source->print(sourceOs);
  }

  unsigned maxThreads = (benchmarkParams.numCompileThreads > 0)
                            ? benchmarkParams.numCompileThreads
                            : std::thread::hardware_concurrency();
  if (maxThreads == 0)
    maxThreads = 4;

  // Thread resources are cached and reused across greedy iterations to avoid
  // re-creating MLIRContexts, PassManagers, and re-parsing the source module.
  std::vector<ThreadResources> threadResources;

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

      // The tuning space may have fallen back to a different kind (e.g. Greedy
      // -> Exhaustive for non-accel), so adjust the iteration count.
      effectiveKind = tuningSpace->effectiveKind;
      numTuningIterations = rock::getNumberOfIterations(effectiveKind);

      for (rock::RockTuningParamAttrInterface tuningAttr :
           tuningSpace->tuningRange) {
        SmallString<64> perfConfig;
        tuningAttr.getPerfConfigStr(perfConfig);
        configs.push_back(perfConfig);
      }
    }

    unsigned numThreads =
        std::min(maxThreads, static_cast<unsigned>(configs.size()));

    // PHASE 2: Pre-initialize thread resources (contexts, PassManagers, parsed
    // modules). This avoids the expensive per-config overhead of creating
    // contexts, parsing modules, and building pipelines.
    // On the first iteration all resources are created; subsequent iterations
    // only grow the pool if more threads are needed.
    // Note: MLIR's MLIRContext cannot be safely shared across parallel pass
    // executions, so each thread needs its own context.
    if (static_cast<size_t>(numThreads) > threadResources.size()) {
      unsigned oldSize = threadResources.size();
      threadResources.resize(numThreads);
      std::atomic<bool> initFailed{false};
      {
        std::vector<std::thread> initThreads;
        initThreads.reserve(numThreads - oldSize);
        for (unsigned i = oldSize; i < numThreads; ++i) {
          initThreads.emplace_back([&, i]() {
            if (!threadResources[i].initialize(
                    sourceModuleStr, applicabilityOpts, compilationKernOpts,
                    backendOpts)) {
              initFailed.store(true, std::memory_order_relaxed);
            }
          });
        }
        for (auto &t : initThreads) {
          t.join();
        }
      }

      if (initFailed.load(std::memory_order_relaxed)) {
        llvm::errs() << "Failed to initialize thread resources\n";
        return failure();
      }
    }

    // PHASE 3: Parallel compilation phase using pre-initialized resources
    // Unbounded queue if we wait for all compilations to avoid blocking
    size_t maxCapacity = benchmarkParams.waitForCompiles ? 0 : numThreads;
    ConcurrentQueue<CompilationResult> compilationResults(maxCapacity);
    std::mutex outputMutex; // For thread-safe console output

    // Compile a single config using pre-initialized thread resources
    auto compileConfig = [&](size_t idx,
                             ThreadResources &res) -> CompilationResult {
      CompilationResult result;
      result.perfConfig = configs[idx];

      if (!res.isValid())
        return result;

      StringAttr perfConfigAttr =
          StringAttr::get(res.ctx.get(), result.perfConfig);

      // Helper to copy IR with perf config set
      auto copyIR = [&](ModuleOp src,
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

      if (doesModuleHaveFusions(res.sourceModule.get()) &&
          !rock::isModuleFusible(res.sourceModule.get(), result.perfConfig)) {
        result.status = CompilationStatus::NotApplicable;
        return result;
      }

      // Applicability check - clone the pre-parsed module
      OwningOpRef<ModuleOp> sourceCopy =
          copyIR(res.sourceModule.get(), perfConfigAttr);
      if (failed(res.applicabilityPM->run(sourceCopy.get()))) {
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

      // Compilation - use pre-built pipeline
      if (failed(res.compilationPM->run(sourceCopy.get()))) {
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
    std::atomic<unsigned> nextThreadId{0};
    std::atomic<size_t> activeThreads{numThreads};
    auto worker = [&] {
      // Each worker gets assigned a unique thread ID for its resources
      unsigned myThreadId =
          nextThreadId.fetch_add(1, std::memory_order_relaxed);
      ThreadResources &myRes = threadResources[myThreadId];

      while (true) {
        if (compilationResults.isTerminated())
          break; // Avoid unnecessary work

        size_t idx = nextIdx.fetch_add(1, std::memory_order_relaxed);
        if (idx >= configs.size())
          break;

        if (!compilationResults.push(compileConfig(idx, myRes)))
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

    llvm::scope_exit threadCleanup([&] {
      // In case of early termination, signal all threads to stop
      compilationResults.terminate();
      for (auto &t : threads) {
        t.join();
      }
    });

    if (benchmarkParams.waitForCompiles) {
      for (auto &t : threads) {
        t.join();
      }
      threads.clear();
    }

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
                           result.blockSizes, result.gridSizes, gpuBuffers,
                           stream, result.perfConfig, benchmarkParams);

      if (failed(timing)) {
        llvm::errs() << "Kernel execution failed\n";
        return failure();
      }
      llvm::outs() << timing.value() << "\n";

      validResults++;
      // Find best config
      if (rock::needToUpdateBest(effectiveKind)) {
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
    StringAttr arch = rock::getArchValue(op);
    module = op->getParentOfType<ModuleOp>();
    module->setAttr("mhal.arch", arch);
    return WalkResult::interrupt();
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
