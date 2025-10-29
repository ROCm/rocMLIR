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

#include <chrono>
#include <cstdlib>
#include <thread>

// Utilities to allocate buffers
#include "../utils/performance/common/benchmarkUtils.h"

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
  } else {
    llvm_unreachable("Kernels only accept ints or floats");
  }
}

// intentionally leaky macro
#define HIPCHECK(expr)                                                         \
  if (hipSuccess != (expr)) {                                                  \
    return failure();                                                          \
  }

size_t flushSize = 0;
void *flushBuffer = nullptr;

static LogicalResult flushL2Cache(hipStream_t stream) {
  if (flushBuffer == nullptr) {
    hipDeviceProp_t props;
    HIPCHECK(hipGetDeviceProperties(&props, 0));
    size_t l2Size = props.l2CacheSize;

    flushSize = l2Size + (l2Size / 5); // 20% margin
    HIPCHECK(hipMalloc(&flushBuffer, flushSize));
  }

  HIPCHECK(hipMemsetAsync(flushBuffer, 0, flushSize, stream));

  return success();
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

// In order to match rocprof, returns time in nanoseconds
static FailureOr<double>
benchmarkKernels(ArrayRef<std::string> binaries,
                 ArrayRef<std::string> funcNames, ArrayRef<uint32_t> blockSizes,
                 ArrayRef<uint32_t> gridSizes, benchmark::DataType dataType,
                 ArrayRef<void *> hostBuffers,
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
    if (failed(flushL2Cache(stream))) {
      return failure();
    }

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

static FailureOr<std::pair<Type, Type>>
extractKernelDataType(ModuleOp op, SmallVectorImpl<func::FuncOp> &kernels) {
  if (!op->hasAttr("mhal.arch")) {
    return op->emitOpError(
        "no architecture set, set mhal.arch on the input module");
  }
  Type toTuneType;
  Type outputType;
  op.walk([&toTuneType, &outputType, &kernels](func::FuncOp f) {
    Attribute kernel = f->getAttr("kernel");
    if (!kernel)
      return;
    kernels.push_back(f);
    if (!toTuneType) {
      f.walk(
          [&toTuneType, &outputType](rock::RockGemmWrapperInterface gemmLike) {
            toTuneType = gemmLike.getAType();
            outputType = gemmLike.getCType();
          });
      f.walk([&toTuneType,
              &outputType](rock::RockGemmGemmWrapperInterface attnLike) {
        toTuneType = cast<MemRefType>(attnLike.getAType()).getElementType();
        outputType = cast<MemRefType>(attnLike.getOutType()).getElementType();
      });
    }
  });

  std::sort(kernels.begin(), kernels.end(),
            [](const func::FuncOp &a, const func::FuncOp &b) {
              int kernelA = toKernelOrder(a->getAttr("kernel"));
              int kernelB = toKernelOrder(b->getAttr("kernel"));
              return kernelA < kernelB;
            });

  if (!toTuneType) {
    return op.emitError("could not find a tunable kernel in the input");
  }
  return std::make_pair(toTuneType, outputType);
}

static LogicalResult runTuningLoop(ModuleOp source) {
  // Verify prerequisites
  SmallVector<func::FuncOp> funcs;
  auto maybeInOutTypes = extractKernelDataType(source, funcs);
  if (failed(maybeInOutTypes))
    return failure();
  Type toTuneType = maybeInOutTypes.value().first;
  Type outType = maybeInOutTypes.value().second;
  // Provisionally use the type of input A to set up the init value - this
  // should be a per-buffer value in the futurue.
  benchmark::DataType dataType = getDataType(toTuneType);
  benchmark::DataType outDataType = getDataType(outType);

  // We need a copy since HIP'll want a C string
  SmallVector<std::string> kernelFuncNames;
  SmallVector<size_t> bufferLengths;
  for (func::FuncOp &funcOp : funcs) {
    kernelFuncNames.push_back(funcOp.getSymName().str());
  }
  for (Type argType : funcs[0].getArgumentTypes()) {
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
    bufferLengths.push_back(sizeInBits / 8);
  }

  // 2. Set up pipelines. Do this only once to save on construction cost.
  MLIRContext *ctx = source->getContext();
  PassManager applicability(source->getName(), PassManager::Nesting::Implicit);
  PassManager compilation(source->getName(), PassManager::Nesting::Implicit);

  rock::KernelOptions applicabilityOpts;
  applicabilityOpts.enableApplicability = true;
  applicabilityOpts.enableFusion = true;
  applicabilityOpts.tuningFallback = false;
  rock::buildKernelPipeline(applicability, applicabilityOpts);

  rock::KernelOptions compilationKernOpts;
  compilationKernOpts.enableApplicability = false;
  compilationKernOpts.enableFusion = true;
  compilationKernOpts.tuningFallback = false;
  rock::buildKernelPipeline(compilation, compilationKernOpts);

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
  rock::buildBackendPipeline(compilation, backendOpts);

  // Now that we're in the kernel execution zone, turn off error messages
  // Register a handler that swallows all diagnostic print
  DiagnosticEngine &engine = ctx->getDiagEngine();
  engine.registerHandler([](Diagnostic &diag) {});

  // 3. Initialize host buffers and allocate device buffers
  std::vector<void *> hostBuffers;
  std::vector<void *> gpuBuffers;
  for (size_t i = 0; i < bufferLengths.size(); i++) {
    benchmark::DataType type =
        (i == bufferLengths.size() - 1 ? dataType : outDataType);
    void *hostBuffer = benchmark::allocAndFill(type, bufferLengths[i]);
    void *gpuBuffer = nullptr;
    HIPCHECK(hipMalloc(&gpuBuffer, bufferLengths[i]));
    hostBuffers.push_back(hostBuffer);
    gpuBuffers.push_back(gpuBuffer);
  }

  auto copyIR = [&](ModuleOp source,
                    StringAttr perfConfigAttr) -> OwningOpRef<ModuleOp> {
    OwningOpRef<ModuleOp> copy = cast<ModuleOp>(source->clone());

    copy->walk([&perfConfigAttr](rock::RockGemmWrapperInterface op) {
      op->setAttr("perf_config", perfConfigAttr);
    });
    copy->walk([&perfConfigAttr](rock::RockGemmGemmWrapperInterface op) {
      op->setAttr("perf_config", perfConfigAttr);
    });
    return copy;
  };

  // 4. Actually tune
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

  for (const auto &perfConfig : configs) {
    llvm::outs() << perfConfig << "\t";
    OwningOpRef<ModuleOp> tuneCopy = cast<ModuleOp>(source->clone());
    StringAttr perfConfigAttr = StringAttr::get(ctx, perfConfig);

    OwningOpRef<ModuleOp> applicabilityCopy = copyIR(source, perfConfigAttr);
    if (rock::isSplitKRequested(applicabilityCopy.get(), perfConfig) &&
        failed(rock::testFusionLegalitySplitK(applicabilityCopy.get()))) {
      llvm::outs() << "N/A\n";
      continue;
    }

    if (failed(applicability.run(applicabilityCopy.get()))) {
      llvm::outs() << "N/A\n";
      continue;
    }

    // We have to get these now, they disappear later. Also, if these attributes
    // aren't set the contract of the applicability pipeline changed and that's
    // a problem.
    SmallVector<uint32_t> blockSizes;
    SmallVector<uint32_t> gridSizes;
    for (auto &fnName : kernelFuncNames) {
      auto tunedFunc = applicabilityCopy->lookupSymbol<func::FuncOp>(fnName);
      if (!tunedFunc) {
        llvm::errs() << "Tuned copy somehow missing kernel function\n";
        return failure();
      }
      blockSizes.push_back(
          tunedFunc->getAttrOfType<IntegerAttr>("block_size").getInt());
      gridSizes.push_back(
          tunedFunc->getAttrOfType<IntegerAttr>("grid_size").getInt());
    }

    OwningOpRef<ModuleOp> compileCopy = copyIR(source, perfConfigAttr);

    // NOTE: Call to run() resets the cl opts
    if (failed(compilation.run(compileCopy.get()))) {
      llvm::errs() << "Backend pipeline failed for config: " << perfConfig
                   << "\n";
      return failure();
    }

    // Extract binary and benchmark
    SmallVector<std::string> hipModules;
    for (const auto &fnName : kernelFuncNames) {
      auto binary =
          compileCopy->lookupSymbol<gpu::BinaryOp>(fnName + "_module");
      if (!binary) {
        llvm::errs() << "could not find the GPU binary\n";
      }
      hipModules.push_back(cast<gpu::ObjectAttr>(binary.getObjects()[0])
                               .getObject()
                               .getValue()
                               .str());
    }

    FailureOr<double> timing = benchmarkKernels(
        hipModules, kernelFuncNames, blockSizes, gridSizes, dataType,
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
  if (flushBuffer) {
    HIPCHECK(hipFree(flushBuffer));
    flushBuffer = nullptr;
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
