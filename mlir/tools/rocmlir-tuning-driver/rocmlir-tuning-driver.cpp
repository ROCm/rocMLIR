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
#include "mlir/Dialect/Rock/IR/Rock.h"
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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <cstdlib>

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
    llvm::cl::init(rock::TuningParamSetKind::Exhaustive));

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
  } else if (inputType.isFloat8E4M3FNUZ()) {
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

// In order to match rocprof, returns time in nanoseconds
static FailureOr<double> benchmarkKernels(
    ArrayRef<std::string> binaries, ArrayRef<std::string> funcNames,
    ArrayRef<uint32_t> blockSizes, ArrayRef<uint32_t> gridSizes,
    benchmark::DataType dataType, ArrayRef<void *> hostBuffers,
    MutableArrayRef<void *> gpuBuffers, ArrayRef<size_t> bufferSizes) {
  constexpr double msToNs = 1e6;
  float milliseconds = 0.0;

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream))

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

  for (auto [binary, funcName, blockSize, gridSize] :
       llvm::zip(binaries, funcNames, blockSizes, gridSizes)) {
    hipModule_t mod;
    HIPCHECK(hipModuleLoadData(&mod, binary.c_str()))
    hipFunction_t func;
    HIPCHECK(hipModuleGetFunction(&func, mod, funcName.c_str()))

    hipEvent_t startEvent, stopEvent;
    HIPCHECK(hipEventCreate(&startEvent))
    HIPCHECK(hipEventCreate(&stopEvent));

    HIPCHECK(hipExtModuleLaunchKernel(
        func, gridSize * blockSize, 1, 1, blockSize, 1, 1, 0, stream,
        argPointers.data(), nullptr, startEvent, stopEvent))
    HIPCHECK(hipStreamSynchronize(stream))
    float currentMilliseconds = 0.0;
    HIPCHECK(hipEventElapsedTime(&currentMilliseconds, startEvent, stopEvent))

    HIPCHECK(hipEventDestroy(stopEvent))
    HIPCHECK(hipEventDestroy(startEvent))

    HIPCHECK(hipModuleUnload(mod))

    milliseconds += currentMilliseconds;
  }

  double ret = msToNs * static_cast<double>(milliseconds);

  HIPCHECK(hipStreamDestroy(stream))

  return ret;
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
    }
    if (!toTuneType) {
      f.walk([&toTuneType, &outputType](rock::AttentionOp attnOp) {
        toTuneType = attnOp.getQueries().getType().getElementType();
        outputType = toTuneType;
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

  // 4. Actually tune
  std::unique_ptr<rock::TuningParamSet> tuningSpace(
      rock::createTunableParamSpace(source, tuningSpaceKind));
  for (rock::RockTuningParamAttrInterface tuningAttr :
       tuningSpace->tuningRange) {
    OwningOpRef<ModuleOp> tuneCopy = cast<ModuleOp>(source->clone());
    // TODO: remove this once perf_config gets parsed earlier
    SmallString<64> perfConfig;
    tuningAttr.getPerfConfigStr(perfConfig);
    llvm::outs() << perfConfig << "\t";
    StringAttr perfConfigAttr = StringAttr::get(ctx, perfConfig);
    tuneCopy->walk([&perfConfigAttr](rock::RockGemmWrapperInterface op) {
      op->setAttr("perf_config", perfConfigAttr);
    });
    tuneCopy->walk([&perfConfigAttr](rock::AttentionOp op) {
      op->setAttr("perf_config", perfConfigAttr);
    });

    if (rock::isSplitKRequested(tuneCopy.get(), perfConfig)) {
      if (failed(rock::testFusionLegality(tuneCopy.get()))) {
        llvm::outs() << "N/A\n";
        continue;
      }
    }

    if (failed(applicability.run(tuneCopy.get()))) {
      llvm::outs() << "N/A\n";
      continue;
    }

    SmallVector<uint32_t> blockSizes;
    SmallVector<uint32_t> gridSizes;
    for (auto &fnName : kernelFuncNames) {
      auto tunedFunc = tuneCopy->lookupSymbol<func::FuncOp>(fnName);
      if (!tunedFunc) {
        llvm::errs() << "Tuned copy somehow missing kernel function\n";
        return failure();
      }
      blockSizes.push_back(
          tunedFunc->getAttrOfType<IntegerAttr>("block_size").getInt());
      gridSizes.push_back(
          tunedFunc->getAttrOfType<IntegerAttr>("grid_size").getInt());
    }
    // We have to get these now, they disappear later. Also, if these attributes
    // aren't set the contract of the applicability pipeline changed and that's
    // a problem.
    if (failed(compilation.run(tuneCopy.get()))) {
      llvm::errs() << "Backend pipeline failed for config: " << perfConfig
                   << "\n";
      return failure();
    }

    // Extract binary and benchmark
    SmallVector<std::string> hipModules;
    for (const auto &fnName : kernelFuncNames) {
      Operation *module = tuneCopy->lookupSymbol(fnName + "_module");
      if (!isa<gpu::GPUModuleOp>(module)) {
        llvm::errs() << "could not find the GPU module\n";
      }
      hipModules.push_back(
          module->getAttrOfType<StringAttr>("gpu.binary").getValue().str());
    }

    FailureOr<double> timing =
        benchmarkKernels(hipModules, kernelFuncNames, blockSizes, gridSizes,
                         dataType, hostBuffers, gpuBuffers, bufferLengths);
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
    HIPCHECK(hipFree(buffer))
  }
  return success();
}
#undef HIPCHECK

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  mlir::registerMLIRCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "rocMLIR tuning driver");

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
