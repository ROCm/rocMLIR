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
#include "mlir/ExecutionEngine/RocmDeviceName.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
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
  } else if (inputType.isF16()) {
    return benchmark::DataType::F16;
  } else if (inputType.isBF16()) {
    return benchmark::DataType::BF16;
  } else if (inputType.isInteger(8)) {
    return benchmark::DataType::I8;
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
static FailureOr<double> benchmarkKernel(const char *binary,
                                         const char *funcName,
                                         uint32_t blockSize, uint32_t gridSize,
                                         benchmark::DataType dataType,
                                         ArrayRef<void *> hostBuffers,
                                         MutableArrayRef<void *> gpuBuffers,
                                         ArrayRef<size_t> bufferSizes) {
  constexpr double msToNs = 1e6;
  hipModule_t mod;
  HIPCHECK(hipModuleLoadData(&mod, binary))
  hipFunction_t func;
  HIPCHECK(hipModuleGetFunction(&func, mod, funcName))

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream))

  // Initialize device buffers
  for (size_t i = 0; i < bufferSizes.size(); i++) {
    HIPCHECK(hipMemcpyAsync(gpuBuffers[i], hostBuffers[i], bufferSizes[i],
                            hipMemcpyHostToDevice, stream));
  }

  hipEvent_t startEvent, stopEvent;
  HIPCHECK(hipEventCreate(&startEvent))
  HIPCHECK(hipEventCreate(&stopEvent));

  // HIP wants an array of pointers to each argument
  std::vector<void *> argPointers;
  for (void *&item : gpuBuffers) {
    argPointers.push_back(reinterpret_cast<void *>(&item));
  }

  HIPCHECK(hipExtModuleLaunchKernel(func, gridSize * blockSize, 1, 1, blockSize,
                                    1, 1, 0, stream, argPointers.data(),
                                    nullptr, startEvent, stopEvent))
  HIPCHECK(hipStreamSynchronize(stream))
  float milliseconds = -1.0;
  HIPCHECK(hipEventElapsedTime(&milliseconds, startEvent, stopEvent))
  double ret = msToNs * static_cast<double>(milliseconds);

  HIPCHECK(hipEventDestroy(stopEvent))
  HIPCHECK(hipEventDestroy(startEvent))
  HIPCHECK(hipStreamDestroy(stream))
  HIPCHECK(hipModuleUnload(mod))

  return ret;
}

static FailureOr<rock::RockGemmWrapperInterface> extractKernel(ModuleOp op) {
  if (!op->hasAttr("mhal.arch")) {
    return op->emitOpError(
        "no architecture set, set mhal.arch on the input module");
  }
  rock::RockGemmWrapperInterface kernel;
  uint32_t nKernels = 0;
  op.walk([&kernel, &nKernels](rock::RockGemmWrapperInterface candidate) {
    nKernels++;
    kernel = candidate;
  });
  if (nKernels == 0)
    return op.emitOpError("input module contains no kernels");
  if (nKernels > 1)
    return op.emitOpError(
        "more than one kernel on the input, don't know what to tune");
  return kernel;
}

static LogicalResult runTuningLoop(ModuleOp source) {
  // Verify prerequisites
  FailureOr<rock::RockGemmWrapperInterface> maybeToTune = extractKernel(source);
  if (failed(maybeToTune))
    return failure();
  rock::RockGemmWrapperInterface toTune = std::move(*maybeToTune);
  // Provisionally use the type of input A to set up the init value - this
  // should be a per-buffer value in the futurue.
  benchmark::DataType dataType = getDataType(toTune.getAType());

  auto kernelFunc = toTune->getParentOfType<func::FuncOp>();
  if (!kernelFunc || !kernelFunc->hasAttr("kernel"))
    return toTune.emitOpError(
        "kernel must be in a function with the kernel attribute");

  // We need a copy since HIP'll want a C string
  std::string kernelFuncName = kernelFunc.getSymName().str();
  std::vector<size_t> bufferLengths;
  for (Type argType : kernelFunc.getArgumentTypes()) {
    auto shapedTy = argType.dyn_cast<ShapedType>();
    if (!shapedTy)
      return kernelFunc.emitOpError("all kernel inputs must be shaped types");
    if (!shapedTy.hasStaticShape())
      return kernelFunc.emitOpError(
          "all kernel arguments must have static shape");
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
  backendOpts.indexBitwidth = 32;
  rock::buildBackendPipeline(compilation, backendOpts);

  // Now that we're in the kernel execution zone, turn off error messages
  // Register a handler that swallows all diagnostic print
  DiagnosticEngine &engine = ctx->getDiagEngine();
  engine.registerHandler([](Diagnostic &diag) {});

  // 3. Initialize host buffers and allocate device buffers
  std::vector<void *> hostBuffers;
  std::vector<void *> gpuBuffers;
  for (size_t i = 0; i < bufferLengths.size(); i++) {
    bool isOut = (i == bufferLengths.size() - 1);
    void *hostBuffer =
        benchmark::allocAndFill(dataType, bufferLengths[i], isOut);
    void *gpuBuffer;
    HIPCHECK(hipMalloc(&gpuBuffer, bufferLengths[i]));
    hostBuffers.push_back(hostBuffer);
    gpuBuffers.push_back(gpuBuffer);
  }

  // 4. Actually tune
  rock::TunableParams *tuningSpace = rock::createTunableParamSpace(source);
  for (rock::RockTuningParamAttrInterface tuningAttr :
       tuningSpace->tuningRange) {
    ModuleOp tuneCopy = cast<ModuleOp>(source->clone());
    // TODO: remove this once perf_config gets parsed earlier
    std::string perfConfig;
    tuningAttr.getPerfConfigStr(perfConfig);
    StringAttr perfConfigAttr = StringAttr::get(ctx, perfConfig);
    tuneCopy->walk([&perfConfigAttr](rock::RockGemmWrapperInterface op) {
      op->setAttr("perf_config", perfConfigAttr);
    });
    if (failed(applicability.run(tuneCopy))) {
      llvm::outs() << perfConfig << "\t"
                   << "N/A\n";
      continue;
    }
    auto tunedFunc = tuneCopy.lookupSymbol<func::FuncOp>(kernelFuncName);
    if (!tunedFunc) {
      llvm::errs() << "Tuned copy somehow missing kernel function\n";
      return failure();
    }
    // We have to get these now, they disappear later. Also, if these attributes
    // aren't set the contract of the applicability pipeline changed and that's
    // a problem.
    uint32_t blockSize =
        tunedFunc->getAttrOfType<IntegerAttr>("block_size").getInt();
    uint32_t gridSize =
        tunedFunc->getAttrOfType<IntegerAttr>("grid_size").getInt();
    if (failed(compilation.run(tuneCopy))) {
      llvm::errs() << "Backend pipeline failed for config: " << perfConfig
                   << "\n";
      return failure();
    }

    // Extract binary and benchmark
    std::string hipModule;
    tuneCopy.walk([&hipModule, &kernelFuncName](gpu::GPUModuleOp op) {
      std::string moduleName = op.getName().str();
      if (moduleName == kernelFuncName + "_module") {
        hipModule =
            op->getAttrOfType<StringAttr>("gpu.binary").getValue().str();
        return WalkResult::interrupt();
      }
      llvm::errs() << "Ignoring utility kernels, benchmark times will not "
                      "match performance tests\n";
      return WalkResult::advance();
    });
    FailureOr<double> timing = benchmarkKernel(
        hipModule.c_str(), kernelFuncName.c_str(), blockSize, gridSize,
        dataType, hostBuffers, gpuBuffers, bufferLengths);
    if (failed(timing)) {
      llvm::errs() << "Kernel execution failed\n";
      return failure();
    }
    llvm::outs() << perfConfig << "\t" << timing << "\n";
    tuneCopy->erase();
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

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
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
  WalkResult findModule = source->walk([&](ModuleOp op) -> WalkResult {
    if (op->hasAttr("mhal.arch")) {
      module = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!findModule.wasInterrupted()) {
    source->emitOpError(
        "no architecture set, set mhal.arch on the input module");
    llvm::errs() << "Tuning loop failed\n";
    return EXIT_FAILURE;
  }

  if (failed(runTuningLoop(module))) {
    llvm::errs() << "Tuning loop failed\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
