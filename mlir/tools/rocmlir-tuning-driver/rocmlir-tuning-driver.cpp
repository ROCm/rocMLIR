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

// Note, that this simplified init value handling will flood int8 output buffers
// with 0x01010101, but that's fine, since they get overwritten and we don't
// actulaly care too much what the values are, so long as they're legal for the type
static std::pair<uint32_t, uint32_t> getInitValue(Type inputType) {
  APInt ret;
  if (auto intType = inputType.dyn_cast<IntegerType>()) {
    ret = APInt(intType.getWidth(), 1);
  } else if (auto floatType = inputType.dyn_cast<FloatType>()) {
    // Avoid getting to inf so as to prevent overly unrealistic benchmarks
    APFloat val(0.01);
    bool dontCare;
    val.convert(floatType.getFloatSemantics(), APFloat::rmNearestTiesToEven, &dontCare);
    ret = val.bitcastToAPInt();
  } else {
    llvm_unreachable("Kernels only accept ints or floats");
  }
  return {ret.getZExtValue(), ret.getBitWidth()};
}

// In order to match rocprof, returns time in nanoseconds
static FailureOr<double> benchmarkKernel(const char *binary,
                                         const char *funcName,
                                         uint32_t blockSize, uint32_t gridSize,
                                         uint32_t initValue, uint32_t bitWidth,
                                         ArrayRef<size_t> bufferSizes) {
  constexpr double msToNs = 1e6;
// intentionally leaky macro
#define HIPCHECK(expr)                                                         \
  if (hipSuccess != (expr)) {                                                  \
    return failure();                                                          \
  }
  hipModule_t mod;
  HIPCHECK(hipModuleLoadData(&mod, binary))
  hipFunction_t func;
  HIPCHECK(hipModuleGetFunction(&func, mod, funcName))

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream))

  // Start allocating buffers
  std::vector<void *> gpuBuffers;
  for (size_t byteLen : bufferSizes) {
    void *buffer = nullptr;
    HIPCHECK(hipMalloc(&buffer, byteLen))
    switch (bitWidth) {
    case 8:
    HIPCHECK(hipMemsetD8Async(buffer, initValue, byteLen, stream))
    break;
    case 16:
    HIPCHECK(hipMemsetD16Async(buffer, initValue, byteLen / 2, stream));
    break;
    case 32:
    HIPCHECK(hipMemsetD32Async(buffer, initValue, byteLen / 4, stream))
    break;
    default:
    llvm_unreachable("Unsupported initial vaule bitwidth");
    }
    gpuBuffers.push_back(buffer);
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

  // cleanup
  for (void *buffer : gpuBuffers) {
    HIPCHECK(hipFree(buffer))
  }
  HIPCHECK(hipEventDestroy(stopEvent))
  HIPCHECK(hipEventDestroy(startEvent))
  HIPCHECK(hipStreamDestroy(stream))
  HIPCHECK(hipModuleUnload(mod))
#undef HIPCHECK

  return ret;
}

static FailureOr<rock::RockGemmWrapperInterface> extractKernel(ModuleOp op) {
  if (!op->hasAttr("xmodel.arch")) {
    return op->emitOpError(
        "no architecture set, set xmodel.arch on the input module");
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
  uint32_t initValue, bitWidth;
  std::tie(initValue, bitWidth) = getInitValue(toTune.getInputType());

  auto kernelFunc = toTune->getParentOfType<func::FuncOp>();
  if (!kernelFunc || !kernelFunc->hasAttr("kernel"))
    return toTune.emitOpError(
        "kernel must be in a function with the kernel attribute");

  // We need a copy since HIP'll want a C string
  std::string kernelFuncName = kernelFunc.getSymName().str();
  SmallVector<size_t, 4> bufferLengths;
  for (Type argType : kernelFunc.getArgumentTypes()) {
    auto shapedTy = argType.dyn_cast<ShapedType>();
    if (!shapedTy)
      return kernelFunc.emitOpError("all kernel inputs must be shaped types");
    if (!shapedTy.hasStaticShape())
      return kernelFunc.emitOpError(
          "all kernel arguments must have static shape");
    bufferLengths.push_back(shapedTy.getSizeInBits() / 8);
  }

  // 2. Set up pipelines. Do this only once to save on construction cost.
  MLIRContext *ctx = source->getContext();
  PassManager applicability(ctx, PassManager::Nesting::Implicit);
  PassManager compilation(ctx, PassManager::Nesting::Implicit);

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
      source->getAttrOfType<StringAttr>("xmodel.arch").getValue();
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

  // 3. Actually tune
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

    // Extract binary anb benchmark
    std::string hipModule;
    tuneCopy.walk([&hipModule](gpu::GPUModuleOp op) {
      hipModule = op->getAttrOfType<StringAttr>("gpu.binary").getValue().str();
      return WalkResult::interrupt();
    });

    FailureOr<double> timing =
        benchmarkKernel(hipModule.c_str(), kernelFuncName.c_str(), blockSize,
                        gridSize, initValue, bitWidth, bufferLengths);
    if (failed(timing)) {
      llvm::errs() << "Kernel execution failed\n";
      return failure();
    }
    llvm::outs() << perfConfig << "\t" << timing << "\n";
    tuneCopy->erase();
  }
  return success();
}

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
  if (failed(runTuningLoop(*source))) {
    llvm::errs() << "Tuning loop failed\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
