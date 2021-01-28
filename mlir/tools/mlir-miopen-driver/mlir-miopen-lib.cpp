#include "mlir-miopen-lib.hpp"
#include "MlirParse.h"
#include "mlir/Dialect/MIOpen/LowerMIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/MIOpenCPP.h"

#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/ExecutionEngine/ROCm/BackendUitls.h"
#include "mlir/InitAllDialects.h"
#include "llvm/Support/TargetSelect.h"

#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <map>
#include <sstream>
#include <string>

namespace mlir {
namespace {
struct MlirHandle_s {
  MlirHandle_s() {
    OpBuilder builder(&context);
    module = ModuleOp::create(builder.getUnknownLoc());
  }
  mlir::ModuleOp getModule() { return module.get(); }
  MLIRContext context;
  mlir::OwningModuleRef module;
  std::string genTxt;
};

static void strToTokens(const std::string &arguments,
                        std::map<std::string, std::string> &argMap) {
  std::istringstream iss(arguments);
  std::string token;
  std::string argKey;
  std::string argVal;
  while (std::getline(iss, token, ' ')) {
    auto pos = token.find("--");
    if (pos != std::string::npos) {
      argKey = token.substr(pos + 2, token.size());
    } else {
      argVal = token;
      if (argKey.empty() || argVal.empty())
        continue;
      argMap[argKey] = argVal;
    }
  }
}
} // namespace

typedef void *MlirHandle;

extern "C" MlirHandle CreateMlirHandle(const char *arguments) {
  mlir::registerAllDialects();
  mlir::registerAllPasses();

  MlirHandle_s *handle = new MlirHandle_s();
  OpBuilder builder(&(handle->context));

  std::map<std::string, std::string> argMap;
  strToTokens(arguments, argMap);

  auto isValid = [&argMap]() {
    std::vector<std::string> validKeys = {
        "operation",     "in_layout",   "out_layout", "fil_layout",
        "batchsize",     "in_channels", "in_h",       "in_w",
        "out_channels",  "out_h",       "out_w",      "fil_w",
        "fil_h",         "dilation_h",  "dilation_w", "conv_stride_h",
        "conv_stride_w", "padding_h",   "padding_w",  "arch",
        "num_cu"};
    return std::all_of(
        validKeys.begin(), validKeys.end(),
        [&argMap](std::string &key) { return argMap.count(key) > 0; });
  };

  // Proceed only if we have a valid argMap. Otherwise leave the handle to be
  // empty
  if (isValid()) {
    auto strToLong = [&argMap](std::string argKey) {
      return std::stoul(argMap[argKey]);
    };

    auto strToInt = [&argMap](std::string argKey) {
      return std::stoi(argMap[argKey]);
    };

    // MIOpen has NCHW as layout string for all three tensors
    std::string inLayout = translateLayout(
        argMap["in_layout"], std::string("NCHW"), std::string("nchw"));
    std::string filLayout = translateLayout(
        argMap["fil_layout"], std::string("NCHW"), std::string("kcyx"));
    std::string outLayout = translateLayout(
        argMap["out_layout"], std::string("NCHW"), std::string("nkhw"));

    SmallString<128> kernelName;
    ModuleOp module = handle->getModule();
    populateConvolutionLogic(
        argMap["arch"], strToInt("num_cu"), argMap["operation"], inLayout,
        outLayout, filLayout, strToLong("batchsize"), strToLong("in_channels"),
        strToLong("in_h"), strToLong("in_w"), strToLong("out_channels"),
        strToLong("out_h"), strToLong("out_w"), strToLong("fil_w"),
        strToLong("fil_h"), strToInt("dilation_h"), strToInt("dilation_w"),
        strToInt("conv_stride_h"), strToInt("conv_stride_w"),
        strToInt("padding_h"), strToInt("padding_w"), module, builder,
        kernelName, mlir::FloatType::getF32(&(handle->context)), false);
  }

  return handle;
}

extern "C" void MlirLowerCpp(MlirHandle mlirHandle) {
  MlirHandle_s *handle = static_cast<MlirHandle_s *>(mlirHandle);
  ModuleOp module = handle->getModule();

  PassManager pm(module.getContext());
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep1Pass());
  pm.run(module);
}

extern "C" void DestroyMlirHandle(MlirHandle mlirHandle) {
  MlirHandle_s *handle = static_cast<MlirHandle_s *>(mlirHandle);
  delete handle;
}

extern "C" const char *MlirGenIgemmSource(MlirHandle mlirHandle) {
  MlirHandle_s *handle = static_cast<MlirHandle_s *>(mlirHandle);
  handle->genTxt = "";
  translateModuleToMIOpenCpp(handle->getModule(), handle->genTxt);
  return (handle->genTxt).c_str();
}

extern "C" const char *MlirGenIgemmHeader(MlirHandle mlirHandle) {
  MlirHandle_s *handle = static_cast<MlirHandle_s *>(mlirHandle);
  handle->genTxt = "";
  translateModuleToMIOpenHeader(handle->getModule(), handle->genTxt);
  return (handle->genTxt).c_str();
}

extern "C" const char *MlirGenIgemmCflags(MlirHandle mlirHandle) {
  MlirHandle_s *handle = static_cast<MlirHandle_s *>(mlirHandle);
  handle->genTxt = "";
  translateModuleToMIOpenCFlags(handle->getModule(), handle->genTxt);
  return (handle->genTxt).c_str();
}

extern "C" void MlirLowerBin(MlirHandle mlirHandle) {
  MlirHandle_s *handle = static_cast<MlirHandle_s *>(mlirHandle);
  ModuleOp module = handle->getModule();

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();

  // Initialize LLVM AMDGPU backend.
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();
  mlir::initializeLLVMPasses();

  PassManager pm(module.getContext());

  BackendUtils utils;

  // Passes for lowering MIOpen dialect.
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep1Pass());
  pm.addPass(mlir::miopen::createAffineTransformPass());
  pm.addPass(mlir::miopen::createAffixTuningParametersPass(0, 0));
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep2Pass());
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep3Pass());
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep4Pass());
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep5Pass());
  pm.addPass(
      mlir::createLowerMIOpenOpsToGPUPass("miopen_conv2d_kcyx_nchw_nkhw"));

  // Passes for lowering linalg dialect.
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createLowerToCFGPass());

  // Passes for lowering ROCDL dialect
  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(createStripDebugInfoPass());
  pm.addPass(createLowerGpuOpsToROCDLOpsPass());
  pm.addPass(createConvertGPUKernelToBlobPass(
      [&utils](Operation *m) { return utils.compileModuleToROCDLIR(m); },
      [&utils](const std::string isa, Location loc, StringRef name) {
        return utils.compileISAToHsaco(isa, loc, name);
      },
      utils.getTriple(), utils.getChip(), utils.getFeatures(),
      /*gpuBinaryAnnotation=*/"rocdl.hsaco"));

  pm.run(module);
}

extern "C" void MlirGenIgemmBin(MlirHandle mlirHandle, char **buffer,
                                size_t *size) {
  if ((buffer == nullptr) || (size == nullptr))
    return;

  MlirHandle_s *handle = static_cast<MlirHandle_s *>(mlirHandle);
  ModuleOp module = handle->getModule();

  module.walk([&](gpu::GPUModuleOp gpuModule) -> WalkResult {
    auto hsaco = gpuModule.getAttrOfType<StringAttr>("rocdl.hsaco");
    if (hsaco) {
      handle->genTxt = hsaco.getValue().str();
      *buffer = &(handle->genTxt[0]);
      *size = hsaco.getValue().size();
    }
    return success();
  });
}
} // namespace mlir
