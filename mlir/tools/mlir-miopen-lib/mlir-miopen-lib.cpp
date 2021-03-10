#include "mlir-miopen-lib.h"
#include "mlir/Dialect/MIOpen/Generator/Conv2dGenerator.h"
#include "mlir/Dialect/MIOpen/LowerMIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
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

namespace {
struct MiirHandle_s {
  MiirHandle_s() {
    context.loadDialect<miopen::MIOpenDialect, StandardOpsDialect>();
    mlir::registerAllDialects(context.getDialectRegistry());
    OpBuilder builder(&context);
    module = ModuleOp::create(builder.getUnknownLoc());
  }
  mlir::ModuleOp getModule() { return module.get(); }
  MLIRContext context;
  mlir::OwningModuleRef module;
  std::string arch;
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

typedef void *MiirHandle;

extern "C" MiirHandle miirCreateHandle(const char *arguments) {
  mlir::registerAllPasses();

  MiirHandle_s *handle = nullptr;

  std::map<std::string, std::string> argMap;
  strToTokens(arguments, argMap);

  auto isValid = [&argMap]() {
    std::vector<std::string> validKeys = {
        "operation",    "batchsize",     "arch",          "num_cu",
        "kernel_name",  "in_layout",     "in_type",       "in_channels",
        "in_h",         "in_w",          "out_layout",    "out_type",
        "out_channels", "out_h",         "out_w",         "fil_layout",
        "fil_type",     "fil_w",         "fil_h",         "padding_h",
        "padding_w",    "conv_stride_h", "conv_stride_w", "dilation_h",
        "dilation_w"};
    return std::all_of(
        validKeys.begin(), validKeys.end(),
        [&argMap](std::string &key) { return argMap.count(key) > 0; });
  };

  auto getType = [](mlir::MLIRContext *context, const std::string &type_s) {
    mlir::Type type;
    if (type_s == "fp32") {
      type = mlir::FloatType::getF32(context);
    } else if (type_s == "fp16") {
      type = mlir::FloatType::getF16(context);
    }
    return type;
  };

  // Proceed only if we have a valid argMap. Otherwise leave the handle to be
  // empty
  if (isValid()) {

    handle = new MiirHandle_s;
    OpBuilder builder(&(handle->context));

    handle->arch = argMap["arch"];

    mlir::Type type = getType(&(handle->context), argMap["out_type"]);
    if (!type) {
      delete handle;
      return nullptr;
    }

    auto strToLong = [&argMap](std::string argKey) {
      return std::stoul(argMap[argKey]);
    };

    auto strToInt = [&argMap](std::string argKey) {
      return std::stoi(argMap[argKey]);
    };

    Conv2dGenerator conv2dGenerator;
    // MIOpen has NCHW as layout string for all three tensors
    std::string inLayout = conv2dGenerator.translateLayout(
        argMap["in_layout"], std::string("NCHW"), std::string("nchw"));
    std::string filLayout = conv2dGenerator.translateLayout(
        argMap["fil_layout"], std::string("NCHW"), std::string("kcyx"));
    std::string outLayout = conv2dGenerator.translateLayout(
        argMap["out_layout"], std::string("NCHW"), std::string("nkhw"));

    ModuleOp module = handle->getModule();
    // Determine dimensions.
    SmallVector<int64_t, 4> filterDimension;
    SmallVector<int64_t, 4> inputDimension;
    SmallVector<int64_t, 4> outputDimension;
    conv2dGenerator.parseConvDims(
        inLayout, outLayout, filLayout, strToLong("batchsize"),
        strToLong("in_channels"), strToLong("in_h"), strToLong("in_w"),
        strToLong("out_channels"), strToLong("out_h"), strToLong("out_w"),
        strToLong("fil_w"), strToLong("fil_h"), filterDimension, inputDimension,
        outputDimension);

    conv2dGenerator.genConvModule(
        argMap["arch"], strToInt("num_cu"), argMap["operation"], inLayout,
        outLayout, filLayout, filterDimension, inputDimension, outputDimension,
        strToInt("dilation_h"), strToInt("dilation_w"),
        strToInt("conv_stride_h"), strToInt("conv_stride_w"),
        strToInt("padding_h"), strToInt("padding_w"), module, builder,
        argMap["kernel_name"], type, false);
  }

  return handle;
}

extern "C" MiirStatus miirDestroyHandle(MiirHandle mlirHandle) {
  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return MIIR_INVALID_PARAM;

  delete handle;
  return MIIR_SUCCESS;
}

extern "C" MiirStatus miirGetExecutionDims(MiirHandle mlirHandle,
                                           size_t *global_size,
                                           size_t *local_size) {
  if (global_size == nullptr || local_size == nullptr)
    return MIIR_INVALID_PARAM;

  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return MIIR_INVALID_PARAM;

  ModuleOp module = handle->getModule();

  LLVM::LLVMFuncOp kernel;
  int count = 0;
  module.walk([&](LLVM::LLVMFuncOp funcOp) -> WalkResult {
    kernel = funcOp;
    count++;
    return WalkResult::advance();
  });
  if (count != 1)
    return MIIR_INVALID_MODULE;

  auto blockSizeAttr = kernel->getAttr("block_size");
  auto gridSizeAttr = kernel->getAttr("grid_size");

  if (blockSizeAttr && gridSizeAttr) {
    auto blockSize = blockSizeAttr.template dyn_cast<IntegerAttr>().getInt();
    auto gridSize = gridSizeAttr.template dyn_cast<IntegerAttr>().getInt();
    *global_size = gridSize * blockSize;
    *local_size = blockSize;
    return MIIR_SUCCESS;
  }
  return MIIR_INVALID_MODULE;
}

extern "C" MiirStatus miirLowerCpp(MiirHandle mlirHandle) {
  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return MIIR_INVALID_PARAM;

  ModuleOp module = handle->getModule();

  PassManager pm(module.getContext(), PassManager::Nesting::Implicit);
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep1Pass());
  pm.run(module);
  return MIIR_SUCCESS;
}

extern "C" const char *miirGenIgemmSource(MiirHandle mlirHandle) {
  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return "";

  handle->genTxt = "";
  translateModuleFromMIOpenToCpp(handle->getModule(), handle->genTxt);
  return (handle->genTxt).c_str();
}

extern "C" const char *miirGenIgemmHeader(MiirHandle mlirHandle) {
  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return "";

  handle->genTxt = "";
  translateModuleFromMIOpenToHeader(handle->getModule(), handle->genTxt);
  return (handle->genTxt).c_str();
}

extern "C" const char *miirGenIgemmCflags(MiirHandle mlirHandle) {
  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return "";

  handle->genTxt = "";
  translateModuleFromMIOpenToCFlags(handle->getModule(), handle->genTxt);
  return (handle->genTxt).c_str();
}

extern "C" MiirStatus miirLowerBin(MiirHandle mlirHandle) {
  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return MIIR_INVALID_PARAM;

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

  PassManager pm(module.getContext(), PassManager::Nesting::Implicit);

  std::string triple = "amdgcn-amd-amdhsa";
  BackendUtils utils(triple, handle->arch, "");

  // Retrieve name of FuncOp from the incoming module and set it
  // as the GpuFuncOps's kernel name
  StringRef kernelName;
  for (auto func : module.getOps<FuncOp>()) {
    kernelName = func.getName();
  }

  // Passes for lowering MIOpen dialect.
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep1Pass());
  pm.addPass(mlir::miopen::createAffineTransformPass());
  pm.addPass(mlir::miopen::createAffixTuningParametersPass(0, 0));
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep2Pass());
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep3Pass());
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep4Pass());
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep5Pass());
  pm.addPass(mlir::createLowerMIOpenOpsToGPUPass(kernelName));

  // Passes for lowering linalg dialect.
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createLowerToCFGPass());

  // Passes for lowering ROCDL dialect
  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(createStripDebugInfoPass());
  pm.addPass(createLowerGpuOpsToROCDLOpsPass());
  pm.addPass(createConvertGPUKernelToBlobPass(
      [&utils](Operation *m, llvm::LLVMContext &llvmContext,
               llvm::StringRef name) {
        return utils.compileModuleToROCDLIR(m, llvmContext, name);
      },
      [&utils](const std::string isa, Location loc, StringRef name) {
        return utils.compileISAToHsaco(isa, loc, name);
      },
      utils.getTriple(), utils.getChip(), utils.getFeatures(),
      /*gpuBinaryAnnotation=*/"rocdl.hsaco"));

  auto status = pm.run(module);

  return status.succeeded() ? MIIR_SUCCESS : MIIR_BUILD_FAILURE;
}

extern "C" MiirStatus miirGenIgemmBin(MiirHandle mlirHandle, char **buffer,
                                      size_t *size) {
  if ((buffer == nullptr) || (size == nullptr))
    return MIIR_INVALID_PARAM;

  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  ModuleOp module = handle->getModule();

  module.walk([&](gpu::GPUModuleOp gpuModule) -> WalkResult {
    auto hsaco = gpuModule->getAttrOfType<StringAttr>("rocdl.hsaco");
    if (hsaco) {
      handle->genTxt = hsaco.getValue().str();
      *buffer = &(handle->genTxt[0]);
      *size = hsaco.getValue().size();
    }
    return success();
  });
  return MIIR_SUCCESS;
}
