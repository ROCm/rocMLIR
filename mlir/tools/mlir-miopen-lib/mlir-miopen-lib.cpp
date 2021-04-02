#include "Miir.h"
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

void strToTokens(const std::string &arguments,
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

// In multi-threaded context, static intialization is guaranteed to
// be thread safe, since C++11. Refer to
// https://en.cppreference.com/w/cpp/language/storage_duration
//
// With this guarantee, we are protected from the possible race
// condition of one thread doing intialization and another doing
// lowering.
bool miirLazyInit() {
  static const bool once = []() {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();

    // Initialize LLVM AMDGPU backend.
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmPrinter();
    mlir::initializeLLVMPasses();
    return true;
  }();
  return once;
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
        "dilation_w", "groupsize"};
    return std::all_of(
        validKeys.cbegin(), validKeys.cend(),
        [&argMap](const std::string &key) { return argMap.count(key) > 0; });
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
        argMap["in_layout"], std::string("NGCHW"), std::string("ngchw"));
    std::string filLayout = conv2dGenerator.translateLayout(
        argMap["fil_layout"], std::string("GKCYX"), std::string("gkcyx"));
    std::string outLayout = conv2dGenerator.translateLayout(
        argMap["out_layout"], std::string("NGKHW"), std::string("ngkhw"));

    ModuleOp module = handle->getModule();
    // Determine dimensions.
    SmallVector<int64_t, 5> filterDimension;
    SmallVector<int64_t, 5> inputDimension;
    SmallVector<int64_t, 5> outputDimension;
    conv2dGenerator.parseConvDims(
        inLayout, outLayout, filLayout, strToLong("groupsize"),
        strToLong("batchsize"), strToLong("in_channels"), strToLong("in_h"),
        strToLong("in_w"), strToLong("out_channels"), strToLong("out_h"),
        strToLong("out_w"), strToLong("fil_w"), strToLong("fil_h"),
        filterDimension, inputDimension, outputDimension);

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
                                           size_t *globalSize,
                                           size_t *localSize) {
  if (globalSize == nullptr || localSize == nullptr)
    return MIIR_INVALID_PARAM;

  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return MIIR_INVALID_PARAM;

  ModuleOp module = handle->getModule();

  auto getSizeAttr = [](const Attribute &attr, int32_t &size) {
    if (attr) {
      size = attr.template dyn_cast<IntegerAttr>().getInt();
      return success();
    } else {
      return failure();
    }
  };

  auto setReturn = [&](int32_t blockSize, int32_t gridSize) {
    *globalSize = gridSize * blockSize;
    *localSize = blockSize;
  };

  int count = 0;
  int32_t blockSize = 0;
  int32_t gridSize = 0;

  // If mlirHandle contains result from miirLowerTuningParams(), it is still
  // a mlir::FuncOp
  module.walk([&](FuncOp funcOp) -> WalkResult {
    auto statusBlock = getSizeAttr(funcOp->getAttr("block_size"), blockSize);
    auto statusGrid = getSizeAttr(funcOp->getAttr("grid_size"), gridSize);
    if (statusBlock.succeeded() && statusGrid.succeeded()) {
      setReturn(blockSize, gridSize);
    }
    ++count;
    return WalkResult::advance();
  });
  if (count == 1)
    return MIIR_SUCCESS;

  count = 0;
  // If mlirHandle contains result from miirLowerTuningBin(), it is
  // a LLVM::LLVMFuncOp
  module.walk([&](LLVM::LLVMFuncOp funcOp) -> WalkResult {
    auto statusBlock = getSizeAttr(funcOp->getAttr("block_size"), blockSize);
    auto statusGrid = getSizeAttr(funcOp->getAttr("grid_size"), gridSize);
    if (statusBlock.succeeded() && statusGrid.succeeded()) {
      setReturn(blockSize, gridSize);
    }
    ++count;
    return WalkResult::advance();
  });
  if (count == 1)
    return MIIR_SUCCESS;

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

extern "C" MiirStatus miirLowerTuningParams(MiirHandle mlirHandle) {
  miirLazyInit();

  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return MIIR_INVALID_PARAM;

  ModuleOp module = handle->getModule();

  PassManager pm(module.getContext(), PassManager::Nesting::Implicit);

  // Passes for lowering MIOpen dialect.
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep1Pass());
  pm.addPass(mlir::miopen::createAffineTransformPass());
  pm.addPass(mlir::miopen::createAffixTuningParametersPass(0, 0));

  auto status = pm.run(module);

  return status.succeeded() ? MIIR_SUCCESS : MIIR_BUILD_FAILURE;
}

extern "C" MiirStatus miirLowerBin(MiirHandle mlirHandle) {
  miirLazyInit();

  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return MIIR_INVALID_PARAM;

  ModuleOp module = handle->getModule();

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

extern "C" MiirStatus miirBufferGet(MiirHandle mlirHandle, char *buffer,
                                    size_t *size) {
  if ((buffer == nullptr) && (size == nullptr))
    return MIIR_INVALID_PARAM;

  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  ModuleOp module = handle->getModule();

  // 1st call: give client the size of buffer to allocate
  if ((buffer == nullptr) && (size != nullptr)) {
    module.walk([&](gpu::GPUModuleOp gpuModule) {
      auto hsacoAttr = gpuModule->getAttrOfType<StringAttr>("rocdl.hsaco");
      if (hsacoAttr) {
        *size = hsacoAttr.getValue().size();
      }
    });
    // 2nd call: copy the hsaco to the target buffer
  } else {
    module.walk([&](gpu::GPUModuleOp gpuModule) {
      auto hsacoAttr = gpuModule->getAttrOfType<StringAttr>("rocdl.hsaco");
      if (hsacoAttr) {
        std::string hsaco = hsacoAttr.getValue().str();
        std::copy(hsaco.begin(), hsaco.end(), buffer);
      }
    });
  }
  return MIIR_SUCCESS;
}
