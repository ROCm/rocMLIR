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
#include <mutex>
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
  int kernelCount;
};

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
static std::mutex mutex;

extern "C" MiirHandle miirCreateHandle(const char *arguments) {
  const std::lock_guard<std::mutex> lock(mutex);
  mlir::registerAllPasses();

  MiirHandle_s *handle = nullptr;

  Conv2dGenerator conv2dGenerator;
  LogicalResult result = LogicalResult::failure();

  if (succeeded(conv2dGenerator.parseConvConfig(arguments))) {

    handle = new MiirHandle_s;
    OpBuilder builder(&(handle->context));

    handle->arch = conv2dGenerator.getConfig().arch;
    handle->kernelCount = conv2dGenerator.getKernelCount();

    ModuleOp module = handle->getModule();

    result = conv2dGenerator.genConvModule(module, builder);
  }

  return (result.succeeded()) ? handle : nullptr;
}

extern "C" int miirGetKernelCount(MiirHandle mlirHandle) {
  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return -1;

  return handle->kernelCount;
}

extern "C" MiirStatus miirDestroyHandle(MiirHandle mlirHandle) {
  const std::lock_guard<std::mutex> lock(mutex);
  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return MIIR_INVALID_PARAM;

  delete handle;
  return MIIR_SUCCESS;
}

extern "C" MiirStatus miirGetExecutionDims(MiirHandle mlirHandle,
                                           size_t *globalSize,
                                           size_t *localSize) {
  const std::lock_guard<std::mutex> lock(mutex);
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
  const std::lock_guard<std::mutex> lock(mutex);
  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return MIIR_INVALID_PARAM;

  ModuleOp module = handle->getModule();

  PassManager pm(module.getContext(), PassManager::Nesting::Implicit);
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep1Pass());
  LogicalResult result = pm.run(module);
  return (result.succeeded()) ? MIIR_SUCCESS : MIIR_BUILD_FAILURE;
}

extern "C" const char *miirGenIgemmSource(MiirHandle mlirHandle) {
  const std::lock_guard<std::mutex> lock(mutex);
  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return "";

  handle->genTxt = "";
  translateModuleFromMIOpenToCpp(handle->getModule(), handle->genTxt);
  return (handle->genTxt).c_str();
}

extern "C" const char *miirGenIgemmHeader(MiirHandle mlirHandle) {
  const std::lock_guard<std::mutex> lock(mutex);
  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return "";

  handle->genTxt = "";
  translateModuleFromMIOpenToHeader(handle->getModule(), handle->genTxt);
  return (handle->genTxt).c_str();
}

extern "C" const char *miirGenIgemmCflags(MiirHandle mlirHandle) {
  const std::lock_guard<std::mutex> lock(mutex);
  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return "";

  handle->genTxt = "";
  translateModuleFromMIOpenToCFlags(handle->getModule(), handle->genTxt);
  return (handle->genTxt).c_str();
}

extern "C" MiirStatus miirLowerTuningParams(MiirHandle mlirHandle) {
  const std::lock_guard<std::mutex> lock(mutex);
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
  const std::lock_guard<std::mutex> lock(mutex);
  miirLazyInit();

  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return MIIR_INVALID_PARAM;

  ModuleOp module = handle->getModule();

  PassManager pm(module.getContext(), PassManager::Nesting::Implicit);

  std::string triple = "amdgcn-amd-amdhsa";
  BackendUtils utils(triple, handle->arch, "");

  // Passes for lowering MIOpen dialect.
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep1Pass());
  pm.addPass(mlir::miopen::createAffineTransformPass());
  pm.addPass(mlir::miopen::createAffixTuningParametersPass(0, 0));
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep2Pass());
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep3Pass());
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep4Pass());
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep5Pass());
  pm.addPass(mlir::createLowerMIOpenOpsToGPUPass());

  // Passes for lowering linalg dialect.
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createLowerToCFGPass());

  // Passes for lowering ROCDL dialect
  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(createStripDebugInfoPass());
  pm.addPass(createLowerGpuOpsToROCDLOpsPass(/*indexBitWidth=*/32));
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
  const std::lock_guard<std::mutex> lock(mutex);
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
