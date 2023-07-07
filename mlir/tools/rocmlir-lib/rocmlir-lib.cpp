#include "Miir.h"
#include "mlir/Dialect/Rock/Generator/Conv2dGenerator.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitRocMLIRDialects.h"
#include "mlir/InitRocMLIRPasses.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/Support/TargetSelect.h"

#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>

using namespace mlir;

namespace {
struct MiirHandle_s {
  MiirHandle_s() {
    DialectRegistry &registry = getRegistry();
    llvm::ThreadPool &pool = getThreadPool();
    context = new MLIRContext(registry, MLIRContext::Threading::DISABLED);
    context->setThreadPool(pool);
    // Turn off all diagnotic printing on op and stacktrace
    // Note: This is not necessary with below handler
    context->printOpOnDiagnostic(false);
    context->printStackTraceOnDiagnostic(false);
    // Register a handler that swallows all diagnostic print
    DiagnosticEngine &engine = context->getDiagEngine();
    engine.registerHandler([](Diagnostic &diag) {});
    context->loadDialect<rock::RockDialect, func::FuncDialect>();

    module = ModuleOp::create(UnknownLoc::get(context));
  }

  ~MiirHandle_s() {
    module.release();
    delete context;
  }

  mlir::MLIRContext &getContext() { return *context; }

  mlir::ModuleOp getModule() { return module.get(); }

  mlir::MLIRContext *context;
  mlir::OwningOpRef<mlir::ModuleOp> module;

  std::string triple;
  std::string chip;
  std::string features;
  std::string genTxt;
  int kernelCount = 0;
  int workspace = 0;

private:
  // In multi-threaded context, static intialization is guaranteed to
  // be thread safe, since C++11. Refer to
  // https://en.cppreference.com/w/cpp/language/storage_duration
  //
  // With this guarantee, we are protected from the possible race
  // condition of one thread doing intialization and another doing
  // lowering.
  DialectRegistry &getRegistry() {
    static DialectRegistry registry;
    static std::once_flag once;
    std::call_once(once, []() {
      registerRocMLIRDialects(registry);
      registerRocMLIRPasses();
    });
    return registry;
  }

  // While we have multiple contexts, we'll only have one thread pool among
  // them.
  llvm::ThreadPool &getThreadPool() {
    static llvm::ThreadPool pool;
    return pool;
  }
};

LogicalResult RockEnabled(const mlir::rock::Conv2dGenerator::Config &conf) {
  const std::string &inLayout = conf.inputLayout;
  const std::string &filLayout = conf.filterLayout;
  const std::string &outLayout = conf.outputLayout;

  const static std::set<std::tuple<std::string, std::string, std::string>>
      supportedLayouts = {{"ngchw", "gkcyx", "ngkhw"},
                          {"nhwgc", "gkyxc", "nhwgk"}};

  bool layoutSupported =
      supportedLayouts.count(std::make_tuple(inLayout, filLayout, outLayout)) >
      0;
  bool noBF16 = conf.inputDataTypeStr != "bf16";
  return LogicalResult::success(layoutSupported && noBF16);
}

} // namespace

typedef void *MiirHandle;
static std::mutex mutex;

extern "C" MiirHandle miirCreateHandle(const char *arguments) {
  const std::lock_guard<std::mutex> lock(mutex);

  MiirHandle_s *handle = new MiirHandle_s;
  ModuleOp module = handle->getModule();
  OpBuilder builder(module.getContext());

  mlir::rock::Conv2dGenerator conv2dGenerator;
  if (failed(conv2dGenerator.parseConvConfig(builder, arguments))) {
    return nullptr;
  }

  if (failed(conv2dGenerator.isApplicable())) {
    return nullptr;
  }

  const auto &config = conv2dGenerator.getConfig();
  if (failed(RockEnabled(config))) {
    return nullptr;
  }

  handle->triple = config.triple;
  handle->chip = config.chip;
  handle->features = config.chipFeatures;

  if (failed(conv2dGenerator.getKernelCount(builder, handle->kernelCount))) {
    return nullptr;
  }

  if (failed(conv2dGenerator.getWorkspaceSize(module, handle->workspace))) {
    return nullptr;
  }

  if (failed(conv2dGenerator.genConvModule(module, config.kernelId))) {
    return nullptr;
  }
  return handle;
}

extern "C" int miirGetKernelCount(MiirHandle mlirHandle) {
  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return -1;

  return handle->kernelCount;
}

extern "C" int miirGetWorkspaceSize(MiirHandle mlirHandle) {
  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return 0;

  return handle->workspace;
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
    if (!attr) {
      return failure();
    }
    size = attr.template dyn_cast<IntegerAttr>().getInt();
    return success();
  };

  auto setReturn = [&](int32_t blockSize, int32_t gridSize) {
    *globalSize = gridSize * blockSize;
    *localSize = blockSize;
  };

  int count = 0;
  int32_t blockSize = 0;
  int32_t gridSize = 0;

  // If mlirHandle contains result from miirLowerTuningParams(), it is still
  // a mlir::func::FuncOp
  module.walk([&](func::FuncOp funcOp) -> WalkResult {
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

extern "C" MiirStatus miirLowerTuningParams(MiirHandle mlirHandle) {
  const std::lock_guard<std::mutex> lock(mutex);

  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return MIIR_INVALID_PARAM;

  ModuleOp module = handle->getModule();

  PassManager pm(module->getName(), PassManager::Nesting::Implicit);

  rock::KernelOptions opts;
  opts.enableApplicability = true;
  rock::buildKernelPipeline(pm, opts);

  auto status = pm.run(module);

  return status.succeeded() ? MIIR_SUCCESS : MIIR_BUILD_FAILURE;
}

extern "C" MiirStatus miirLowerBin(MiirHandle mlirHandle) {
  const std::lock_guard<std::mutex> lock(mutex);

  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return MIIR_INVALID_PARAM;

  ModuleOp module = handle->getModule();

  PassManager pm(module->getName(), PassManager::Nesting::Implicit);

  rock::buildKernelPipeline(pm);

  rock::BackendOptions opts;
  opts.triple = handle->triple;
  opts.chip = handle->chip;
  opts.features = handle->features;
  rock::buildBackendPipeline(pm, opts);

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
      auto hsacoAttr = gpuModule->getAttrOfType<StringAttr>(
          gpu::getDefaultGpuBinaryAnnotation());
      if (hsacoAttr) {
        *size = hsacoAttr.getValue().size();
      }
    });
    // 2nd call: copy the hsaco to the target buffer
  } else {
    module.walk([&](gpu::GPUModuleOp gpuModule) {
      auto hsacoAttr = gpuModule->getAttrOfType<StringAttr>(
          gpu::getDefaultGpuBinaryAnnotation());
      if (hsacoAttr) {
        std::string hsaco = hsacoAttr.getValue().str();
        std::copy(hsaco.begin(), hsaco.end(), buffer);
      }
    });
  }
  return MIIR_SUCCESS;
}
