#include "Miir.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/Rock/Generator/ConvGenerator.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitRocMLIRDialects.h"
#include "mlir/InitRocMLIRPasses.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/Rock/IR/ConvolutionDims.h"
#include "mlir/Dialect/Rock/Winograd/WinogradArgLayout.h"
#include "mlir/Dialect/Rock/Winograd/WinogradAssembler.h"
#include "mlir/Dialect/Rock/Winograd/WinogradConvProblem.h"
#include "mlir/Dialect/Rock/Winograd/WinogradSolver.h"

#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
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
    llvm::StdThreadPool &pool = getThreadPool();
    context = new MLIRContext(registry, MLIRContext::Threading::DISABLED);
    context->setThreadPool(pool);
    // Turn off all diagnotic printing on op and stacktrace
    // Note: This is not necessary with below handler
    context->printOpOnDiagnostic(false);
    context->printStackTraceOnDiagnostic(false);
    // Register a handler that swallows all diagnostic print
    DiagnosticEngine &engine = context->getDiagEngine();
    engine.registerHandler([](Diagnostic &diag) {});
    context->loadDialect<rock::RockDialect, func::FuncDialect,
                         mhal::MHALDialect>();

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

  // Winograd state (populated if Winograd path is selected)
  std::optional<mlir::rock::winograd::WinogradKernelSelection>
      winogradSelection;
  bool isWinogradPath = false;
  mlir::rock::ConvGenerator::Config convConfig;

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
  llvm::StdThreadPool &getThreadPool() {
    static llvm::StdThreadPool pool;
    return pool;
  }
};

LogicalResult RockEnabled(const mlir::rock::ConvGenerator::Config &conf) {
  const std::string &inLayout = conf.inputLayout;
  const std::string &filLayout = conf.filterLayout;
  const std::string &outLayout = conf.outputLayout;

  const static std::set<std::tuple<std::string, std::string, std::string>>
      supportedLayouts = {{"ngchw", "gkcyx", "ngkhw"},
                          {"nhwgc", "gkyxc", "nhwgk"},
                          {"ngc01", "gkc01", "ngk01"},
                          {"n01gc", "gk01c", "n01gk"}};

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

  mlir::rock::ConvGenerator convGenerator;
  if (failed(convGenerator.parseConvConfig(builder, arguments))) {
    return nullptr;
  }

  if (failed(convGenerator.isApplicable())) {
    return nullptr;
  }

  const auto &config = convGenerator.getConfig();
  if (failed(RockEnabled(config))) {
    return nullptr;
  }

  handle->triple = config.triple;
  handle->chip = config.chip;
  handle->features = config.chipFeatures;
  handle->convConfig = config;

  if (failed(convGenerator.getKernelCount(builder, handle->kernelCount))) {
    return nullptr;
  }

  if (failed(convGenerator.getWorkspaceSize(module, handle->workspace))) {
    return nullptr;
  }

  if (failed(convGenerator.genConvModule(module, config.kernelId))) {
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
    size = dyn_cast<IntegerAttr>(attr).getInt();
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
  // a gpu::BinaryOp
  module.walk([&](gpu::BinaryOp binary) -> WalkResult {
    gpu::KernelTableAttr metadata =
        cast<gpu::ObjectAttr>(binary.getObjects()[0]).getKernels();
    for (auto kernel : metadata) {
      auto statusBlock = getSizeAttr(kernel.getAttr("block_size"), blockSize);
      auto statusGrid = getSizeAttr(kernel.getAttr("grid_size"), gridSize);
      if (statusBlock.succeeded() && statusGrid.succeeded()) {
        setReturn(blockSize, gridSize);
      }
      ++count;
    }
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
  opts.applicabilityMode = mlir::rock::ApplicabilityMode::Applicability;
  rock::buildKernelPipeline(pm, opts);

  auto status = pm.run(module);

  return status.succeeded() ? MIIR_SUCCESS : MIIR_BUILD_FAILURE;
}

/// Walk the module looking for rock.conv ops with a perf_config attribute
/// starting with "winograd:". Returns the perf_config string if found,
/// or an empty string otherwise.
std::string tryExtractWinogradPerfConfig(ModuleOp module) {
  std::string result;
  module.walk([&](Operation *op) -> WalkResult {
    if (auto perfConfigAttr = op->getAttrOfType<StringAttr>("perf_config")) {
      StringRef perfConfig = perfConfigAttr.getValue();
      if (perfConfig.starts_with("winograd:")) {
        result = perfConfig.str();
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return result;
}

/// Build a WinogradConvProblem from the ConvGenerator config, mapping arch,
/// data types, spatial dimensions, strides, padding, and dilation.
rock::winograd::WinogradConvProblem
buildWinogradConvProblem(const rock::ConvGenerator::Config &config) {
  using namespace rock::winograd;
  WinogradConvProblem problem;
  problem.arch = config.arch;

  auto convDims = rock::ConvGenerator::getConvolutionDims(&config);
  problem.N = convDims.n;
  problem.C = convDims.c;
  problem.K = convDims.k;
  problem.groupCount = convDims.g;

  problem.H = convDims.in.size() > 0 ? convDims.in[0] : 1;
  problem.W = convDims.in.size() > 1 ? convDims.in[1] : 1;
  problem.R = convDims.fil.size() > 0 ? convDims.fil[0] : 1;
  problem.S = convDims.fil.size() > 1 ? convDims.fil[1] : 1;
  problem.outH = convDims.out.size() > 0 ? convDims.out[0] : 1;
  problem.outW = convDims.out.size() > 1 ? convDims.out[1] : 1;

  problem.padH = config.paddingLeftDims.size() > 0 ? config.paddingLeftDims[0]
                                                   : 0;
  problem.padW = config.paddingLeftDims.size() > 1 ? config.paddingLeftDims[1]
                                                   : 0;
  problem.strideH = config.strideDims.size() > 0 ? config.strideDims[0] : 1;
  problem.strideW = config.strideDims.size() > 1 ? config.strideDims[1] : 1;
  problem.dilationH =
      config.dilationDims.size() > 0 ? config.dilationDims[0] : 1;
  problem.dilationW =
      config.dilationDims.size() > 1 ? config.dilationDims[1] : 1;

  problem.isFp16 = (config.inputDataTypeStr == "f16");
  problem.isFp32 = (config.inputDataTypeStr == "f32");
  problem.isBf16 = (config.inputDataTypeStr == "bf16");
  problem.numCU = config.num_cu.value_or(64);
  problem.isXnackEnabled =
      config.chipFeatures.find("+xnack") != std::string::npos;

  // Check if input layout is NCHW-compatible (channel before spatial).
  // NCHW layouts have 'c' before '0' in the input layout string.
  const auto &inLayout = config.inputLayout;
  auto cPos = inLayout.find('c');
  auto hPos = inLayout.find('0');
  problem.isNCHW = (cPos != std::string::npos && hPos != std::string::npos &&
                    cPos < hPos);

  if (config.operation.has_value()) {
    switch (config.operation.value()) {
    case rock::ConvOpType::Fwd:
      problem.direction = WinogradDirection::Forward;
      break;
    case rock::ConvOpType::BwdData:
      problem.direction = WinogradDirection::BackwardData;
      break;
    case rock::ConvOpType::BwdWeight:
      problem.direction = WinogradDirection::BackwardWeight;
      break;
    }
  } else {
    problem.direction = WinogradDirection::Forward;
  }

  return problem;
}

/// Lower the module to a gpu.binary using the Winograd assembly path.
/// Parses the perf_config, assembles the kernel, and creates the IR
/// structures needed for downstream consumption (gpu.binary + func.func).
LogicalResult lowerWinogradToBinary(MiirHandle_s *handle,
                                    const std::string &perfConfig) {
  using namespace rock::winograd;

  ModuleOp module = handle->getModule();
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);
  Location loc = module.getLoc();

  // 1. Build the convolution problem description
  WinogradConvProblem problem = buildWinogradConvProblem(handle->convConfig);

  // 2. Resolve kernel selection from perf_config
  auto selection = WinogradSolver::resolveFromPerfConfig(problem, perfConfig);
  if (!selection)
    return failure();

  // 3. Assemble the Winograd kernel to HSACO
  auto hsaco = assembleWinogradKernel(*selection, handle->chip, handle->triple,
                                      handle->features);
  if (!hsaco)
    return failure();

  // 4. Store the selection in the handle
  handle->winogradSelection = *selection;
  handle->isWinogradPath = true;

  // 5. Clear existing conv IR from the module
  SmallVector<Operation *, 4> toErase;
  module.walk([&](func::FuncOp funcOp) { toErase.push_back(funcOp); });
  for (auto *op : toErase)
    op->erase();

  module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                  UnitAttr::get(ctx));

  // 6. Build kernel metadata for the gpu.binary
  Type elemTy;
  if (problem.isFp16)
    elemTy = builder.getF16Type();
  else if (problem.isBf16)
    elemTy = builder.getBF16Type();
  else
    elemTy = builder.getF32Type();

  // Build a function type matching the Winograd kernel's pointer arguments.
  // Each pointer slot (data, filter, output) maps to a memref<?xelemTy>.
  auto argLayout = (selection->abiVersion == 2)
                       ? WinogradArgLayout::createV2()
                       : WinogradArgLayout::createV1();
  auto pointerSlots = argLayout.getPointerSlots();
  SmallVector<Type> argTypes;
  for (size_t i = 0; i < pointerSlots.size(); ++i) {
    auto memrefTy = MemRefType::get({ShapedType::kDynamic}, elemTy);
    argTypes.push_back(memrefTy);
  }
  auto funcType = builder.getFunctionType(argTypes, {});

  NamedAttrList kernelMetadataAttrs;
  kernelMetadataAttrs.append("block_size",
                             builder.getI32IntegerAttr(selection->blockSize));
  kernelMetadataAttrs.append("grid_size",
                             builder.getI32IntegerAttr(selection->gridSize));

  auto kernelNameAttr = builder.getStringAttr(selection->kernelName);
  auto metadataAttr = gpu::KernelMetadataAttr::get(
      kernelNameAttr, funcType, /*argAttrs=*/nullptr,
      builder.getDictionaryAttr(kernelMetadataAttrs));

  SmallVector<gpu::KernelMetadataAttr> kernels = {metadataAttr};
  auto kernelTable = gpu::KernelTableAttr::get(ctx, kernels, /*isSorted=*/true);

  // 7. Create the gpu.binary with an ObjectAttr containing the HSACO
  auto targetAttr = ROCDL::ROCDLTargetAttr::get(
      ctx, /*optLevel=*/2, handle->triple, handle->chip, handle->features);

  auto objectData =
      builder.getStringAttr(StringRef(hsaco->data(), hsaco->size()));
  auto objectAttr = gpu::ObjectAttr::get(ctx, targetAttr,
                                         gpu::CompilationTarget::Binary,
                                         objectData, /*properties=*/nullptr,
                                         kernelTable);

  builder.setInsertionPointToEnd(module.getBody());
  gpu::BinaryOp::create(builder, loc, selection->kernelName,
                         /*offloadingHandler=*/nullptr,
                         ArrayRef<Attribute>{objectAttr});

  // 8. Create a func.func with kernel launch attributes so that
  //    miirGetExecutionDims can read block_size/grid_size.
  auto funcOp =
      func::FuncOp::create(builder, loc, selection->kernelName, funcType);
  funcOp->setAttr("block_size",
                   builder.getI32IntegerAttr(selection->blockSize));
  funcOp->setAttr("grid_size",
                   builder.getI32IntegerAttr(selection->gridSize));
  module.push_back(funcOp);

  return success();
}

extern "C" MiirStatus miirLowerBin(MiirHandle mlirHandle) {
  const std::lock_guard<std::mutex> lock(mutex);

  MiirHandle_s *handle = static_cast<MiirHandle_s *>(mlirHandle);
  if (handle == nullptr)
    return MIIR_INVALID_PARAM;

  ModuleOp module = handle->getModule();

  // Check if this is a Winograd assembly path
  std::string winoPerfConfig = tryExtractWinogradPerfConfig(module);
  if (!winoPerfConfig.empty()) {
    auto status = lowerWinogradToBinary(handle, winoPerfConfig);
    return status.succeeded() ? MIIR_SUCCESS : MIIR_BUILD_FAILURE;
  }

  // Existing MLIR pipeline path
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
    module.walk([&](gpu::BinaryOp binary) {
      auto object = llvm::cast<mlir::gpu::ObjectAttr>(binary.getObjects()[0]);
      *size = object.getObject().getValue().size();
    });
    // 2nd call: copy the hsaco to the target buffer
  } else {
    module.walk([&](gpu::BinaryOp binary) {
      auto object = llvm::cast<mlir::gpu::ObjectAttr>(binary.getObjects()[0]);
      llvm::StringRef hsaco = object.getObject().getValue();
      std::copy(hsaco.begin(), hsaco.end(), buffer);
    });
  }
  return MIIR_SUCCESS;
}
