#include "mlir/Dialect/Rock/Generator/ConvGenerator.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/ExecutionEngine/RocmDeviceName.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <functional>
#include <numeric>

using namespace mlir;
using namespace mlir::rock;

#define DEBUG_TYPE "conv2d-gen"

ConvGenerator::ConvGenerator(
    const std::string &arch, const std::string &chip, const std::string &triple,
    const std::string &chipFeatures, const std::string &perfConfig,
    std::optional<int> num_cu, bool reverseGrid, GemmFeatures features,
    const std::optional<ConvOpType> operation,
    const std::string &filterDataTypeStr, const std::string &inputDataTypeStr,
    const std::string &outputDataTypeStr, ArrayRef<int> dilations,
    ArrayRef<int> strides, ArrayRef<int> paddingLeft,
    ArrayRef<int> paddingRight, const std::string &filterLayout,
    const std::string &inputLayout, const std::string &outputLayout,
    const std::string &kernelBaseName)
    : config{arch,
             chip,
             triple,
             chipFeatures,
             perfConfig,
             num_cu,
             reverseGrid,
             features,
             operation,
             filterDataTypeStr,
             inputDataTypeStr,
             outputDataTypeStr,
             {dilations.begin(), dilations.end()},
             {strides.begin(), strides.end()},
             {paddingLeft.begin(), paddingLeft.end()},
             {paddingRight.begin(), paddingRight.end()},
             filterLayout,
             inputLayout,
             outputLayout,
             kernelBaseName,
             -1,
             {},
             {},
             {},
             {}} {}

ConvGenerator::ConvGenerator(const ConvGenerator::Config &_config)
    : config(_config) {}

static void strToTokens(const std::string &arguments,
                        std::map<std::string, std::string> &argMap) {
  std::istringstream iss(arguments);
  std::string token;
  std::string argKey;
  while (iss >> token) {
    auto pos = token.find("--");
    if (pos != std::string::npos) {
      argKey = token.substr(pos + 2);
    } else {
      if (!argKey.empty()) {
        argMap[argKey] = token;
        argKey.clear();
      }
    }
  }
}

static llvm::StringMap<int64_t> canonicalizeDims(const ArrayRef<int64_t> dims,
                                                 const StringRef layout) {
  llvm::StringMap<int64_t> ret;
  for (const auto &[keych, dim] : llvm::zip(layout, dims)) {
    StringRef key(&keych, 1);
    ret.insert_or_assign(key, dim);
  }
  return ret;
}

static LogicalResult hasDimensions(const llvm::StringMap<int64_t> &map,
                                   const StringRef wantedLayout,
                                   const StringRef operation) {
  for (size_t i = 0; i < wantedLayout.size(); ++i) {
    auto key = wantedLayout.slice(i, i + 1);
    if (map.count(key) == 0) {
      LLVM_DEBUG(llvm::dbgs() << "Layout for " << operation
                              << " tensor missing dimension: " << key << "\n");
      return failure();
    }
  }
  return success();
}

LogicalResult ConvGenerator::isApplicable(bool checkChip) const {
  if (failed(hasValidDimension())) {
    return failure();
  }

  if (checkChip && failed(hasValidChip())) {
    return failure();
  }

  return success();
}

LogicalResult ConvGenerator::hasValidDimension() const {
  if (any_of(
          llvm::concat<const int64_t>(config.dilationDims, config.strideDims),
          [](const int64_t &a) { return a <= 0; })) {
    LLVM_DEBUG(llvm::dbgs()
               << "Dilation and stride must be a positive integer\n");
    return failure();
  }

  if (any_of(llvm::concat<const int64_t>(config.paddingLeftDims,
                                         config.paddingRightDims),
             [](const int64_t &a) { return a < 0; })) {
    LLVM_DEBUG(llvm::dbgs() << "Padding values cannot be negative\n");
    return failure();
  }

  static const llvm::StringMap<size_t> typeWidths{
      {"f32", sizeof(float)},     {"fp32", sizeof(float)},
      {"fp16", sizeof(uint16_t)}, {"f16", sizeof(uint16_t)},
      {"bf16", sizeof(uint16_t)}, {"i8", sizeof(int8_t)},
      {"fp8", sizeof(uint8_t)},   {"bf8", sizeof(int8_t)}};

  for (const std::string &type :
       {config.filterDataTypeStr, config.inputDataTypeStr,
        config.outputDataTypeStr}) {
    if (typeWidths.count(type) == 0) {
      LLVM_DEBUG(llvm::dbgs() << type << " is not a known datatype\n");
    }
  }

  auto checkDimSizes = [](const ArrayRef<int64_t> dims) -> bool {
    return all_of(dims, [](const int64_t &a) { return a > 0; });
  };

  if (!checkDimSizes(config.inputDimension)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Input tensor dimensions must be strictly positive\n");
    return failure();
  }
  if (!checkDimSizes(config.filterDimension)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Filter tensor dimensions must be strictly positive\n");
  }
  if (!checkDimSizes(config.outputDimension)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Output tensor dimensions must be strictly positive\n");
    return failure();
  }

  auto inDim = canonicalizeDims(config.inputDimension, config.inputLayout);
  auto filDim = canonicalizeDims(config.filterDimension, config.filterLayout);
  auto outDim = canonicalizeDims(config.outputDimension, config.outputLayout);

  // Note: hasDimensions() prints error messages
  if (failed(hasDimensions(inDim, "ngc01", "input")) ||
      failed(hasDimensions(filDim, "gkc01", "filter")) ||
      failed(hasDimensions(outDim, "ngk01", "output"))) {
    return failure();
  }

  if (inDim["n"] != outDim["n"]) {
    LLVM_DEBUG(llvm::dbgs() << "Input and output batch sizes don't match\n");
    return failure();
  }
  if (inDim["g"] != outDim["g"] || inDim["g"] != filDim["g"]) {
    LLVM_DEBUG(llvm::dbgs() << "Group sizes are not consistent between input, "
                               "output, and filter\n");
    return failure();
  }
  if (inDim["c"] != filDim["c"]) {
    LLVM_DEBUG(llvm::dbgs()
               << "Number of channels in input doesn't match number of "
                  "channels in filter\n");
    return failure();
  }
  if (filDim["k"] != outDim["k"]) {
    LLVM_DEBUG(llvm::dbgs()
               << "Number of channels in output doesn't match number of "
                  "channels in filter\n");
    return failure();
  }

  assert(config.strideDims.size() == config.dilationDims.size() &&
         config.strideDims.size() == config.paddingLeftDims.size() &&
         config.strideDims.size() == config.paddingRightDims.size());

  for (size_t i = 0; i < config.strideDims.size(); i++) {
    auto ii = std::to_string(i);
    int64_t expected =
        outputDim(inDim[ii], filDim[ii], config.paddingLeftDims[i],
                  config.paddingRightDims[i], config.strideDims[i],
                  config.dilationDims[i]);
    if (outDim[ii] != expected) {
      LLVM_DEBUG(llvm::dbgs() << "Output dimension " << i << " " << outDim[ii]
                              << " doesn't match " << expected
                              << " computed from other parameters\n");
      return failure();
    }
  }

  for (size_t i = 0; i < config.paddingLeftDims.size(); i++) {
    auto ii = std::to_string(i);
    if (inDim[ii] + config.paddingLeftDims[i] + config.paddingRightDims[i] <
        filDim[ii]) {
      LLVM_DEBUG(llvm::dbgs() << "Input, including padding, is smaller than "
                                 "the filter in dimension "
                              << i << "\n");
      return failure();
    }
  }

  return success();
}

LogicalResult ConvGenerator::hasValidChip() const {
  // We support in between gfx900 to gfx908 and gfx1030 for nonxdlops algorithm
  // For example, gfx803, gfx90c are unsupported now
  unsigned int chipHexNumber = 0;
  if (sscanf(config.chip.c_str(), "gfx%x", &chipHexNumber) != 1)
    return failure();

  constexpr size_t NUM_SUPPORTED_CHIPS = 12;
  static const unsigned int supportedChips[NUM_SUPPORTED_CHIPS] = {
      0x900, 0x906,  0x908,  0x90a,  0x940,  0x941,
      0x942, 0x1030, 0x1100, 0x1101, 0x1102, 0x1103};
  const unsigned int *ptr;
  ptr = std::find(supportedChips, supportedChips + NUM_SUPPORTED_CHIPS,
                  chipHexNumber);
  if (ptr == supportedChips + NUM_SUPPORTED_CHIPS)
    return failure();

  // XDLOPS are only supported on MI-100 (gfx908) and MI-200 (gfx90a)
  if (bitEnumContainsAll(config.features, GemmFeatures::mfma) &&
      (chipHexNumber != 0x908 && chipHexNumber != 0x90a))
    return failure();

  // WMMA is only supported on gfx11xx
  if (bitEnumContainsAll(config.features, GemmFeatures::wmma) &&
      (chipHexNumber > 0x1103))
    return failure();
  return success();
}

LogicalResult ConvGenerator::getKernelCount(OpBuilder &builder,
                                            int &kernelCount) const {
  if (config.kernelId > 0) { // generate only 1 specified kernel
    kernelCount = 1;
    return success();
  }
  assert(config.operation.has_value());
  switch (config.operation.value()) {
  case ConvOpType::BwdData:
    kernelCount = getBwdDataKernelCount();
    return success();
  case ConvOpType::Fwd:
    kernelCount = 1;
    return success();
  case ConvOpType::BwdWeight:
    LogicalResult res = getBwdWeightKernelCount(builder, kernelCount);
    return res;
  }
  llvm_unreachable("Invalid conv2d operation");
}

LogicalResult ConvGenerator::getBwdWeightKernelCount(OpBuilder &builder,
                                                     int &kernelCount) const {
  assert(config.operation.value() == ConvOpType::BwdWeight);

  kernelCount = 1;
  if (isAccel(config.features)) {
    bool needExtraPad = false;
    if (failed(needExtraPadBwdWeight(builder, needExtraPad))) {
      return failure();
    }
    if (!needExtraPad) {
      Type dataType = getInputDataType(builder);
      if (dataType.isF32()) {
        // For the following case, use 2 kernels:
        // - backward weight
        // - XDLOPS
        // - fp32
        // - No need extra pad along Gemm M/N/K
        // The first kernel will 0-initialize the output (filter tensor).
        // The second kernel will conduct the actual backward weight
        // convolution, using atomic add instructions.
        kernelCount = 2;
      } else if (dataType.isF16()) {
        // For the following case, use 3 kernels:
        // - backward weight
        // - XDLOPS
        // - fp16
        // - No need extra pad along Gemm M/N/K
        // The first kernel will 0-initialize the workspace.
        // The second kernel will conduct the actual backward weight
        // convolution, using atomic add instructions. The third kernel will do
        // elementwise conversion from fp32 in the workspace to fp16 in the
        // actual output (filter tensor).
        kernelCount = 3;
      }
    }
  }
  return success();
}

int ConvGenerator::getBwdDataKernelCount() const {
  llvm::SmallVector<int64_t> gemmIds = backwardDataKernelIds(
      config.strideDims, config.dilationDims, config.filterDims);
  return static_cast<int>(gemmIds.size());
}

static Type strToType(StringRef dataTypeStr, OpBuilder &builder) {
  std::optional<Type> type =
      llvm::StringSwitch<std::optional<Type>>(dataTypeStr)
          .Case("f32", builder.getF32Type())
          .Case("fp32", builder.getF32Type())
          .Case("f16", builder.getF16Type())
          .Case("fp16", builder.getF16Type())
          .Case("bf16", builder.getBF16Type())
          .Case("i32", builder.getI32Type())
          .Case("i8", builder.getI8Type())
          .Cases("f8E5M2FNUZ", "bf8", builder.getFloat8E5M2FNUZType())
          .Cases("f8E4M3FNUZ", "fp8", builder.getFloat8E4M3FNUZType())
          .Default(std::nullopt);
  if (!type) {
    llvm::errs() << "Unknown data type: " << dataTypeStr << "\n";
    exit(1);
  }
  return *type;
}

Type ConvGenerator::getFilterDataType(OpBuilder &builder) const {
  if (config.filterDataTypeStr.empty())
    return getInputDataType(builder);
  return strToType(config.filterDataTypeStr, builder);
}

Type ConvGenerator::getInputDataType(OpBuilder &builder) const {
  return strToType(config.inputDataTypeStr, builder);
}

Type ConvGenerator::getOutputDataType(OpBuilder &builder) const {
  if (config.outputDataTypeStr.empty()) {
    // Special-case i8 -> i32 translation as a default
    Type inType = getInputDataType(builder);
    if (inType.isInteger(8))
      return builder.getIntegerType(32);
  }
  return strToType(config.outputDataTypeStr, builder);
}

LogicalResult ConvGenerator::needExtraPadBwdWeight(OpBuilder &builder,
                                                   bool &needExtraPad) const {
  Type dataType = getInputDataType(builder);
  ConvOpType dir = config.operation.value();
  assert(dir == ConvOpType::BwdWeight &&
         "This method should only be called for wrw ops");

  ConvolutionDims convDims = getConvolutionDims();
  GemmSize gemmSize = GemmSize::fromConvolution(dir, convDims);

  needExtraPad = false;
  // TODO: support mixed-type fp8 here too.
  PopulateParamsInfo info{/*gemmSize=*/gemmSize,
                          /*arch*=*/config.arch,
                          /*gemmFeatures=*/config.features,
                          /*gemmAType=*/dataType,
                          /*gemmBType=*/dataType,
                          /*kernelType=*/KernelType::ConvBwdWeight,
                          /*batchSize=*/convDims.n,
                          /*numCU=*/getNumCU()};

  if (isAccel(config.features)) {
    auto populateParamsAccelPtr = PopulateParamsAccel::select(config.features);
    InitParamsAccel validParams;
    auto res = populateParamsAccelPtr->obtainTuningParameters(
        builder, info, config.perfConfig, validParams);
    if (succeeded(res)) {
      needExtraPad = (populateParamsAccelPtr->calculatePaddingAmount(
                          validParams, gemmSize) != 0);
      return success();
    }
  } else {
    PopulateParams populateParams;
    InitParamsNonAccel validParams;
    auto res = populateParams.obtainTuningParameters(
        builder, info, config.perfConfig, validParams);

    if (succeeded(res)) {
      needExtraPad =
          (populateParams.calculatePaddingAmount(validParams, gemmSize) != 0);
      return success();
    }
  }
  return failure();
}

LogicalResult ConvGenerator::hasWorkspace(OpBuilder &builder,
                                          bool &needWorkspace) const {
  // Decide if a workspace is needed.
  // Preconditions:
  // - data type: fp16
  // - operation: backward weight conv2d.
  // - use XDLOPS.
  // - No need to pad along Gemm M/N/K dimension.

  needWorkspace = false;
  if (config.operation.has_value()) {
    Type dataType = getInputDataType(builder);
    ConvOpType dir = config.operation.value();
    if ((dir == ConvOpType::BwdWeight) && isAccel(config.features) &&
        (dataType == builder.getF16Type())) {
      // In case we need extra padding, do not use workspace.
      bool needPadding = false;
      if (failed(needExtraPadBwdWeight(builder, needPadding))) {
        return failure();
      }
      needWorkspace = !needPadding;
    }
  }
  return success();
}

LogicalResult ConvGenerator::getWorkspaceSize(ModuleOp &module,
                                              int &workspaceSize) const {
  // Currently onlt in the following condition would a workspace is needed.
  // - data type: fp16
  // - operation: backward weight conv2d.
  // - use XDLOPS.
  // - No need to pad along Gemm M/N/K dimension.
  // Workspace size is the same as the filter dimension, with fp32 type.
  bool needWorkspace = false;
  OpBuilder builder(module.getContext());
  if (failed(hasWorkspace(builder, needWorkspace))) {
    return failure();
  }
  if (needWorkspace) {
    workspaceSize = std::accumulate(config.filterDimension.begin(),
                                    config.filterDimension.end(), 1,
                                    std::multiplies<int>()) *
                    builder.getF32Type().getWidth() / 8;
  }
  return success();
}

uint32_t ConvGenerator::getNumCU() const {
  return config.num_cu.has_value() ? config.num_cu.value()
                                   : rock::lookupArchInfo(config.arch).minNumCU;
}

LogicalResult ConvGenerator::parseConvConfig(OpBuilder &builder,
                                             const char *arguments) {
  std::map<std::string, std::string> argMap;
  strToTokens(arguments, argMap);

  auto isValid = [&argMap]() {
    // only require tensor configs
    static std::vector<std::string> validKeys = {
        "batchsize",   "groupsize",    "in_layout", "in_type",
        "in_channels", "in_h",         "in_w",      "out_layout",
        "out_type",    "out_channels", "out_h",     "out_w",
        "fil_layout",  "fil_type",     "fil_w",     "fil_h"};
    if (argMap["in_layout"].length() > 5) { // Ie, 3-D.
      validKeys.push_back("in_d");
      validKeys.push_back("out_d");
      validKeys.push_back("fil_d");
    }
    auto isPresent = [&argMap](const std::string &key) {
      return argMap.count(key) > 0;
    };
    if (!llvm::all_of(validKeys, isPresent)) {
      return false;
    }
    return (argMap["fil_layout"].length() == argMap["in_layout"].length()) &&
           (argMap["in_layout"].length() == argMap["out_layout"].length());

  };

  // Proceed only if we have a valid argMap. Otherwise leave the handle to be
  // empty
  if (!isValid()) {
    return failure();
  }

  auto strToLong = [&argMap](const std::string &argKey) {
    return std::stol(argMap[argKey]);
  };

  auto strToInt = [&argMap](const std::string &key, auto &setting) {
    if (argMap.find(key) != argMap.end()) {
      setting = std::stoi(argMap[key]);
    }
  };

  auto strToStr = [&argMap](const std::string &key, std::string &setting) {
    if (argMap.find(key) != argMap.end()) {
      setting = argMap[key];
    }
  };

  std::string arch;
  strToStr("arch", arch);
  RocmDeviceName splitter;
  if (failed(splitter.parse(arch))) {
    return failure();
  }
  // Canonicalize architecture name
  SmallString<64> canonicalArch;
  splitter.getFullName(canonicalArch);
  arch = canonicalArch.str();

  config.arch = arch;
  config.chip = splitter.getChip().str();
  config.chipFeatures = splitter.getFeaturesForBackend();
  config.triple = splitter.getTriple().str();

  strToStr("perf_config", config.perfConfig);
  strToInt("num_cu", config.num_cu);
  strToInt(rock::ReverseGridAttrAttr::getMnemonic().str(), config.reverseGrid);

  // conv settings
  auto const op = getConvOpTypeForName(argMap["operation"]);
  if (!op.has_value()) {
    return failure();
  }

  auto canonicalizeDataType = [](const std::string &type) {
    if (type == "fp32")
      return std::string("f32");
    if (type == "fp16")
      return std::string("f16");
    if (type == "f8E5M2FNUZ")
      return std::string("bf8");
    if (type == "f8E4M3FNUZ")
      return std::string("fp8");
    return type;
  };
  config.operation = op.value();
  strToInt("kernel_id", config.kernelId);
  config.filterDataTypeStr = canonicalizeDataType(argMap["fil_type"]);
  config.inputDataTypeStr = canonicalizeDataType(argMap["in_type"]);
  config.outputDataTypeStr = canonicalizeDataType(argMap["out_type"]);
  strToInt("dilation_h", config.dilationDims[DIM::HEIGHT]);
  strToInt("dilation_w", config.dilationDims[DIM::WIDTH]);
  if (config.dilationDims.size() > DIM::DEPTH)
    strToInt("dilation_d", config.dilationDims[DIM::DEPTH]);
  strToInt("conv_stride_h", config.strideDims[DIM::HEIGHT]);
  strToInt("conv_stride_w", config.strideDims[DIM::WIDTH]);
  if (config.strideDims.size() > DIM::DEPTH)
    strToInt("conv_stride_d", config.strideDims[DIM::DEPTH]);
  strToInt("padding_h", config.paddingLeftDims[DIM::HEIGHT]);
  strToInt("padding_h", config.paddingRightDims[DIM::HEIGHT]);
  strToInt("padding_w", config.paddingLeftDims[DIM::WIDTH]);
  strToInt("padding_w", config.paddingRightDims[DIM::WIDTH]);
  if (config.paddingLeftDims.size() > DIM::DEPTH)
    strToInt("padding_d", config.paddingLeftDims[DIM::DEPTH]);
  if (config.paddingRightDims.size() > DIM::DEPTH)
    strToInt("padding_d", config.paddingRightDims[DIM::DEPTH]);

  strToStr("kernel_name", config.kernelBaseName);

  // Get the default features associated with the chip (and with the data type)
  AmdArchInfo archInfo = lookupArchInfo(splitter.getChip());
  Type filterDataType = getFilterDataType(builder);
  Type inputDataType = getInputDataType(builder);
  Type filterElemType = mlir::getElementTypeOrSelf(filterDataType);
  Type inputElemType = mlir::getElementTypeOrSelf(inputDataType);
  Type dataType = inputDataType;
  config.features = archInfo.getDefaultFeatures(dataType);
  // Disable acceleration for mixed types
  if (filterElemType.getIntOrFloatBitWidth() !=
      inputElemType.getIntOrFloatBitWidth()) {
    config.features = bitEnumClear(config.features, GemmFeatures::mfma);
    config.features = bitEnumClear(config.features, GemmFeatures::wmma);
  }
  // Force acceleration if that's what the client wants
  int hasAccel = 0;
  strToInt("x2", hasAccel);
  config.features =
      bitEnumSet(config.features, GemmFeatures::mfma, hasAccel == 1);
  config.features =
      bitEnumSet(config.features, GemmFeatures::wmma, hasAccel == 2);

  // Allow only fwd direction for 8-bit types. Reject other directions.
  if (config.operation.value() != ConvOpType::Fwd &&
      (config.inputDataTypeStr == "i8" || config.inputDataTypeStr == "fp8" ||
       config.inputDataTypeStr == "bf8")) {
    return failure();
  }

  // Rock has NCHW as layout string for all three tensors
  config.inputLayout = translateLayout(
      argMap["in_layout"], std::string("NGCHWD012"), std::string("ngchwd012"));
  config.filterLayout = translateLayout(
      argMap["fil_layout"], std::string("GNCHWD012"), std::string("gkcyxz012"));
  config.outputLayout = translateLayout(
      argMap["out_layout"], std::string("NGCHWD012"), std::string("ngkhwd012"));

  // Determine tensor dimensions.
  SmallVector<int64_t> inDims{strToLong("in_h"), strToLong("in_w")};
  if (argMap.count("in_d") > 0)
    inDims.push_back(strToLong("in_d"));
  SmallVector<int64_t> outDims{strToLong("out_h"), strToLong("out_w")};
  if (argMap.count("out_d") > 0)
    outDims.push_back(strToLong("out_d"));
  SmallVector<int64_t> filDims{strToLong("fil_h"), strToLong("fil_w")};
  if (argMap.count("fil_d") > 0)
    filDims.push_back(strToLong("fil_d"));
  auto status = parseConvDims(strToLong("batchsize"), strToLong("groupsize"),
                              strToLong("in_channels"), inDims,
                              strToLong("out_channels"), outDims, filDims);

  if (status.failed()) {
    return failure();
  }

  return success();
}

LogicalResult ConvGenerator::parseConvDims(int64_t batchSize, int64_t groupSize,
                                           int64_t inputChannel,
                                           ArrayRef<int64_t> inputDims,
                                           int64_t outputChannel,
                                           ArrayRef<int64_t> outputDims,
                                           ArrayRef<int64_t> filterDims) {
  config.filterDims.clear();
  for (auto dim : filterDims)
    config.filterDims.push_back(dim);

  llvm::StringMap<int64_t> filterMap = {{"k", outputChannel / groupSize},
                                        {"g", groupSize},
                                        {"c", inputChannel / groupSize},
                                        {"y", filterDims[0]},
                                        {"x", filterDims[1]}};
  for (size_t i = 0; i < filterDims.size(); i++)
    filterMap[std::to_string(i)] = filterDims[i];

  llvm::StringMap<int64_t> inputMap = {{"n", batchSize},
                                       {"g", groupSize},
                                       {"c", inputChannel / groupSize},
                                       {"h", inputDims[0]},
                                       {"w", inputDims[1]}};
  for (size_t i = 0; i < inputDims.size(); i++)
    inputMap[std::to_string(i)] = inputDims[i];

  llvm::StringMap<int64_t> outputMap = {{"n", batchSize},
                                        {"g", groupSize},
                                        {"k", outputChannel / groupSize},
                                        {"h", outputDims[0]},
                                        {"w", outputDims[1]}};
  for (size_t i = 0; i < outputDims.size(); i++)
    outputMap[std::to_string(i)] = outputDims[i];

  auto convertLayout = [](char &key, llvm::StringMap<int64_t> &kmap,
                          auto &dims) {
    auto keyl = std::string{static_cast<char>(std::tolower(key))};
    if (!kmap.contains(keyl) && !isdigit(key)) {
      keyl = "k";
      if (!kmap.contains(keyl))
        return false;
    }
    dims.push_back(kmap[keyl]);
    key = keyl[0];
    return true;
  };

  size_t layoutLen = config.filterLayout.length();
  if (layoutLen != config.inputLayout.length() ||
      layoutLen != config.outputLayout.length()) {
    return failure();
  }
  // Determine dimensions.
  for (size_t i = 0; i < layoutLen; ++i) {
    if (!convertLayout(config.filterLayout[i], filterMap,
                       config.filterDimension)) {
      return failure();
    }
    if (!convertLayout(config.inputLayout[i], inputMap,
                       config.inputDimension)) {
      return failure();
    }
    if (!convertLayout(config.outputLayout[i], outputMap,
                       config.outputDimension)) {
      return failure();
    }
  }

  // Determine kernel name, if there isn't one.
  if (config.kernelBaseName.empty()) {
    assert(config.operation.has_value());
    auto opType = config.operation.value();
    config.kernelBaseName = std::string("rock_") +
                            getNameForConvOpType(opType).str() + "_" +
                            config.filterLayout + "_" + config.inputLayout +
                            "_" + config.outputLayout;
  }

  return success();
}

void ConvGenerator::setKernelName(const std::string &newName) {
  config.kernelBaseName = newName;
}

void ConvGenerator::setDataTypes(const std::string &dataTypeStr) {
  config.filterDataTypeStr = config.inputDataTypeStr =
      config.outputDataTypeStr = dataTypeStr;
}

void ConvGenerator::flipAccel() {
  config.features = bitEnumClear(config.features, GemmFeatures::mfma);
  config.features = bitEnumClear(config.features, GemmFeatures::wmma);
}

void ConvGenerator::setPerfConfig(StringRef perfConfig) {
  config.perfConfig = perfConfig.str();
}

ConvolutionDims ConvGenerator::getConvolutionDims() const {
  auto inDim = canonicalizeDims(config.inputDimension, config.inputLayout);
  auto filDim = canonicalizeDims(config.filterDimension, config.filterLayout);
  auto outDim = canonicalizeDims(config.outputDimension, config.outputLayout);

  SmallVector<int64_t> inDims;
  for (size_t i = 0; i < config.inputLayout.size() - 3; i++)
    inDims.push_back(inDim[std::to_string(i)]);
  SmallVector<int64_t> filDims;
  for (size_t i = 0; i < config.filterLayout.size() - 3; i++)
    filDims.push_back(filDim[std::to_string(i)]);
  SmallVector<int64_t> outDims;
  for (size_t i = 0; i < config.outputLayout.size() - 3; i++)
    outDims.push_back(outDim[std::to_string(i)]);

  return ConvolutionDims(filDims, outDims, inDims, filDim["k"], filDim["c"],
                         inDim["n"], inDim["g"]);
}

LogicalResult ConvGenerator::genConvModule(ModuleOp &module, int rawKernelId,
                                           bool is_verifier,
                                           bool ignoreTuning) {
  OpBuilder builder(module.getContext());

  Type filterDataType = getFilterDataType(builder);
  Type inputDataType = getInputDataType(builder);
  Type outputDataType = getOutputDataType(builder);
  if (!filterDataType || !inputDataType || !outputDataType)
    return failure();

  // Construct a new FuncOp.
  auto filterArgType =
      MemRefType::get(ArrayRef<int64_t>(config.filterDimension.begin(),
                                        config.filterDimension.end()),
                      filterDataType);
  auto inputArgType =
      MemRefType::get(ArrayRef<int64_t>(config.inputDimension.begin(),
                                        config.inputDimension.end()),
                      inputDataType);
  auto outputArgType =
      MemRefType::get(ArrayRef<int64_t>(config.outputDimension.begin(),
                                        config.outputDimension.end()),
                      outputDataType);

  bool hasWorkspace = false;
  if (failed(this->hasWorkspace(builder, hasWorkspace))) {
    return failure();
  }
  Type workspaceArgType;
  if (hasWorkspace) {
    workspaceArgType =
        MemRefType::get(ArrayRef<int64_t>(config.filterDimension.begin(),
                                          config.filterDimension.end()),
                        builder.getF32Type());
  }

  SmallVector<Type, 3> logicalFuncArgTypes = {filterArgType, inputArgType,
                                              outputArgType};
  if (hasWorkspace) {
    logicalFuncArgTypes = {filterArgType, inputArgType, outputArgType,
                           workspaceArgType};
  }
  SmallVector<Type, 3> physicalFuncArgTypes =
      llvm::map_to_vector(logicalFuncArgTypes, getFlattenedType);
  auto funcType = builder.getFunctionType(physicalFuncArgTypes, {});

  std::string kernelName = config.kernelBaseName;
  if (is_verifier) {
    kernelName += "_ver";
  }

  func::FuncOp func = module.lookupSymbol<func::FuncOp>(kernelName);
  if (func) {
    assert(func.isDeclaration());
    func.erase();
  }

  // Fix raw kernel ID in case it is less than 0.
  // The only case this could happen is to query the number of kernels needed
  // from MIIR API, where the kernel ID is not yet unknown.
  if (rawKernelId < 0)
    rawKernelId = 0;

  // Annotate kernel attribute to the FuncOp.
  StringAttr archStrAttr = builder.getStringAttr(config.arch);
  NamedAttribute archAttr = builder.getNamedAttr("mhal.arch", archStrAttr);

  SmallVector<NamedAttribute, 2> kernelAttrs = {
      builder.getNamedAttr("kernel", builder.getI32IntegerAttr(rawKernelId)),
      archAttr};

  // Construct the FuncOp.
  func = func::FuncOp::create(builder.getUnknownLoc(), kernelName, funcType,
                              ArrayRef<NamedAttribute>(kernelAttrs));
  if (config.reverseGrid) {
    func->setAttr(rock::ReverseGridAttrAttr::getMnemonic(),
                  builder.getUnitAttr());
  }
  module.push_back(func);
  if (!is_verifier)
    module->setAttr(archAttr.getName(), archAttr.getValue());
  if (func.getName() != kernelName) {
    return failure();
  }
  kernelFunc = func;

  // Construct a new Block.
  Block *block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);

  // Construct a new ConvOp.
  SmallVector<StringAttr, 5> filterLayoutSpec;
  SmallVector<StringAttr, 5> inputLayoutSpec;
  SmallVector<StringAttr, 5> outputLayoutSpec;
  for (auto &key : config.filterLayout)
    filterLayoutSpec.push_back(builder.getStringAttr(StringRef(&key, 1)));
  for (auto &key : config.inputLayout)
    inputLayoutSpec.push_back(builder.getStringAttr(StringRef(&key, 1) + "i"));
  for (auto &key : config.outputLayout)
    outputLayoutSpec.push_back(builder.getStringAttr(StringRef(&key, 1) + "o"));

  // Set kernel ID to  be the same as the raw kernel ID.
  // For backward data convolution, additional processing is needed below.
  int64_t kernelId = rawKernelId;

  // Obtain kernel ID as used by backwards data kernels from the raw, 0-indexed
  // kernel ID.
  if (config.operation.value() == ConvOpType::BwdData) {
    llvm::SmallVector<int64_t> kernelIds = backwardDataKernelIds(
        config.strideDims, config.dilationDims, config.filterDims);
    assert(kernelIds.size() > static_cast<size_t>(rawKernelId));
    kernelId = kernelIds[rawKernelId];
  }

  std::vector<NamedAttribute> attributes{
      builder.getNamedAttr("arch", archStrAttr),

      builder.getNamedAttr(
          "filter_layout",
          builder.getArrayAttr(ArrayRef<Attribute>(filterLayoutSpec.begin(),
                                                   filterLayoutSpec.end()))),
      builder.getNamedAttr(
          "input_layout", builder.getArrayAttr(ArrayRef<Attribute>(
                              inputLayoutSpec.begin(), inputLayoutSpec.end()))),
      builder.getNamedAttr(
          "output_layout",
          builder.getArrayAttr(ArrayRef<Attribute>(outputLayoutSpec.begin(),
                                                   outputLayoutSpec.end()))),
      builder.getNamedAttr("numCU", builder.getI32IntegerAttr(getNumCU())),
  };

  // The backwards data kernel needs to know its kernel ID, as there are
  // multiple copies of it that compute different parts of the input tensor in
  // some contexts. No other kernel has a meaningful use for the kernel ID.
  if (config.operation.value() == ConvOpType::BwdData) {
    attributes.push_back(
        builder.getNamedAttr("kernelId", builder.getIndexAttr(kernelId)));
  }
  // features
  GemmFeaturesAttr features =
      builder.getAttr<GemmFeaturesAttr>(config.features);
  attributes.push_back(builder.getNamedAttr("features", features));

  SmallVector<int64_t, 8> paddingArray;
  for (const auto &[left, right] :
       zip(config.paddingLeftDims, config.paddingRightDims)) {
    paddingArray.push_back(left);
    paddingArray.push_back(right);
  }

  attributes.push_back(
      builder.getNamedAttr("padding", builder.getIndexArrayAttr(paddingArray)));

  attributes.push_back(builder.getNamedAttr(
      "strides", builder.getIndexArrayAttr(config.strideDims)));

  attributes.push_back(builder.getNamedAttr(
      "dilations", builder.getIndexArrayAttr(config.dilationDims)));

  // perf_config
  if (!ignoreTuning && !config.perfConfig.empty()) {
    attributes.push_back(builder.getNamedAttr(
        "perf_config", builder.getStringAttr(config.perfConfig)));
  }

  SmallVector<SmallVector<StringRef>, 4> argDimNameRefs;
  argDimNameRefs.reserve(logicalFuncArgTypes.size());
  auto referenceNames = [&](ArrayRef<StringAttr> layout) {
    argDimNameRefs.push_back(llvm::map_to_vector(
        layout, [](StringAttr sa) { return sa.getValue(); }));
  };
  referenceNames(filterLayoutSpec);
  referenceNames(inputLayoutSpec);
  referenceNames(outputLayoutSpec);
  if (hasWorkspace)
    referenceNames(filterLayoutSpec);

  SmallVector<Value, 4> args;
  expandFlatFunctionArguments(builder, func, argDimNameRefs,
                              logicalFuncArgTypes, args);
  switch (config.operation.value()) {
  case ConvOpType::Fwd: {
    builder.create<ConvOp>(builder.getUnknownLoc(), ArrayRef<Type>{}, args,
                           attributes);
  } break;
  case ConvOpType::BwdData: {
    if (kernelId < 0) {
      // zero init input tensor
      builder.create<InitKernelOp>(
          builder.getUnknownLoc(), /*resultType=*/TypeRange{},
          func.getArgument(1), features, /*initValueAttr=*/nullptr,
          /*blockSize=*/nullptr, /*gridSize=*/nullptr,
          /*elemsPerThread=*/nullptr);
    } else {
      builder.create<ConvBwdDataOp>(builder.getUnknownLoc(), ArrayRef<Type>{},
                                    args, attributes);
    }
  } break;
  case ConvOpType::BwdWeight: {
    int kernelCount = 0;
    if (failed(getBwdWeightKernelCount(builder, kernelCount))) {
      return failure();
    }
    bool hasUtilities = (kernelCount > 1);
    if (hasUtilities && kernelId == 0) {
      // If there is a workspace, zero-init it, otherwise fill the filter tensor
      builder.create<InitKernelOp>(builder.getUnknownLoc(),
                                   /*resultType=*/TypeRange{},
                                   func.getArgument(hasWorkspace ? 3 : 0),
                                   features, /*initValueAttr=*/nullptr,
                                   /*blockSize=*/nullptr, /*gridSize=*/nullptr,
                                   /*elemsPerThread=*/nullptr);
    } else if (hasUtilities && kernelId == 2) {
      // Workspace -> filter tensor
      builder.create<ConvertingCopyKernelOp>(
          builder.getUnknownLoc(), /*resultType=*/TypeRange{},
          func.getArgument(3), func.getArgument(0), features,
          /*blockSize=*/nullptr, /*gridSize=*/nullptr,
          /*elemsPerThread=*/nullptr);
    } else {
      builder.create<ConvBwdWeightOp>(builder.getUnknownLoc(), ArrayRef<Type>{},
                                      args, attributes);
    }
  } break;
  }

  builder.create<func::ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  return success();
}

func::FuncOp ConvGenerator::getKernelFunc() const { return kernelFunc; }
