#include "mlir/Dialect/Rock/Generator/Conv2dGenerator.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/Generator/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/ExecutionEngine/RocmDeviceName.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
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

Conv2dGenerator::Conv2dGenerator(
    const std::string &arch, const std::string &chip, const std::string &triple,
    const std::string &chipFeatures, const std::string &perfConfig, int num_cu,
    GemmFeatures features, const Optional<ConvOpType> operation,
    const std::string &dataTypeStr, int dilationHeight, int dilationWidth,
    int strideHeight, int strideWidth, int paddingHeightLeft,
    int paddingHeightRight, int paddingWidthLeft, int paddingWidthRight,
    const std::string &filterLayout, const std::string &inputLayout,
    const std::string &outputLayout, const std::string &kernelBaseName)
    : config{arch,
             chip,
             triple,
             chipFeatures,
             perfConfig,
             num_cu,
             features,
             operation,
             dataTypeStr,
             dilationHeight,
             dilationWidth,
             strideHeight,
             strideWidth,
             paddingHeightLeft,
             paddingHeightRight,
             paddingWidthLeft,
             paddingWidthRight,
             filterLayout,
             inputLayout,
             outputLayout,
             kernelBaseName,
             -1,
             {},
             {},
             {},
             -1,
             -1} {}

Conv2dGenerator::Conv2dGenerator(const Conv2dGenerator::Config &_config)
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
  for (const auto &tuple : llvm::zip(layout, dims)) {
    StringRef key(&std::get<0>(tuple), 1);
    ret.insert_or_assign(key, std::get<1>(tuple));
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

LogicalResult Conv2dGenerator::isApplicable(bool checkChip) const {
  if (failed(hasValidDimension())) {
    return failure();
  }

  if (checkChip && failed(hasValidChip())) {
    return failure();
  }

  return success();
}

LogicalResult Conv2dGenerator::hasValidDimension() const {
  static const SmallVector<int64_t, 4> strictlyPositiveParams{
      config.dilationHeight, config.dilationWidth, config.strideHeight,
      config.strideWidth};
  if (std::any_of(strictlyPositiveParams.begin(), strictlyPositiveParams.end(),
                  [](const int64_t &a) { return a <= 0; })) {
    LLVM_DEBUG(llvm::dbgs()
               << "Dilation and stride must be a positive integer\n");
    return failure();
  }

  static const SmallVector<int64_t, 4> nonNegativeParams{
      config.paddingHeightLeft, config.paddingHeightRight,
      config.paddingWidthLeft, config.paddingWidthRight};
  if (std::any_of(nonNegativeParams.begin(), nonNegativeParams.end(),
                  [](const int64_t &a) { return a < 0; })) {
    LLVM_DEBUG(llvm::dbgs() << "Padding values cannot be negative\n");
    return failure();
  }

  static const llvm::StringMap<size_t> typeWidths{
      {"f32", sizeof(float)},     {"fp32", sizeof(float)},
      {"fp16", sizeof(uint16_t)}, {"f16", sizeof(uint16_t)},
      {"bf16", sizeof(uint16_t)}, {"i8", sizeof(int8_t)}};

  auto checkDimSizes = [](const SmallVector<int64_t, 5> &dims) -> bool {
    return std::all_of(dims.begin(), dims.end(),
                       [](const int64_t &a) { return a > 0; });
  };

  if (typeWidths.count(config.dataTypeStr) == 0) {
    LLVM_DEBUG(llvm::dbgs()
               << config.dataTypeStr << " is not a known datatype\n");
  }
  size_t elementWidth = typeWidths.lookup(config.dataTypeStr);

  if (!checkDimSizes(config.inputDimension)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Input tensor dimensions must be strictly positive\n");
    return failure();
  }
  if (!checkDimSizes(config.filterDimension)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Filter tensoor dimensions must be strictly positive\n");
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
  if (failed(hasDimensions(inDim, "ngchw", "input")) ||
      failed(hasDimensions(filDim, "gkcyx", "filter")) ||
      failed(hasDimensions(outDim, "ngkhw", "output"))) {
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

  int64_t expectedOutHeight = outputDim(
      inDim["h"], filDim["y"], config.paddingHeightLeft,
      config.paddingHeightRight, config.strideHeight, config.dilationHeight);
  int64_t expectedOutWidth = outputDim(
      inDim["w"], filDim["x"], config.paddingWidthLeft,
      config.paddingWidthRight, config.strideWidth, config.dilationWidth);
  if (outDim["h"] != expectedOutHeight) {
    LLVM_DEBUG(llvm::dbgs()
               << "Output height " << outDim["h"] << " doesn't match height "
               << expectedOutHeight << " computed from other parameters\n");
    return failure();
  }
  if (outDim["w"] != expectedOutWidth) {
    LLVM_DEBUG(llvm::dbgs()
               << "Output width " << outDim["w"] << " doesn't match width "
               << expectedOutWidth << " computed from other parameters\n");
    return failure();
  }

  if (inDim["h"] + config.paddingHeightLeft + config.paddingHeightRight <
      filDim["y"]) {
    LLVM_DEBUG(llvm::dbgs()
               << "Input, including padding, is shorter than the filter\n");
    return failure();
  }

  if (inDim["w"] + config.paddingWidthLeft + config.paddingWidthRight <
      filDim["x"]) {
    LLVM_DEBUG(llvm::dbgs()
               << "Input, including padding, is narrower than the filter\n");
    return failure();
  }

  return success();
}

LogicalResult Conv2dGenerator::hasValidChip() const {
  // We support in between gfx900 to gfx908 and gfx1030 for nonxdlops algorithm
  // For example, gfx803, gfx90c are unsupported now
  unsigned int chipHexNumber = 0;
  if (sscanf(config.chip.c_str(), "gfx%x", &chipHexNumber) != 1)
    return failure();

  constexpr size_t NUM_SUPPORTED_CHIPS = 5;
  static const unsigned int supportedChips[NUM_SUPPORTED_CHIPS] = {
      0x900, 0x906, 0x908, 0x90a, 0x1030};
  const unsigned int *ptr;
  ptr = std::find(supportedChips, supportedChips + NUM_SUPPORTED_CHIPS,
                  chipHexNumber);
  if (ptr == supportedChips + NUM_SUPPORTED_CHIPS)
    return failure();

  // XDLOPS are only supported on MI-100 (gfx908) and MI-200 (gfx90a)
  if (bitEnumContainsAll(config.features, GemmFeatures::mfma) &&
      (chipHexNumber != 0x908 && chipHexNumber != 0x90a))
    return failure();
  return success();
}

int Conv2dGenerator::getKernelCount(OpBuilder &builder) const {
  if (config.kernelId > 0) { // generate only 1 specified kernel
    return 1;
  }
  assert(config.operation.has_value());
  switch (config.operation.value()) {
  case ConvOpType::BwdData:
    return getBwdDataKernelCount();
  case ConvOpType::Fwd:
    return 1;
  case ConvOpType::BwdWeight:
    return getBwdWeightKernelCount(builder);
  }
  llvm_unreachable("Invalid conv2d operation");
}

int Conv2dGenerator::getBwdWeightKernelCount(OpBuilder &builder) const {
  assert(config.operation.value() == ConvOpType::BwdWeight);

  if (bitEnumContainsAll(config.features, GemmFeatures::mfma)) {
    Type dataType = getDataType(builder);
    if (!needExtraPadBwdWeight(builder)) {
      if (dataType == builder.getF32Type()) {
        // For the following case, use 2 kernels:
        // - backward weight
        // - XDLOPS
        // - fp32
        // - No need extra pad along Gemm M/N/K
        // The first kernel will 0-initialize the output (filter tensor).
        // The second kernel will conduct the actual backward weight
        // convolution, using atomic add instructions.
        return 2;
      } else if (dataType == builder.getF16Type()) {
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
        return 3;
      }
    }
  }
  return 1;
}

int Conv2dGenerator::getBwdDataKernelCount() const {
  llvm::SmallVector<int64_t> gemmIds = populateBackwardDataGemmIds(
      config.strideHeight, config.strideWidth, config.dilationHeight,
      config.dilationWidth, config.filterHeight, config.filterWidth);
  return static_cast<int>(gemmIds.size());
}

Type Conv2dGenerator::getDataType(OpBuilder &builder) const {
  Type dataType;
  if (config.dataTypeStr == "f32" || config.dataTypeStr == "fp32") {
    dataType = builder.getF32Type();
  } else if (config.dataTypeStr == "f16" || config.dataTypeStr == "fp16") {
    dataType = builder.getF16Type();
  } else if (config.dataTypeStr == "bf16") {
    dataType = builder.getBF16Type();
  } else if (config.dataTypeStr == "i8") {
    dataType = builder.getI8Type();
  }
  return dataType;
}

// The function is used to compute extra padding sizes.
// For example, if gemmM size is 3 and gemmMPerBlock is 64,
// we set gemmMExtra be 64 so (gemmM+gemmMExtra)%gemmMPerBlock=0.
//
// If padding is needed, returns a GemmSize containing the number of elements
// needed to pad the M, N, and K dimensions (**not** the new gemm size).
// Otherwise, returns None
template <typename T>
static Optional<GemmSize>
calculatePaddingKernelSize(GemmSize gemmSize, ConvOpType dir, Type dataType,
                           T populateParams) {
  bool needExtraPad = false;
  int64_t gemmMExtra, gemmNExtra, gemmKExtra;
  gemmMExtra = gemmNExtra = gemmKExtra = 0;

  auto configParams = populateParams.getTuningParameters(dir, dataType);
  size_t numOfFailedConfigs = 0;
  for (auto &params : configParams) {
    if (gemmSize.m % params.gemmMPerBlock == 0 &&
        gemmSize.k % params.gemmKPerBlock == 0 &&
        gemmSize.n % params.gemmNPerBlock == 0) {
      break;
    }
    numOfFailedConfigs++;
  }

  auto extraParams = populateParams.getUniversalParameters();
  if (numOfFailedConfigs == configParams.size()) {
    needExtraPad = true;
    int64_t gemmMRemain, gemmKRemain, gemmNRemain;

    gemmMRemain = gemmSize.m % extraParams.gemmMPerBlock;
    if (gemmMRemain != 0)
      gemmMExtra = extraParams.gemmMPerBlock - gemmMRemain;

    gemmNRemain = gemmSize.n % extraParams.gemmNPerBlock;
    if (gemmNRemain != 0)
      gemmNExtra = extraParams.gemmNPerBlock - gemmNRemain;

    gemmKRemain = gemmSize.k % extraParams.gemmKPerBlock;
    if (gemmKRemain != 0)
      gemmKExtra = extraParams.gemmKPerBlock - gemmKRemain;

    // llvm::errs() << "gemmMExtra: " << gemmMExtra << "gemmNExtra: " <<
    // gemmNExtra << "gemmKExtra: " << gemmKExtra << "\n";
  }

  if (needExtraPad)
    return GemmSize(gemmSize.g, gemmMExtra, gemmKExtra, gemmNExtra);
  return llvm::None;
}

bool Conv2dGenerator::needExtraPadBwdWeight(OpBuilder &builder) const {
  Type dataType = getDataType(builder);
  ConvOpType dir = config.operation.value();
  assert(dir == ConvOpType::BwdWeight &&
         "This method should only be called for wrw ops");

  ConvolutionDims convDims = getConvolutionDims();
  GemmSize gemmSize = GemmSize::fromConvolution(dir, convDims);

  bool needExtraPad = false;
  if (!bitEnumContainsAll(config.features, GemmFeatures::mfma)) {
    PopulateParams populateParams;
    InitParamsNonXDL validParams;
    // If there is a perfConfig present in the configuration
    // we just need to see if padding will be used in that config
    if (validParams.deserialize(config.perfConfig)) {
      needExtraPad =
          calculatePadding(validParams.gemmKPerBlock, validParams.gemmMPerBlock,
                           validParams.gemmNPerBlock, gemmSize)
              .has_value();
    } else {
      // This function will go through the list and pick a valid configuration
      // the following code will return if the resulting config needs padding
      needExtraPad =
          calculatePaddingKernelSize(gemmSize, dir, dataType, populateParams)
              .has_value();
    }
  } else {
    PopulateParamsXDL populateParamsXDL;
    InitParamsXDL validParams;
    // If there is a perfConfig present in the configuration
    // we just need to see if padding will be used in that config
    if (validParams.deserialize(config.perfConfig)) {
      needExtraPad =
          calculatePadding(validParams.gemmKPerBlock, validParams.gemmMPerBlock,
                           validParams.gemmNPerBlock, gemmSize,
                           validParams.gemmKPack)
              .has_value();
    } else {
      // This function will go through the list and pick a valid configuration
      // the following code will return if the resulting config needs padding
      needExtraPad =
          calculatePaddingKernelSize(gemmSize, dir, dataType, populateParamsXDL)
              .has_value();
    }
  }
  return needExtraPad;
}

bool Conv2dGenerator::hasWorkspace(OpBuilder &builder) const {
  // Decide if a workspace is needed.
  // Preconditions:
  // - data type: fp16
  // - operation: backward weight conv2d.
  // - use XDLOPS.
  // - No need to pad along Gemm M/N/K dimension.
  bool result = false;

  if (config.operation.has_value()) {
    Type dataType = getDataType(builder);
    ConvOpType dir = config.operation.value();
    if ((dir == ConvOpType::BwdWeight) &&
        bitEnumContainsAll(config.features, GemmFeatures::mfma) &&
        (dataType == builder.getF16Type())) {
      // In case we need extra padding, do not use workspace.
      result = (needExtraPadBwdWeight(builder) == false);
    }
  }
  return result;
}

int Conv2dGenerator::getWorkspaceSize(ModuleOp &module) const {
  // Currently onlt in the following condition would a workspace is needed.
  // - data type: fp16
  // - operation: backward weight conv2d.
  // - use XDLOPS.
  // - No need to pad along Gemm M/N/K dimension.
  // Workspace size is the same as the filter dimension, with fp32 type.
  int result = 0;
  OpBuilder builder(module.getContext());
  if (hasWorkspace(builder)) {
    result = std::accumulate(config.filterDimension.begin(),
                             config.filterDimension.end(), 1,
                             std::multiplies<int>()) *
             builder.getF32Type().getWidth() / 8;
  }
  return result;
}

LogicalResult Conv2dGenerator::parseConvConfig(const char *arguments) {
  std::map<std::string, std::string> argMap;
  strToTokens(arguments, argMap);

  auto isValid = [&argMap]() {
    // only require tensor configs
    static const std::vector<std::string> validKeys = {
        "batchsize",   "groupsize",    "in_layout", "in_type",
        "in_channels", "in_h",         "in_w",      "out_layout",
        "out_type",    "out_channels", "out_h",     "out_w",
        "fil_layout",  "fil_type",     "fil_w",     "fil_h"};
    if (!std::all_of(validKeys.cbegin(), validKeys.cend(),
                     [&argMap](const std::string &key) {
                       return argMap.count(key) > 0;
                     })) {
      return false;
    }
    static const std::vector<std::string> layoutArgs = {
        "fil_layout", "in_layout", "out_layout"};

    if (!std::all_of(layoutArgs.cbegin(), layoutArgs.cend(),
                     [&argMap](const std::string &key) {
                       return argMap[key].length() == 5;
                     })) {
      return false;
    }

    bool noMixedTypes =
        (argMap["in_type"] == argMap["fil_type"] &&
         argMap["out_type"] == argMap["in_type"]) ||
        (argMap["in_type"] == "i8" && argMap["fil_type"] == "i8" &&
         argMap["out_type"] == "i32");
    return noMixedTypes;
  };

  // Proceed only if we have a valid argMap. Otherwise leave the handle to be
  // empty
  if (!isValid())
    return failure();

  auto strToLong = [&argMap](std::string argKey) {
    return std::stoul(argMap[argKey]);
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
  AmdArchInfo archInfo = lookupArchInfo(splitter.getChip());
  config.features = archInfo.defaultFeatures;

  strToStr("perf_config", config.perfConfig);
  strToInt("num_cu", config.num_cu);
  int hasXdlops = 0;
  strToInt("x2", hasXdlops);
  config.features = bitEnumSet(config.features, GemmFeatures::mfma, hasXdlops);

  // conv settings
  auto const op = getConvOpTypeForName(argMap["operation"]);
  if (!op.has_value()) {
    return failure();
  }

  auto canonicalizeDataType = [](const std::string type) {
    if (type == "fp32")
      return std::string("f32");
    else if (type == "fp16")
      return std::string("f16");
    else
      return type;
  };
  config.operation = op.value();
  strToInt("kernel_id", config.kernelId);
  config.dataTypeStr = canonicalizeDataType(argMap["in_type"]);
  strToInt("dilation_h", config.dilationHeight);
  strToInt("dilation_w", config.dilationWidth);
  strToInt("conv_stride_h", config.strideHeight);
  strToInt("conv_stride_w", config.strideWidth);
  strToInt("padding_h", config.paddingHeightLeft);
  strToInt("padding_h", config.paddingHeightRight);
  strToInt("padding_w", config.paddingWidthLeft);
  strToInt("padding_w", config.paddingWidthRight);

  strToStr("kernel_name", config.kernelBaseName);

  // Allow only fwd direction for int8. Reject other directions.
  if (config.operation.value() != ConvOpType::Fwd &&
      config.dataTypeStr == "i8") {
    return failure();
  }

  // Rock has NCHW as layout string for all three tensors
  config.inputLayout = translateLayout(
      argMap["in_layout"], std::string("NGCHW"), std::string("ngchw"));
  config.filterLayout = translateLayout(
      argMap["fil_layout"], std::string("GNCHW"), std::string("gkcyx"));
  config.outputLayout = translateLayout(
      argMap["out_layout"], std::string("NGCHW"), std::string("ngkhw"));

  // Determine tensor dimensions.
  auto status = parseConvDims(strToLong("batchsize"), strToLong("groupsize"),
                              strToLong("in_channels"), strToLong("in_h"),
                              strToLong("in_w"), strToLong("out_channels"),
                              strToLong("out_h"), strToLong("out_w"),
                              strToLong("fil_w"), strToLong("fil_h"));

  if (status.failed()) {
    return failure();
  }

  return success();
}

LogicalResult
Conv2dGenerator::parseConvDims(int64_t batchSize, int64_t groupSize,
                               int64_t inputChannel, int64_t inputHeight,
                               int64_t inputWidth, int64_t outputChannel,
                               int64_t outputHeight, int64_t outputWidth,
                               int64_t filterHeight, int64_t filterWidth) {
  config.filterHeight = filterHeight;
  config.filterWidth = filterWidth;
  static const std::string filterKeys = "kgcyx";
  int64_t filterVals[] = {outputChannel / groupSize, groupSize,
                          inputChannel / groupSize, filterHeight, filterWidth};

  static const std::string inputKeys = "ngchw";
  int64_t inputVals[] = {batchSize, groupSize, inputChannel / groupSize,
                         inputHeight, inputWidth};

  static const std::string outputKeys = "ngkhw";
  int64_t outputVals[] = {batchSize, groupSize, outputChannel / groupSize,
                          outputHeight, outputWidth};

  auto convertLayout = [](char &key, const std::string &kmap, int64_t vals[],
                          auto &dims) {
    auto keyl = std::tolower(key);
    auto ii = kmap.find(keyl);
    if (ii == std::string::npos) {
      static std::string nchw = "ngchw";
      ii = nchw.find(keyl);
      if (ii == std::string::npos)
        return false;
    }
    dims.push_back(vals[ii]);
    key = kmap[ii];
    return true;
  };

  size_t layoutLen = config.filterLayout.length();
  if (layoutLen != config.inputLayout.length() ||
      layoutLen != config.outputLayout.length()) {
    return failure();
  }
  // Determine dimensions.
  for (size_t i = 0; i < layoutLen; ++i) {
    if (!convertLayout(config.filterLayout[i], filterKeys, filterVals,
                       config.filterDimension) ||
        !convertLayout(config.inputLayout[i], inputKeys, inputVals,
                       config.inputDimension) ||
        !convertLayout(config.outputLayout[i], outputKeys, outputVals,
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

void Conv2dGenerator::setKernelName(const std::string &newName) {
  config.kernelBaseName = newName;
}

void Conv2dGenerator::setDataType(std::string newType) {
  config.dataTypeStr = newType;
}

void Conv2dGenerator::flipXdlops() {
  config.features = config.features ^ GemmFeatures::mfma;
}

ConvolutionDims Conv2dGenerator::getConvolutionDims() const {
  auto inDim = canonicalizeDims(config.inputDimension, config.inputLayout);
  auto filDim = canonicalizeDims(config.filterDimension, config.filterLayout);
  auto outDim = canonicalizeDims(config.outputDimension, config.outputLayout);
  return ConvolutionDims(filDim["y"], filDim["x"], outDim["h"], outDim["w"],
                         inDim["h"], inDim["w"], filDim["k"], filDim["c"],
                         inDim["n"], inDim["g"]);
}

LogicalResult Conv2dGenerator::genConvModule(ModuleOp &module, int kernel_id,
                                             bool is_verifier,
                                             bool ignoreTuning) {
  OpBuilder builder(module.getContext());

  Type dataType = getDataType(builder);
  if (!dataType) {
    return failure();
  }

  Type outputDataType = dataType;
  if (dataType.isInteger(8)) {
    outputDataType = builder.getIntegerType(32);
  }
  // Construct a new FuncOp.
  auto filterArgType =
      MemRefType::get(ArrayRef<int64_t>(config.filterDimension.begin(),
                                        config.filterDimension.end()),
                      dataType);
  auto inputArgType =
      MemRefType::get(ArrayRef<int64_t>(config.inputDimension.begin(),
                                        config.inputDimension.end()),
                      dataType);
  auto outputArgType =
      MemRefType::get(ArrayRef<int64_t>(config.outputDimension.begin(),
                                        config.outputDimension.end()),
                      outputDataType);

  bool hasWorkspace = this->hasWorkspace(builder);
  Type workspaceArgType;
  if (hasWorkspace) {
    workspaceArgType =
        MemRefType::get(ArrayRef<int64_t>(config.filterDimension.begin(),
                                          config.filterDimension.end()),
                        builder.getF32Type());
  }

  SmallVector<Type, 3> funcArgTypes = {filterArgType, inputArgType,
                                       outputArgType};
  if (hasWorkspace) {
    funcArgTypes = {filterArgType, inputArgType, outputArgType,
                    workspaceArgType};
  }
  auto funcType = builder.getFunctionType(funcArgTypes, {});

  std::string kernelName = config.kernelBaseName;
  if (is_verifier) {
    kernelName += "_ver";
  }

  func::FuncOp func = module.lookupSymbol<func::FuncOp>(kernelName);
  if (func) {
    assert(func.isDeclaration());
    func.erase();
  }

  // Fix kernel_id in case it is less than 0.
  // The only case this could happen is to query the number of kernels needed
  // from MIIR API, where the kernel_id is not yet unknown.
  if (kernel_id < 0)
    kernel_id = 0;

  // Annotate kernel attribute to the FuncOp.
  NamedAttribute archAttr =
      builder.getNamedAttr("kernel.arch", builder.getStringAttr(config.arch));
  SmallVector<NamedAttribute, 2> kernelAttrs = {
      builder.getNamedAttr("kernel", builder.getI32IntegerAttr(kernel_id)),
      archAttr};

  // Construct the FuncOp.
  func = func::FuncOp::create(builder.getUnknownLoc(), kernelName, funcType,
                              ArrayRef<NamedAttribute>(kernelAttrs));
  module.push_back(func);
  if (!is_verifier)
    module->setAttr(archAttr.getName(), archAttr.getValue());
  if (func.getName() != kernelName) {
    return failure();
  }
  kernelFunc = func;

  // Construct a new Block.
  Block *block = func.addEntryBlock();

  // Construct a new Conv2DOp.
  SmallVector<StringAttr, 5> filterLayoutSpec;
  SmallVector<StringAttr, 5> inputLayoutSpec;
  SmallVector<StringAttr, 5> outputLayoutSpec;
  for (size_t i = 0; i < 5; ++i) {
    filterLayoutSpec.push_back(
        builder.getStringAttr(StringRef(&config.filterLayout[i], 1)));
    inputLayoutSpec.push_back(builder.getStringAttr(
        (StringRef(&config.inputLayout[i], 1) + "i").str()));
    outputLayoutSpec.push_back(builder.getStringAttr(
        (StringRef(&config.outputLayout[i], 1) + "o").str()));
  }

  // Set gemm ID be the same as kernel ID.
  // For backward data convolution, additional processing is needed below.
  int64_t gemmId = kernel_id;

  // Obtain gemm ID from kernel_id for backward data convolution.
  if (config.operation.value() == ConvOpType::BwdData) {
    llvm::SmallVector<int64_t> gemmIds = populateBackwardDataGemmIds(
        config.strideHeight, config.strideWidth, config.dilationHeight,
        config.dilationWidth, config.filterHeight, config.filterWidth);
    assert(gemmIds.size() > static_cast<size_t>(kernel_id));
    gemmId = gemmIds[kernel_id];
  }

  std::vector<NamedAttribute> attributes{
      builder.getNamedAttr("gemm_id", builder.getI32IntegerAttr(gemmId)),
      builder.getNamedAttr("arch", builder.getStringAttr(config.chip)),

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

      builder.getNamedAttr("dilations",
                           builder.getArrayAttr({
                               builder.getI32IntegerAttr(config.dilationHeight),
                               builder.getI32IntegerAttr(config.dilationWidth),
                           })),
      builder.getNamedAttr("strides",
                           builder.getArrayAttr({
                               builder.getI32IntegerAttr(config.strideHeight),
                               builder.getI32IntegerAttr(config.strideWidth),
                           })),
      builder.getNamedAttr(
          "padding", builder.getArrayAttr({
                         builder.getI32IntegerAttr(config.paddingHeightLeft),
                         builder.getI32IntegerAttr(config.paddingHeightRight),
                         builder.getI32IntegerAttr(config.paddingWidthLeft),
                         builder.getI32IntegerAttr(config.paddingWidthRight),
                     })),
  };

  if (config.operation.value() == ConvOpType::BwdWeight) {
    attributes.push_back(builder.getNamedAttr(
        "numCu", builder.getI32IntegerAttr(config.num_cu)));
  }

  // features
  attributes.push_back(builder.getNamedAttr(
      "features", builder.getAttr<GemmFeaturesAttr>(config.features)));

  // perf_config
  if (!ignoreTuning && !config.perfConfig.empty()) {
    attributes.push_back(builder.getNamedAttr(
        "perf_config", builder.getStringAttr(config.perfConfig)));
  }

  SmallVector<Value, 3> args = {func.getArgument(0), func.getArgument(1),
                                func.getArgument(2)};
  if (hasWorkspace) {
    args = {func.getArgument(0), func.getArgument(1), func.getArgument(2),
            func.getArgument(3)};
  }
  switch (config.operation.value()) {
  case ConvOpType::Fwd: {
    auto convOp = builder.create<Conv2DOp>(builder.getUnknownLoc(),
                                           ArrayRef<Type>{}, args, attributes);
    block->push_front(convOp);
  } break;
  case ConvOpType::BwdData: {
    auto convOp = builder.create<Conv2DBwdDataOp>(
        builder.getUnknownLoc(), ArrayRef<Type>{}, args, attributes);
    block->push_front(convOp);
  } break;
  case ConvOpType::BwdWeight: {
    auto convOp = builder.create<Conv2DBwdWeightOp>(
        builder.getUnknownLoc(), ArrayRef<Type>{}, args, attributes);
    block->push_back(convOp);
  } break;
  }

  auto returnOp =
      builder.create<func::ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  block->push_back(returnOp);

  return success();
}

func::FuncOp Conv2dGenerator::getKernelFunc() const { return kernelFunc; }
