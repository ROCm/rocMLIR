#include "mlir/Dialect/MIOpen/Generator/Conv2dGenerator.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/utility/math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/ROCm/IsaNameParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/PassManager.h"
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

Conv2dGenerator::Conv2dGenerator(
    const std::string &chip, const std::string &triple,
    const std::string &features, const std::string &perfConfig, int num_cu,
    bool xdlops, const miopen::ConvOpType operation,
    const std::string &dataTypeStr, int dilationHeight, int dilationWidth,
    int strideHeight, int strideWidth, int paddingHeightLeft,
    int paddingHeightRight, int paddingWidthLeft, int paddingWidthRight,
    const std::string &filterLayout, const std::string &inputLayout,
    const std::string &outputLayout, const std::string &kernelBaseName)
    : config{chip,
             triple,
             features,
             perfConfig,
             num_cu,
             xdlops,
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

Conv2dGenerator::Conv2dGenerator(const Conv2dGenerator::Config &_config) : config(_config) {}

namespace {

void strToTokens(const std::string &arguments,
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

llvm::StringMap<int64_t> canonicalizeDims(const ArrayRef<int64_t> dims,
                                          const StringRef layout) {
  llvm::StringMap<int64_t> ret;
  for (const auto &tuple : llvm::zip(layout, dims)) {
    StringRef key(&std::get<0>(tuple), 1);
    ret.insert_or_assign(key, std::get<1>(tuple));
  }
  return ret;
}

LogicalResult hasDimensions(const llvm::StringMap<int64_t> &map,
                            const StringRef wantedLayout,
                            const StringRef operation) {
  for (size_t i = 0; i < wantedLayout.size(); ++i) {
    auto key = wantedLayout.slice(i, i + 1);
    if (map.count(key) == 0) {
      llvm::errs() << "Layout for " << operation
                   << " tensor missing dimension: " << key;
      return failure();
    }
  }
  return success();
}

LogicalResult smallEnough(const ArrayRef<int64_t> dims, size_t elemWidth,
                          StringRef name) {
  int64_t size = std::accumulate(dims.begin(), dims.end(), 1LL,
                                 std::multiplies<int64_t>()) *
                 elemWidth;
  if (size >= (1LL << 31)) { // 2^31 = 2 GB
    llvm::dbgs() << name << " tensor cannot be larger than 2 GB\n";
    return failure();
  }
  return success();
}
} // namespace

LogicalResult Conv2dGenerator::isApplicable() const {
  if (failed(hasValidDimension())) {
    return failure();
  }

  if (failed(hasValidChip())) {
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
    llvm::errs() << "Dilation and stride must be a positive integer\n";
    return failure();
  }

  static const SmallVector<int64_t, 4> nonNegativeParams{
      config.paddingHeightLeft, config.paddingHeightRight,
      config.paddingWidthLeft, config.paddingWidthRight};
  if (std::any_of(nonNegativeParams.begin(), nonNegativeParams.end(),
                  [](const int64_t &a) { return a < 0; })) {
    llvm::errs() << "Padding values cannot be negative\n";
    return failure();
  }

  static const llvm::StringMap<size_t> typeWidths{{"f32", sizeof(float)},
                                                  {"fp32", sizeof(float)},
                                                  {"fp16", 2},
                                                  {"f16", 2},
                                                  {"bf16", sizeof(uint16_t)}};

  auto checkDimSizes = [](const SmallVector<int64_t, 5> &dims) -> bool {
    return std::all_of(dims.begin(), dims.end(),
                       [](const int64_t &a) { return a > 0; });
  };

  if (typeWidths.count(config.dataTypeStr) == 0) {
    llvm::errs() << config.dataTypeStr << " is not a known datatype";
  }
  size_t elementWidth = typeWidths.lookup(config.dataTypeStr);

  if (!checkDimSizes(config.inputDimension)) {
    llvm::errs() << "Input tensor dimensions must be strictly positive\n";
    return failure();
  }
  if (!checkDimSizes(config.filterDimension)) {
    llvm::errs() << "Filter tensoor dimensions must be strictly positive\n";
  }
  if (!checkDimSizes(config.outputDimension)) {
    llvm::errs() << "Output tensor dimensions must be strictly positive\n";
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
    llvm::errs() << "Input and output batch sizes don't match\n";
    return failure();
  }
  if (inDim["g"] != outDim["g"] || inDim["g"] != filDim["g"]) {
    llvm::errs()
        << "Group sizes are not consistent between input, output, and filter\n";
    return failure();
  }
  if (inDim["c"] != filDim["c"]) {
    llvm::errs() << "Number of channels in input doesn't match number of "
                    "channels in filter\n";
    return failure();
  }
  if (filDim["k"] != outDim["k"]) {
    llvm::errs() << "Number of channels in output doesn't match number of "
                    "channels in filter\n";
    return failure();
  }

  int64_t expectedOutHeight = outputDim(
      inDim["h"], filDim["y"], config.paddingHeightLeft,
      config.paddingHeightRight, config.strideHeight, config.dilationHeight);
  int64_t expectedOutWidth = outputDim(
      inDim["w"], filDim["x"], config.paddingWidthLeft,
      config.paddingWidthRight, config.strideWidth, config.dilationWidth);
  if (outDim["h"] != expectedOutHeight) {
    llvm::errs() << "Output height " << outDim["h"] << " doesn't match height "
                 << expectedOutHeight << " computed from other parameters\n";
    return failure();
  }
  if (outDim["w"] != expectedOutWidth) {
    llvm::errs() << "Output width " << outDim["w"] << " doesn't match width "
                 << expectedOutWidth << " computed from other parameters\n";
    return failure();
  }

  if (inDim["h"] + config.paddingHeightLeft + config.paddingHeightRight <
      filDim["y"]) {
    llvm::errs() << "Input, including padding, is shorter than the filter\n";
    return failure();
  }

  if (inDim["w"] + config.paddingWidthLeft + config.paddingWidthRight <
      filDim["x"]) {
    llvm::errs() << "Input, including padding, is narrower than the filter\n";
    return failure();
  }

  if (failed(smallEnough(config.inputDimension, elementWidth, "input")) ||
      failed(smallEnough(config.filterDimension, elementWidth, "filter")) ||
      failed(smallEnough(config.outputDimension, elementWidth, "output"))) {
    return failure();
  }

  return success();
}

LogicalResult Conv2dGenerator::hasValidChip() const {
  // We support xdlops iff it is a gfx908 chip, fail otherwise
  if (config.xdlops && config.chip != "gfx908")
    return failure();

  // We support in between gfx900 to gfx908 for nonxdlops algorithm
  // For example, gfx803, gfx90a and gfx1030 are unsupported now
  unsigned int chipHexNumber = 0;
  if (sscanf(config.chip.c_str(), "gfx%x", &chipHexNumber) != 1)
    return failure();

  if ((chipHexNumber > 0x908) || (chipHexNumber < 0x900))
    return failure();

  return success();
}

int Conv2dGenerator::getKernelCount() const {
  if (config.kernelId > 0) { // generate only 1 specified kernel
    return 1;
  }
  switch (config.operation) {
  case miopen::ConvOpType::BwdData:
    return getBwdDataKernelCount();
  case miopen::ConvOpType::Fwd:
  case miopen::ConvOpType::BwdWeight:
    return 1;
  }
  llvm_unreachable("Invalid conv2d operation");
}

int Conv2dGenerator::getBwdDataKernelCount() const {
  auto gcdStrideDilationH =
      math_util::gcd(config.strideHeight, config.dilationHeight);
  auto gcdStrideDilationW = math_util::gcd(config.strideWidth, config.dilationWidth);

  auto yTilda = config.strideHeight / gcdStrideDilationH;
  auto xTilda = config.strideWidth / gcdStrideDilationW;

  auto y = config.filterHeight;
  auto x = config.filterWidth;
  int count = 0;
  for (int gemmId = 0; gemmId < yTilda * xTilda; gemmId++) {
    // gemm_k size is different for each GEMM
    const auto iYTilda = gemmId / xTilda;
    const auto iXTilda = gemmId % xTilda;

    auto yDotSlice = math_util::integer_divide_ceil(y - iYTilda, yTilda);
    auto xDotSlice = math_util::integer_divide_ceil(x - iXTilda, xTilda);
    // gemmK must > 0, otherwise not need to run
    if (yDotSlice * xDotSlice > 0)
      count++;
  }

  return count;
}
Type Conv2dGenerator::getDataType(OpBuilder &builder) const {
  mlir::Type dataType;
  if (config.dataTypeStr == "f32" || config.dataTypeStr == "fp32") {
    dataType = builder.getF32Type();
  } else if (config.dataTypeStr == "f16" || config.dataTypeStr == "fp16") {
    dataType = builder.getF16Type();
  } else if (config.dataTypeStr == "bf16") {
    dataType = builder.getIntegerType(16);
  }
  return dataType;
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
    if (!std::all_of(
        validKeys.cbegin(), validKeys.cend(),
        [&argMap](const std::string &key) { return argMap.count(key) > 0; })) {
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

    bool noMixedTypes = argMap["in_type"] == argMap["out_type"] && argMap["fil_type"] == argMap["out_type"];
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
  IsaNameParser parser(arch);
  if (failed(
          parser.parseIsaName(config.chip, config.triple, config.features))) {
    return failure();
  }

  strToStr("perf_config", config.perfConfig);
  strToInt("num_cu", config.num_cu);
  strToInt("x2", config.xdlops);

  // conv settings
  auto const op = miopen::getConvOpTypeForName(argMap["operation"]);
  if (!op.hasValue()) {
    return failure();
  }
  config.operation = op.getValue();
  strToInt("kernel_id", config.kernelId);
  config.dataTypeStr = argMap["out_type"];
  strToInt("dilation_h", config.dilationHeight);
  strToInt("dilation_w", config.dilationWidth);
  strToInt("conv_stride_h", config.strideHeight);
  strToInt("conv_stride_w", config.strideWidth);
  strToInt("padding_h", config.paddingHeightLeft);
  strToInt("padding_h", config.paddingHeightRight);
  strToInt("padding_w", config.paddingWidthLeft);
  strToInt("padding_w", config.paddingWidthRight);

  strToStr("kernel_name", config.kernelBaseName);

  // MIOpen has NCHW as layout string for all three tensors
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
    config.kernelBaseName = std::string("miopen_") +
                        miopen::getNameForConvOpType(config.operation) + "_" +
                        config.filterLayout + "_" + config.inputLayout + "_" +
                        config.outputLayout;
  }

  return success();
}

void Conv2dGenerator::setDataType(std::string newType) {
  config.dataTypeStr = newType;
}

void Conv2dGenerator::flipXdlops() { config.xdlops = !config.xdlops; }

LogicalResult Conv2dGenerator::genConvModule(ModuleOp &module,
                                             int kernel_id, bool is_verifier) {
  OpBuilder builder(module.getContext());

  Type dataType = getDataType(builder);
  if (!dataType) {
    return failure();
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
                      dataType);
  auto funcType =
      builder.getFunctionType({filterArgType, inputArgType, outputArgType}, {});

  std::string kernelName = config.kernelBaseName;
  kernelName += "_";
  kernelName += std::to_string(kernel_id);
  if (is_verifier) {
    kernelName += "_ver";
  }

  FuncOp func = module.lookupSymbol<FuncOp>(kernelName);
  if (func) {
    assert(func.isDeclaration());
    func.erase();
  }
  
  // Annotate kernel attribute to the FuncOp.
  SmallVector<NamedAttribute, 1> kernelAttrs{
      builder.getNamedAttr("kernel", builder.getI32IntegerAttr(kernel_id)),
  };

  // Construct the FuncOp.
  func = FuncOp::create(builder.getUnknownLoc(), kernelName, funcType,
                             ArrayRef<NamedAttribute>(kernelAttrs));
  module.push_back(func);
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

  std::vector<NamedAttribute> attributes{
      builder.getNamedAttr("gemm_id", builder.getI32IntegerAttr(kernel_id)),
      builder.getNamedAttr("arch", builder.getStringAttr(config.chip)),
      builder.getNamedAttr("num_cu", builder.getI32IntegerAttr(config.num_cu)),

      builder.getNamedAttr(
          "filter_layout",
          builder.getArrayAttr(ArrayRef<mlir::Attribute>(
              filterLayoutSpec.begin(), filterLayoutSpec.end()))),
      builder.getNamedAttr(
          "input_layout", builder.getArrayAttr(ArrayRef<mlir::Attribute>(
                              inputLayoutSpec.begin(), inputLayoutSpec.end()))),
      builder.getNamedAttr(
          "output_layout",
          builder.getArrayAttr(ArrayRef<mlir::Attribute>(
              outputLayoutSpec.begin(), outputLayoutSpec.end()))),

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

  // xdlops v2.
  if (config.xdlops)
    attributes.push_back(
        builder.getNamedAttr("xdlopsV2", builder.getBoolAttr(true)));

  switch (config.operation) {
  case miopen::ConvOpType::Fwd: {
    auto convOp = builder.create<miopen::Conv2DOp>(
        builder.getUnknownLoc(), ArrayRef<mlir::Type>{},
        ValueRange{func.getArgument(0), func.getArgument(1),
                   func.getArgument(2)},
        attributes);
    block->push_front(convOp);
  } break;
  case miopen::ConvOpType::BwdData: {
    auto convOp = builder.create<miopen::Conv2DBwdDataOp>(
        builder.getUnknownLoc(), ArrayRef<mlir::Type>{},
        ValueRange{func.getArgument(0), func.getArgument(1),
                   func.getArgument(2)},
        attributes);
    block->push_front(convOp);
  } break;
  case miopen::ConvOpType::BwdWeight: {
    auto convOp = builder.create<miopen::Conv2DBwdWeightOp>(
        builder.getUnknownLoc(), ArrayRef<mlir::Type>{},
        ValueRange{func.getArgument(0), func.getArgument(1),
                   func.getArgument(2)},
        attributes);
    block->push_back(convOp);
  } break;
  }

  auto returnOp =
      builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  block->push_back(returnOp);

  return success();
}

FuncOp Conv2dGenerator::getKernelFunc() const {
  return kernelFunc;
}
