#include "mlir/Dialect/MIOpen/Generator/Conv2dGenerator.h"
#include "mlir/Dialect/MIOpen/utility/math.hpp"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

Conv2dGenerator::Conv2dGenerator(
    const std::string &arch, int num_cu, bool xdlops,
    const std::string &operation, const std::string &dataTypeStr,
    int dilationHeight, int dilationWidth, int strideHeight, int strideWidth,
    int paddingHeightLeft, int paddingHeightRight, int paddingWidthLeft,
    int paddingWidthRight, const std::string &filterLayout,
    const std::string &inputLayout, const std::string &outputLayout,
    const std::string &kernelName)
    : config{arch,
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
             kernelName,
             -1} {}

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

} // namespace

llvm::SmallVector<int> Conv2dGenerator::getKernelCount() const {
  llvm::SmallVector<int> gemmIds;
  if (config.kernelId > 0) { // generate only 1 specified kernel
    gemmIds.push_back(0);
  } else if (config.operation == "conv2d") {
    gemmIds.push_back(0);
  } else if (config.operation == "conv2d_bwd_data") {
    gemmIds = getBwdDataKernelCount();
  } else if (config.operation == "conv2d_bwd_weight") {
    gemmIds.push_back(0);
  } else if (config.operation == "conv2d_dummy") {
    gemmIds.push_back(0);
  }
  return gemmIds;
}
// calculate gemmId of gemm that can be runned. In some case, the gemmIds may
// not be consecutive
llvm::SmallVector<int> Conv2dGenerator::getBwdDataKernelCount() const {
  auto gcdStrideDilationH =
      math::gcd(config.strideHeight, config.dilationHeight);
  auto gcdStrideDilationW = math::gcd(config.strideWidth, config.dilationWidth);

  auto yTilda = config.strideHeight / gcdStrideDilationH;
  auto xTilda = config.strideWidth / gcdStrideDilationW;

  auto y = config.filterHeight;
  auto x = config.filterWidth;
  llvm::SmallVector<int> gemmIds;
  for (int gemmId = 0; gemmId < yTilda * xTilda; gemmId++) {
    // gemm_k size is different for each GEMM
    const auto iYTilda = gemmId / xTilda;
    const auto iXTilda = gemmId % xTilda;

    auto yDotSlice = math::integer_divide_ceil(y - iYTilda, yTilda);
    auto xDotSlice = math::integer_divide_ceil(x - iXTilda, xTilda);
    // gemmK must > 0, otherwise not need to run
    if (yDotSlice * xDotSlice > 0)
      gemmIds.push_back(gemmId);
  }

  return gemmIds;
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
    return std::all_of(
        validKeys.cbegin(), validKeys.cend(),
        [&argMap](const std::string &key) { return argMap.count(key) > 0; });
  };

  // Proceed only if we have a valid argMap. Otherwise leave the handle to be
  // empty
  if (isValid()) {

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

    // arch settings
    strToStr("arch", config.arch);
    strToInt("num_cu", config.num_cu);
    strToInt("x2", config.xdlops);

    // conv settings
    config.operation = argMap["operation"];
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

    strToStr("kernel_name", config.kernelName);

    // MIOpen has NCHW as layout string for all three tensors
    config.inputLayout = translateLayout(
        argMap["in_layout"], std::string("NGCHW"), std::string("ngchw"));
    config.filterLayout = translateLayout(
        argMap["fil_layout"], std::string("GNCHW"), std::string("gkcyx"));
    config.outputLayout = translateLayout(
        argMap["out_layout"], std::string("NGCHW"), std::string("ngkhw"));

    // Determine tensor dimensions.
    return parseConvDims(strToLong("batchsize"), strToLong("groupsize"),
                         strToLong("in_channels"), strToLong("in_h"),
                         strToLong("in_w"), strToLong("out_channels"),
                         strToLong("out_h"), strToLong("out_w"),
                         strToLong("fil_w"), strToLong("fil_h"));
  }

  return failure();
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

  // Determine dimensions.
  for (size_t i = 0; i < 5; ++i) {
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
  if (config.kernelName.empty()) {
    int id = std::max(config.kernelId, 0);
    config.kernelName = "miopen_" + config.operation + "_" +
                        config.filterLayout + "_" + config.inputLayout + "_" +
                        config.outputLayout + "_" + std::to_string(id);
  }

  return success();
}

void Conv2dGenerator::setKernelName(std::string newName) {
  config.kernelName = newName;
}

LogicalResult Conv2dGenerator::genConvModule(ModuleOp &module,
                                             OpBuilder &builder,
                                             int kernel_id) {
  if (kernel_id == -1) {
    kernel_id = config.kernelId;
  }

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

  std::string kernelName = config.kernelName;

  // Annotate kernel attribute to the FuncOp.
  SmallVector<NamedAttribute, 1> kernelAttrs{
      builder.getNamedAttr("kernel", builder.getI32IntegerAttr(kernel_id)),
  };

  // Construct the FuncOp.
  auto func = FuncOp::create(builder.getUnknownLoc(), kernelName, funcType,
                             ArrayRef<NamedAttribute>(kernelAttrs));
  module.push_back(func);

  if (func.getName() != kernelName) {
    return failure();
  }

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
      builder.getNamedAttr("arch", builder.getStringAttr(config.arch)),
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

  if (config.operation == "conv2d") {
    auto convOp = builder.create<miopen::Conv2DOp>(
        builder.getUnknownLoc(), ArrayRef<mlir::Type>{},
        ValueRange{func.getArgument(0), func.getArgument(1),
                   func.getArgument(2)},
        attributes);
    block->push_front(convOp);
  } else if (config.operation == "conv2d_bwd_data") {
    auto convOp = builder.create<miopen::Conv2DBwdDataOp>(
        builder.getUnknownLoc(), ArrayRef<mlir::Type>{},
        ValueRange{func.getArgument(0), func.getArgument(1),
                   func.getArgument(2)},
        attributes);
    block->push_front(convOp);
  } else if (config.operation == "conv2d_bwd_weight") {
    auto convOp = builder.create<miopen::Conv2DBwdWeightOp>(
        builder.getUnknownLoc(), ArrayRef<mlir::Type>{},
        ValueRange{func.getArgument(0), func.getArgument(1),
                   func.getArgument(2)},
        attributes);
    block->push_back(convOp);
  } else if (config.operation == "conv2d_dummy") {
    auto convOp = builder.create<miopen::Conv2DDummyOp>(
        builder.getUnknownLoc(), ArrayRef<mlir::Type>{},
        ValueRange{func.getArgument(0), func.getArgument(1),
                   func.getArgument(2)},
        attributes);
    block->push_front(convOp);
  }

  auto returnOp =
      builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  block->push_back(returnOp);

  return success();
}
