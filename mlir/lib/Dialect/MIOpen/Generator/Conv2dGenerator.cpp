#include "mlir/Dialect/MIOpen/Generator/Conv2dGenerator.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;


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

LogicalResult Conv2dGenerator::parseConvConfig(const char *arguments) {
  std::map<std::string, std::string> argMap;
  strToTokens(arguments, argMap);

  auto isValid = [&argMap]() {
    // only require tensor configs
    static const std::vector<std::string> validKeys = {
        "batchsize", "in_layout",  "in_type",  "in_channels",  "in_h",
        "in_w",      "out_layout", "out_type", "out_channels", "out_h",
        "out_w",     "fil_layout", "fil_type", "fil_w",        "fil_h"};
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
    strToStr("arch", arch);
    strToInt("num_cu", num_cu);
    strToInt("x2", xdlops);

    // conv settings
    operation = argMap["operation"];
    dataTypeStr = argMap["out_type"];
    strToInt("dilation_h", dilationHeight);
    strToInt("dilation_w", dilationWidth);
    strToInt("conv_stride_h", strideHeight);
    strToInt("conv_stride_w", strideWidth);
    strToInt("padding_h", paddingHeight);
    strToInt("padding_w", paddingWidth);

    strToStr("kernel_name", kernelName);

    // MIOpen has NCHW as layout string for all three tensors
    strToStr("fil_layout", filterLayout);
    strToStr("in_layout", inputLayout);
    strToStr("out_layout", outputLayout);

    // Determine tensor dimensions.
    return parseConvDims(
        strToLong("batchsize"), strToLong("in_channels"), strToLong("in_h"),
        strToLong("in_w"), strToLong("out_channels"), strToLong("out_h"),
        strToLong("out_w"), strToLong("fil_w"), strToLong("fil_h"));
  }

  return failure();
}

LogicalResult Conv2dGenerator::parseConvDims(
    int64_t batchSize, int64_t inputChannel, int64_t inputHeight,
    int64_t inputWidth, int64_t outputChannel, int64_t outputHeight,
    int64_t outputWidth, int64_t filterHeight, int64_t filterWidth) {

  static const std::string filterKeys = "kcyx";
  int64_t filterVals[] = {outputChannel, inputChannel, filterHeight,
                          filterWidth};

  static const std::string inputKeys = "nchw";
  int64_t inputVals[] = {batchSize, inputChannel, inputHeight, inputWidth};

  static const std::string outputKeys = "nkhw";
  int64_t outputVals[] = {batchSize, outputChannel, outputHeight, outputWidth};

  auto convertLayout = [](char key, const std::string &kmap, int64_t vals[],
                          auto &dims) {
    auto keyl = std::tolower(key);
    auto ii = kmap.find(keyl);
    if (ii == std::string::npos) {
      static std::string nchw = "nchw";
      ii = nchw.find(keyl);
      if (ii == std::string::npos)
        return '\0';
    }
    dims.push_back(vals[ii]);
    return kmap[ii];
  };

  // Determine dimensions.
  for (size_t i = 0; i < 4; ++i) {
    filterLayout[i] =
        convertLayout(filterLayout[i], filterKeys, filterVals, filterDimension);
    inputLayout[i] =
        convertLayout(inputLayout[i], inputKeys, inputVals, inputDimension);
    outputLayout[i] =
        convertLayout(outputLayout[i], outputKeys, outputVals, outputDimension);
  }

  return success();
}

LogicalResult Conv2dGenerator::genConvModule(ModuleOp &module,
                                             OpBuilder &builder) {

  mlir::Type dataType;
  if (dataTypeStr == "f32" || dataTypeStr == "fp32") {
    dataType = builder.getF32Type();
  } else if (dataTypeStr == "f16" || dataTypeStr == "fp16") {
    dataType = builder.getF16Type();
  } else if (dataTypeStr == "bf16") {
    dataType = builder.getIntegerType(16);
  } else {
    return failure();
  }

  // Construct a new FuncOp.
  auto filterArgType = MemRefType::get(
      ArrayRef<int64_t>(filterDimension.begin(), filterDimension.end()),
      dataType);
  auto inputArgType = MemRefType::get(
      ArrayRef<int64_t>(inputDimension.begin(), inputDimension.end()),
      dataType);
  auto outputArgType = MemRefType::get(
      ArrayRef<int64_t>(outputDimension.begin(), outputDimension.end()),
      dataType);
  auto funcType =
      builder.getFunctionType({filterArgType, inputArgType, outputArgType}, {});

  // Determine kernel name, if there isn't one.
  if (kernelName.size() == 0) {
    kernelName = "miopen_" + operation + "_" + filterLayout + "_" +
                 inputLayout + "_" + outputLayout;
  }

  // Annotate kernel attribute to the FuncOp.
  SmallVector<NamedAttribute, 1> kernelAttrs{
      builder.getNamedAttr("kernel", builder.getUnitAttr()),
  };

  // Construct the FuncOp.
  auto func = FuncOp::create(builder.getUnknownLoc(), kernelName, funcType,
                             ArrayRef<NamedAttribute>(kernelAttrs));
  module.push_back(func);

  // Construct a new Block.
  Block *block = func.addEntryBlock();

  // Construct a new Conv2DOp.
  SmallVector<StringAttr, 5> filterLayoutSpec;
  SmallVector<StringAttr, 5> inputLayoutSpec;
  SmallVector<StringAttr, 5> outputLayoutSpec;
  for (size_t i = 0; i < 5; ++i) {
    filterLayoutSpec.push_back(
        builder.getStringAttr(StringRef(&filterLayout[i], 1)));
    inputLayoutSpec.push_back(
        builder.getStringAttr((StringRef(&inputLayout[i], 1) + "i").str()));
    outputLayoutSpec.push_back(
        builder.getStringAttr((StringRef(&outputLayout[i], 1) + "o").str()));
  }

  std::vector<NamedAttribute> attributes{
      builder.getNamedAttr("arch", builder.getStringAttr(arch)),
      builder.getNamedAttr("num_cu", builder.getI32IntegerAttr(num_cu)),

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
                               builder.getI32IntegerAttr(dilationHeight),
                               builder.getI32IntegerAttr(dilationWidth),
                           })),
      builder.getNamedAttr("strides",
                           builder.getArrayAttr({
                               builder.getI32IntegerAttr(strideHeight),
                               builder.getI32IntegerAttr(strideWidth),
                           })),
      builder.getNamedAttr("padding",
                           builder.getArrayAttr({
                               builder.getI32IntegerAttr(paddingHeight),
                               builder.getI32IntegerAttr(paddingWidth),
                           })),
  };

  // xdlops v2.
  if (xdlops)
    attributes.push_back(
        builder.getNamedAttr("xdlopsV2", builder.getBoolAttr(true)));

  if (operation == "conv2d") {
    auto convOp = builder.create<miopen::Conv2DOp>(
        builder.getUnknownLoc(), ArrayRef<mlir::Type>{},
        ValueRange{func.getArgument(0), func.getArgument(1),
                   func.getArgument(2)},
        attributes);
    block->push_front(convOp);
  } else if (operation == "conv2d_bwd_data") {
    auto convOp = builder.create<miopen::Conv2DBwdDataOp>(
        builder.getUnknownLoc(), ArrayRef<mlir::Type>{},
        ValueRange{func.getArgument(0), func.getArgument(1),
                   func.getArgument(2)},
        attributes);
    block->push_front(convOp);
  } else if (operation == "conv2d_bwd_weight") {
    auto convOp = builder.create<miopen::Conv2DBwdWeightOp>(
        builder.getUnknownLoc(), ArrayRef<mlir::Type>{},
        ValueRange{func.getArgument(0), func.getArgument(1),
                   func.getArgument(2)},
        attributes);
    block->push_back(convOp);
  } else if (operation.compare("conv2d_dummy") == 0) {
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
