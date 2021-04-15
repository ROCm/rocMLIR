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

LogicalResult Conv2dGenerator::parseConvDims(
    std::string &inputLayout, std::string &outputLayout,
    std::string &filterLayout, int64_t groupSize, int64_t batchSize,
    int64_t inputChannel, int64_t inputHeight, int64_t inputWidth,
    int64_t outputChannel, int64_t outputHeight, int64_t outputWidth,
    int64_t filterHeight, int64_t filterWidth,
    SmallVector<int64_t, 5> &filterDimension,
    SmallVector<int64_t, 5> &inputDimension,
    SmallVector<int64_t, 5> &outputDimension) {
  // Determine dimensions.
  for (size_t i = 0; i < 5; ++i) {
    auto &filterDim = filterLayout[i];
    auto &inputDim = inputLayout[i];
    auto &outputDim = outputLayout[i];

    if (filterDim == 'k') {
      filterDimension.push_back(outputChannel / groupSize);
    } else if (filterDim == 'c') {
      filterDimension.push_back(inputChannel / groupSize);
    } else if (filterDim == 'y') {
      filterDimension.push_back(filterHeight);
    } else if (filterDim == 'x') {
      filterDimension.push_back(filterWidth);
    } else if (filterDim == 'g') {
      filterDimension.push_back(groupSize);
    }

    if (inputDim == 'n') {
      inputDimension.push_back(batchSize);
    } else if (inputDim == 'c') {
      inputDimension.push_back(inputChannel / groupSize);
    } else if (inputDim == 'h') {
      inputDimension.push_back(inputHeight);
    } else if (inputDim == 'w') {
      inputDimension.push_back(inputWidth);
    } else if (inputDim == 'g') {
      inputDimension.push_back(groupSize);
    }

    if (outputDim == 'n') {
      outputDimension.push_back(batchSize);
    } else if (outputDim == 'k') {
      outputDimension.push_back(outputChannel / groupSize);
    } else if (outputDim == 'h') {
      outputDimension.push_back(outputHeight);
    } else if (outputDim == 'w') {
      outputDimension.push_back(outputWidth);
    } else if (outputDim == 'g') {
      outputDimension.push_back(groupSize);
    }
  }

  return success();
}

LogicalResult Conv2dGenerator::genConvModule(
    std::string &arch, int num_cu, std::string &operation,
    std::string &inputLayout, std::string &outputLayout,
    std::string &filterLayout, const SmallVector<int64_t, 5> &filterDimension,
    const SmallVector<int64_t, 5> &inputDimension,
    const SmallVector<int64_t, 5> &outputDimension, int dilationHeight,
    int dilationWidth, int strideHeight, int strideWidth, int paddingHeight,
    int paddingWidth, ModuleOp &module, OpBuilder &builder,
    std::string &kernelName, mlir::Type dataType, bool xdlops) {

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

  auto func = FuncOp::create(builder.getUnknownLoc(), kernelName, funcType);
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

  if (operation.compare("conv2d") == 0) {
    auto convOp = builder.create<miopen::Conv2DOp>(
        builder.getUnknownLoc(), ArrayRef<mlir::Type>{},
        ValueRange{func.getArgument(0), func.getArgument(1),
                   func.getArgument(2)},
        attributes);
    block->push_front(convOp);
  } else if (operation.compare("conv2d_bwd_data") == 0) {
    auto convOp = builder.create<miopen::Conv2DBwdDataOp>(
        builder.getUnknownLoc(), ArrayRef<mlir::Type>{},
        ValueRange{func.getArgument(0), func.getArgument(1),
                   func.getArgument(2)},
        attributes);
    block->push_front(convOp);
  } else if (operation.compare("conv2d_bwd_weight") == 0) {
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
