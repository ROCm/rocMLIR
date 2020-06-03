//===- mlir-miopen-driver.cpp - MLIR MIOpen Dialect Driver ----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-miopen-driver.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<std::string>
    operation("operation",
              cl::desc("Convolution operation, eg: conv2d, conv2d_bwd_data..."),
              cl::value_desc("convolution flavor string"), cl::init("conv2d"));

static cl::opt<std::string> filterLayout("fil_layout", cl::desc("Filter layout"),
                                              cl::value_desc("layout string"),
                                              cl::init("kcyx"));

static cl::opt<std::string> inputLayout("in_layout", cl::desc("Input layout"),
                                              cl::value_desc("layout string"),
                                              cl::init("nchw"));

static cl::opt<std::string> outputLayout("out_layout", cl::desc("Output layout"),
                                              cl::value_desc("layout string"),
                                              cl::init("nkhw"));

// N
static cl::opt<int64_t> batchSize("batchsize", cl::desc("Batch size"),
                                              cl::value_desc("dimension value"),
                                              cl::init(-1));

// C
static cl::opt<int64_t> inputChannel("in_channels", cl::desc("Input channels"),
                                              cl::value_desc("dimension value"),
                                              cl::init(-1));

// Hi
static cl::opt<int64_t> inputHeight("in_h", cl::desc("Input height"),
                                              cl::value_desc("dimension value"),
                                              cl::init(-1));

// Wi
static cl::opt<int64_t> inputWidth("in_w", cl::desc("Input width"),
                                              cl::value_desc("dimension value"),
                                              cl::init(-1));

// K
static cl::opt<int64_t> outputChannel("out_channels", cl::desc("Output channels"),
                                              cl::value_desc("dimension value"),
                                              cl::init(-1));

// Y
static cl::opt<int64_t> filterWidth("fil_w", cl::desc("Filter width"),
                                              cl::value_desc("dimension value"),
                                              cl::init(-1));

// X
static cl::opt<int64_t> filterHeight("fil_h", cl::desc("Filter height"),
                                              cl::value_desc("dimension value"),
                                              cl::init(-1));

// Ho
static cl::opt<int64_t> outputHeight("out_h", cl::desc("Output height"),
                                              cl::value_desc("dimension value"),
                                              cl::init(-1));

// Wo
static cl::opt<int64_t> outputWidth("out_w", cl::desc("Output width"),
                                              cl::value_desc("dimension value"),
                                              cl::init(-1));

// dilation height
static cl::opt<int> dilationHeight("dilation_h", cl::desc("Dilation height"),
                                                     cl::value_desc("attribute value"),
                                                     cl::init(1));

// dilation width
static cl::opt<int> dilationWidth("dilation_w", cl::desc("Dilation width"),
                                                     cl::value_desc("attribute value"),
                                                     cl::init(1));

// stride height
static cl::opt<int> strideHeight("conv_stride_h", cl::desc("Stride height"),
                                                     cl::value_desc("attribute value"),
                                                     cl::init(1));

// stride width
static cl::opt<int> strideWidth("conv_stride_w", cl::desc("Stride width"),
                                                     cl::value_desc("attribute value"),
                                                     cl::init(1));

// padding height
static cl::opt<int> paddingHeight("padding_h", cl::desc("Padding height"),
                                                     cl::value_desc("attribute value"),
                                                     cl::init(0));

// padding width
static cl::opt<int> paddingWidth("padding_w", cl::desc("Padding width"),
                                                     cl::value_desc("attribute value"),
                                                     cl::init(0));

// populate default values
static cl::opt<bool> populateDefaultValues("p", cl::desc("To populate default values"),
                                                cl::value_desc("To populate default values"),
                                                cl::init(false));

static LogicalResult populateModule(ModuleOp &module, OpBuilder &builder,
                                    MLIRContext &context) {
  // Populate default parameters if necessary.
  if (populateDefaultValues.getValue() == true) {
    batchSize.setValue(128);
    inputChannel.setValue(8);
    outputChannel.setValue(128);
    inputHeight.setValue(32); 
    inputWidth.setValue(32);
    outputHeight.setValue(30);
    outputWidth.setValue(30);
    filterHeight.setValue(3);
    filterWidth.setValue(3);
    dilationHeight.setValue(1);
    dilationWidth.setValue(1);
    strideHeight.setValue(1);
    strideWidth.setValue(1);
    paddingHeight.setValue(0);
    paddingWidth.setValue(0);
  }

  // Determine dimensions.
  llvm::SmallVector<int64_t, 4> filterDimension;
  llvm::SmallVector<int64_t, 4> inputDimension;
  llvm::SmallVector<int64_t, 4> outputDimension;
  for (size_t i = 0; i < 4; ++i) {
    auto &filterDim = filterLayout.getValue()[i];
    auto &inputDim = inputLayout.getValue()[i];
    auto &outputDim = outputLayout.getValue()[i];

    if (filterDim == 'k') {
      filterDimension.push_back(outputChannel.getValue());
    } else if (filterDim == 'c') {
      filterDimension.push_back(inputChannel.getValue());
    } else if (filterDim == 'y') {
      filterDimension.push_back(filterWidth.getValue());
    } else if (filterDim == 'x') {
      filterDimension.push_back(filterHeight.getValue());
    }

    if (inputDim == 'n') {
      inputDimension.push_back(batchSize.getValue());
    } else if (inputDim == 'c') {
      inputDimension.push_back(inputChannel.getValue());
    } else if (inputDim == 'h') {
      inputDimension.push_back(inputWidth.getValue());
    } else if (inputDim == 'w') {
      inputDimension.push_back(inputHeight.getValue());
    }

    if (outputDim == 'n') {
      outputDimension.push_back(batchSize.getValue());
    } else if (outputDim == 'k') {
      outputDimension.push_back(outputChannel.getValue());
    } else if (outputDim == 'h') {
      outputDimension.push_back(outputWidth.getValue());
    } else if (outputDim == 'w') {
      outputDimension.push_back(outputHeight.getValue());
    }
  }

  // Construct a new FuncOp.
  auto filterArgType = MemRefType::get(ArrayRef<int64_t>(filterDimension.begin(), filterDimension.end()), builder.getF32Type());
  auto inputArgType = MemRefType::get(ArrayRef<int64_t>(inputDimension.begin(), inputDimension.end()), builder.getF32Type());
  auto outputArgType = MemRefType::get(ArrayRef<int64_t>(outputDimension.begin(), outputDimension.end()), builder.getF32Type());
  auto funcType = builder.getFunctionType({filterArgType, inputArgType, outputArgType}, {});
  auto func =
      FuncOp::create(builder.getUnknownLoc(),
                     "miopen_" + operation.getValue() + "_" + filterLayout +
                         "_" + inputLayout + "_" + outputLayout,
                     funcType);
  module.push_back(func);

  // Construct a new Block.
  auto *block = func.addEntryBlock();

  // Construct a new Conv2DOp.
  llvm::SmallVector<StringAttr, 4> filterLayoutSpec;
  llvm::SmallVector<StringAttr, 4> inputLayoutSpec;
  llvm::SmallVector<StringAttr, 4> outputLayoutSpec;
  for (size_t i = 0; i < 4; ++i) {
    filterLayoutSpec.push_back(builder.getStringAttr(StringRef(&filterLayout.getValue()[i], 1)));
    inputLayoutSpec.push_back(builder.getStringAttr((StringRef(&inputLayout.getValue()[i], 1) + "i").str()));
    outputLayoutSpec.push_back(builder.getStringAttr((StringRef(&outputLayout.getValue()[i], 1) + "o").str()));
  }

  std::vector<NamedAttribute> attributes{
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

      builder.getNamedAttr(
          "dilations", builder.getArrayAttr({
                           builder.getI32IntegerAttr(dilationHeight.getValue()),
                           builder.getI32IntegerAttr(dilationWidth.getValue()),
                       })),
      builder.getNamedAttr(
          "strides", builder.getArrayAttr({
                         builder.getI32IntegerAttr(strideHeight.getValue()),
                         builder.getI32IntegerAttr(strideWidth.getValue()),
                     })),
      builder.getNamedAttr(
          "padding", builder.getArrayAttr({
                         builder.getI32IntegerAttr(paddingHeight.getValue()),
                         builder.getI32IntegerAttr(paddingWidth.getValue()),
                     })),
  };

  if (operation.getValue().compare("conv2d") == 0) {
    auto convOp = builder.create<miopen::Conv2DOp>(
        builder.getUnknownLoc(), ArrayRef<mlir::Type>{},
        ValueRange{block->getArgument(0), block->getArgument(1),
                   block->getArgument(2)},
        attributes);
    block->push_back(convOp);
  } else if (operation.getValue().compare("conv2d_bwd_data") == 0) {
    auto convOp = builder.create<miopen::Conv2DBwdDataOp>(
        builder.getUnknownLoc(), ArrayRef<mlir::Type>{},
        ValueRange{block->getArgument(0), block->getArgument(1),
                   block->getArgument(2)},
        attributes);
    block->push_back(convOp);
  }

  // Construct a new ReturnOp.
  auto returnOp = builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  block->push_back(returnOp);

  return success();
}

static LogicalResult runMLIRPasses(ModuleOp &module, mlir::PassPipelineCLParser &passPipeline) {
  PassManager pm(module.getContext());
  applyPassManagerCLOptions(pm);
  if (failed(passPipeline.addToPipeline(pm)))
      return failure();

  return pm.run(module);
}

int main(int argc, char **argv) {
  mlir::registerAllDialects();
  mlir::registerAllPasses();
  InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipeline("", "compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR MIOpen Dialect driver\n");

  // Construct a new ModuleOp.
  MLIRContext context;
  OpBuilder builder(&context);
  ModuleOp module = ModuleOp::create(builder.getUnknownLoc());

  // Populate the module.
  if (failed(populateModule(module, builder, context))) {
    llvm::errs() << "Module population failed\n";
    exit(1);
  }

  // Apply passes.
  if (failed(runMLIRPasses(module, passPipeline))) {
    llvm::errs() << "Lowering failed\n";
    exit(1);
  }

  // Set up the output file.
  std::string errorMessage;
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  module.print(output->os());
  output->keep();
  return 0;
}
