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

#include "mlir/Conversion/MIOpenPasses.h"
#include "mlir/Conversion/MIOpenToGPU/MIOpenToGPU.h"
#include "mlir/Dialect/MIOpen/Generator/Conv2dGenerator.h"
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "bf16convert.hpp"
#include "mlir/Dialect/SCF/EDSC/Builders.h"
#include <unordered_map>

using namespace llvm;
using namespace mlir;

static cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                          llvm::cl::desc("<input file>"),
                                          llvm::cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<mlir::miopen::ConvOpType> operation(
    "operation", cl::desc("Convolution operation,"),
    cl::values(clEnumValN(miopen::Conv2DOpType, "conv2d",
                          "Forward convolution"),
               clEnumValN(miopen::Conv2DBwdDataOpType, "conv2d_bwd_data",
                          "Backpropogate convolution data"),
               clEnumValN(miopen::Conv2DBwdWeightOpType, "conv2d_bwd_weight",
                          "Backpropogate convolution weights")),
    cl::value_desc("convolution type"), cl::init(miopen::Conv2DOpType));

static cl::opt<std::string>
    arch("arch",
         cl::desc("amdgpu architecture, eg: gfx803, gfx900, gfx906, gfx908"),
         cl::value_desc("GFX architecture string"), cl::init("gfx906"));

static cl::opt<std::string>
    perfConfig("perf_config",
               cl::desc("performance config data used for tuning"),
               cl::value_desc("Serialized tuning parameters"), cl::init(""));

static cl::opt<int>
    num_cu("num_cu",
           cl::desc("Number of compute units, valid combinations include: "
                    "gfx803(36/64), gfx900(56/64), "
                    "gfx906(60/64), gfx908(120)"),
           cl::value_desc("compute unit value"), cl::init(64));

static cl::opt<std::string> filterLayout("fil_layout",
                                         cl::desc("Filter layout"),
                                         cl::value_desc("layout string"),
                                         cl::init("gkcyx"));

static cl::opt<std::string> inputLayout("in_layout", cl::desc("Input layout"),
                                        cl::value_desc("layout string"),
                                        cl::init("ngchw"));

static cl::opt<std::string> outputLayout("out_layout",
                                         cl::desc("Output layout"),
                                         cl::value_desc("layout string"),
                                         cl::init("ngkhw"));

static cl::opt<int64_t> groupSize("groupsize", cl::desc("Group size"),
                                  cl::value_desc("dimension value"),
                                  cl::init(1));
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
static cl::opt<int64_t> outputChannel("out_channels",
                                      cl::desc("Output channels"),
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
static cl::opt<int64_t>
    outputHeight("out_h", cl::desc("Output height"),
                 cl::value_desc("ouput dimension value, does not need to set."),
                 cl::init(-1));

// Wo
static cl::opt<int64_t>
    outputWidth("out_w", cl::desc("Output width"),
                cl::value_desc("ouput dimension value, does not need to set."),
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
                                cl::value_desc("attribute value"), cl::init(1));

// padding height
static cl::opt<int> paddingHeight("padding_h", cl::desc("Padding height"),
                                  cl::value_desc("attribute value"),
                                  cl::init(0));

static cl::opt<int> paddingHeightLeft("padding_h_l",
                                      cl::desc("Padding height Left"),
                                      cl::value_desc("attribute value"),
                                      cl::init(0));

static cl::opt<int> paddingHeightRight("padding_h_r",
                                       cl::desc("Padding height Right"),
                                       cl::value_desc("attribute value"),
                                       cl::init(0));
// padding width
static cl::opt<int> paddingWidth("padding_w", cl::desc("Padding width"),
                                 cl::value_desc("attribute value"),
                                 cl::init(0));

static cl::opt<int> paddingWidthLeft("padding_w_l",
                                     cl::desc("Padding width Left"),
                                     cl::value_desc("attribute value"),
                                     cl::init(0));

static cl::opt<int> paddingWidthRight("padding_w_r",
                                      cl::desc("Padding width Right"),
                                      cl::value_desc("attribute value"),
                                      cl::init(0));

// conv-config
static cl::opt<std::string> populateConvConfig(
    "conv-config",
    cl::desc("Populate full config settings (overrides all specific settings)"),
    cl::value_desc("config settings"), cl::init(""));

// populate default values
static cl::opt<bool>
    populateDefaultValues("p", cl::desc("To populate default values"),
                          cl::value_desc("To populate default values"),
                          cl::init(false));

// populate entry point
static cl::opt<std::string>
    populateEntryPoint("entry-point", cl::desc("Populate entry point function"),
                       cl::value_desc("Populate entry point function"),
                       cl::init("conv2d"));

// populate entry point function for gpu validation
static cl::opt<std::string> populateValidateEntryPoint(
    "validate-entry-point",
    cl::desc("Populate entry point function for validation"),
    cl::value_desc("Populate entry point function for validation"),
    cl::init("conv2d_1"));
// Set up lowering pipeline.
// The default lowering pipeline compiles down to GPU dialect.
// The output of the pipeline can be piped to mlir-rocm-runner for execution.
//
// When users specify "-c -target=rocdl", compiles down to LLVM dialect.
// The output of the pipeline can be piped to mlir-translate for translation to
// LLVM IR.
static cl::opt<bool> loweringWithBuiltinPipeline(
    "c", cl::desc("Compile with the specified pipeline"),
    cl::value_desc("By default, compiles down to GPU dialect. Set "
                   "-target=rocdl compiles to ROCDL dialect."),
    cl::init(false));

static cl::opt<std::string> loweringTargetDialect(
    "target",
    cl::desc("By default, compiles down to GPU dialect. Set "
             "-target=rocdl compiles to ROCDL dialect."),
    cl::value_desc("Target dialect"), cl::init("gpu"));

// use host harness program.
static cl::opt<bool> useHostHarness(
    "host", cl::desc("To use host harness"),
    cl::value_desc("To use host harness"), cl::init(false));

static cl::opt<bool> populateHostHarness(
    "ph", cl::desc("To populate host harness logic"),
    cl::value_desc("To populate host harness logic"), cl::init(false));

// populate host validation logic.
static cl::opt<bool> populateValidation(
    "pv", cl::desc("To populate host validation logic for conv2d"),
    cl::value_desc("To populate host validation logic"), cl::init(false));

static cl::opt<bool> populateValidationWithGPU(
    "pv_with_gpu",
    cl::desc("To populate host validation logic using gpu kernels"),
    cl::value_desc("To populate host validation logic"), cl::init(false));

static cl::opt<bool>
    printResultTensor("pr", cl::desc("To print result tensor for verification"),
                      cl::value_desc("To print result tensor for verification"),
                      cl::init(false));

static cl::opt<bool> populateCpuConvolution(
    "prc", cl::desc("To run cpu conv2d and print results for verification"),
    cl::value_desc("To run cpu conv2d and print results for verification"),
    cl::init(false));

static cl::opt<int> blockSize("block_size", cl::desc("Block size"),
                              cl::value_desc("Block size"), cl::init(0));

static cl::opt<int> gridSize("grid_size", cl::desc("Grid size"),
                             cl::value_desc("Grid size"), cl::init(0));

// use XDLOPS
static cl::opt<bool>
    xdlopsV2("x2", cl::desc("To use XDLOPS V2 lowering pipeline"),
             cl::value_desc("To use XDLOPS V2 lowering pipeline"),
             cl::init(false));

// data type
static cl::opt<std::string>
    tensorDataType("t", cl::desc("Data type for convolution"),
                   cl::value_desc("Data type for convolution"),
                   cl::init("f32"));

static cl::opt<std::string> randomSeed(
    "rand",
    cl::desc(
        "A positive integer or zero indicates the seed of random data generator"
        "for convolution inputs, e.g. -rand 1. If not specifed, or -rand none, "
        "use all ones. If 0, use time(0) as the seed."),
    cl::value_desc("seed"), cl::init("none"));

static cl::opt<std::string>
    randomDataType("rand_type",
                   cl::desc("To specify data type for random number generator,"
                            "e.g. -rand_type float, -rand_type int (default)."),
                   cl::value_desc("type"), cl::init("int"));

static cl::opt<std::string> randomSide(
    "rand_side",
    cl::desc("To populate random numbers to a specified tensor: "
             "For conv2d, -rand_side filter or -rand_side input; "
             "For conv2d_bwd_data, -rand_side filter or -rand_side output; "
             "For conv2d_bwd_weight, -rand_side input or -rand_side output. "
             "By default, populate random numbers to both tensors."),
    cl::value_desc("tensor"), cl::init("both"));

static void correctParameters() {
  std::string filterLayoutValue = filterLayout.getValue();
  std::string inputLayoutValue = inputLayout.getValue();
  std::string outputLayoutValue = outputLayout.getValue();
  if (filterLayoutValue.size() == 4) { // yxcgk not implement yet
    if (filterLayoutValue == "kcyx")
      filterLayout.setValue("gkcyx");
    else if (filterLayoutValue == "kyxc")
      filterLayout.setValue("gkyxc");
    else
      filterLayout.setValue("g" + filterLayoutValue);
  }
  if (outputLayoutValue.size() == 4) {
    if (outputLayoutValue == "nkhw")
      outputLayout.setValue("ngkhw");
    else if (outputLayoutValue == "nhwk")
      outputLayout.setValue("nhwgk");
    else
      outputLayout.setValue("g" + outputLayoutValue);
  }

  if (inputLayoutValue.size() == 4) {
    if (inputLayoutValue == "nchw")
      inputLayout.setValue("ngchw");
    else if (inputLayoutValue == "nhwc")
      inputLayout.setValue("nhwgc");
    else
      inputLayout.setValue("g" + inputLayoutValue);
  }

  // we can use paddingHeight or paddingHeightLeft + paddingHeightRight
  // if use paddingHeight , paddingHeightLeft and paddingHeightRight =
  // paddingHeight if use paddingHeightLeft + paddingHeightRight , please
  // assigne value
  if (paddingHeight.getValue() > 0) {
    if (paddingHeightLeft.getValue() == 0 &&
        paddingHeightRight.getValue() == 0) {
      paddingHeightLeft.setValue(paddingHeight.getValue());
      paddingHeightRight.setValue(paddingHeight.getValue());
    } else {
      if (paddingHeightLeft.getValue() != paddingHeight.getValue() ||
          paddingHeightRight.getValue() != paddingHeight.getValue()) {
        llvm::errs()
            << "you can't use both padding_h and (padding_h_l,padding_h_r).\n";
      }
    }
  }

  // we can use paddingWidth or paddingWidthLeft + paddingWidthRight
  // if use paddingWidth , paddingWidthLeft and paddingWidthRight = paddingWidth
  // if use paddingWidthLeft + paddingWidthRight , please assigne value
  if (paddingWidth.getValue() > 0) {
    if (paddingWidthLeft.getValue() == 0 && paddingWidthRight.getValue() == 0) {
      paddingWidthLeft.setValue(paddingWidth.getValue());
      paddingWidthRight.setValue(paddingWidth.getValue());
    } else {
      if (paddingWidthLeft.getValue() != paddingWidth.getValue() ||
          paddingWidthRight.getValue() != paddingWidth.getValue()) {
        llvm::errs()
            << "you can't use both padding_w and (padding_w_l,padding_w_r).\n";
      }
    }
  }

  // adjust the padding size
  // getOutputDim can give us correct output size
  // output size = input size+ padding size
  // then -(filter size-1) * dilation size -1
  // ,/ stride size and add 1
  auto getOutputDim = [](int64_t inputLen, int64_t filLen, int leftPadLen,
                         int rightPadLen, int strideLen, int dilLen) {
    return (inputLen + leftPadLen + rightPadLen - (filLen - 1) * dilLen - 1) /
               strideLen +
           1;
  };

  int hi = inputHeight.getValue();
  int y = filterHeight.getValue();
  int in_left_pad_h = paddingHeightLeft.getValue();
  int conv_stride_h = strideHeight.getValue();
  int conv_dilation_h = dilationHeight.getValue();
  int ho = getOutputDim(hi, y, in_left_pad_h, paddingHeightRight.getValue(),
                        conv_stride_h, conv_dilation_h);
  int hi_padded = 1 + (y - 1) * conv_dilation_h + (ho - 1) * conv_stride_h;
  // we got correct output size via getOutputDim, before adjusting size
  // we have original output size from user , but we need to check the padding
  // size so we use output size to calculate original input size add pad size,
  // hi_padded if hi_padded is equal to hi + in_left_pad_h , no adjusting but if
  // not equal, need extra padding hi_padded - (hi + in_left_pad_h)
  int in_right_pad_h =
      hi_padded > (hi + in_left_pad_h) ? hi_padded - (hi + in_left_pad_h) : 0;
  paddingHeightRight.setValue(in_right_pad_h);

  int wi = inputWidth.getValue();
  int x = filterWidth.getValue();
  int in_left_pad_w = paddingWidthLeft.getValue();
  int conv_stride_w = strideWidth.getValue();
  int conv_dilation_w = dilationWidth.getValue();
  int wo = getOutputDim(wi, x, in_left_pad_w, paddingWidthRight.getValue(),
                        conv_stride_w, conv_dilation_w);

  int wi_padded = 1 + (x - 1) * conv_dilation_w + (wo - 1) * conv_stride_w;
  int in_right_pad_w =
      wi_padded > (wi + in_left_pad_w) ? wi_padded - (wi + in_left_pad_w) : 0;
  paddingWidthRight.setValue(in_right_pad_w);
}

static void verifyLayout() {
  std::string filterLayoutValue = filterLayout.getValue();
  std::string inputLayoutValue = inputLayout.getValue();

  if (filterLayoutValue.find("yx") == std::string::npos &&
      filterLayoutValue.find("xy") == std::string::npos) {
    llvm::errs() << "Unsupported filter layout: disjointed yx!\n";
    exit(1);
  }

  if (inputLayoutValue.find("hw") == std::string::npos &&
      inputLayoutValue.find("wh") == std::string::npos) {

    llvm::errs() << "Unsupported input layout: disjointed hw!\n";
    exit(1);
  }
}

static void populateDefaults() {
  // arch is a required field to make lowering succeed. However,
  // 1. mlir-miopen-lib get it from client
  // 2. mlir-rocm-runner get it from the host machine
  // We don't particularly care about this field in the lowering
  // process unless it is tuning related. Therefore, setting this
  // field to a default value regardless.
  arch.setValue("amdgcn-amd-amdhsa:gfx900");

  if (populateDefaultValues == true) {
    if (xdlopsV2.getValue() == false) {
      groupSize.setValue(1);
      batchSize.setValue(128);
      inputChannel.setValue(8);
      outputChannel.setValue(128);
      inputHeight.setValue(32);
      inputWidth.setValue(32);
      filterHeight.setValue(3);
      filterWidth.setValue(3);
      dilationHeight.setValue(1);
      dilationWidth.setValue(1);
      strideHeight.setValue(1);
      strideWidth.setValue(1);
      paddingHeightLeft.setValue(0);
      paddingHeightRight.setValue(0);
      paddingWidthLeft.setValue(0);
      paddingWidthRight.setValue(0);
    } else {
      groupSize.setValue(1);
      batchSize.setValue(128);
      inputChannel.setValue(1024);
      outputChannel.setValue(1024);
      inputHeight.setValue(14);
      inputWidth.setValue(14);
      filterHeight.setValue(1);
      filterWidth.setValue(1);
      dilationHeight.setValue(1);
      dilationWidth.setValue(1);
      strideHeight.setValue(1);
      strideWidth.setValue(1);
      paddingHeightLeft.setValue(0);
      paddingHeightRight.setValue(0);
      paddingWidthLeft.setValue(0);
      paddingWidthRight.setValue(0);
      num_cu.setValue(120);
      arch.setValue("amdgcn-amd-amdhsa:gfx908");
    }
  }

  if (xdlopsV2.getValue() == true) {
    num_cu.setValue(120);
    arch.setValue("amdgcn-amd-amdhsa:gfx908");
  }

  if (outputHeight.getNumOccurrences() == 0) {
    outputHeight.setValue(Conv2dGenerator::outputDim(
        inputHeight.getValue(), filterHeight.getValue(),
        paddingHeightLeft.getValue(), paddingHeightRight.getValue(),
        strideHeight.getValue(), dilationHeight.getValue()));
  }
  if (outputWidth.getNumOccurrences() == 0) {
    outputWidth.setValue(Conv2dGenerator::outputDim(
        inputWidth.getValue(), filterWidth.getValue(),
        paddingWidthLeft.getValue(), paddingWidthRight.getValue(),
        strideWidth.getValue(), dilationWidth.getValue()));
  }
}

static LogicalResult detectMissingArguments() {
  const static std::vector<const cl::opt<int64_t> *> requiredArgs = {
      &groupSize,  &batchSize,     &inputChannel, &inputHeight,
      &inputWidth, &outputChannel, &filterWidth,  &filterHeight};
  for (auto *arg : requiredArgs) {
    if (arg->getValue() < 0) {
      llvm::errs() << "Value for: " << arg->ArgStr << " not specified\n";
      return failure();
    }
  }
  return success();
}

static FuncOp makeFuncDecl(OpBuilder &builder, StringRef funcName,
                           TypeRange inputs, TypeRange results) {
  auto func = FuncOp::create(builder.getUnknownLoc(), funcName,
                             builder.getFunctionType(inputs, results));
  func.sym_visibilityAttr(builder.getStringAttr("private"));
  return func;
}

static mlir::Value allocAndInitializeTensor(OpBuilder &builder, Block *block,
                                            mlir::Type dataType,
                                            mlir::FuncOp &mcpuMemset5DFuncOp,
                                            mlir::MemRefType &memRefType,
                                            mlir::Value &memsetMinValue,
                                            mlir::Value &memsetMaxValue,
                                            mlir::ConstantOp &seedValue) {
  auto fiveDimUnknownSizeMemRefType =
      MemRefType::get({-1, -1, -1, -1, -1}, dataType);

  // Emit CPU alloc
  auto cpuAllocOp =
      builder.create<AllocOp>(builder.getUnknownLoc(), memRefType);
  block->push_back(cpuAllocOp);

  // Emit memref cast
  auto cpuMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), cpuAllocOp, fiveDimUnknownSizeMemRefType);
  block->push_back(cpuMemRefCastOp);

  // Populate initial values
  auto cpuMemsetOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mcpuMemset5DFuncOp,
      ValueRange{cpuMemRefCastOp, memsetMinValue, memsetMaxValue, seedValue});
  block->push_back(cpuMemsetOp);

  return cpuAllocOp;
}

static FuncOp createConvertTensor(ModuleOp &module, OpBuilder &builder,
                                  mlir::MemRefType &originalMemRefType,
                                  mlir::MemRefType &convertedMemRefType) {
  std::string funcName = "convert_tensor";
  auto originalMemRefShape = originalMemRefType.getShape();
  funcName += std::to_string(originalMemRefShape[0]);
  for (unsigned i = 1; i < originalMemRefShape.size(); ++i) {
    auto dim = originalMemRefShape[i];
    funcName += "x" + std::to_string(dim);
  }
  auto originalDataType = originalMemRefType.getElementType();
  if (originalDataType == builder.getF32Type())
    funcName += "xf32";
  else if (originalDataType == builder.getF16Type())
    funcName += "xf16";
  else
    funcName += "xi16";

  FuncOp convertFuncOp;
  // Does a conversion function already exist?
  auto walkResult = module.walk([&](FuncOp funcOp) -> WalkResult {
    if (funcOp.getName() == funcName) {
      convertFuncOp = funcOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) { // Reuse an existing function
    return convertFuncOp;
  }

  // Otherwise create a new conversion function
  auto convertResultFuncOp = FuncOp::create(
      builder.getUnknownLoc(), StringRef(funcName),
      builder.getFunctionType({originalMemRefType, convertedMemRefType}, {}));
  module.push_back(convertResultFuncOp);

  // Construct a new Block.
  Block *block = convertResultFuncOp.addEntryBlock();

  // Insert loop to convert data.
  auto printMemRefShape = convertedMemRefType.getShape();
  auto zero = builder.create<ConstantIndexOp>(builder.getUnknownLoc(), 0);
  auto one = builder.create<ConstantIndexOp>(builder.getUnknownLoc(), 1);
  block->push_back(zero);
  block->push_back(one);
  SmallVector<mlir::Value, 4> boundVector;
  SmallVector<mlir::scf::ForOp, 4> loopOpVector;
  SmallVector<mlir::Value, 4> loopIVVector;

  // Emit loop bounds.
  for (unsigned i = 0; i < printMemRefShape.size(); ++i) {
    auto dim = printMemRefShape[i];
    auto bound = builder.create<ConstantIndexOp>(builder.getUnknownLoc(), dim);
    block->push_back(bound);
    boundVector.push_back(bound);
  }

  // Emit loops.
  for (unsigned i = 0; i < printMemRefShape.size(); ++i) {
    if (i == 0) {
      auto loopOp = builder.create<scf::ForOp>(builder.getUnknownLoc(), zero,
                                               boundVector[i], one);
      block->push_back(loopOp);
      loopOpVector.push_back(loopOp);
      loopIVVector.push_back(loopOp.getInductionVar());
    } else {
      auto lb = OpBuilder::atBlockBegin(
          loopOpVector[loopOpVector.size() - 1].getBody());
      auto loopOp = lb.create<scf::ForOp>(builder.getUnknownLoc(), zero,
                                          boundVector[i], one);
      loopOpVector.push_back(loopOp);
      loopIVVector.push_back(loopOp.getInductionVar());
    }
  }

  // Emit loop body in the innermost loop.
  auto innermostLoopBuilder =
      OpBuilder::atBlockBegin(loopOpVector[loopOpVector.size() - 1].getBody());
  auto convertedDataType = convertedMemRefType.getElementType();
  auto arguments = convertResultFuncOp.getArguments();
  mlir::Value sourceMemRef = arguments[0];
  mlir::Value convertedMemRef = arguments[1];
  mlir::Value sourceValue = innermostLoopBuilder.create<LoadOp>(
      builder.getUnknownLoc(), originalDataType, sourceMemRef, loopIVVector);
  mlir::Value convertedValue;
  if (originalDataType == convertedDataType) {
    convertedValue = sourceValue;
  } else {
    auto f16Type = builder.getF16Type();
    auto f32Type = builder.getF32Type();
    // f16->f32, emit fpext
    if (originalDataType == f16Type && convertedDataType == f32Type)
      convertedValue = innermostLoopBuilder.create<FPExtOp>(
          builder.getUnknownLoc(), sourceValue, f32Type);
    // f32->f16, emit fptrunc
    else if (originalDataType == f32Type && convertedDataType == f16Type)
      convertedValue = innermostLoopBuilder.create<FPTruncOp>(
          builder.getUnknownLoc(), sourceValue, f16Type);
    else if (originalDataType == builder.getIntegerType(16))
      // Treat I16 as BF16.
      // TBD: Implement proper conversion logic. Force cast for now.
      convertedValue = innermostLoopBuilder.create<SIToFPOp>(
          builder.getUnknownLoc(), sourceValue, f32Type);
  }
  innermostLoopBuilder.create<StoreOp>(builder.getUnknownLoc(), convertedValue,
                                       convertedMemRef, loopIVVector);

  // Emit return op.
  auto returnOp =
      builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  block->push_back(returnOp);

  return convertResultFuncOp;
}

// Alloc CPU memory of f32 and copy values from sourceOriginalAllocOp
static mlir::Value
allocAndCopyTensor(ModuleOp &module, OpBuilder &builder, Block *block,
                   mlir::FuncOp &mcpuMemCopy5DFuncOp,
                   mlir::Value sourceOriginalAllocOp,
                   const SmallVector<int64_t, 5> &sourceDimension) {
  auto sourceMemRefType =
      sourceOriginalAllocOp.getType().template dyn_cast<MemRefType>();
  auto dataType = sourceMemRefType.getElementType();

  if (dataType == builder.getF32Type())
    return sourceOriginalAllocOp;

  auto floatType = builder.getF32Type();
  auto fiveDimUnknownSizeFloatType =
      MemRefType::get({-1, -1, -1, -1, -1}, floatType);

  // Emit CPU alloc of Float type
  auto floatMemRefType = MemRefType::get(
      ArrayRef<int64_t>(sourceDimension.begin(), sourceDimension.end()),
      floatType);
  auto cpuAllocOp =
      builder.create<AllocOp>(builder.getUnknownLoc(), floatMemRefType);
  block->push_back(cpuAllocOp);

  if (dataType == builder.getF16Type()) { // f16
    // Create conversion routine
    auto convertResultFuncOp =
        createConvertTensor(module, builder, sourceMemRefType, floatMemRefType);
    auto convertResultCallOp =
        builder.create<CallOp>(builder.getUnknownLoc(), convertResultFuncOp,
                               ValueRange{sourceOriginalAllocOp, cpuAllocOp});
    block->push_back(convertResultCallOp);
  } else { // bf16
    // Emit memref cast
    auto cpuMemRefCastOp = builder.create<MemRefCastOp>(
        builder.getUnknownLoc(), cpuAllocOp, fiveDimUnknownSizeFloatType);
    block->push_back(cpuMemRefCastOp);

    auto sourceUnknownSizeMemRefType =
        MemRefType::get({-1, -1, -1, -1, -1}, dataType);
    auto sourceMemRefCastOp = builder.create<MemRefCastOp>(
        builder.getUnknownLoc(), sourceOriginalAllocOp,
        sourceUnknownSizeMemRefType);
    block->push_back(sourceMemRefCastOp);

    // Copy values of sourceAllocOp into cpuAllocOp
    auto cpuMemCopyCallOp =
        builder.create<CallOp>(builder.getUnknownLoc(), mcpuMemCopy5DFuncOp,
                               ValueRange{sourceMemRefCastOp, cpuMemRefCastOp});
    block->push_back(cpuMemCopyCallOp);
  }

  return cpuAllocOp;
}

static FuncOp
createCPUConvolution(ModuleOp &module, OpBuilder &builder,
                     const mlir::Conv2dGenerator::Config &genConfig) {
  auto filterDimension = genConfig.filterDimension;
  auto inputDimension = genConfig.inputDimension;
  auto outputDimension = genConfig.outputDimension;
  auto convOpType = genConfig.operation;

  auto floatType = builder.getF32Type();
  auto filterMemRefType = MemRefType::get(
      ArrayRef<int64_t>(filterDimension.begin(), filterDimension.end()),
      floatType);
  auto inputMemRefType = MemRefType::get(
      ArrayRef<int64_t>(inputDimension.begin(), inputDimension.end()),
      floatType);
  auto outputMemRefType = MemRefType::get(
      ArrayRef<int64_t>(outputDimension.begin(), outputDimension.end()),
      floatType);

  // Create conv2d_host function
  auto cpuConvFuncOp = FuncOp::create(
      builder.getUnknownLoc(), StringRef("conv2d_host"),
      builder.getFunctionType(
          {filterMemRefType, inputMemRefType, outputMemRefType}, {}));
  module.push_back(cpuConvFuncOp);

  // Construct a new Block.
  Block *cpuConvBlock = cpuConvFuncOp.addEntryBlock();

  // Emit memref_cast.
  // %a0 = memref_cast %arg0 : memref<128x8x3x3xf32> to memref<*xf32>
  // %a1 = memref_cast %arg1 : memref<128x8x32x32xf32> to memref<*xf32>
  // %a2 = memref_cast %arg2 : memref<128x128x30x30xf32> to memref<*xf32>
  auto unrankedMemRefType =
      UnrankedMemRefType::get(filterMemRefType.getElementType(), 0);

  auto filterMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), cpuConvBlock->getArgument(0),
      unrankedMemRefType);
  auto inputMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), cpuConvBlock->getArgument(1),
      unrankedMemRefType);
  auto outputMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), cpuConvBlock->getArgument(2),
      unrankedMemRefType);
  cpuConvBlock->push_back(filterMemRefCastOp);
  cpuConvBlock->push_back(inputMemRefCastOp);
  cpuConvBlock->push_back(outputMemRefCastOp);

  // Emit ConstantOps to be used for strides, paddings and dilations
  auto intType = builder.getIntegerType(32);

  auto strideHeightConstantOp = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), genConfig.strideHeight, intType);

  auto strideWidthConstantOp = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), genConfig.strideWidth, intType);

  auto paddingHeightLeftConstantOp = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), genConfig.paddingHeightLeft, intType);

  auto paddingHeightRightConstantOp = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), genConfig.paddingHeightRight, intType);

  auto paddingWidthLeftConstantOp = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), genConfig.paddingWidthLeft, intType);

  auto paddingWidthRightConstantOp = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), genConfig.paddingWidthRight, intType);

  auto dilationHeightConstantOp = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), genConfig.dilationHeight, intType);

  auto dilationWidthConstantOp = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), genConfig.dilationWidth, intType);

  cpuConvBlock->push_back(strideHeightConstantOp);
  cpuConvBlock->push_back(strideWidthConstantOp);
  cpuConvBlock->push_back(paddingHeightLeftConstantOp);
  cpuConvBlock->push_back(paddingHeightRightConstantOp);
  cpuConvBlock->push_back(paddingWidthLeftConstantOp);
  cpuConvBlock->push_back(paddingWidthRightConstantOp);
  cpuConvBlock->push_back(dilationHeightConstantOp);
  cpuConvBlock->push_back(dilationWidthConstantOp);

  // Emit ConstantIndex ops
  // %c_0 = constant 0 : index
  // %c_1 = constant 1 : index
  // %c_2 = constant 2 : index
  // %c_3 = constant 3 : index
  // %c_4 = constant 4 : index
  std::vector<ConstantIndexOp> indexOpVec;
  for (int i = 0; i < 5; i++) {
    auto indexOp = builder.create<ConstantIndexOp>(builder.getUnknownLoc(), i);
    cpuConvBlock->push_back(indexOp);
    indexOpVec.push_back(indexOp);
  }

  auto charType = builder.getIntegerType(8);
  // Emit Constant ops for letters used in layouts
  //  these numbers are ascii codes , g = 103, k = 107
  //  %g = constant 103: i8
  //  %k = constant 107 : i8
  //  %c = constant 99 : i8
  //  %y = constant 121 : i8
  //  %x = constant 120 : i8
  //  %n = constant 110 : i8
  //  %h = constant 104 : i8
  //  %w = constant 119 : i8
  auto kConstantOp =
      builder.create<ConstantIntOp>(builder.getUnknownLoc(), 'k', charType);

  auto cConstantOp =
      builder.create<ConstantIntOp>(builder.getUnknownLoc(), 'c', charType);

  auto yConstantOp =
      builder.create<ConstantIntOp>(builder.getUnknownLoc(), 'y', charType);

  auto xConstantOp =
      builder.create<ConstantIntOp>(builder.getUnknownLoc(), 'x', charType);

  auto nConstantOp =
      builder.create<ConstantIntOp>(builder.getUnknownLoc(), 'n', charType);

  auto hConstantOp =
      builder.create<ConstantIntOp>(builder.getUnknownLoc(), 'h', charType);

  auto wConstantOp =
      builder.create<ConstantIntOp>(builder.getUnknownLoc(), 'w', charType);

  auto gConstantOp =
      builder.create<ConstantIntOp>(builder.getUnknownLoc(), 'g', charType);

  cpuConvBlock->push_back(gConstantOp);
  cpuConvBlock->push_back(kConstantOp);
  cpuConvBlock->push_back(cConstantOp);
  cpuConvBlock->push_back(yConstantOp);
  cpuConvBlock->push_back(xConstantOp);
  cpuConvBlock->push_back(nConstantOp);
  cpuConvBlock->push_back(hConstantOp);
  cpuConvBlock->push_back(wConstantOp);

  std::unordered_map<char, mlir::ConstantOp> layoutConstOps;
  layoutConstOps['g'] = gConstantOp;
  layoutConstOps['k'] = kConstantOp;
  layoutConstOps['c'] = cConstantOp;
  layoutConstOps['y'] = yConstantOp;
  layoutConstOps['x'] = xConstantOp;
  layoutConstOps['n'] = nConstantOp;
  layoutConstOps['h'] = hConstantOp;
  layoutConstOps['w'] = wConstantOp;

  // %3   = alloca() : memref<5xi8>
  // %4   = alloca() : memref<5xi8>
  // %5   = alloca() : memref<5xi8>
  SmallVector<int64_t, 5> layoutVector({5});
  auto layoutMemRefType = MemRefType::get(
      ArrayRef<int64_t>(layoutVector.begin(), layoutVector.end()), charType);
  auto filLayoutAllocOp =
      builder.create<AllocaOp>(builder.getUnknownLoc(), layoutMemRefType);
  auto inLayoutAllocOp =
      builder.create<AllocaOp>(builder.getUnknownLoc(), layoutMemRefType);
  auto outLayoutAllocOp =
      builder.create<AllocaOp>(builder.getUnknownLoc(), layoutMemRefType);
  cpuConvBlock->push_back(filLayoutAllocOp);
  cpuConvBlock->push_back(inLayoutAllocOp);
  cpuConvBlock->push_back(outLayoutAllocOp);

  // Store layouts into layoutAllocOp
  // store %k, %3[%c_0]: memref<5xi32>
  std::string fil_layout = genConfig.filterLayout;
  std::string in_layout = genConfig.inputLayout;
  std::string out_layout = genConfig.outputLayout;
  for (int i = 0; i < 5; i++) {
    auto storeOp = builder.create<StoreOp>(
        builder.getUnknownLoc(), layoutConstOps[fil_layout[i]],
        filLayoutAllocOp, ValueRange{indexOpVec[i]});
    cpuConvBlock->push_back(storeOp);
  }

  for (int i = 0; i < 5; i++) {
    auto storeOp = builder.create<StoreOp>(
        builder.getUnknownLoc(), layoutConstOps[in_layout[i]], inLayoutAllocOp,
        ValueRange{indexOpVec[i]});
    cpuConvBlock->push_back(storeOp);
  }

  for (int i = 0; i < 5; i++) {
    auto storeOp = builder.create<StoreOp>(
        builder.getUnknownLoc(), layoutConstOps[out_layout[i]],
        outLayoutAllocOp, ValueRange{indexOpVec[i]});
    cpuConvBlock->push_back(storeOp);
  }

  // Emit memref_cast
  // %6 = memref_cast %3 : memref<5xi8> to memref<*xi8>
  // %7 = memref_cast %4 : memref<5xi8> to memref<*xi8>
  // %8 = memref_cast %5 : memref<5xi8> to memref<*xi8>
  auto unrankedLayoutMemRefType = UnrankedMemRefType::get(charType, 0);
  auto filLayoutMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), filLayoutAllocOp, unrankedLayoutMemRefType);
  auto inLayoutMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), inLayoutAllocOp, unrankedLayoutMemRefType);
  auto outLayoutMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), outLayoutAllocOp, unrankedLayoutMemRefType);

  cpuConvBlock->push_back(filLayoutMemRefCastOp);
  cpuConvBlock->push_back(inLayoutMemRefCastOp);
  cpuConvBlock->push_back(outLayoutMemRefCastOp);

  std::string mcpuFuncName;

  switch (convOpType) {
  case mlir::miopen::Conv2DOpType:
    mcpuFuncName = "mcpuConv2d";
    break;
  case miopen::Conv2DBwdDataOpType:
    mcpuFuncName = "mcpuConv2dBwdData";
    break;
  case miopen::Conv2DBwdWeightOpType:
    mcpuFuncName = "mcpuConv2dBwdWeight";
    break;
  }

  // Emit cpu convolution function call op
  auto mcpuConv2dFuncOp =
      makeFuncDecl(builder, mcpuFuncName,
                   {unrankedMemRefType, unrankedMemRefType, unrankedMemRefType,
                    unrankedLayoutMemRefType, unrankedLayoutMemRefType,
                    unrankedLayoutMemRefType, intType, intType, intType,
                    intType, intType, intType, intType, intType},
                   {});

  auto mcpuConv2dCallOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mcpuConv2dFuncOp,
      ValueRange{filterMemRefCastOp, inputMemRefCastOp, outputMemRefCastOp,
                 filLayoutMemRefCastOp, inLayoutMemRefCastOp,
                 outLayoutMemRefCastOp, strideHeightConstantOp,
                 strideWidthConstantOp, paddingHeightLeftConstantOp,
                 paddingHeightRightConstantOp, paddingWidthLeftConstantOp,
                 paddingWidthRightConstantOp, dilationHeightConstantOp,
                 dilationWidthConstantOp});

  module.push_back(mcpuConv2dFuncOp);
  cpuConvBlock->push_back(mcpuConv2dCallOp);

  // Emit return op
  auto cpuConvFuncOpReturnOp =
      builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  cpuConvBlock->push_back(cpuConvFuncOpReturnOp);

  return cpuConvFuncOp;
}

// Cconvert CPU results of f32 to f16 or bf16
static mlir::Value
getConvertedCpuResults(ModuleOp &module, OpBuilder &builder, Block *block,
                       mlir::Value cpuOriginalAllocOp,
                       mlir::MemRefType &convertedMemRefType) {
  auto dataType = convertedMemRefType.getElementType();

  if (dataType == builder.getF32Type())
    return cpuOriginalAllocOp;

  // Emit allocOp for converted CPU results.
  auto cpuConvertedResults =
      builder.create<AllocOp>(builder.getUnknownLoc(), convertedMemRefType);
  block->push_back(cpuConvertedResults);

  if (dataType == builder.getF16Type()) { // f16
    auto originalMemRefType =
        cpuOriginalAllocOp.getType().template dyn_cast<MemRefType>();

    // Create conversion routine f32->f16
    auto convertFuncOp = createConvertTensor(
        module, builder, originalMemRefType, convertedMemRefType);
    auto convertResultCallOp = builder.create<CallOp>(
        builder.getUnknownLoc(), convertFuncOp,
        ValueRange{cpuOriginalAllocOp, cpuConvertedResults});
    block->push_back(convertResultCallOp);
  } else { // bf16
    StringRef convertFuncName = "mcpuMem5DFloatConvertBF16";

    auto fiveDimUnknownSizeMemRefFloat =
        MemRefType::get({-1, -1, -1, -1, -1}, builder.getF32Type());
    auto fiveDimUnknownSizeMemRefType =
        MemRefType::get({-1, -1, -1, -1, -1}, dataType);

    auto convertFuncOp = makeFuncDecl(
        builder, convertFuncName,
        {fiveDimUnknownSizeMemRefFloat, fiveDimUnknownSizeMemRefType}, {});
    module.push_back(convertFuncOp);

    // Emit memref cast
    auto cpuConvertedMemRefCastOp = builder.create<MemRefCastOp>(
        builder.getUnknownLoc(), cpuConvertedResults,
        fiveDimUnknownSizeMemRefType);
    block->push_back(cpuConvertedMemRefCastOp);

    // Emit memref_cast
    auto cpuOrigianlMemRefCastOp = builder.create<MemRefCastOp>(
        builder.getUnknownLoc(), cpuOriginalAllocOp,
        fiveDimUnknownSizeMemRefFloat);
    block->push_back(cpuOrigianlMemRefCastOp);

    // Emit function call to convert Cpu results from F32 to desired data type
    auto convertCallOp = builder.create<CallOp>(
        builder.getUnknownLoc(), convertFuncOp,
        ValueRange{cpuOrigianlMemRefCastOp, cpuConvertedMemRefCastOp});
    block->push_back(convertCallOp);
  }
  return cpuConvertedResults;
}

static FuncOp createVerifyFuncOp(ModuleOp &module, OpBuilder &builder,
                                 const SmallVector<int64_t, 5> &outputDimension,
                                 mlir::Value cpuAllocOp,
                                 mlir::Value gpuAllocOp) {
  // Emit verify_results function call
  auto outputMemRefType = cpuAllocOp.getType().template dyn_cast<MemRefType>();

  auto verifyFuncOp = FuncOp::create(
      builder.getUnknownLoc(), StringRef("verify_results"),
      builder.getFunctionType({outputMemRefType, outputMemRefType}, {}));
  module.push_back(verifyFuncOp);

  // Emit verification logic.
  // Create a new block
  Block *verifyResultsBlock = verifyFuncOp.addEntryBlock();

  // %c0 = constant 0: index
  auto c0IndexOp = builder.create<ConstantIndexOp>(builder.getUnknownLoc(), 0);
  verifyResultsBlock->push_back(c0IndexOp);

  // %result = alloca() : memref<1xi32>
  SmallVector<int64_t, 1> oneElementVector({1});
  auto resultMemRefType = MemRefType::get(
      ArrayRef<int64_t>(oneElementVector.begin(), oneElementVector.end()),
      builder.getIntegerType(32));
  auto cmpResultAllocOp =
      builder.create<AllocaOp>(builder.getUnknownLoc(), resultMemRefType);
  verifyResultsBlock->push_back(cmpResultAllocOp);

  // %c0_i32 = constant 0 : i32
  // %c1_i32 = constant 1 : i32
  auto c0ConstantInt32Op = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), 0, builder.getIntegerType(32));
  verifyResultsBlock->push_back(c0ConstantInt32Op);
  auto c1ConstantInt32Op = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), 1, builder.getIntegerType(32));
  verifyResultsBlock->push_back(c1ConstantInt32Op);

  // store %c1_i32, %result[%c0] : memref<1xi32>
  auto storeOp1 =
      builder.create<StoreOp>(builder.getUnknownLoc(), c1ConstantInt32Op,
                              cmpResultAllocOp, ValueRange{c0IndexOp});
  verifyResultsBlock->push_back(storeOp1);

  // %%c1 = constant 1 : index
  auto c1IndexOp = builder.create<ConstantIndexOp>(builder.getUnknownLoc(), 1);
  verifyResultsBlock->push_back(c1IndexOp);

  // Emit constant index Ops for loop upper bounds
  auto indexOp0 = builder.create<ConstantIndexOp>(builder.getUnknownLoc(),
                                                  outputDimension[0]);
  verifyResultsBlock->push_back(indexOp0);
  auto indexOp1 = builder.create<ConstantIndexOp>(builder.getUnknownLoc(),
                                                  outputDimension[1]);
  verifyResultsBlock->push_back(indexOp1);
  auto indexOp2 = builder.create<ConstantIndexOp>(builder.getUnknownLoc(),
                                                  outputDimension[2]);
  verifyResultsBlock->push_back(indexOp2);
  auto indexOp3 = builder.create<ConstantIndexOp>(builder.getUnknownLoc(),
                                                  outputDimension[3]);
  verifyResultsBlock->push_back(indexOp3);

  auto indexOp4 = builder.create<ConstantIndexOp>(builder.getUnknownLoc(),
                                                  outputDimension[4]);
  verifyResultsBlock->push_back(indexOp4);

  // clang-format off
  // scf.for %arg0 = %c0 to %c128 step %c1 {
  //  scf.for %arg1 = %c0 to %c30 step %c1 {
  //    scf.for %arg2 = %c0 to %c30 step %c1 {
  //      scf.for %arg3 = %c0 to %c1 step %c1 {
  //        scf.for %arg4 = %c0 to %c128 step %c1 {
  //          %2 = load %CpuResults[%arg0, %arg1, %arg2, %arg3, %arg4] : memref<128x30x30x1x128xf32>
  //          %3 = load %GpuResults[%arg0, %arg1, %arg2, %arg3, %arg4] : memref<128x30x30x1x128xf32>
  //          %cst = constant 1.000000e-07 : f32
  //          %4 = subf %2, %3 : f32
  //          %5 = absf %4 : f32
  //          %6 = cmpf ugt, %5, %cst : f32
  //          scf.if %6 {
  //            store %c0_i32, %result[%c0] : memref<1xi32>
  //          }
  //        }
  //      }
  //    }
  //  }
  //}
  // clang-format on

  auto loop0 = builder.create<scf::ForOp>(builder.getUnknownLoc(), c0IndexOp,
                                          indexOp0, c1IndexOp);
  auto bt0 = OpBuilder::atBlockTerminator(loop0.getBody());
  auto iv0 = loop0.getInductionVar();

  auto loop1 = bt0.create<scf::ForOp>(builder.getUnknownLoc(), c0IndexOp,
                                      indexOp1, c1IndexOp);
  auto bt1 = OpBuilder::atBlockTerminator(loop1.getBody());
  auto iv1 = loop1.getInductionVar();

  auto loop2 = bt1.create<scf::ForOp>(builder.getUnknownLoc(), c0IndexOp,
                                      indexOp2, c1IndexOp);
  auto bt2 = OpBuilder::atBlockTerminator(loop2.getBody());
  auto iv2 = loop2.getInductionVar();

  auto loop3 = bt2.create<scf::ForOp>(builder.getUnknownLoc(), c0IndexOp,
                                      indexOp3, c1IndexOp);
  auto bt3 = OpBuilder::atBlockTerminator(loop3.getBody());
  auto iv3 = loop3.getInductionVar();

  auto loop4 = bt3.create<scf::ForOp>(builder.getUnknownLoc(), c0IndexOp,
                                      indexOp4, c1IndexOp);

  auto bt4 = OpBuilder::atBlockTerminator(loop4.getBody());
  auto iv4 = loop4.getInductionVar();
  auto cpuLoadOp = bt4.create<LoadOp>(builder.getUnknownLoc(),
                                      verifyResultsBlock->getArgument(0),
                                      ValueRange{iv0, iv1, iv2, iv3, iv4});

  auto gpuLoadOp = bt4.create<LoadOp>(builder.getUnknownLoc(),
                                      verifyResultsBlock->getArgument(1),
                                      ValueRange{iv0, iv1, iv2, iv3, iv4});

  scf::IfOp ifOp;
  if (outputMemRefType.getElementType() == builder.getIntegerType(16)) {
    auto cmpOp = bt4.create<CmpIOp>(builder.getUnknownLoc(), CmpIPredicate::ne,
                                    cpuLoadOp, gpuLoadOp);
    ifOp = bt4.create<scf::IfOp>(builder.getUnknownLoc(), cmpOp, false);
  } else {
    if (randomSeed.getValue() != "none" &&
        randomDataType.getValue() == "float") {
      float delta = 0.0000001;
      if (outputMemRefType.getElementType() == builder.getF16Type())
        delta = 0.0001;

      auto deltaConstantOp = bt4.create<ConstantOp>(
          builder.getUnknownLoc(), outputMemRefType.getElementType(),
          builder.getFloatAttr(outputMemRefType.getElementType(), delta));
      auto subfOp =
          bt4.create<SubFOp>(builder.getUnknownLoc(), cpuLoadOp, gpuLoadOp);
      auto absfOp = bt4.create<AbsFOp>(builder.getUnknownLoc(), subfOp);
      auto cmpOp = bt4.create<CmpFOp>(
          builder.getUnknownLoc(), CmpFPredicate::UGT, absfOp, deltaConstantOp);
      ifOp = bt4.create<scf::IfOp>(builder.getUnknownLoc(), cmpOp, false);
    } else {
      auto cmpOp = bt4.create<CmpFOp>(builder.getUnknownLoc(),
                                      CmpFPredicate::UNE, cpuLoadOp, gpuLoadOp);
      ifOp = bt4.create<scf::IfOp>(builder.getUnknownLoc(), cmpOp, false);
    }
  }
  auto thenBody = ifOp.getThenBodyBuilder();

  thenBody.create<StoreOp>(builder.getUnknownLoc(), c0ConstantInt32Op,
                           cmpResultAllocOp, ValueRange{c0IndexOp});

  verifyResultsBlock->push_back(loop0);

  // Emit print function call
  auto unrankedMemRefType =
      UnrankedMemRefType::get(builder.getIntegerType(32), 0);
  auto printMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), cmpResultAllocOp, unrankedMemRefType);
  auto printMemRefFuncOp =
      makeFuncDecl(builder, "print_memref_i32", {unrankedMemRefType}, {});
  auto printMemRefCallOp =
      builder.create<CallOp>(builder.getUnknownLoc(), printMemRefFuncOp,
                             ValueRange{printMemRefCastOp});
  module.push_back(printMemRefFuncOp);
  verifyResultsBlock->push_back(printMemRefCastOp);
  verifyResultsBlock->push_back(printMemRefCallOp);

  auto returnOp =
      builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  verifyResultsBlock->push_back(returnOp);

  return verifyFuncOp;
}

static FuncOp launchGPUConvolution(ModuleOp &module, OpBuilder &builder,
                                   mlir::Type dataType,
                                   mlir::miopen::ConvOpType convOpType,
                                   mlir::Value filterHostAllocOp,
                                   mlir::Value inputHostAllocOp,
                                   mlir::Value outputHostAllocOp,
                                   StringRef entryPoint) {
  auto filterMemRefType =
      filterHostAllocOp.getType().template dyn_cast<MemRefType>();
  auto inputMemRefType =
      inputHostAllocOp.getType().template dyn_cast<MemRefType>();
  auto outputMemRefType =
      outputHostAllocOp.getType().template dyn_cast<MemRefType>();
  // Create gpu_conv function

  std::string funcName = "gpu_conv";
  if (entryPoint == populateValidateEntryPoint.getValue())
    funcName += "_verifier";
  auto gpuConvFuncOp = FuncOp::create(
      builder.getUnknownLoc(), StringRef(funcName),
      builder.getFunctionType(
          {filterMemRefType, inputMemRefType, outputMemRefType}, {}));
  module.push_back(gpuConvFuncOp);
  // Emit gpu convolution logic.
  // Create a new block
  Block *gpuConvBlock = gpuConvFuncOp.addEntryBlock();

  // Emit memref_cast.
  auto fiveDimUnknownSizeMemRefType =
      MemRefType::get({-1, -1, -1, -1, -1}, dataType);

  auto filterMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), gpuConvBlock->getArgument(0),
      fiveDimUnknownSizeMemRefType);
  auto inputMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), gpuConvBlock->getArgument(1),
      fiveDimUnknownSizeMemRefType);
  auto outputMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), gpuConvBlock->getArgument(2),
      fiveDimUnknownSizeMemRefType);
  gpuConvBlock->push_back(filterMemRefCastOp);
  gpuConvBlock->push_back(inputMemRefCastOp);
  gpuConvBlock->push_back(outputMemRefCastOp);

  // Emit GPU memory allocation function calls.
  StringRef gpuMemAllocFuncName;
  if (dataType == builder.getF32Type()) {
    gpuMemAllocFuncName = "mgpuMemAlloc5DFloat";
  } else if (dataType == builder.getF16Type()) {
    gpuMemAllocFuncName = "mgpuMemAlloc5DHalf";
  } else if (dataType == builder.getIntegerType(16)) {
    gpuMemAllocFuncName = "mgpuMemAlloc5DBF16";
  }

  FuncOp mgpuMemAlloc5DFuncOp;
  auto walkResult = module.walk([&](FuncOp funcOp) -> WalkResult {
    if (funcOp.getName() == gpuMemAllocFuncName) {
      mgpuMemAlloc5DFuncOp = funcOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  // Build an alloc function if it didn't already exist
  if (!walkResult.wasInterrupted()) {
    mgpuMemAlloc5DFuncOp = makeFuncDecl(builder, gpuMemAllocFuncName,
                                        {fiveDimUnknownSizeMemRefType},
                                        {fiveDimUnknownSizeMemRefType});

    module.push_back(mgpuMemAlloc5DFuncOp);
  }

  auto filterGpuAllocOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mgpuMemAlloc5DFuncOp,
                             ValueRange{filterMemRefCastOp});
  auto inputGpuAllocOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mgpuMemAlloc5DFuncOp,
                             ValueRange{inputMemRefCastOp});
  auto outputGpuAllocOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mgpuMemAlloc5DFuncOp,
                             ValueRange{outputMemRefCastOp});
  gpuConvBlock->push_back(filterGpuAllocOp);
  gpuConvBlock->push_back(inputGpuAllocOp);
  gpuConvBlock->push_back(outputGpuAllocOp);

  // Emit some constant values for HIP runtime API calls.
  auto oneConstantI32Op = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), 1, builder.getIntegerType(32));
  auto twoConstantI32Op = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), 2, builder.getIntegerType(32));
  gpuConvBlock->push_back(oneConstantI32Op);
  gpuConvBlock->push_back(twoConstantI32Op);

  // Emit CPU->GPU memcpy function calls.
  StringRef gpuMemCopyFuncName;
  if (dataType == builder.getF32Type()) {
    gpuMemCopyFuncName = "mgpuMemCopy5DFloat";
  } else if (dataType == builder.getF16Type()) {
    gpuMemCopyFuncName = "mgpuMemCopy5DHalf";
  } else if (dataType == builder.getIntegerType(16)) {
    gpuMemCopyFuncName = "mgpuMemCopy5DBF16";
  }

  FuncOp mgpuMemCopy5DFuncOp;
  walkResult = module.walk([&](FuncOp funcOp) -> WalkResult {
    if (funcOp.getName() == gpuMemCopyFuncName) {
      mgpuMemCopy5DFuncOp = funcOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!walkResult.wasInterrupted()) {
    mgpuMemCopy5DFuncOp =
        makeFuncDecl(builder, gpuMemCopyFuncName,
                     {fiveDimUnknownSizeMemRefType,
                      fiveDimUnknownSizeMemRefType, builder.getIntegerType(32)},
                     {});
    module.push_back(mgpuMemCopy5DFuncOp);
  }

  auto filterCpuToGpuCopyOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mgpuMemCopy5DFuncOp,
      ValueRange{filterMemRefCastOp, filterGpuAllocOp.getResult(0),
                 oneConstantI32Op});
  auto inputCpuToGpuCopyOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mgpuMemCopy5DFuncOp,
      ValueRange{inputMemRefCastOp, inputGpuAllocOp.getResult(0),
                 oneConstantI32Op});
  auto outputCpuToGpuCopyOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mgpuMemCopy5DFuncOp,
      ValueRange{outputMemRefCastOp, outputGpuAllocOp.getResult(0),
                 oneConstantI32Op});
  gpuConvBlock->push_back(filterCpuToGpuCopyOp);
  gpuConvBlock->push_back(inputCpuToGpuCopyOp);
  gpuConvBlock->push_back(outputCpuToGpuCopyOp);

  // Emit memref_cast.

  auto filterGpuMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), filterGpuAllocOp.getResult(0), filterMemRefType);
  auto inputGpuMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), inputGpuAllocOp.getResult(0), inputMemRefType);
  auto outputGpuMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), outputGpuAllocOp.getResult(0), outputMemRefType);
  gpuConvBlock->push_back(filterGpuMemRefCastOp);
  gpuConvBlock->push_back(inputGpuMemRefCastOp);
  gpuConvBlock->push_back(outputGpuMemRefCastOp);

  // Emit host stub function.
  auto kernelStubFuncOp = FuncOp::create(
      builder.getUnknownLoc(), entryPoint, // populateEntryPoint,
      builder.getFunctionType(
          {filterMemRefType, inputMemRefType, outputMemRefType}, {}));
  module.push_back(kernelStubFuncOp);

  // Construct a new Block.
  Block *kernelStubFuncOpBlock = kernelStubFuncOp.addEntryBlock();
  auto kernelStubFuncOpReturnOp =
      builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  kernelStubFuncOpBlock->push_back(kernelStubFuncOpReturnOp);

  // Emit conv2d function call.
  auto kernelCallOp = builder.create<CallOp>(
      builder.getUnknownLoc(), kernelStubFuncOp,
      ValueRange{filterGpuMemRefCastOp, inputGpuMemRefCastOp,
                 outputGpuMemRefCastOp});
  gpuConvBlock->push_back(kernelCallOp);

  // Emit mgpuMemCopy5DFloat function call.
  mlir::Value resultGpuValue, resultCpuValue;
  switch (convOpType) {
  case miopen::Conv2DOpType:
    resultGpuValue = outputGpuAllocOp.getResult(0);
    resultCpuValue = outputMemRefCastOp;
    break;
  case miopen::Conv2DBwdDataOpType:
    resultGpuValue = inputGpuAllocOp.getResult(0);
    resultCpuValue = inputMemRefCastOp;
    break;
  case miopen::Conv2DBwdWeightOpType:
    resultGpuValue = filterGpuAllocOp.getResult(0);
    resultCpuValue = filterMemRefCastOp;
    break;
  }

  auto outputGpuToCpuCopyOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mgpuMemCopy5DFuncOp,
      ValueRange{resultGpuValue, resultCpuValue, twoConstantI32Op});
  gpuConvBlock->push_back(outputGpuToCpuCopyOp);

  // Emit GPU memory deallocation function calls.
  StringRef gpuMemDeallocFuncName;
  if (dataType == builder.getF32Type()) {
    gpuMemDeallocFuncName = "mgpuMemDealloc5DFloat";
  } else if (dataType == builder.getF16Type()) {
    gpuMemDeallocFuncName = "mgpuMemDealloc5DHalf";
  } else if (dataType == builder.getIntegerType(16)) {
    gpuMemDeallocFuncName = "mgpuMemDealloc5DBF16";
  }

  FuncOp mgpuMemDealloc5DFuncOp;
  walkResult = module.walk([&](FuncOp funcOp) -> WalkResult {
    if (funcOp.getName() == gpuMemDeallocFuncName) {
      mgpuMemDealloc5DFuncOp = funcOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!walkResult.wasInterrupted()) {
    mgpuMemDealloc5DFuncOp = makeFuncDecl(builder, gpuMemDeallocFuncName,
                                          {fiveDimUnknownSizeMemRefType}, {});
    module.push_back(mgpuMemDealloc5DFuncOp);
  }

  auto filterGpuDeallocOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mgpuMemDealloc5DFuncOp,
                             ValueRange{filterMemRefCastOp});
  auto inputGpuDeallocOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mgpuMemDealloc5DFuncOp,
                             ValueRange{inputMemRefCastOp});
  auto outputGpuDeallocOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mgpuMemDealloc5DFuncOp,
                             ValueRange{outputMemRefCastOp});
  gpuConvBlock->push_back(filterGpuDeallocOp);
  gpuConvBlock->push_back(inputGpuDeallocOp);
  gpuConvBlock->push_back(outputGpuDeallocOp);

  auto returnOp =
      builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  gpuConvBlock->push_back(returnOp);

  return gpuConvFuncOp;
}

// Determine the range and seed for the random data generator
static std::tuple<short, short, int> configRandomTestData() {
  short min, max;
  int seed = 1;
  if (randomSeed.getValue() == "none") {
    min = 1;
    max = 1;
  } else {
    if (randomDataType.getValue() == "int") {
      // generate random integer in [-5, 5)
      min = -5;
      max = 5;
    } else {
      // generate random floats in [-1, 1)
      min = -1;
      max = 1;
    }
    std::string rseed = randomSeed.getValue();
    if (rseed[0] >= '0' and rseed[0] <= '9')
      seed = std::stoi(rseed);
    else
      seed = -1;
  }
  return std::make_tuple(min, max, seed);
}

static std::string getMemsetFuncName(OpBuilder &builder, mlir::Type dataType) {
  std::string memsetFuncName;
  if (dataType == builder.getF32Type()) {
    memsetFuncName = "mcpuMemset5DFloatRand";
  } else if (dataType == builder.getF16Type()) {
    memsetFuncName = "mcpuMemset5DHalfRand";
  } else if (dataType == builder.getIntegerType(16)) {
    memsetFuncName = "mcpuMemset5DBF16Rand";
  }
  if (randomDataType == "float")
    memsetFuncName += "Float";
  else
    memsetFuncName += "Int";
  return memsetFuncName;
}

static void generateTensorInitValues(
    OpBuilder &builder, Block *block, mlir::FuncOp &mcpuMemset5DFuncOp,
    mlir::Value &filterMemsetMinValue, mlir::Value &filterMemsetMaxValue,
    mlir::Value &inputMemsetMinValue, mlir::Value &inputMemsetMaxValue,
    mlir::Value &outputMemsetMinValue, mlir::Value &outputMemsetMaxValue,
    mlir::ConstantOp &seedConstantIntOp, mlir::miopen::ConvOpType convOpType) {
  auto int16Type = builder.getIntegerType(16);
  auto int32Type = builder.getIntegerType(32);
  unsigned short zero = 0, one = 1;
  short min, max;
  int seed = 1;
  std::tie(min, max, seed) = configRandomTestData();

  mlir::ConstantOp oneConstantIntOp;
  if (randomSeed.getValue() != "none" && randomSide.getValue() != "both") {
    oneConstantIntOp = builder.create<ConstantOp>(
        builder.getUnknownLoc(), int16Type, builder.getI16IntegerAttr(one));
    block->push_back(oneConstantIntOp);
  }

  auto zeroConstantIntOp = builder.create<ConstantOp>(
      builder.getUnknownLoc(), int16Type, builder.getI16IntegerAttr(zero));
  auto minConstantIntOp = builder.create<ConstantOp>(
      builder.getUnknownLoc(), int16Type, builder.getI16IntegerAttr(min));
  auto maxConstantIntOp = builder.create<ConstantOp>(
      builder.getUnknownLoc(), int16Type, builder.getI16IntegerAttr(max));
  seedConstantIntOp = builder.create<ConstantOp>(
      builder.getUnknownLoc(), int32Type, builder.getI32IntegerAttr(seed));

  block->push_back(zeroConstantIntOp);
  block->push_back(minConstantIntOp);
  block->push_back(maxConstantIntOp);
  block->push_back(seedConstantIntOp);

  switch (convOpType) {
  case miopen::Conv2DOpType:
    if (randomSeed.getValue() == "none" || // min & max are already set to 1
        (randomSeed.getValue() != "none" &&
         (randomSide.getValue() == "filter" ||
          randomSide.getValue() == "both"))) {
      filterMemsetMinValue = minConstantIntOp;
      filterMemsetMaxValue = maxConstantIntOp;
    } else {
      filterMemsetMinValue = oneConstantIntOp;
      filterMemsetMaxValue = oneConstantIntOp;
    }

    if (randomSeed.getValue() == "none" ||
        (randomSeed.getValue() != "none" &&
         (randomSide.getValue() == "input" ||
          randomSide.getValue() == "both"))) {
      inputMemsetMinValue = minConstantIntOp;
      inputMemsetMaxValue = maxConstantIntOp;
    } else {
      inputMemsetMinValue = oneConstantIntOp;
      inputMemsetMaxValue = oneConstantIntOp;
    }

    outputMemsetMinValue = zeroConstantIntOp;
    outputMemsetMaxValue = zeroConstantIntOp;
    break;
  case miopen::Conv2DBwdDataOpType:
    if (randomSeed.getValue() == "none" ||
        (randomSeed.getValue() != "none" &&
         (randomSide.getValue() == "filter" ||
          randomSide.getValue() == "both"))) {
      filterMemsetMinValue = minConstantIntOp;
      filterMemsetMaxValue = maxConstantIntOp;
    } else {
      filterMemsetMinValue = oneConstantIntOp;
      filterMemsetMaxValue = oneConstantIntOp;
    }

    if (randomSeed.getValue() == "none" ||
        (randomSeed.getValue() != "none" &&
         (randomSide.getValue() == "output" ||
          randomSide.getValue() == "both"))) {
      outputMemsetMinValue = minConstantIntOp;
      outputMemsetMaxValue = maxConstantIntOp;
    } else {
      outputMemsetMinValue = oneConstantIntOp;
      outputMemsetMaxValue = oneConstantIntOp;
    }

    inputMemsetMinValue = zeroConstantIntOp;
    inputMemsetMaxValue = zeroConstantIntOp;
    break;
  case miopen::Conv2DBwdWeightOpType:
    if (randomSeed.getValue() == "none" ||
        (randomSeed.getValue() != "none" &&
         (randomSide.getValue() == "input" ||
          randomSide.getValue() == "both"))) {
      inputMemsetMinValue = minConstantIntOp;
      inputMemsetMaxValue = maxConstantIntOp;
    } else {
      inputMemsetMinValue = oneConstantIntOp;
      inputMemsetMaxValue = oneConstantIntOp;
    }

    if (randomSeed.getValue() == "none" ||
        (randomSeed.getValue() != "none" &&
         (randomSide.getValue() == "output" ||
          randomSide.getValue() == "both"))) {
      outputMemsetMinValue = minConstantIntOp;
      outputMemsetMaxValue = maxConstantIntOp;
    } else {
      outputMemsetMinValue = oneConstantIntOp;
      outputMemsetMaxValue = oneConstantIntOp;
    }

    filterMemsetMinValue = zeroConstantIntOp;
    filterMemsetMaxValue = zeroConstantIntOp;
    break;
  }
  return;
}

static LogicalResult populateHostHarnessLogic(
    ModuleOp &module, OpBuilder &builder, MLIRContext &context,
    const mlir::Conv2dGenerator::Config &genConfig, mlir::Type dataType) {

  auto filterDimension = genConfig.filterDimension;
  auto inputDimension = genConfig.inputDimension;
  auto outputDimension = genConfig.outputDimension;
  auto convOpType = genConfig.operation;

  // Construct main function.
  auto func = FuncOp::create(builder.getUnknownLoc(), "main",
                             builder.getFunctionType({}, {}));
  module.push_back(func);

  // Construct a new Block.
  Block *block = func.addEntryBlock();

  auto filterMemRefType = MemRefType::get(
      ArrayRef<int64_t>(filterDimension.begin(), filterDimension.end()),
      dataType);
  auto inputMemRefType = MemRefType::get(
      ArrayRef<int64_t>(inputDimension.begin(), inputDimension.end()),
      dataType);
  auto outputMemRefType = MemRefType::get(
      ArrayRef<int64_t>(outputDimension.begin(), outputDimension.end()),
      dataType);
  auto fiveDimUnknownSizeMemRefType =
      MemRefType::get({-1, -1, -1, -1, -1}, dataType);

  // Determine types of memref to be printed out.
  // Forward convolution: output tensor.
  // Backward data convolution: input tensor.
  // Backward weight convolution: filter tensor.
  MemRefType printMemRefType;
  switch (convOpType) {
  case miopen::Conv2DOpType:
    printMemRefType = MemRefType::get(
        ArrayRef<int64_t>(outputDimension.begin(), outputDimension.end()),
        builder.getF32Type());
    break;
  case miopen::Conv2DBwdDataOpType:
    printMemRefType = MemRefType::get(
        ArrayRef<int64_t>(inputDimension.begin(), inputDimension.end()),
        builder.getF32Type());
    break;
  case miopen::Conv2DBwdWeightOpType:
    printMemRefType = MemRefType::get(
        ArrayRef<int64_t>(filterDimension.begin(), filterDimension.end()),
        builder.getF32Type());
    break;
  }

  // Emit CPU alloc.
  auto filterHostAllocOp =
      builder.create<AllocOp>(builder.getUnknownLoc(), filterMemRefType);
  auto inputHostAllocOp =
      builder.create<AllocOp>(builder.getUnknownLoc(), inputMemRefType);
  auto outputHostAllocOp =
      builder.create<AllocOp>(builder.getUnknownLoc(), outputMemRefType);
  block->push_back(filterHostAllocOp);
  block->push_back(inputHostAllocOp);
  block->push_back(outputHostAllocOp);

  // Emit memref_cast.
  auto filterMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), filterHostAllocOp, fiveDimUnknownSizeMemRefType);
  auto inputMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), inputHostAllocOp, fiveDimUnknownSizeMemRefType);
  auto outputMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), outputHostAllocOp, fiveDimUnknownSizeMemRefType);
  block->push_back(filterMemRefCastOp);
  block->push_back(inputMemRefCastOp);
  block->push_back(outputMemRefCastOp);

  auto int16Type = builder.getIntegerType(16);
  auto int32Type = builder.getIntegerType(32);

  // Emit CPU memset function calls.
  std::string memsetFuncName = getMemsetFuncName(builder, dataType);

  auto mcpuMemset5DFuncOp = makeFuncDecl(
      builder, memsetFuncName,
      {fiveDimUnknownSizeMemRefType, int16Type, int16Type, int32Type}, {});
  module.push_back(mcpuMemset5DFuncOp);

  // Populate initial values.
  mlir::Value filterMemsetMinValue, inputMemsetMinValue, outputMemsetMinValue;
  mlir::Value filterMemsetMaxValue, inputMemsetMaxValue, outputMemsetMaxValue;
  mlir::ConstantOp seedConstantIntOp;

  generateTensorInitValues(builder, block, mcpuMemset5DFuncOp,
                           filterMemsetMinValue, filterMemsetMaxValue,
                           inputMemsetMinValue, inputMemsetMaxValue,
                           outputMemsetMinValue, outputMemsetMaxValue,
                           seedConstantIntOp, convOpType);

  auto filterCpuMemsetOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mcpuMemset5DFuncOp,
      ValueRange{filterMemRefCastOp, filterMemsetMinValue, filterMemsetMaxValue,
                 seedConstantIntOp});
  auto inputCpuMemsetOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mcpuMemset5DFuncOp,
      ValueRange{inputMemRefCastOp, inputMemsetMinValue, inputMemsetMaxValue,
                 seedConstantIntOp});
  auto outputCpuMemsetOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mcpuMemset5DFuncOp,
      ValueRange{outputMemRefCastOp, outputMemsetMinValue, outputMemsetMaxValue,
                 seedConstantIntOp});
  block->push_back(filterCpuMemsetOp);
  block->push_back(inputCpuMemsetOp);
  block->push_back(outputCpuMemsetOp);

  // launch gpu_conv
  auto gpuConvFuncOp = launchGPUConvolution(
      module, builder, dataType, convOpType, filterHostAllocOp,
      inputHostAllocOp, outputHostAllocOp, populateEntryPoint);

  auto gpuConvCallOp = builder.create<CallOp>(
      builder.getUnknownLoc(), gpuConvFuncOp,
      ValueRange{filterHostAllocOp, inputHostAllocOp, outputHostAllocOp});
  block->push_back(gpuConvCallOp);

  mlir::Value resultCpuValue, resultOriginalCpuValue;
  mlir::MemRefType resultOriginalCpuType;
  switch (convOpType) {
  case miopen::Conv2DOpType:
    resultCpuValue = outputMemRefCastOp;
    resultOriginalCpuValue = outputHostAllocOp;
    resultOriginalCpuType = outputMemRefType;
    break;
  case miopen::Conv2DBwdDataOpType:
    resultCpuValue = inputMemRefCastOp;
    resultOriginalCpuValue = inputHostAllocOp;
    resultOriginalCpuType = inputMemRefType;
    break;
  case miopen::Conv2DBwdWeightOpType:
    resultCpuValue = filterMemRefCastOp;
    resultOriginalCpuValue = filterHostAllocOp;
    resultOriginalCpuType = filterMemRefType;
    break;
  }

  // Print the result if be specified.
  if (printResultTensor.getValue()) {
    // Emit CPU alloc for memref to be printed out.
    auto printHostAllocOp =
        builder.create<AllocOp>(builder.getUnknownLoc(), printMemRefType);
    block->push_back(printHostAllocOp);

    if (dataType == builder.getIntegerType(16)) { // i16 only
      auto floatType = builder.getF32Type();
      auto unknownSizeMemRefFloatType =
          MemRefType::get({-1, -1, -1, -1, -1}, floatType);

      auto printUnkownSizeMemRefCastOp = builder.create<MemRefCastOp>(
          builder.getUnknownLoc(), printHostAllocOp,
          unknownSizeMemRefFloatType);
      block->push_back(printUnkownSizeMemRefCastOp);

      auto cpuMemConvertOp = makeFuncDecl(
          builder, "mcpuMem5DBF16ConvertFloat",
          {fiveDimUnknownSizeMemRefType, unknownSizeMemRefFloatType}, {});
      module.push_back(cpuMemConvertOp);

      auto printMemConvertCallOp = builder.create<CallOp>(
          builder.getUnknownLoc(), cpuMemConvertOp,
          ValueRange{resultCpuValue, printUnkownSizeMemRefCastOp});
      block->push_back(printMemConvertCallOp);

    } else { // f32 or f16
      // Emit type conversion routine to convert every element to f32.
      auto convertResultFuncOp = createConvertTensor(
          module, builder, resultOriginalCpuType, printMemRefType);
      auto convertResultCallOp = builder.create<CallOp>(
          builder.getUnknownLoc(), convertResultFuncOp,
          ValueRange{resultOriginalCpuValue, printHostAllocOp});
      block->push_back(convertResultCallOp);
    }

    // Emit print function call.
    StringRef printMemRefFuncName = "print_memref_f32";
    auto unrankedMemRefType = UnrankedMemRefType::get(builder.getF32Type(), 0);
    auto printMemRefCastOp = builder.create<MemRefCastOp>(
        builder.getUnknownLoc(), printHostAllocOp, unrankedMemRefType);
    auto printMemRefFuncOp =
        makeFuncDecl(builder, printMemRefFuncName, {unrankedMemRefType}, {});
    auto printMemRefCallOp =
        builder.create<CallOp>(builder.getUnknownLoc(), printMemRefFuncOp,
                               ValueRange{printMemRefCastOp});
    module.push_back(printMemRefFuncOp);
    block->push_back(printMemRefCastOp);
    block->push_back(printMemRefCallOp);

    auto printHostDeallocOp =
        builder.create<DeallocOp>(builder.getUnknownLoc(), printHostAllocOp);
    block->push_back(printHostDeallocOp);
  }

  // Emit CPU dealloc.
  auto filterHostDeallocOp =
      builder.create<DeallocOp>(builder.getUnknownLoc(), filterHostAllocOp);
  auto inputHostDeallocOp =
      builder.create<DeallocOp>(builder.getUnknownLoc(), inputHostAllocOp);
  auto outputHostDeallocOp =
      builder.create<DeallocOp>(builder.getUnknownLoc(), outputHostAllocOp);
  block->push_back(filterHostDeallocOp);
  block->push_back(inputHostDeallocOp);
  block->push_back(outputHostDeallocOp);

  auto returnOp =
      builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  block->push_back(returnOp);

  return success();
}

static LogicalResult populateValidationLogic(
    ModuleOp &module, OpBuilder &builder, MLIRContext &context,
    const mlir::Conv2dGenerator::Config &genConfig, mlir::Type dataType) {

  auto filterDimension = genConfig.filterDimension;
  auto inputDimension = genConfig.inputDimension;
  auto outputDimension = genConfig.outputDimension;
  auto convOpType = genConfig.operation;
  auto xdlops = genConfig.xdlops;

  // Construct main function.
  auto func = FuncOp::create(builder.getUnknownLoc(), "main",
                             builder.getFunctionType({}, {}));
  module.push_back(func);

  // Construct a new Block.
  Block *block = func.addEntryBlock();

  auto filterMemRefType = MemRefType::get(
      ArrayRef<int64_t>(filterDimension.begin(), filterDimension.end()),
      dataType);
  auto inputMemRefType = MemRefType::get(
      ArrayRef<int64_t>(inputDimension.begin(), inputDimension.end()),
      dataType);
  auto outputMemRefType = MemRefType::get(
      ArrayRef<int64_t>(outputDimension.begin(), outputDimension.end()),
      dataType);
  auto fiveDimUnknownSizeMemRefType =
      MemRefType::get({-1, -1, -1, -1, -1}, dataType);

  // Emit CPU alloc.
  auto filterHostAllocOp =
      builder.create<AllocOp>(builder.getUnknownLoc(), filterMemRefType);
  auto inputHostAllocOp =
      builder.create<AllocOp>(builder.getUnknownLoc(), inputMemRefType);
  auto outputHostAllocOp =
      builder.create<AllocOp>(builder.getUnknownLoc(), outputMemRefType);
  block->push_back(filterHostAllocOp);
  block->push_back(inputHostAllocOp);
  block->push_back(outputHostAllocOp);

  // Emit memref_cast.
  auto filterMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), filterHostAllocOp, fiveDimUnknownSizeMemRefType);
  auto inputMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), inputHostAllocOp, fiveDimUnknownSizeMemRefType);
  auto outputMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), outputHostAllocOp, fiveDimUnknownSizeMemRefType);
  block->push_back(filterMemRefCastOp);
  block->push_back(inputMemRefCastOp);
  block->push_back(outputMemRefCastOp);

  auto int16Type = builder.getIntegerType(16);
  auto int32Type = builder.getIntegerType(32);

  // Emit CPU memset function calls.
  std::string memsetFuncName = getMemsetFuncName(builder, dataType);

  auto mcpuMemset5DFuncOp = makeFuncDecl(
      builder, memsetFuncName,
      {fiveDimUnknownSizeMemRefType, int16Type, int16Type, int32Type}, {});
  module.push_back(mcpuMemset5DFuncOp);

  // Populate initial values.
  mlir::Value filterMemsetMinValue, inputMemsetMinValue, outputMemsetMinValue;
  mlir::Value filterMemsetMaxValue, inputMemsetMaxValue, outputMemsetMaxValue;
  mlir::ConstantOp seedConstantIntOp;

  generateTensorInitValues(builder, block, mcpuMemset5DFuncOp,
                           filterMemsetMinValue, filterMemsetMaxValue,
                           inputMemsetMinValue, inputMemsetMaxValue,
                           outputMemsetMinValue, outputMemsetMaxValue,
                           seedConstantIntOp, convOpType);

  auto filterCpuMemsetOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mcpuMemset5DFuncOp,
      ValueRange{filterMemRefCastOp, filterMemsetMinValue, filterMemsetMaxValue,
                 seedConstantIntOp});
  auto inputCpuMemsetOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mcpuMemset5DFuncOp,
      ValueRange{inputMemRefCastOp, inputMemsetMinValue, inputMemsetMaxValue,
                 seedConstantIntOp});
  auto outputCpuMemsetOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mcpuMemset5DFuncOp,
      ValueRange{outputMemRefCastOp, outputMemsetMinValue, outputMemsetMaxValue,
                 seedConstantIntOp});
  block->push_back(filterCpuMemsetOp);
  block->push_back(inputCpuMemsetOp);
  block->push_back(outputCpuMemsetOp);

  // Populate host harness logic
  auto gpuConvFuncOp = launchGPUConvolution(
      module, builder, dataType, convOpType, filterHostAllocOp,
      inputHostAllocOp, outputHostAllocOp, populateEntryPoint);

  auto gpuConvCallOp = builder.create<CallOp>(
      builder.getUnknownLoc(), gpuConvFuncOp,
      ValueRange{filterHostAllocOp, inputHostAllocOp, outputHostAllocOp});
  block->push_back(gpuConvCallOp);

  mlir::Value gpuOriginalResults;
  MemRefType gpuOriginalResultType;
  switch (convOpType) {
  case miopen::Conv2DOpType:
    gpuOriginalResults = outputHostAllocOp;
    gpuOriginalResultType = outputMemRefType;
    break;
  case miopen::Conv2DBwdDataOpType:
    gpuOriginalResults = inputHostAllocOp;
    gpuOriginalResultType = inputMemRefType;
    break;
  case miopen::Conv2DBwdWeightOpType:
    gpuOriginalResults = filterHostAllocOp;
    gpuOriginalResultType = filterMemRefType;
    break;
  }

  // Produce CPU or GPU convolution logic on F32 type
  auto floatType = builder.getF32Type();

  filterMemRefType = MemRefType::get(
      ArrayRef<int64_t>(filterDimension.begin(), filterDimension.end()),
      floatType);
  inputMemRefType = MemRefType::get(
      ArrayRef<int64_t>(inputDimension.begin(), inputDimension.end()),
      floatType);
  outputMemRefType = MemRefType::get(
      ArrayRef<int64_t>(outputDimension.begin(), outputDimension.end()),
      floatType);

  auto fiveDimUnknownSizeFloatType =
      MemRefType::get({-1, -1, -1, -1, -1}, floatType);

  // Emit CPU memcopy function calls
  mlir::FuncOp mcpuMemCopy5DFuncOp;
  if (dataType == builder.getIntegerType(16)) { // bf16
    mcpuMemCopy5DFuncOp = makeFuncDecl(
        builder, "mcpuMem5DBF16ConvertFloat",
        {fiveDimUnknownSizeMemRefType, fiveDimUnknownSizeFloatType}, {});
  } else { // fp32 or fp16
    mcpuMemCopy5DFuncOp = makeFuncDecl(
        builder, "mcpuMemCopy5DFloat",
        {fiveDimUnknownSizeFloatType, fiveDimUnknownSizeFloatType}, {});
  }
  module.push_back(mcpuMemCopy5DFuncOp);

  // Emit CPU memset function calls
  if (dataType != builder.getF32Type()) {
    StringRef funcName;
    if (randomDataType.getValue() == "float")
      funcName = "mcpuMemset5DFloatRandFloat";
    else
      funcName = "mcpuMemset5DFloatRandInt";
    mcpuMemset5DFuncOp = makeFuncDecl(
        builder, funcName,
        {fiveDimUnknownSizeFloatType, int16Type, int16Type, int32Type}, {});
    module.push_back(mcpuMemset5DFuncOp);
  }

  // Prepare input data for verifier convolution.
  mlir::Value verifierFilterHostAllocOp, verifierInputHostAllocOp,
      verifierOutputHostAllocOp;
  if (randomSeed.getValue() == "none") {
    // If not using random data, emit CPU alloc and initialization
    verifierFilterHostAllocOp = allocAndInitializeTensor(
        builder, block, floatType, mcpuMemset5DFuncOp, filterMemRefType,
        filterMemsetMinValue, filterMemsetMaxValue, seedConstantIntOp);
    verifierInputHostAllocOp = allocAndInitializeTensor(
        builder, block, floatType, mcpuMemset5DFuncOp, inputMemRefType,
        inputMemsetMinValue, inputMemsetMaxValue, seedConstantIntOp);
    verifierOutputHostAllocOp = allocAndInitializeTensor(
        builder, block, floatType, mcpuMemset5DFuncOp, outputMemRefType,
        outputMemsetMinValue, outputMemsetMaxValue, seedConstantIntOp);
  } else {
    auto zeroConstantIntOp = builder.create<ConstantOp>(
        builder.getUnknownLoc(), int16Type, builder.getI16IntegerAttr(0));
    block->push_back(zeroConstantIntOp);

    mlir::Value memsetValue = zeroConstantIntOp;

    // If using random data, emit CPU alloc and copy input data
    switch (convOpType) {
    case miopen::Conv2DOpType:
      verifierFilterHostAllocOp =
          allocAndCopyTensor(module, builder, block, mcpuMemCopy5DFuncOp,
                             filterHostAllocOp, filterDimension);
      verifierInputHostAllocOp =
          allocAndCopyTensor(module, builder, block, mcpuMemCopy5DFuncOp,
                             inputHostAllocOp, inputDimension);
      verifierOutputHostAllocOp = allocAndInitializeTensor(
          builder, block, floatType, mcpuMemset5DFuncOp, outputMemRefType,
          memsetValue, memsetValue, seedConstantIntOp);
      break;
    case miopen::Conv2DBwdDataOpType:
      verifierFilterHostAllocOp =
          allocAndCopyTensor(module, builder, block, mcpuMemCopy5DFuncOp,
                             filterHostAllocOp, filterDimension);
      verifierInputHostAllocOp = allocAndInitializeTensor(
          builder, block, floatType, mcpuMemset5DFuncOp, inputMemRefType,
          memsetValue, memsetValue, seedConstantIntOp);
      verifierOutputHostAllocOp =
          allocAndCopyTensor(module, builder, block, mcpuMemCopy5DFuncOp,
                             outputHostAllocOp, outputDimension);
      break;
    case miopen::Conv2DBwdWeightOpType:
      verifierFilterHostAllocOp = allocAndInitializeTensor(
          builder, block, floatType, mcpuMemset5DFuncOp, filterMemRefType,
          memsetValue, memsetValue, seedConstantIntOp);
      verifierInputHostAllocOp =
          allocAndCopyTensor(module, builder, block, mcpuMemCopy5DFuncOp,
                             inputHostAllocOp, inputDimension);
      verifierOutputHostAllocOp =
          allocAndCopyTensor(module, builder, block, mcpuMemCopy5DFuncOp,
                             outputHostAllocOp, outputDimension);
      break;
    }
  }

  // Populate host validation logic
  if (populateValidationWithGPU.getValue() &&
      (!(!xdlops && dataType == builder.getF32Type()))) {
    // Verify with GPU convolution of f32
    auto verifierConvFuncOp = launchGPUConvolution(
        module, builder, floatType, convOpType, verifierFilterHostAllocOp,
        verifierInputHostAllocOp, verifierOutputHostAllocOp,
        populateValidateEntryPoint);

    // Emit gpu_conv_verifier function call
    auto verifierConvCallOp = builder.create<CallOp>(
        builder.getUnknownLoc(), verifierConvFuncOp,
        ValueRange{verifierFilterHostAllocOp, verifierInputHostAllocOp,
                   verifierOutputHostAllocOp});
    block->push_back(verifierConvCallOp);
  } else {
    // Otherwise (-pv, or -pv_with_gpu with non-XDLops and f32 ), verity with
    // CPU convolution of f32
    auto cpuConvFuncOp = createCPUConvolution(module, builder, genConfig);

    // Emit conv2d_host function call.
    auto cpuConvCallOp = builder.create<CallOp>(
        builder.getUnknownLoc(), cpuConvFuncOp,
        ValueRange{verifierFilterHostAllocOp, verifierInputHostAllocOp,
                   verifierOutputHostAllocOp});
    block->push_back(cpuConvCallOp);
  }

  mlir::Value verifierResults;
  switch (convOpType) {
  case miopen::Conv2DOpType:
    verifierResults = verifierOutputHostAllocOp;
    break;
  case miopen::Conv2DBwdDataOpType:
    verifierResults = verifierInputHostAllocOp;
    break;
  case miopen::Conv2DBwdWeightOpType:
    verifierResults = verifierFilterHostAllocOp;
    break;
  }

  // Convert CPU results
  mlir::Value verifierConvertedResults;
  if (dataType == builder.getF32Type()) {
    verifierConvertedResults = verifierResults;
  } else {
    verifierConvertedResults = getConvertedCpuResults(
        module, builder, block, verifierResults, gpuOriginalResultType);
  }

  mlir::FuncOp verifyFuncOp;
  switch (convOpType) {
  case miopen::Conv2DOpType:
    verifyFuncOp =
        createVerifyFuncOp(module, builder, outputDimension,
                           verifierConvertedResults, outputHostAllocOp);
    break;
  case miopen::Conv2DBwdDataOpType:
    verifyFuncOp =
        createVerifyFuncOp(module, builder, inputDimension,
                           verifierConvertedResults, inputHostAllocOp);
    break;
  case miopen::Conv2DBwdWeightOpType:
    verifyFuncOp =
        createVerifyFuncOp(module, builder, filterDimension,
                           verifierConvertedResults, filterHostAllocOp);
    break;
  }

  // Compare the results
  auto verifyCallOp = builder.create<CallOp>(
      builder.getUnknownLoc(), verifyFuncOp,
      ValueRange{verifierConvertedResults, gpuOriginalResults});
  block->push_back(verifyCallOp);

  // Emit CPU dealloc.
  if (dataType != builder.getF32Type()) {
    auto verifierResultsDeallocOp = builder.create<DeallocOp>(
        builder.getUnknownLoc(), verifierConvertedResults);
    block->push_back(verifierResultsDeallocOp);

    auto verifierFilterHostDeallocOp = builder.create<DeallocOp>(
        builder.getUnknownLoc(), verifierFilterHostAllocOp);
    auto verifierInputHostDeallocOp = builder.create<DeallocOp>(
        builder.getUnknownLoc(), verifierInputHostAllocOp);
    auto verifierOutputHostDeallocOp = builder.create<DeallocOp>(
        builder.getUnknownLoc(), verifierOutputHostAllocOp);
    block->push_back(verifierFilterHostDeallocOp);
    block->push_back(verifierInputHostDeallocOp);
    block->push_back(verifierOutputHostDeallocOp);
  }

  auto filterHostDeallocOp =
      builder.create<DeallocOp>(builder.getUnknownLoc(), filterHostAllocOp);
  auto inputHostDeallocOp =
      builder.create<DeallocOp>(builder.getUnknownLoc(), inputHostAllocOp);
  auto outputHostDeallocOp =
      builder.create<DeallocOp>(builder.getUnknownLoc(), outputHostAllocOp);
  block->push_back(filterHostDeallocOp);
  block->push_back(inputHostDeallocOp);
  block->push_back(outputHostDeallocOp);

  auto returnOp =
      builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  block->push_back(returnOp);

  return success();
}

static LogicalResult populateCpuConvolutionLogic(
    ModuleOp &module, OpBuilder &builder, MLIRContext &context,
    const mlir::Conv2dGenerator::Config &genConfig, mlir::Type dataType) {

  auto filterDimension = genConfig.filterDimension;
  auto inputDimension = genConfig.inputDimension;
  auto outputDimension = genConfig.outputDimension;
  auto convOpType = genConfig.operation;

  // Construct main function.
  auto func = FuncOp::create(builder.getUnknownLoc(), "main",
                             builder.getFunctionType({}, {}));
  module.push_back(func);

  // Construct a new Block.
  Block *block = func.addEntryBlock();

  // CPU tensors of dataType
  auto filterMemRefType = MemRefType::get(
      ArrayRef<int64_t>(filterDimension.begin(), filterDimension.end()),
      dataType);
  auto inputMemRefType = MemRefType::get(
      ArrayRef<int64_t>(inputDimension.begin(), inputDimension.end()),
      dataType);
  auto outputMemRefType = MemRefType::get(
      ArrayRef<int64_t>(outputDimension.begin(), outputDimension.end()),
      dataType);

  auto int16Type = builder.getIntegerType(16);
  auto int32Type = builder.getIntegerType(32);

  auto fiveDimUnknownSizeMemRefType =
      MemRefType::get({-1, -1, -1, -1, -1}, dataType);

  // Emit CPU memset function calls.
  std::string memsetFuncName = getMemsetFuncName(builder, dataType);

  auto mcpuMemset5DFuncOp = makeFuncDecl(
      builder, memsetFuncName,
      {fiveDimUnknownSizeMemRefType, int16Type, int16Type, int32Type}, {});
  module.push_back(mcpuMemset5DFuncOp);

  mlir::Value filterMemsetMinValue, inputMemsetMinValue, outputMemsetMinValue;
  mlir::Value filterMemsetMaxValue, inputMemsetMaxValue, outputMemsetMaxValue;
  mlir::ConstantOp seedConstantIntOp;

  generateTensorInitValues(builder, block, mcpuMemset5DFuncOp,
                           filterMemsetMinValue, filterMemsetMaxValue,
                           inputMemsetMinValue, inputMemsetMaxValue,
                           outputMemsetMinValue, outputMemsetMaxValue,
                           seedConstantIntOp, convOpType);

  // Emit CPU alloc and populate initial values.
  auto cpuFilterAllocOp = allocAndInitializeTensor(
      builder, block, dataType, mcpuMemset5DFuncOp, filterMemRefType,
      filterMemsetMinValue, filterMemsetMaxValue, seedConstantIntOp);

  auto cpuInputAllocOp = allocAndInitializeTensor(
      builder, block, dataType, mcpuMemset5DFuncOp, inputMemRefType,
      inputMemsetMinValue, inputMemsetMaxValue, seedConstantIntOp);

  auto cpuOutputAllocOp = allocAndInitializeTensor(
      builder, block, dataType, mcpuMemset5DFuncOp, outputMemRefType,
      outputMemsetMinValue, outputMemsetMaxValue, seedConstantIntOp);

  auto floatType = builder.getF32Type();
  auto fiveDimUnknownSizeFloatType =
      MemRefType::get({-1, -1, -1, -1, -1}, floatType);

  // Convert CPU tensors of f16 or bf16 to f32
  auto cpuFilterHostAllocOp = cpuFilterAllocOp;
  auto cpuInputHostAllocOp = cpuInputAllocOp;
  auto cpuOutputHostAllocOp = cpuOutputAllocOp;

  mlir::FuncOp mcpuMemCopy5DFuncOp;
  if (dataType != builder.getF32Type()) {
    // CPU bf16->f32 conversion function
    mcpuMemCopy5DFuncOp = makeFuncDecl(
        builder, "mcpuMem5DBF16ConvertFloat",
        {fiveDimUnknownSizeMemRefType, fiveDimUnknownSizeFloatType}, {});
    if (dataType == builder.getIntegerType(16))
      module.push_back(mcpuMemCopy5DFuncOp);

    // Emit CPU alloc and conversion function calls
    cpuFilterHostAllocOp =
        allocAndCopyTensor(module, builder, block, mcpuMemCopy5DFuncOp,
                           cpuFilterAllocOp, filterDimension);
    cpuInputHostAllocOp =
        allocAndCopyTensor(module, builder, block, mcpuMemCopy5DFuncOp,
                           cpuInputAllocOp, inputDimension);
    cpuOutputHostAllocOp =
        allocAndCopyTensor(module, builder, block, mcpuMemCopy5DFuncOp,
                           cpuOutputAllocOp, outputDimension);
  }

  //
  auto filterMemRefFloatType = MemRefType::get(
      ArrayRef<int64_t>(filterDimension.begin(), filterDimension.end()),
      floatType);
  auto inputMemRefFloatType = MemRefType::get(
      ArrayRef<int64_t>(inputDimension.begin(), inputDimension.end()),
      floatType);
  auto outputMemRefFloatType = MemRefType::get(
      ArrayRef<int64_t>(outputDimension.begin(), outputDimension.end()),
      floatType);

  // Populate host validation logic
  auto cpuConvFuncOp = createCPUConvolution(module, builder, genConfig);

  // Emit conv2d_host function call.
  auto cpuConvCallOp = builder.create<CallOp>(
      builder.getUnknownLoc(), cpuConvFuncOp,
      ValueRange{cpuFilterHostAllocOp, cpuInputHostAllocOp,
                 cpuOutputHostAllocOp});
  block->push_back(cpuConvCallOp);

  auto cpuResults = cpuOutputHostAllocOp;
  mlir::MemRefType dataTypeMemRefType, floatMemRefType;
  switch (convOpType) {
  case miopen::Conv2DOpType:
    cpuResults = cpuOutputHostAllocOp;
    floatMemRefType = outputMemRefFloatType;
    dataTypeMemRefType = MemRefType::get(
        ArrayRef<int64_t>(outputDimension.begin(), outputDimension.end()),
        dataType);
    break;
  case miopen::Conv2DBwdDataOpType:
    cpuResults = cpuInputHostAllocOp;
    floatMemRefType = inputMemRefFloatType;
    dataTypeMemRefType = MemRefType::get(
        ArrayRef<int64_t>(inputDimension.begin(), inputDimension.end()),
        dataType);
    break;
  case miopen::Conv2DBwdWeightOpType:
    cpuResults = cpuFilterHostAllocOp;
    floatMemRefType = filterMemRefFloatType;
    dataTypeMemRefType = MemRefType::get(
        ArrayRef<int64_t>(filterDimension.begin(), filterDimension.end()),
        dataType);
    break;
  }

  // mlir::Value cpuConvertedResults;
  auto cpuConvertedResults = cpuResults;
  if (dataType != builder.getF32Type())
    // Convert Cpu results of f32 to desired dataType
    cpuConvertedResults = getConvertedCpuResults(
        module, builder, block, cpuResults, dataTypeMemRefType);

  // Convert CPU results to f32 for printing
  auto printAllocOp = cpuConvertedResults;

  if (dataType != builder.getF32Type()) {
    auto allocOp =
        builder.create<AllocOp>(builder.getUnknownLoc(), floatMemRefType);
    block->push_back(allocOp);
    printAllocOp = allocOp;

    if (dataType == builder.getIntegerType(16)) { // bf16 only

      auto printUnkownSizeMemRefCastOp = builder.create<MemRefCastOp>(
          builder.getUnknownLoc(), printAllocOp, fiveDimUnknownSizeFloatType);
      block->push_back(printUnkownSizeMemRefCastOp);

      auto dataTypeMemRefCastOp = builder.create<MemRefCastOp>(
          builder.getUnknownLoc(), cpuConvertedResults,
          fiveDimUnknownSizeMemRefType);
      block->push_back(dataTypeMemRefCastOp);

      auto printMemConvertCallOp = builder.create<CallOp>(
          builder.getUnknownLoc(), mcpuMemCopy5DFuncOp,
          ValueRange{dataTypeMemRefCastOp, printUnkownSizeMemRefCastOp});
      block->push_back(printMemConvertCallOp);

    } else { // f16
      // Emit type conversion routine to convert every element to f32.
      auto convertResultFuncOp = createConvertTensor(
          module, builder, dataTypeMemRefType, floatMemRefType);
      auto convertResultCallOp =
          builder.create<CallOp>(builder.getUnknownLoc(), convertResultFuncOp,
                                 ValueRange{cpuConvertedResults, printAllocOp});
      block->push_back(convertResultCallOp);
    }
  }

  // Emit print function call.
  StringRef printMemRefFuncName = "print_memref_f32";
  auto unrankedMemRefType = UnrankedMemRefType::get(builder.getF32Type(), 0);
  auto printMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), printAllocOp, unrankedMemRefType);
  auto printMemRefFuncOp =
      makeFuncDecl(builder, printMemRefFuncName, {unrankedMemRefType}, {});
  auto printMemRefCallOp =
      builder.create<CallOp>(builder.getUnknownLoc(), printMemRefFuncOp,
                             ValueRange{printMemRefCastOp});
  module.push_back(printMemRefFuncOp);
  block->push_back(printMemRefCastOp);
  block->push_back(printMemRefCallOp);

  // Emit CPU dealloc.
  auto filterDeallocOp =
      builder.create<DeallocOp>(builder.getUnknownLoc(), cpuFilterAllocOp);
  auto inputDeallocOp =
      builder.create<DeallocOp>(builder.getUnknownLoc(), cpuInputAllocOp);
  auto outputDeallocOp =
      builder.create<DeallocOp>(builder.getUnknownLoc(), cpuOutputAllocOp);
  block->push_back(filterDeallocOp);
  block->push_back(inputDeallocOp);
  block->push_back(outputDeallocOp);

  if (dataType != builder.getF32Type()) {
    auto filterHostDeallocOp = builder.create<DeallocOp>(
        builder.getUnknownLoc(), cpuFilterHostAllocOp);
    auto inputHostDeallocOp =
        builder.create<DeallocOp>(builder.getUnknownLoc(), cpuInputHostAllocOp);
    auto outputHostDeallocOp = builder.create<DeallocOp>(
        builder.getUnknownLoc(), cpuOutputHostAllocOp);
    block->push_back(filterHostDeallocOp);
    block->push_back(inputHostDeallocOp);
    block->push_back(outputHostDeallocOp);

    auto cpuResultsDeallocOp =
        builder.create<DeallocOp>(builder.getUnknownLoc(), cpuConvertedResults);
    block->push_back(cpuResultsDeallocOp);

    auto printDeallocOp =
        builder.create<DeallocOp>(builder.getUnknownLoc(), printAllocOp);
    block->push_back(printDeallocOp);
  }

  auto returnOp =
      builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  block->push_back(returnOp);

  return success();
}

static LogicalResult populateKernelLaunchLogic(
    ModuleOp &module, OpBuilder &builder, MLIRContext &context,
    const SmallVector<std::string, 4> &kernels, const std::string entryPoint) {
  // Check if populate entry point exist.
  FuncOp theFunc;
  auto walkResult = module.walk([&](FuncOp funcOp) -> WalkResult {
    if (funcOp.getName() == entryPoint) {
      theFunc = funcOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (!walkResult.wasInterrupted()) {
    // do not fail for now. silent exit.
    // return failure();
    return success();
  }

  std::unordered_map<std::string, gpu::GPUFuncOp> gpuKernelMap;
  // Check if kernel to be launched exist.
  module.walk([&](gpu::GPUModuleOp gpuModule) -> WalkResult {
    module.walk([&](gpu::GPUFuncOp gpuFunc) -> WalkResult {
      auto fi = std::find(kernels.begin(), kernels.end(), gpuFunc.getName());
      if (fi != kernels.end()) {
        gpuKernelMap.emplace(*fi, gpuFunc);
      }
      return WalkResult::advance();
    });
    return WalkResult::advance();
  });

  if (gpuKernelMap.size() != kernels.size()) {
    // do not fail for now. silent exit.
    // return failure();
    return success();
  }

  Block *block = &(theFunc.getBody().front());
  block->clear();

  auto genLaunchCode = [&builder, &theFunc, &block](gpu::GPUFuncOp theGpuFunc) {
    auto blockSizeAttr = theGpuFunc->getAttr("block_size")
                             .template dyn_cast<IntegerAttr>()
                             .getInt();
    auto gridSizeAttr = theGpuFunc->getAttr("grid_size")
                            .template dyn_cast<IntegerAttr>()
                            .getInt();
    auto cstOne = builder.create<ConstantIndexOp>(builder.getUnknownLoc(), 1);
    auto cstBlockSize =
        builder.create<ConstantIndexOp>(builder.getUnknownLoc(), blockSizeAttr);
    auto cstGridSize =
        builder.create<ConstantIndexOp>(builder.getUnknownLoc(), gridSizeAttr);
    block->push_back(cstOne);
    block->push_back(cstBlockSize);
    block->push_back(cstGridSize);

    auto gpuLaunchFuncOp = builder.create<gpu::LaunchFuncOp>(
        builder.getUnknownLoc(), theGpuFunc,
        gpu::KernelDim3{cstGridSize, cstOne, cstOne},
        gpu::KernelDim3{cstBlockSize, cstOne, cstOne},
        ValueRange{theFunc.getArgument(0), theFunc.getArgument(1),
                   theFunc.getArgument(2)});
    gpuLaunchFuncOp->setAttr(
        "operand_segment_sizes",
        builder.getI32VectorAttr({static_cast<int32_t>(0),    // sync
                                  static_cast<int32_t>(1),    // gridX
                                  static_cast<int32_t>(1),    // gridY
                                  static_cast<int32_t>(1),    // gridZ
                                  static_cast<int32_t>(1),    // blockX
                                  static_cast<int32_t>(1),    // blockY
                                  static_cast<int32_t>(1),    // blockZ
                                  static_cast<int32_t>(3)})); // arg count

    block->push_back(gpuLaunchFuncOp);
  };

  for (auto gpuKernel : kernels) {
    genLaunchCode(gpuKernelMap[gpuKernel]);
  }

  auto returnOp = builder.create<ReturnOp>(builder.getUnknownLoc());
  block->push_back(returnOp);

  return success();
}

static void populateTuningPipeline(PassManager &pm,
                                   const std::string &perfConfig) {
  pm.addPass(mlir::miopen::createAffixTuningParametersPass(blockSize, gridSize,
                                                           perfConfig));
}

static void populateDefaultLoweringPipeline(PassManager &pm,
                                            const std::string &perfConfig) {
  // Passes for lowering MIOpen dialect.
  populateTuningPipeline(pm, perfConfig);
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep2Pass());
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep3Pass());
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep4Pass());
  pm.addPass(mlir::miopen::createLowerMIOpenOpsStep5Pass());
  pm.addPass(mlir::createLowerMIOpenOpsToGPUPass());
  // pm.addPass(mlir::createGpuKernelOutliningPass());

  // Passes for lowering linalg dialect.
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createLowerToCFGPass());
}

static LogicalResult runMLIRPasses(ModuleOp &module,
                                   mlir::PassPipelineCLParser &passPipeline,
                                   const std::string &perfConfig) {
  PassManager pm(module.getContext(), PassManager::Nesting::Implicit);
  applyPassManagerCLOptions(pm);

  // Set up lowering pipeline.
  bool toUseBuiltinPipeline = loweringWithBuiltinPipeline.getValue();
  if (toUseBuiltinPipeline) {
    StringRef pipeline = loweringTargetDialect.getValue();
    if (pipeline == "tuning") {
      // Set up the default lowering pipeline which goes down to affix tuning
      // parameters
      populateTuningPipeline(pm, perfConfig);
    } else if (pipeline == "gpu") {
      // Set up the default lowering pipeline which goes down to GPU dialect.
      populateDefaultLoweringPipeline(pm, perfConfig);
    } else if (pipeline == "rocdl") {
      // Set up the lowering pipeline which goes down to ROCDL dialect.
      populateDefaultLoweringPipeline(pm, perfConfig);
      pm.addPass(createLowerGpuOpsToROCDLOpsPass(/*indexBitWidth=*/32));
    }
  } else {
    auto errorHandler = [&](const Twine &msg) {
      emitError(UnknownLoc::get(module.getContext())) << msg;
      return failure();
    };

    // Use lowering pipeline specified at command line.
    if (failed(passPipeline.addToPipeline(pm, errorHandler)))
      return failure();
  }

  return pm.run(module);
}

int main(int argc, char **argv) {
  MLIRContext context;
  mlir::registerAllDialects(context.getDialectRegistry());
  context.loadDialect<miopen::MIOpenDialect, StandardOpsDialect,
                      scf::SCFDialect, AffineDialect>();
  mlir::registerAllPasses();
  mlir::registerMIOpenConversionPasses();
  miopen::registerPasses();
  InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipeline("", "compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR MIOpen Dialect driver\n");

  if (populateValidationWithGPU.getValue())
    populateValidation.setValue(true);

  OpBuilder builder(&context);
  ModuleOp module;

  std::string errorMessage;
  SourceMgr sourceMgr;
  OwningModuleRef moduleRef;
  if (useHostHarness.getValue()) {
    // Set up the input file.
    auto file = openInputFile(inputFilename, &errorMessage);
    if (!file) {
      llvm::errs() << errorMessage << "\n";
      exit(1);
    }

    // Parse the input file.
    sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
    moduleRef = parseSourceFile(sourceMgr, &context);
    if (!moduleRef) {
      llvm::errs() << "Parse host harness " << inputFilename << " failed.\n";
      exit(1);
    }
    module = moduleRef.get();
  } else {
    // Construct a new ModuleOp.
    module = ModuleOp::create(builder.getUnknownLoc());
  }

  verifyLayout();
  correctParameters();
  populateDefaults();

  auto convConfig = populateConvConfig.getValue();

  if (convConfig.empty() && failed(detectMissingArguments())) {
    exit(1);
  }

  Conv2dGenerator conv2dGenerator(
      arch.getValue(), perfConfig.getValue(), num_cu.getValue(),
      xdlopsV2.getValue(), operation.getValue(), tensorDataType.getValue(),
      dilationHeight.getValue(), dilationWidth.getValue(),
      strideHeight.getValue(), strideWidth.getValue(),
      paddingHeightLeft.getValue(), paddingHeightRight.getValue(),
      paddingWidthLeft.getValue(), paddingWidthRight.getValue(),
      filterLayout.getValue(), inputLayout.getValue(), outputLayout.getValue());

  if (!convConfig.empty()) {
    if (failed(conv2dGenerator.parseConvConfig(convConfig.c_str()))) {
      llvm::errs() << "Module population failed.\n";
      exit(1);
    }
  } else {
    LogicalResult parseSucceded = conv2dGenerator.parseConvDims(
        batchSize, groupSize, inputChannel, inputHeight, inputWidth,
        outputChannel, outputHeight, outputWidth, filterHeight, filterWidth);
    if (failed(parseSucceded)) {
      llvm::errs() << "Could not form a valid convolution\n";
      exit(1);
    }
  }

  const auto &genConfig = conv2dGenerator.getConfig();

  SmallVector<std::string, 4> kernels;

  // Populate the module.
  if (!populateCpuConvolution.getValue()) {
    if (genConfig.kernelId < 0) {
      // generate all sub-kernels, and get corresponding gemmId
      int kernelCount = conv2dGenerator.getKernelCount();
      auto knSize = genConfig.kernelName.size();
      std::string kernelBaseName = genConfig.kernelName.substr(0, knSize - 1);
      for (int i = 0; i < kernelCount; ++i) {
        std::string kName = kernelBaseName + std::to_string(i);
        conv2dGenerator.setKernelName(kName);
        if (failed(conv2dGenerator.genConvModule(module, builder, i))) {
          llvm::errs() << "Module population failed.\n";
          exit(1);
        }
        kernels.push_back(kName);
      }
    } else {
      // generate a specific kernel (kernel_id >= 0)
      if (failed(conv2dGenerator.genConvModule(module, builder))) {
        llvm::errs() << "Module population failed.\n";
        exit(1);
      }
      kernels.push_back(genConfig.kernelName);
    }
  }

  // Determine data type.
  mlir::Type dataType = conv2dGenerator.getDataType(builder);

  // populate host harness and host validation.
  if (populateValidation.getValue()) {
    if (failed(populateValidationLogic(module, builder, context, genConfig,
                                       dataType))) {
      llvm::errs() << "Host validation populated failed.\n";
      exit(1);
    }
  }

  // Populate the module for gpu validation.
  SmallVector<std::string, 4> kernels_v;
  if (populateValidationWithGPU.getValue() &&
      (genConfig.xdlops || dataType != builder.getF32Type())) {
    // use non-xdlops kernels to verify xdlops kernels
    if (genConfig.xdlops)
      conv2dGenerator.flipXdlops();
    // use f32 data type to verify non-f32 or xdlops f32 kernels
    conv2dGenerator.setDataType("f32");

    if (genConfig.kernelId < 0) {
      // generate all sub-kernels, and get corresponding gemmId
      int kernelCount = conv2dGenerator.getKernelCount();
      if (kernelCount > 1000) {
        llvm::errs() << "Populating gpu kernels for validation failed.\n";
        exit(1);
      }

      auto knSize = genConfig.kernelName.size();
      std::string kernelBaseName = genConfig.kernelName.substr(0, knSize - 1);
      for (int i = 0; i < kernelCount; ++i) {
        std::string kName = kernelBaseName + std::to_string(1000 + i);
        conv2dGenerator.setKernelName(kName);
        if (failed(conv2dGenerator.genConvModule(module, builder, i))) {
          llvm::errs() << "Module population failed.\n";
          exit(1);
        }
        kernels_v.push_back(kName);
      }
    } else {
      // generate a specific kernel (kernel_id >= 0)
      std::string kName = genConfig.kernelName + std::to_string(1000);
      conv2dGenerator.setKernelName(kName);
      if (failed(conv2dGenerator.genConvModule(module, builder))) {
        llvm::errs() << "Module population failed.\n";
        exit(1);
      }
      kernels_v.push_back(kName);
    }
  }

  // populate CPU convolution and print the results.
  if (populateCpuConvolution.getValue()) {
    if (failed(populateCpuConvolutionLogic(module, builder, context, genConfig,
                                           dataType))) {
      llvm::errs() << "Cpu Convolution populated failed.\n";
      exit(1);
    }
  }

  // Run MLIR passes with passed in tuning parameters
  if (failed(runMLIRPasses(module, passPipeline, genConfig.perfConfig))) {
    llvm::errs() << "Lowering failed.\n";
    exit(1);
  }

  // populate host logic.
  if (populateHostHarness.getValue()) {
    if (failed(populateHostHarnessLogic(module, builder, context, genConfig,
                                        dataType))) {
      llvm::errs() << "Host logic populated failed.\n";
      exit(1);
    }
  }

  // populate host launch logic.
  if (useHostHarness.getValue() || populateHostHarness.getValue() ||
      populateValidation.getValue()) {
    if (failed(populateKernelLaunchLogic(module, builder, context, kernels,
                                         populateEntryPoint.getValue()))) {
      llvm::errs() << "Host kernel launch logic populated failed.\n";
      exit(1);
    }
  }

  if (populateValidationWithGPU.getValue()) {
    if (failed(
            populateKernelLaunchLogic(module, builder, context, kernels_v,
                                      populateValidateEntryPoint.getValue()))) {
      llvm::errs() << "Host kernel launch logic populated failed.\n";
      exit(1);
    }
  }

  // Set up the output file.
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  module.print(output->os());
  output->keep();
  return 0;
}
