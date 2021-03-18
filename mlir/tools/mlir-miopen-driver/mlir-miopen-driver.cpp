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

#include "mlir/Dialect/MIOpen/Generator/Conv2dGenerator.h"
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
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

static cl::opt<std::string>
    operation("operation",
              cl::desc("Convolution operation, eg: conv2d, conv2d_bwd_data, "
                       "conv2d_bwd_weight..."),
              cl::value_desc("convolution flavor string"), cl::init("conv2d"));

static cl::opt<std::string>
    arch("arch",
         cl::desc("amdgpu architecture, eg: gfx803, gfx900, gfx906, gfx908"),
         cl::value_desc("GFX architecture string"), cl::init("gfx906"));

static cl::opt<int>
    num_cu("num_cu",
           cl::desc("Number of compute units, valid combinations include: "
                    "gfx803(36/64), gfx900(56/64), "
                    "gfx906(60/64), gfx908(120)"),
           cl::value_desc("compute unit value"), cl::init(64));

static cl::opt<std::string> filterLayout("fil_layout",
                                         cl::desc("Filter layout"),
                                         cl::value_desc("layout string"),
                                         cl::init("kcyx"));

static cl::opt<std::string> inputLayout("in_layout", cl::desc("Input layout"),
                                        cl::value_desc("layout string"),
                                        cl::init("nchw"));

static cl::opt<std::string> outputLayout("out_layout",
                                         cl::desc("Output layout"),
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

// padding width
static cl::opt<int> paddingWidth("padding_w", cl::desc("Padding width"),
                                 cl::value_desc("attribute value"),
                                 cl::init(0));

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

// lowering pipeline setup.
static cl::opt<bool> loweringWithDefaultPipeline(
    "c", cl::desc("To lower with default pipeline"),
    cl::value_desc("To lower with default pipeline"), cl::init(false));

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

static cl::opt<bool> printResultTensor(
    "pr", cl::desc("To print result tensor for verification"),
     cl::value_desc("To print result tensor for verification"), cl::init(false));

static cl::opt<bool> populateCpuConvolution(
    "prc", cl::desc("To run cpu conv2d and print results for verification"),
    cl::value_desc("To run cpu conv2d and print results for verification"),
    cl::init(false));

static cl::opt<int> blockSize("block_size", cl::desc("Block size"),
                              cl::value_desc("Block size"), cl::init(0));

static cl::opt<int> gridSize("grid_size", cl::desc("Grid size"),
                             cl::value_desc("Grid size"), cl::init(0));

// use XDLOPS
static cl::opt<bool> xdlopsV2("x2", cl::desc("To use XDLOPS V2 lowering pipeline"),
                             cl::value_desc("To use XDLOPS V2 lowering pipeline"),
                             cl::init(false));

// data type
static cl::opt<std::string> tensorDataType("t", cl::desc("Data type for convolution"),
                                           cl::value_desc("Data type for convolution"),
                                           cl::init("f32"));

static void populateDefaults() {
  if (populateDefaultValues == true) {
    if (xdlopsV2.getValue() == false) {
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
      paddingHeight.setValue(0);
      paddingWidth.setValue(0);
    } else {
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
      paddingHeight.setValue(0);
      paddingWidth.setValue(0);
      num_cu.setValue(120);
    }
    arch.setValue("gfx908");
  }

  auto getOutputDim = [](int64_t inputLen, int64_t filLen, int padLen,
                         int strideLen, int dilLen) {
    return (inputLen + 2 * padLen - (filLen - 1) * dilLen - 1) / strideLen + 1;
  };
  outputHeight.setValue(getOutputDim(
      inputHeight.getValue(), filterHeight.getValue(), paddingHeight.getValue(),
      strideHeight.getValue(), dilationHeight.getValue()));
  outputWidth.setValue(getOutputDim(
      inputWidth.getValue(), filterWidth.getValue(), paddingWidth.getValue(),
      strideWidth.getValue(), dilationWidth.getValue()));
}

static FuncOp makeFuncDecl(OpBuilder &builder, StringRef funcName,
                           TypeRange inputs, TypeRange results) {
  auto func = FuncOp::create(builder.getUnknownLoc(), funcName,
                             builder.getFunctionType(inputs, results));
  func.sym_visibilityAttr(builder.getStringAttr("private"));
  return func;
}

static AllocOp initializeCPUConvResult(OpBuilder &builder, Block *block,
                                       mlir::Type dataType,
                                       mlir::FuncOp &mcpuMemset4DFuncOp,
                                       mlir::MemRefType &resultMemRefType,
                                       mlir::Value &resultMemsetValue) {
  auto fourDimUnknownSizeMemRefType =
      MemRefType::get({-1, -1, -1, -1}, dataType);

  // Emit CPU alloc for the result tensor
  auto cpuResultHostAllocOp =
      builder.create<AllocOp>(builder.getUnknownLoc(), resultMemRefType);
  block->push_back(cpuResultHostAllocOp);

  // Emit memref cast
  auto cpuResultMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), cpuResultHostAllocOp,
      fourDimUnknownSizeMemRefType);
  block->push_back(cpuResultMemRefCastOp);

  // Populate initial values of the output tensor
  auto cpuResultMemsetOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mcpuMemset4DFuncOp,
      ValueRange{cpuResultMemRefCastOp, resultMemsetValue});
  block->push_back(cpuResultMemsetOp);

  return cpuResultHostAllocOp;
}

static FuncOp createConvertResult(ModuleOp &module, OpBuilder &builder,
                                  mlir::MemRefType &originalMemRefType,
                                  mlir::MemRefType &convertedMemRefType) {
  // Create print_result function
  auto convertResultFuncOp = FuncOp::create(
      builder.getUnknownLoc(), StringRef("convert_result"),
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
  auto dataType = originalMemRefType.getElementType();
  auto arguments = convertResultFuncOp.getArguments();
  mlir::Value sourceMemRef = arguments[0];
  mlir::Value convertedMemRef = arguments[1];
  mlir::Value sourceValue = innermostLoopBuilder.create<LoadOp>(
      builder.getUnknownLoc(), dataType, sourceMemRef, loopIVVector);
  mlir::Value convertedValue;
  if (dataType == builder.getF16Type()) {
    // Emit fpext.
    convertedValue = innermostLoopBuilder.create<FPExtOp>(
        builder.getUnknownLoc(), sourceValue, builder.getF32Type());
  } else if (dataType == builder.getIntegerType(16)) {
    // Treat I16 as BF16.
    // TBD: Implement proper conversion logic. Force cast for now.
    convertedValue = innermostLoopBuilder.create<SIToFPOp>(
        builder.getUnknownLoc(), sourceValue, builder.getF32Type());
  } else {
    convertedValue = sourceValue;
  }
  innermostLoopBuilder.create<StoreOp>(builder.getUnknownLoc(), convertedValue,
                                       convertedMemRef, loopIVVector);

  // Emit return op.
  auto returnOp =
      builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  block->push_back(returnOp);

  return convertResultFuncOp;
}

static FuncOp createCPUConvolution(ModuleOp &module, OpBuilder &builder,
                                   mlir::MemRefType &filterMemRefType,
                                   mlir::MemRefType &inputMemRefType,
                                   mlir::MemRefType &cpuOutputMemRefType) {
  // Create conv2d_host function
  auto cpuConvFuncOp = FuncOp::create(
      builder.getUnknownLoc(), StringRef("conv2d_host"),
      builder.getFunctionType(
          {filterMemRefType, inputMemRefType, cpuOutputMemRefType}, {}));
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
      builder.getUnknownLoc(), strideHeight.getValue(), intType);

  auto strideWidthConstantOp = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), strideWidth.getValue(), intType);

  auto paddingHeightConstantOp = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), paddingHeight.getValue(), intType);

  auto paddingWidthConstantOp = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), paddingWidth.getValue(), intType);

  auto dilationHeightConstantOp = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), dilationHeight.getValue(), intType);

  auto dilationWidthConstantOp = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), dilationWidth.getValue(), intType);

  cpuConvBlock->push_back(strideHeightConstantOp);
  cpuConvBlock->push_back(strideWidthConstantOp);
  cpuConvBlock->push_back(paddingHeightConstantOp);
  cpuConvBlock->push_back(paddingWidthConstantOp);
  cpuConvBlock->push_back(dilationHeightConstantOp);
  cpuConvBlock->push_back(dilationWidthConstantOp);

  // Emit ConstantIndex ops
  // %c_0 = constant 0 : index
  // %c_1 = constant 1 : index
  // %c_2 = constant 2 : index
  // %c_3 = constant 3 : index
  std::vector<ConstantIndexOp> indexOpVec;
  for (int i = 0; i < 4; i++) {
    auto indexOp = builder.create<ConstantIndexOp>(builder.getUnknownLoc(), i);
    cpuConvBlock->push_back(indexOp);
    indexOpVec.push_back(indexOp);
  }

  auto charType = builder.getIntegerType(8);
  // Emit Constant ops for letters used in layouts
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

  cpuConvBlock->push_back(kConstantOp);
  cpuConvBlock->push_back(cConstantOp);
  cpuConvBlock->push_back(yConstantOp);
  cpuConvBlock->push_back(xConstantOp);
  cpuConvBlock->push_back(nConstantOp);
  cpuConvBlock->push_back(hConstantOp);
  cpuConvBlock->push_back(wConstantOp);

  std::unordered_map<char, mlir::ConstantOp> layoutConstOps;
  layoutConstOps['k'] = kConstantOp;
  layoutConstOps['c'] = cConstantOp;
  layoutConstOps['y'] = yConstantOp;
  layoutConstOps['x'] = xConstantOp;
  layoutConstOps['n'] = nConstantOp;
  layoutConstOps['h'] = hConstantOp;
  layoutConstOps['w'] = wConstantOp;

  // %3   = alloca() : memref<4xi8>
  // %4   = alloca() : memref<4xi8>
  // %5   = alloca() : memref<4xi8>
  SmallVector<int64_t, 4> layoutVector({4});
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
  // store %k, %3[%c_0]: memref<4xi32>
  std::string fil_layout = filterLayout.getValue();
  std::string in_layout = inputLayout.getValue();
  std::string out_layout = outputLayout.getValue();
  for (int i = 0; i < 4; i++) {
    auto storeOp = builder.create<StoreOp>(
        builder.getUnknownLoc(), layoutConstOps[fil_layout[i]],
        filLayoutAllocOp, ValueRange{indexOpVec[i]});
    cpuConvBlock->push_back(storeOp);
  }

  for (int i = 0; i < 4; i++) {
    auto storeOp = builder.create<StoreOp>(
        builder.getUnknownLoc(), layoutConstOps[in_layout[i]], inLayoutAllocOp,
        ValueRange{indexOpVec[i]});
    cpuConvBlock->push_back(storeOp);
  }

  for (int i = 0; i < 4; i++) {
    auto storeOp = builder.create<StoreOp>(
        builder.getUnknownLoc(), layoutConstOps[out_layout[i]],
        outLayoutAllocOp, ValueRange{indexOpVec[i]});
    cpuConvBlock->push_back(storeOp);
  }

  // Emit memref_cast
  // %6 = memref_cast %3 : memref<4xi8> to memref<*xi8>
  // %7 = memref_cast %4 : memref<4xi8> to memref<*xi8>
  // %8 = memref_cast %5 : memref<4xi8> to memref<*xi8>
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

  if (operation.getValue() == "conv2d") {
    mcpuFuncName = "mcpuConv2d";
  } else if (operation.getValue() == "conv2d_bwd_data") {
    mcpuFuncName = "mcpuConv2dBwdData";
  } else if (operation.getValue() == "conv2d_bwd_weight") {
    mcpuFuncName = "mcpuConv2dBwdWeight";
  }

  // Emit cpu convolution function call op
  auto mcpuConv2dFuncOp =
      makeFuncDecl(builder, mcpuFuncName,
                   {unrankedMemRefType, unrankedMemRefType, unrankedMemRefType,
                    unrankedLayoutMemRefType, unrankedLayoutMemRefType,
                    unrankedLayoutMemRefType, intType, intType, intType,
                    intType, intType, intType},
                   {});

  auto mcpuConv2dCallOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mcpuConv2dFuncOp,
      ValueRange{filterMemRefCastOp, inputMemRefCastOp, outputMemRefCastOp,
                 filLayoutMemRefCastOp, inLayoutMemRefCastOp,
                 outLayoutMemRefCastOp, strideHeightConstantOp,
                 strideWidthConstantOp, paddingHeightConstantOp,
                 paddingWidthConstantOp, dilationHeightConstantOp,
                 dilationWidthConstantOp});

  module.push_back(mcpuConv2dFuncOp);
  cpuConvBlock->push_back(mcpuConv2dCallOp);

  // Emit return op
  auto cpuConvFuncOpReturnOp =
      builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  cpuConvBlock->push_back(cpuConvFuncOpReturnOp);

  return cpuConvFuncOp;
}

static FuncOp createVerifyFuncOp(ModuleOp &module, OpBuilder &builder,
                                 SmallVector<int64_t, 4> &outputDimension,
                                 mlir::AllocOp &cpuAllocOp,
                                 mlir::AllocOp &gpuAllocOp) {
  // Emit verify_results function call
  auto outputMemRefType = cpuAllocOp.getType();

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

  // scf.for %arg0 = %c0 to %c128 step %c1 {
  //  scf.for %arg1 = %c0 to %c30 step %c1 {
  //    scf.for %arg2 = %c0 to %c30 step %c1 {
  //      scf.for %arg3 = %c0 to %c128 step %c1 {
  //        %cpu_result = load %CpuResults[%arg0, %arg1, %arg2, %arg3] :
  //        memref<128x30x30x128xf32> %gpu_result = load %GpuRsults[%arg0,
  //        %arg1, %arg2, %arg3] : memref<128x30x30x128xf32> %cmp_result = cmpf
  //        "oeq", %cpu_result, %gpu_result : f32
  //
  //        scf.if %cmp_result {
  //        } else {
  //          store %c0_i32, %result[%c0] : memref<1xi32>
  //        }
  //      }
  //    }
  //  }
  //}
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

  auto cpuLoadOp = bt3.create<LoadOp>(builder.getUnknownLoc(),
                                      verifyResultsBlock->getArgument(0),
                                      ValueRange{iv0, iv1, iv2, iv3});
  auto gpuLoadOp = bt3.create<LoadOp>(builder.getUnknownLoc(),
                                      verifyResultsBlock->getArgument(1),
                                      ValueRange{iv0, iv1, iv2, iv3});
  auto cmpOp = bt3.create<CmpFOp>(builder.getUnknownLoc(), CmpFPredicate::UNE,
                                  cpuLoadOp, gpuLoadOp);
  auto ifOp = bt3.create<scf::IfOp>(builder.getUnknownLoc(), cmpOp, false);
  auto thenBody = ifOp.getThenBodyBuilder();

  auto storeOp0 =
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
                                   mlir::AllocOp &filterHostAllocOp,
                                   mlir::AllocOp &inputHostAllocOp,
                                   mlir::AllocOp &outputHostAllocOp) {
  auto filterMemRefType = filterHostAllocOp.getType();
  auto inputMemRefType = inputHostAllocOp.getType();
  auto outputMemRefType = outputHostAllocOp.getType();
  // Create gpu_conv function
  auto gpuConvFuncOp = FuncOp::create(
      builder.getUnknownLoc(), StringRef("gpu_conv"),
      builder.getFunctionType(
          {filterMemRefType, inputMemRefType, outputMemRefType}, {}));
  module.push_back(gpuConvFuncOp);
  // Emit gpu convolution logic.
  // Create a new block
  Block *gpuConvBlock = gpuConvFuncOp.addEntryBlock();

  // Emit memref_cast.
  auto fourDimUnknownSizeMemRefType =
      MemRefType::get({-1, -1, -1, -1}, dataType);

  auto filterMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), gpuConvBlock->getArgument(0),
      fourDimUnknownSizeMemRefType);
  auto inputMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), gpuConvBlock->getArgument(1),
      fourDimUnknownSizeMemRefType);
  auto outputMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), gpuConvBlock->getArgument(2),
      fourDimUnknownSizeMemRefType);
  gpuConvBlock->push_back(filterMemRefCastOp);
  gpuConvBlock->push_back(inputMemRefCastOp);
  gpuConvBlock->push_back(outputMemRefCastOp);

  // Emit GPU memory allocation function calls.
  StringRef gpuMemAllocFuncName;
  if (dataType == builder.getF32Type()) {
    gpuMemAllocFuncName = "mgpuMemAlloc4DFloat";
  } else if (dataType == builder.getF16Type()) {
    gpuMemAllocFuncName = "mgpuMemAlloc4DHalf";
  } else if (dataType == builder.getIntegerType(16)) {
    gpuMemAllocFuncName = "mgpuMemAlloc4DBF16";
  }
  auto mgpuMemAlloc4DFuncOp =
      makeFuncDecl(builder, gpuMemAllocFuncName, {fourDimUnknownSizeMemRefType},
                   {fourDimUnknownSizeMemRefType});

  module.push_back(mgpuMemAlloc4DFuncOp);

  auto filterGpuAllocOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mgpuMemAlloc4DFuncOp,
                             ValueRange{filterMemRefCastOp});
  auto inputGpuAllocOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mgpuMemAlloc4DFuncOp,
                             ValueRange{inputMemRefCastOp});
  auto outputGpuAllocOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mgpuMemAlloc4DFuncOp,
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
    gpuMemCopyFuncName = "mgpuMemCopy4DFloat";
  } else if (dataType == builder.getF16Type()) {
    gpuMemCopyFuncName = "mgpuMemCopy4DHalf";
  } else if (dataType == builder.getIntegerType(16)) {
    gpuMemCopyFuncName = "mgpuMemCopy4DBF16";
  }
  auto mgpuMemCopy4DFuncOp =
      makeFuncDecl(builder, gpuMemCopyFuncName,
                   {fourDimUnknownSizeMemRefType, fourDimUnknownSizeMemRefType,
                    builder.getIntegerType(32)},
                   {});
  module.push_back(mgpuMemCopy4DFuncOp);

  auto filterCpuToGpuCopyOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mgpuMemCopy4DFuncOp,
      ValueRange{filterMemRefCastOp, filterGpuAllocOp.getResult(0),
                 oneConstantI32Op});
  auto inputCpuToGpuCopyOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mgpuMemCopy4DFuncOp,
      ValueRange{inputMemRefCastOp, inputGpuAllocOp.getResult(0),
                 oneConstantI32Op});
  auto outputCpuToGpuCopyOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mgpuMemCopy4DFuncOp,
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
      builder.getUnknownLoc(), populateEntryPoint,
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

  // Emit mgpuMemCopy4DFloat function call.
  mlir::Value resultGpuValue, resultCpuValue;
  if (operation.getValue() == "conv2d") {
    resultGpuValue = outputGpuAllocOp.getResult(0);
    resultCpuValue = outputMemRefCastOp;
  } else if (operation.getValue() == "conv2d_bwd_data") {
    resultGpuValue = inputGpuAllocOp.getResult(0);
    resultCpuValue = inputMemRefCastOp;
  } else if (operation.getValue() == "conv2d_bwd_weight") {
    resultGpuValue = filterGpuAllocOp.getResult(0);
    resultCpuValue = filterMemRefCastOp;
  }
  auto outputGpuToCpuCopyOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mgpuMemCopy4DFuncOp,
                             ValueRange{resultGpuValue, resultCpuValue,
                                        twoConstantI32Op});
  gpuConvBlock->push_back(outputGpuToCpuCopyOp);

  // Emit GPU memory deallocation function calls.
  StringRef gpuMemDeallocFuncName;
  if (dataType == builder.getF32Type()) {
    gpuMemDeallocFuncName = "mgpuMemDealloc4DFloat";
  } else if (dataType == builder.getF16Type()) {
    gpuMemDeallocFuncName = "mgpuMemDealloc4DHalf";
  } else if (dataType == builder.getIntegerType(16)) {
    gpuMemDeallocFuncName = "mgpuMemDealloc4DBF16";
  }
  auto mgpuMemDealloc4DFuncOp = makeFuncDecl(
      builder, gpuMemDeallocFuncName, {fourDimUnknownSizeMemRefType}, {});
  module.push_back(mgpuMemDealloc4DFuncOp);

  auto filterGpuDeallocOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mgpuMemDealloc4DFuncOp,
                             ValueRange{filterMemRefCastOp});
  auto inputGpuDeallocOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mgpuMemDealloc4DFuncOp,
                             ValueRange{inputMemRefCastOp});
  auto outputGpuDeallocOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mgpuMemDealloc4DFuncOp,
                             ValueRange{outputMemRefCastOp});
  gpuConvBlock->push_back(filterGpuDeallocOp);
  gpuConvBlock->push_back(inputGpuDeallocOp);
  gpuConvBlock->push_back(outputGpuDeallocOp);

  auto returnOp =
      builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  gpuConvBlock->push_back(returnOp);

  return gpuConvFuncOp;
}

static LogicalResult populateHostHarnessLogic(
    ModuleOp &module, OpBuilder &builder, MLIRContext &context,
    const SmallVector<int64_t, 4> &filterDimension,
    const SmallVector<int64_t, 4> &inputDimension,
    const SmallVector<int64_t, 4> &outputDimension, mlir::Type dataType) {
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
  auto fourDimUnknownSizeMemRefType =
      MemRefType::get({-1, -1, -1, -1}, dataType);

  // Determine types of memref to be printed out.
  // Forward convolution: output tensor.
  // Backward data convolution: input tensor.
  // Backward weight convolution: filter tensor.
  MemRefType printMemRefType;
  if (operation.getValue() == "conv2d") {
    printMemRefType = MemRefType::get(
        ArrayRef<int64_t>(outputDimension.begin(), outputDimension.end()),
        builder.getF32Type());
  } else if (operation.getValue() == "conv2d_bwd_data") {
    printMemRefType = MemRefType::get(
        ArrayRef<int64_t>(inputDimension.begin(), inputDimension.end()),
        builder.getF32Type());
  } else if (operation.getValue() == "conv2d_bwd_weight") {
    printMemRefType = MemRefType::get(
        ArrayRef<int64_t>(filterDimension.begin(), filterDimension.end()),
        builder.getF32Type());
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

  // Emit CPU alloc for memref to be printed out.
  auto printHostAllocOp =
      builder.create<AllocOp>(builder.getUnknownLoc(), printMemRefType);
  block->push_back(printHostAllocOp);

  // Emit memref_cast.
  auto filterMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), filterHostAllocOp, fourDimUnknownSizeMemRefType);
  auto inputMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), inputHostAllocOp, fourDimUnknownSizeMemRefType);
  auto outputMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), outputHostAllocOp, fourDimUnknownSizeMemRefType);
  block->push_back(filterMemRefCastOp);
  block->push_back(inputMemRefCastOp);
  block->push_back(outputMemRefCastOp);

  auto getOneConstOp = [&]() {
    if (dataType == builder.getIntegerType(16)) {
      const ushort one = float_to_bfloat16(1.0);
      return builder.create<ConstantOp>(builder.getUnknownLoc(), dataType,
                                        builder.getI16IntegerAttr(one));
    } else {
      return builder.create<ConstantOp>(builder.getUnknownLoc(), dataType,
                                        builder.getFloatAttr(dataType, 1.0));
    }
  };
  auto getZeroConstOp = [&]() {
    if (dataType == builder.getIntegerType(16)) {
      const ushort zero = float_to_bfloat16(0.0);
      return builder.create<ConstantOp>(builder.getUnknownLoc(), dataType,
                                        builder.getI16IntegerAttr(zero));
    } else {
      return builder.create<ConstantOp>(builder.getUnknownLoc(), dataType,
                                        builder.getFloatAttr(dataType, 0.0));
    }
  };

  auto oneConstantFloatOp = getOneConstOp();
  auto zeroConstantFloatOp = getZeroConstOp();
  block->push_back(oneConstantFloatOp);
  block->push_back(zeroConstantFloatOp);

  // Emit CPU memset function calls.
  StringRef memsetFuncName;
  if (dataType == builder.getF32Type()) {
    memsetFuncName = "mcpuMemset4DFloat";
  } else if (dataType == builder.getF16Type()) {
    memsetFuncName = "mcpuMemset4DHalf";
  } else if (dataType == builder.getIntegerType(16)) {
    memsetFuncName = "mcpuMemset4DBF16";
  }

  auto mcpuMemset4DFuncOp = makeFuncDecl(
      builder, memsetFuncName, {fourDimUnknownSizeMemRefType, dataType}, {});
  module.push_back(mcpuMemset4DFuncOp);

  // Populate initial values.
  mlir::Value filterMemsetValue, inputMemsetValue, outputMemsetValue;
  if (operation.getValue() == "conv2d") {
    filterMemsetValue = oneConstantFloatOp;
    inputMemsetValue = oneConstantFloatOp;
    outputMemsetValue = zeroConstantFloatOp;
  } else if (operation.getValue() == "conv2d_bwd_data") {
    filterMemsetValue = oneConstantFloatOp;
    inputMemsetValue = zeroConstantFloatOp;
    outputMemsetValue = oneConstantFloatOp;
  } else if (operation.getValue() == "conv2d_bwd_weight") {
    filterMemsetValue = zeroConstantFloatOp;
    inputMemsetValue = oneConstantFloatOp;
    outputMemsetValue = oneConstantFloatOp;
  }
  auto filterCpuMemsetOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mcpuMemset4DFuncOp,
                             ValueRange{filterMemRefCastOp, filterMemsetValue});
  auto inputCpuMemsetOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mcpuMemset4DFuncOp,
                             ValueRange{inputMemRefCastOp, inputMemsetValue});
  auto outputCpuMemsetOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mcpuMemset4DFuncOp,
                             ValueRange{outputMemRefCastOp, outputMemsetValue});
  block->push_back(filterCpuMemsetOp);
  block->push_back(inputCpuMemsetOp);
  block->push_back(outputCpuMemsetOp);

  // launch gpu_conv
  auto gpuConvFuncOp =
      launchGPUConvolution(module, builder, dataType, filterHostAllocOp,
                           inputHostAllocOp, outputHostAllocOp);

  auto gpuConvCallOp = builder.create<CallOp>(
      builder.getUnknownLoc(), gpuConvFuncOp,
      ValueRange{filterHostAllocOp, inputHostAllocOp, outputHostAllocOp});
  block->push_back(gpuConvCallOp);

  mlir::Value resultCpuValue, resultOriginalCpuValue;
  mlir::MemRefType resultOriginalCpuType;
  if (operation.getValue() == "conv2d") {
    resultCpuValue = outputMemRefCastOp;
    resultOriginalCpuValue = outputHostAllocOp;
    resultOriginalCpuType = outputMemRefType;
  } else if (operation.getValue() == "conv2d_bwd_data") {
    resultCpuValue = inputMemRefCastOp;
    resultOriginalCpuValue = inputHostAllocOp;
    resultOriginalCpuType = inputMemRefType;
  } else if (operation.getValue() == "conv2d_bwd_weight") {
    resultCpuValue = filterMemRefCastOp;
    resultOriginalCpuValue = filterHostAllocOp;
    resultOriginalCpuType = filterMemRefType;
  }

  // Print the result if be specified.
  if (printResultTensor.getValue()) {
    // Emit type conversion routine to convert every element to f32.
    mlir::Value printHostValue = printHostAllocOp;
    auto convertResultFuncOp = createConvertResult(
        module, builder, resultOriginalCpuType, printMemRefType);
    auto convertResultCallOp = builder.create<CallOp>(
        builder.getUnknownLoc(), convertResultFuncOp,
        ValueRange{resultOriginalCpuValue, printHostAllocOp});
    block->push_back(convertResultCallOp);

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

  auto printHostDeallocOp =
      builder.create<DeallocOp>(builder.getUnknownLoc(), printHostAllocOp);
  block->push_back(printHostDeallocOp);

  auto returnOp =
      builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  block->push_back(returnOp);

  return success();
}

static LogicalResult populateValidationLogic(
    ModuleOp &module, OpBuilder &builder, MLIRContext &context,
    SmallVector<int64_t, 4> &filterDimension,
    SmallVector<int64_t, 4> &inputDimension,
    SmallVector<int64_t, 4> &outputDimension, mlir::Type dataType) {
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
  auto fourDimUnknownSizeMemRefType =
      MemRefType::get({-1, -1, -1, -1}, dataType);

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
      builder.getUnknownLoc(), filterHostAllocOp, fourDimUnknownSizeMemRefType);
  auto inputMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), inputHostAllocOp, fourDimUnknownSizeMemRefType);
  auto outputMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), outputHostAllocOp, fourDimUnknownSizeMemRefType);
  block->push_back(filterMemRefCastOp);
  block->push_back(inputMemRefCastOp);
  block->push_back(outputMemRefCastOp);

  auto getOneConstOp = [&]() {
    if (dataType == builder.getIntegerType(16)) {
      const ushort one = float_to_bfloat16(1.0);
      return builder.create<ConstantOp>(builder.getUnknownLoc(), dataType,
                                        builder.getI16IntegerAttr(one));
    } else {
      return builder.create<ConstantOp>(builder.getUnknownLoc(), dataType,
                                        builder.getFloatAttr(dataType, 1.0));
    }
  };
  auto getZeroConstOp = [&]() {
    if (dataType == builder.getIntegerType(16)) {
      const ushort zero = float_to_bfloat16(0.0);
      return builder.create<ConstantOp>(builder.getUnknownLoc(), dataType,
                                        builder.getI16IntegerAttr(zero));
    } else {
      return builder.create<ConstantOp>(builder.getUnknownLoc(), dataType,
                                        builder.getFloatAttr(dataType, 0.0));
    }
  };

  auto oneConstantFloatOp = getOneConstOp();
  auto zeroConstantFloatOp = getZeroConstOp();
  block->push_back(oneConstantFloatOp);
  block->push_back(zeroConstantFloatOp);

  // Emit CPU memset function calls.
  StringRef memsetFuncName;
  if (dataType == builder.getF32Type()) {
    memsetFuncName = "mcpuMemset4DFloat";
  } else if (dataType == builder.getF16Type()) {
    memsetFuncName = "mcpuMemset4DHalf";
  } else if (dataType == builder.getIntegerType(16)) {
    memsetFuncName = "mcpuMemset4DBF16";
  }

  auto mcpuMemset4DFuncOp = makeFuncDecl(
      builder, memsetFuncName, {fourDimUnknownSizeMemRefType, dataType}, {});
  module.push_back(mcpuMemset4DFuncOp);

  // Populate initial values.
  mlir::Value filterMemsetValue, inputMemsetValue, outputMemsetValue;
  if (operation.getValue() == "conv2d") {
    filterMemsetValue = oneConstantFloatOp;
    inputMemsetValue = oneConstantFloatOp;
    outputMemsetValue = zeroConstantFloatOp;
  } else if (operation.getValue() == "conv2d_bwd_data") {
    filterMemsetValue = oneConstantFloatOp;
    inputMemsetValue = zeroConstantFloatOp;
    outputMemsetValue = oneConstantFloatOp;
  } else if (operation.getValue() == "conv2d_bwd_weight") {
    filterMemsetValue = zeroConstantFloatOp;
    inputMemsetValue = oneConstantFloatOp;
    outputMemsetValue = oneConstantFloatOp;
  }
  auto filterCpuMemsetOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mcpuMemset4DFuncOp,
                             ValueRange{filterMemRefCastOp, filterMemsetValue});
  auto inputCpuMemsetOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mcpuMemset4DFuncOp,
                             ValueRange{inputMemRefCastOp, inputMemsetValue});
  auto outputCpuMemsetOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mcpuMemset4DFuncOp,
                             ValueRange{outputMemRefCastOp, outputMemsetValue});
  block->push_back(filterCpuMemsetOp);
  block->push_back(inputCpuMemsetOp);
  block->push_back(outputCpuMemsetOp);

  // Populate host harness logic
  auto gpuConvFuncOp =
      launchGPUConvolution(module, builder, dataType, filterHostAllocOp,
                           inputHostAllocOp, outputHostAllocOp);

  auto gpuConvCallOp = builder.create<CallOp>(
      builder.getUnknownLoc(), gpuConvFuncOp,
      ValueRange{filterHostAllocOp, inputHostAllocOp, outputHostAllocOp});
  block->push_back(gpuConvCallOp);

  // create f32 data
  auto getFloatDataFromBF16 = [&](mlir::MemRefCastOp &memRefCastOp) {
    // alloc new memory for verify function
    auto floatType = builder.getF32Type();
    auto verifyMemRefType = MemRefType::get(
        ArrayRef<int64_t>(outputDimension.begin(), outputDimension.end()),
        floatType);

    auto verifyHostAllocOp =
        builder.create<AllocOp>(builder.getUnknownLoc(), verifyMemRefType);
    block->push_back(verifyHostAllocOp);

    auto filterMemRefForBf16Type = MemRefType::get(
        ArrayRef<int64_t>(filterDimension.begin(), filterDimension.end()),
        floatType);
    auto inputMemRefForBf16Type = MemRefType::get(
        ArrayRef<int64_t>(inputDimension.begin(), inputDimension.end()),
        floatType);
    auto outputMemRefForBf16Type = MemRefType::get(
        ArrayRef<int64_t>(outputDimension.begin(), outputDimension.end()),
        floatType);

    auto unknownSizeMemRefFloatType =
        MemRefType::get({-1, -1, -1, -1}, floatType);

    auto verifyUnkownSizeMemRefCastOp = builder.create<MemRefCastOp>(
        builder.getUnknownLoc(), verifyHostAllocOp, unknownSizeMemRefFloatType);
    block->push_back(verifyUnkownSizeMemRefCastOp);

    auto cpuMemConvertOp = FuncOp::create(
        builder.getUnknownLoc(), "mcpuMemBF16ConvertFloat",
        builder.getFunctionType(
            {fourDimUnknownSizeMemRefType, unknownSizeMemRefFloatType}, {}));
    module.push_back(cpuMemConvertOp);

    auto verifyMemConvertCallOp = builder.create<CallOp>(
        builder.getUnknownLoc(), cpuMemConvertOp,
        ValueRange{memRefCastOp, verifyUnkownSizeMemRefCastOp});
    block->push_back(verifyMemConvertCallOp);

    return verifyHostAllocOp;
  };

  mlir::AllocOp gpuResults;
  mlir::AllocOp gpuResultsBf16;
  if (operation.getValue() == "conv2d") {
    if (builder.getIntegerType(16) == dataType) {
      gpuResultsBf16 = getFloatDataFromBF16(outputMemRefCastOp);
    } else
      gpuResults = outputHostAllocOp;
  } else if (operation.getValue() == "conv2d_bwd_data") {
    gpuResults = inputHostAllocOp;
    // Reset the input tensor
    auto inputCpuMemsetOp =
        builder.create<CallOp>(builder.getUnknownLoc(), mcpuMemset4DFuncOp,
                               ValueRange{inputMemRefCastOp, inputMemsetValue});
    block->push_back(inputCpuMemsetOp);
  } else if (operation.getValue() == "conv2d_bwd_weight") {
    gpuResults = filterHostAllocOp;
    // Reset the filter tensor
    auto filterCpuMemsetOp = builder.create<CallOp>(
        builder.getUnknownLoc(), mcpuMemset4DFuncOp,
        ValueRange{filterMemRefCastOp, filterMemsetValue});
    block->push_back(filterCpuMemsetOp);
  }

  mlir::AllocOp cpuResults;
  mlir::AllocOp filterHostForBf16AllocOp, inputHostForBf16AllocOp,
      cpuOutputHostForBf16AllocOp;
  mlir::MemRefType filterMemRefForBf16Type, inputMemRefForBf16Type,
      outputMemRefForBf16Type;
  if (dataType == builder.getIntegerType(16)) {
    mlir::Type dataTypeForBf16 = builder.getF32Type();
    auto fourDimUnknownSizeMemRefForBf16Type =
        MemRefType::get({-1, -1, -1, -1}, dataTypeForBf16);

    filterMemRefForBf16Type = MemRefType::get(
        ArrayRef<int64_t>(filterDimension.begin(), filterDimension.end()),
        dataTypeForBf16);

    inputMemRefForBf16Type = MemRefType::get(
        ArrayRef<int64_t>(inputDimension.begin(), inputDimension.end()),
        dataTypeForBf16);
    outputMemRefForBf16Type = MemRefType::get(
        ArrayRef<int64_t>(outputDimension.begin(), outputDimension.end()),
        dataTypeForBf16);

    filterHostForBf16AllocOp = builder.create<AllocOp>(builder.getUnknownLoc(),
                                                       filterMemRefForBf16Type);
    block->push_back(filterHostForBf16AllocOp);
    inputHostForBf16AllocOp = builder.create<AllocOp>(builder.getUnknownLoc(),
                                                      inputMemRefForBf16Type);
    block->push_back(inputHostForBf16AllocOp);

    auto filterMemRefForBf16CastOp = builder.create<MemRefCastOp>(
        builder.getUnknownLoc(), filterHostForBf16AllocOp,
        fourDimUnknownSizeMemRefForBf16Type);
    block->push_back(filterMemRefForBf16CastOp);
    auto inputMemRefForBf16CastOp = builder.create<MemRefCastOp>(
        builder.getUnknownLoc(), inputHostForBf16AllocOp,
        fourDimUnknownSizeMemRefForBf16Type);
    block->push_back(inputMemRefForBf16CastOp);

    // bf16 1.0,change to float 32 and compute
    const ushort oneForBf16 = float_to_bfloat16(1.0);
    const float oneBf16 = bfloat16_to_float(oneForBf16);
    auto oneConstantFloatForBf16Op = builder.create<ConstantOp>(
        builder.getUnknownLoc(), dataTypeForBf16,
        builder.getFloatAttr(dataTypeForBf16, oneBf16));

    const ushort zeroForBf16 = float_to_bfloat16(0.0);
    const float zeroBf16 = bfloat16_to_float(zeroForBf16);
    auto zeroConstantFloatForBf16Op = builder.create<ConstantOp>(
        builder.getUnknownLoc(), dataTypeForBf16,
        builder.getFloatAttr(dataTypeForBf16, zeroBf16));

    auto oneConstantForBf16FloatOp = oneConstantFloatForBf16Op;
    auto zeroConstantForBf16FloatOp = zeroConstantFloatForBf16Op;
    block->push_back(oneConstantForBf16FloatOp);
    block->push_back(zeroConstantForBf16FloatOp);

    auto mcpuMemset4DForBf16FuncOp = FuncOp::create(
        builder.getUnknownLoc(), "mcpuMemset4DFloat",
        builder.getFunctionType(
            {fourDimUnknownSizeMemRefForBf16Type, dataTypeForBf16}, {}));
    module.push_back(mcpuMemset4DForBf16FuncOp);

    mlir::Value filterMemsetForBf16Value, inputMemsetForBf16Value,
        cpuOutputMemsetForBf16Value;
    ;

    if (operation.getValue() == "conv2d") {
      filterMemsetForBf16Value = oneConstantForBf16FloatOp;
      inputMemsetForBf16Value = oneConstantForBf16FloatOp;
      //    outputMemsetValue = zeroConstantFloatOp;
      cpuOutputMemsetForBf16Value = zeroConstantForBf16FloatOp;
    } else if (operation.getValue() == "conv2d_bwd_data") {
      filterMemsetForBf16Value = oneConstantForBf16FloatOp;
      inputMemsetForBf16Value = zeroConstantForBf16FloatOp;
      //    outputMemsetValue = oneConstantFloatOp;
      cpuOutputMemsetForBf16Value = oneConstantForBf16FloatOp;
    } else if (operation.getValue() == "conv2d_bwd_weight") {
      filterMemsetForBf16Value = zeroConstantForBf16FloatOp;
      inputMemsetForBf16Value = oneConstantForBf16FloatOp;
      //    outputMemsetValue = oneConstantFloatOp;
      cpuOutputMemsetForBf16Value = oneConstantForBf16FloatOp;
    }

    auto filterCpuMemsetForBf16Op = builder.create<CallOp>(
        builder.getUnknownLoc(), mcpuMemset4DForBf16FuncOp,
        ValueRange{filterMemRefForBf16CastOp, filterMemsetForBf16Value});

    auto inputCpuMemsetForBf16Op = builder.create<CallOp>(
        builder.getUnknownLoc(), mcpuMemset4DForBf16FuncOp,
        ValueRange{inputMemRefForBf16CastOp, inputMemsetForBf16Value});
    block->push_back(filterCpuMemsetForBf16Op);

    block->push_back(inputCpuMemsetForBf16Op);
    // Emit CPU alloc for CPU convolution
    cpuOutputHostForBf16AllocOp = builder.create<AllocOp>(
        builder.getUnknownLoc(), outputMemRefForBf16Type);
    block->push_back(cpuOutputHostForBf16AllocOp);

    // Emit memref cast
    auto cpuOutputMemRefForBf16CastOp = builder.create<MemRefCastOp>(
        builder.getUnknownLoc(), cpuOutputHostForBf16AllocOp,
        fourDimUnknownSizeMemRefForBf16Type);
    block->push_back(cpuOutputMemRefForBf16CastOp);

    // Populate initial values
    auto cpuOutputCpuMemsetForBf16Op = builder.create<CallOp>(
        builder.getUnknownLoc(), mcpuMemset4DForBf16FuncOp,
        ValueRange{cpuOutputMemRefForBf16CastOp, cpuOutputMemsetForBf16Value});
    block->push_back(cpuOutputCpuMemsetForBf16Op);

  } // i16 end

  mlir::AllocOp cpuFilterHostAllocOp, cpuInputHostAllocOp, cpuOutputHostAllocOp;
  if (dataType == builder.getIntegerType(16)) {
    cpuFilterHostAllocOp = filterHostForBf16AllocOp;
    cpuInputHostAllocOp = inputHostForBf16AllocOp;
    cpuOutputHostAllocOp = cpuOutputHostForBf16AllocOp;

    auto cpuConvFuncOp =
        createCPUConvolution(module, builder, filterMemRefForBf16Type,
                             inputMemRefForBf16Type, outputMemRefForBf16Type);

    // Emit conv2d_host function call.
    auto cpuConvForBf16CallOp = builder.create<CallOp>(
        builder.getUnknownLoc(), cpuConvFuncOp,
        ValueRange{cpuFilterHostAllocOp, cpuInputHostAllocOp,
                   cpuOutputHostAllocOp});
    block->push_back(cpuConvForBf16CallOp);

  } else {
    if (operation.getValue() == "conv2d") {
      cpuFilterHostAllocOp = filterHostAllocOp;
      cpuInputHostAllocOp = inputHostAllocOp;
      cpuOutputHostAllocOp =
          initializeCPUConvResult(builder, block, dataType, mcpuMemset4DFuncOp,
                                  outputMemRefType, outputMemsetValue);
    } else if (operation.getValue() == "conv2d_bwd_data") {
      cpuFilterHostAllocOp = filterHostAllocOp;
      cpuOutputHostAllocOp = outputHostAllocOp;
      cpuInputHostAllocOp =
          initializeCPUConvResult(builder, block, dataType, mcpuMemset4DFuncOp,
                                  inputMemRefType, inputMemsetValue);
    } else if (operation.getValue() == "conv2d_bwd_weight") {
      cpuInputHostAllocOp = inputHostAllocOp;
      cpuOutputHostAllocOp = outputHostAllocOp;
      cpuFilterHostAllocOp =
          initializeCPUConvResult(builder, block, dataType, mcpuMemset4DFuncOp,
                                  filterMemRefType, filterMemsetValue);
    }

    // Populate host validation logic, bf16 and others should not the same
    auto cpuConvFuncOp = createCPUConvolution(
        module, builder, filterMemRefType, inputMemRefType, outputMemRefType);

    // Emit conv2d_host function call.
    auto cpuConvCallOp = builder.create<CallOp>(
        builder.getUnknownLoc(), cpuConvFuncOp,
        ValueRange{cpuFilterHostAllocOp, cpuInputHostAllocOp,
                   cpuOutputHostAllocOp});
    block->push_back(cpuConvCallOp);
  }

  if (dataType == builder.getIntegerType(16)) {
    if (operation.getValue() == "conv2d") {
      cpuResults = cpuOutputHostForBf16AllocOp;
    } else if (operation.getValue() == "conv2d_bwd_data") {
      cpuResults = inputHostForBf16AllocOp;
    } else if (operation.getValue() == "conv2d_bwd_weight") {
      cpuResults = filterHostForBf16AllocOp;
    }
  } else {
    if (operation.getValue() == "conv2d") {
      cpuResults = cpuOutputHostAllocOp;
    } else if (operation.getValue() == "conv2d_bwd_data") {
      cpuResults = cpuInputHostAllocOp;
    } else if (operation.getValue() == "conv2d_bwd_weight") {
      cpuResults = cpuFilterHostAllocOp;
    }
  }

  mlir::FuncOp verifyFuncOp;
  if (operation.getValue() == "conv2d") {
    verifyFuncOp = createVerifyFuncOp(module, builder, outputDimension,
                                      cpuResults, gpuResults);
  } else if (operation.getValue() == "conv2d_bwd_data") {
    verifyFuncOp = createVerifyFuncOp(module, builder, inputDimension,
                                      cpuResults, gpuResults);
  } else if (operation.getValue() == "conv2d_bwd_weight") {
    verifyFuncOp = createVerifyFuncOp(module, builder, filterDimension,
                                      cpuResults, gpuResults);
  }

  // Compare the results
  auto verifyCallOp =
      builder.create<CallOp>(builder.getUnknownLoc(), verifyFuncOp,
                             ValueRange{cpuResults, gpuResults});
  block->push_back(verifyCallOp);

  // Emit CPU dealloc.
  auto filterHostDeallocOp =
      builder.create<DeallocOp>(builder.getUnknownLoc(), filterHostAllocOp);
  auto inputHostDeallocOp =
      builder.create<DeallocOp>(builder.getUnknownLoc(), inputHostAllocOp);
  auto outputHostDeallocOp =
      builder.create<DeallocOp>(builder.getUnknownLoc(), outputHostAllocOp);
  auto cpuResultsDeallocOp =
      builder.create<DeallocOp>(builder.getUnknownLoc(), cpuResults);
  block->push_back(filterHostDeallocOp);
  block->push_back(inputHostDeallocOp);
  block->push_back(outputHostDeallocOp);
  block->push_back(cpuResultsDeallocOp);

  if (dataType == builder.getIntegerType(16)) {
    auto filterHostForBf16DeallocOp = builder.create<DeallocOp>(
        builder.getUnknownLoc(), filterHostForBf16AllocOp);
    auto inputHostForBf16DeallocOp = builder.create<DeallocOp>(
        builder.getUnknownLoc(), inputHostForBf16AllocOp);
    auto cpuOutputHostForBf16DeallocOp = builder.create<DeallocOp>(
        builder.getUnknownLoc(), cpuOutputHostForBf16AllocOp);

    block->push_back(filterHostForBf16DeallocOp);
    block->push_back(inputHostForBf16DeallocOp);
    block->push_back(cpuOutputHostForBf16DeallocOp);
  }

  auto returnOp =
      builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  block->push_back(returnOp);

  return success();
}

static LogicalResult
populateCpuConvolutionLogic(ModuleOp &module, OpBuilder &builder,
                            MLIRContext &context,
                            SmallVector<int64_t, 4> &filterDimension,
                            SmallVector<int64_t, 4> &inputDimension,
                            SmallVector<int64_t, 4> &outputDimension) {
  // Construct main function.
  auto func = FuncOp::create(builder.getUnknownLoc(), "main",
                             builder.getFunctionType({}, {}));
  module.push_back(func);

  // Construct a new Block.
  Block *block = func.addEntryBlock();

  // Only produce convolution logic for F32 type
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

  auto fourDimUnknownSizeMemRefType =
      MemRefType::get({-1, -1, -1, -1}, floatType);

  // Emit CPU alloc.
  auto cpuFilterHostAllocOp =
      builder.create<AllocOp>(builder.getUnknownLoc(), filterMemRefType);
  auto cpuInputHostAllocOp =
      builder.create<AllocOp>(builder.getUnknownLoc(), inputMemRefType);
  auto cpuOutputHostAllocOp =
      builder.create<AllocOp>(builder.getUnknownLoc(), outputMemRefType);
  block->push_back(cpuFilterHostAllocOp);
  block->push_back(cpuInputHostAllocOp);
  block->push_back(cpuOutputHostAllocOp);

  // Emit memref_cast.
  auto filterMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), cpuFilterHostAllocOp,
      fourDimUnknownSizeMemRefType);
  auto inputMemRefCastOp =
      builder.create<MemRefCastOp>(builder.getUnknownLoc(), cpuInputHostAllocOp,
                                   fourDimUnknownSizeMemRefType);
  auto outputMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), cpuOutputHostAllocOp,
      fourDimUnknownSizeMemRefType);
  block->push_back(filterMemRefCastOp);
  block->push_back(inputMemRefCastOp);
  block->push_back(outputMemRefCastOp);

  auto oneConstantFloatOp = builder.create<ConstantOp>(
      builder.getUnknownLoc(), floatType, builder.getFloatAttr(floatType, 1.0));
  auto zeroConstantFloatOp = builder.create<ConstantOp>(
      builder.getUnknownLoc(), floatType, builder.getFloatAttr(floatType, 0.0));

  block->push_back(oneConstantFloatOp);
  block->push_back(zeroConstantFloatOp);

  // Emit CPU memset function calls.
  StringRef memsetFuncName = memsetFuncName = "mcpuMemset4DFloat";
  auto mcpuMemset4DFuncOp = makeFuncDecl(
      builder, memsetFuncName, {fourDimUnknownSizeMemRefType, floatType}, {});
  module.push_back(mcpuMemset4DFuncOp);

  // Populate initial values.
  mlir::Value filterMemsetValue, inputMemsetValue, outputMemsetValue;
  if (operation.getValue() == "conv2d") {
    filterMemsetValue = oneConstantFloatOp;
    inputMemsetValue = oneConstantFloatOp;
    outputMemsetValue = zeroConstantFloatOp;
  } else if (operation.getValue() == "conv2d_bwd_data") {
    filterMemsetValue = oneConstantFloatOp;
    inputMemsetValue = zeroConstantFloatOp;
    outputMemsetValue = oneConstantFloatOp;
  } else if (operation.getValue() == "conv2d_bwd_weight") {
    filterMemsetValue = zeroConstantFloatOp;
    inputMemsetValue = oneConstantFloatOp;
    outputMemsetValue = oneConstantFloatOp;
  }
  auto filterCpuMemsetOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mcpuMemset4DFuncOp,
                             ValueRange{filterMemRefCastOp, filterMemsetValue});
  auto inputCpuMemsetOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mcpuMemset4DFuncOp,
                             ValueRange{inputMemRefCastOp, inputMemsetValue});
  auto outputCpuMemsetOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mcpuMemset4DFuncOp,
                             ValueRange{outputMemRefCastOp, outputMemsetValue});
  block->push_back(filterCpuMemsetOp);
  block->push_back(inputCpuMemsetOp);
  block->push_back(outputCpuMemsetOp);

  // Populate host validation logic
  auto cpuConvFuncOp = createCPUConvolution(module, builder, filterMemRefType,
                                            inputMemRefType, outputMemRefType);

  // Emit conv2d_host function call.
  auto cpuConvCallOp = builder.create<CallOp>(
      builder.getUnknownLoc(), cpuConvFuncOp,
      ValueRange{cpuFilterHostAllocOp, cpuInputHostAllocOp,
                 cpuOutputHostAllocOp});
  block->push_back(cpuConvCallOp);

  mlir::AllocOp cpuResults;
  if (operation.getValue() == "conv2d") {
    cpuResults = cpuOutputHostAllocOp;
  } else if (operation.getValue() == "conv2d_bwd_data") {
    cpuResults = cpuInputHostAllocOp;
  } else if (operation.getValue() == "conv2d_bwd_weight") {
    cpuResults = cpuFilterHostAllocOp;
  }

  // Emit print function call.
  StringRef printMemRefFuncName = "print_memref_f32";
  auto unrankedMemRefType = UnrankedMemRefType::get(builder.getF32Type(), 0);
  auto printMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), cpuResults, unrankedMemRefType);
  auto printMemRefFuncOp =
      makeFuncDecl(builder, printMemRefFuncName, {unrankedMemRefType}, {});
  auto printMemRefCallOp =
      builder.create<CallOp>(builder.getUnknownLoc(), printMemRefFuncOp,
                             ValueRange{printMemRefCastOp});
  module.push_back(printMemRefFuncOp);
  block->push_back(printMemRefCastOp);
  block->push_back(printMemRefCallOp);

  // Emit CPU dealloc.
  auto filterHostDeallocOp =
      builder.create<DeallocOp>(builder.getUnknownLoc(), cpuFilterHostAllocOp);
  auto inputHostDeallocOp =
      builder.create<DeallocOp>(builder.getUnknownLoc(), cpuInputHostAllocOp);
  auto outputHostDeallocOp =
      builder.create<DeallocOp>(builder.getUnknownLoc(), cpuOutputHostAllocOp);
  block->push_back(filterHostDeallocOp);
  block->push_back(inputHostDeallocOp);
  block->push_back(outputHostDeallocOp);

  auto returnOp =
      builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  block->push_back(returnOp);

  return success();
}

static LogicalResult populateKernelLaunchLogic(ModuleOp &module,
                                               OpBuilder &builder,
                                               MLIRContext &context,
                                               StringRef kernelName) {
  // Check if populate entry point exist.
  FuncOp theFunc;
  bool entryPointExist = false;
  module.walk([&](FuncOp funcOp) -> WalkResult {
    if (funcOp.getName() == populateEntryPoint.getValue()) {
      entryPointExist = true;
      theFunc = funcOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (!entryPointExist) {
    // do not fail for now. silent exit.
    // return failure();
    return success();
  }

  // Check if kernel to be launched exist.
  gpu::GPUFuncOp theGpuFunc;
  bool gpuKernelExist = false;
  module.walk([&](gpu::GPUModuleOp gpuModule) -> WalkResult {
    module.walk([&](gpu::GPUFuncOp gpuFunc) -> WalkResult {
      if (gpuFunc.getName() == kernelName) {
        gpuKernelExist = true;
        theGpuFunc = gpuFunc;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (gpuKernelExist)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  if (!gpuKernelExist) {
    // do not fail for now. silent exit.
    // return failure();
    return success();
  }

  Block *block = &(theFunc.getBody().front());
  block->clear();

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

  auto returnOp = builder.create<ReturnOp>(builder.getUnknownLoc());
  block->push_back(returnOp);

  return success();
}

static LogicalResult runMLIRPasses(ModuleOp &module, mlir::PassPipelineCLParser &passPipeline, StringRef kernelName) {
  PassManager pm(module.getContext(), PassManager::Nesting::Implicit);
  applyPassManagerCLOptions(pm);

  if (loweringWithDefaultPipeline.getValue()) {
    // Use fixed lowering pipeline.

    // Passes for lowering MIOpen dialect.
    pm.addPass(mlir::miopen::createLowerMIOpenOpsStep1Pass());
    pm.addPass(mlir::miopen::createAffineTransformPass());
    pm.addPass(
        mlir::miopen::createAffixTuningParametersPass(blockSize, gridSize));
    pm.addPass(mlir::miopen::createLowerMIOpenOpsStep2Pass());
    pm.addPass(mlir::miopen::createLowerMIOpenOpsStep3Pass());
    pm.addPass(mlir::miopen::createLowerMIOpenOpsStep4Pass());
    pm.addPass(mlir::miopen::createLowerMIOpenOpsStep5Pass());
    pm.addPass(mlir::createLowerMIOpenOpsToGPUPass(kernelName));

    // Passes for lowering linalg dialect.
    pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createLowerToCFGPass());
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
  InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipeline("", "compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR MIOpen Dialect driver\n");

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

  // Determine data type.
  mlir::Type dataType = builder.getF32Type();
  if (tensorDataType == "f32") {
    dataType = builder.getF32Type();
  } else if (tensorDataType == "f16") {
    dataType = builder.getF16Type();
  } else if (tensorDataType == "bf16") {
    dataType = builder.getIntegerType(16);
  }

  populateDefaults();

  Conv2dGenerator conv2dGenerator;
  // Determine dimensions.
  SmallVector<int64_t, 4> filterDimension;
  SmallVector<int64_t, 4> inputDimension;
  SmallVector<int64_t, 4> outputDimension;
  conv2dGenerator.parseConvDims(
      inputLayout, outputLayout, filterLayout, batchSize, inputChannel,
      inputHeight, inputWidth, outputChannel, outputHeight, outputWidth,
      filterWidth, filterHeight, filterDimension, inputDimension,
      outputDimension);

  // Populate the module.
  std::string kernelName;
  if (!populateCpuConvolution.getValue()) {
    if (failed(conv2dGenerator.genConvModule(
            arch.getValue(), num_cu.getValue(), operation.getValue(),
            inputLayout, outputLayout, filterLayout, filterDimension,
            inputDimension, outputDimension, dilationHeight.getValue(),
            dilationWidth.getValue(), strideHeight.getValue(),
            strideWidth.getValue(), paddingHeight.getValue(),
            paddingWidth.getValue(), module, builder, kernelName, dataType,
            xdlopsV2.getValue()))) {
      llvm::errs() << "Module population failed.\n";
      exit(1);
    }
  }

  // populate host harness and host validation.
  if (populateValidation.getValue()) {
    if (failed(populateValidationLogic(module, builder, context,
                                       filterDimension, inputDimension,
                                       outputDimension, dataType))) {
      llvm::errs() << "Host validation populated failed.\n";
      exit(1);
    }
  }

  // populate CPU convolution and print the results.
  if (populateCpuConvolution.getValue()) {
    if (failed(populateCpuConvolutionLogic(module, builder, context,
                                           filterDimension, inputDimension,
                                           outputDimension))) {
      llvm::errs() << "Cpu Convolution populated failed.\n";
      exit(1);
    }
  }

  // Apply passes.
  if (failed(runMLIRPasses(module, passPipeline, kernelName))) {
    llvm::errs() << "Lowering failed.\n";
    exit(1);
  }

  // populate host logic.
  if (populateHostHarness.getValue()) {
    if (failed(populateHostHarnessLogic(module, builder, context,
                                        filterDimension, inputDimension,
                                        outputDimension, dataType))) {
      llvm::errs() << "Host logic populated failed.\n";
      exit(1);
    }
  }

  // populate host launch logic.
  if (useHostHarness.getValue() || populateHostHarness.getValue() ||
      populateValidation.getValue()) {
    if (failed(
            populateKernelLaunchLogic(module, builder, context, kernelName))) {
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
