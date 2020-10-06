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

#include "MlirParse.h"
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
                          cl::init(true));

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
      dilationHeight.setValue(0);
      dilationWidth.setValue(0);
      strideHeight.setValue(1);
      strideWidth.setValue(1);
      paddingHeight.setValue(0);
      paddingWidth.setValue(0);
    }
  }

  auto getOutputDim = [](int64_t inputLen, int64_t filLen, int padLen,
                         int strideLen) {
    return (inputLen + 2 * padLen - filLen) / strideLen + 1;
  };
  outputHeight.setValue(
      getOutputDim(inputHeight.getValue(), filterHeight.getValue(),
                   paddingHeight.getValue(), strideHeight.getValue()));
  outputWidth.setValue(
      getOutputDim(inputWidth.getValue(), filterWidth.getValue(),
                   paddingWidth.getValue(), strideWidth.getValue()));
}

static LogicalResult populateHostHarnessLogic(ModuleOp &module, OpBuilder &builder,
                                              MLIRContext &context, mlir::FloatType dataType) {
  // Construct main function.
  auto func = FuncOp::create(builder.getUnknownLoc(), "main", builder.getFunctionType({}, {}));
  module.push_back(func);

  // Construct a new Block.
  Block *block = func.addEntryBlock();

  // Determine dimensions.
  SmallVector<int64_t, 4> filterDimension;
  SmallVector<int64_t, 4> inputDimension;
  SmallVector<int64_t, 4> outputDimension;
  populateConvolutionConfiguration(
      inputLayout.getValue(), outputLayout.getValue(), filterLayout.getValue(),
      batchSize.getValue(), inputChannel.getValue(), inputHeight.getValue(),
      inputWidth.getValue(), outputChannel.getValue(), outputHeight.getValue(),
      outputWidth.getValue(), filterWidth.getValue(), filterHeight.getValue(),
      filterDimension, inputDimension, outputDimension);

  auto filterMemRefType = MemRefType::get(
      ArrayRef<int64_t>(filterDimension.begin(), filterDimension.end()), dataType);
  auto inputMemRefType = MemRefType::get(
      ArrayRef<int64_t>(inputDimension.begin(), inputDimension.end()), dataType);
  auto outputMemRefType = MemRefType::get(
      ArrayRef<int64_t>(outputDimension.begin(), outputDimension.end()), dataType);
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

  // Populate initial values.
  auto oneConstantFloatOp = builder.create<ConstantOp>(
      builder.getUnknownLoc(), dataType, builder.getFloatAttr(dataType, 1.0));
  auto zeroConstantFloatOp = builder.create<ConstantOp>(
      builder.getUnknownLoc(), dataType, builder.getFloatAttr(dataType, 0.0));
  block->push_back(oneConstantFloatOp);
  block->push_back(zeroConstantFloatOp);

  // Emit CPU memset function calls.
  StringRef memsetFuncName;
  if (dataType == builder.getF32Type()) {
    memsetFuncName = "mcpuMemset4DFloat";
  } else if (dataType == builder.getF16Type()) {
    memsetFuncName = "mcpuMemset4DHalf";
  } else if (dataType == builder.getBF16Type()) {
    memsetFuncName = "mcpuMemset4DBF16";
  }
  auto mcpuMemset4DFuncOp = FuncOp::create(
      builder.getUnknownLoc(), memsetFuncName,
      builder.getFunctionType(
          {fourDimUnknownSizeMemRefType, dataType}, {}));
  module.push_back(mcpuMemset4DFuncOp);

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
  auto filterCpuMemsetOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mcpuMemset4DFuncOp,
      ValueRange{filterMemRefCastOp, filterMemsetValue});
  auto inputCpuMemsetOp =
      builder.create<CallOp>(builder.getUnknownLoc(), mcpuMemset4DFuncOp,
                             ValueRange{inputMemRefCastOp, inputMemsetValue});
  auto outputCpuMemsetOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mcpuMemset4DFuncOp,
      ValueRange{outputMemRefCastOp, outputMemsetValue});
  block->push_back(filterCpuMemsetOp);
  block->push_back(inputCpuMemsetOp);
  block->push_back(outputCpuMemsetOp);

  // Emit GPU memory allocation function calls.
  StringRef gpuMemAllocFuncName;
  if (dataType == builder.getF32Type()) {
    gpuMemAllocFuncName = "mgpuMemAlloc4DFloat";
  } else if (dataType == builder.getF16Type()) {
    gpuMemAllocFuncName = "mgpuMemAlloc4DHalf";
  } else if (dataType == builder.getBF16Type()) {
    gpuMemAllocFuncName = "mgpuMemAlloc4DBF16";
  }
  auto mgpuMemAlloc4DFuncOp =
      FuncOp::create(builder.getUnknownLoc(), gpuMemAllocFuncName,
                     builder.getFunctionType({fourDimUnknownSizeMemRefType},
                                             {fourDimUnknownSizeMemRefType}));
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
  block->push_back(filterGpuAllocOp);
  block->push_back(inputGpuAllocOp);
  block->push_back(outputGpuAllocOp);

  // Emit some constant values for HIP runtime API calls.
  auto oneConstantI32Op = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), 1, builder.getIntegerType(32));
  auto twoConstantI32Op = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), 2, builder.getIntegerType(32));
  block->push_back(oneConstantI32Op);
  block->push_back(twoConstantI32Op);

  // Emit CPU->GPU memcpy function calls.
  StringRef gpuMemCopyFuncName;
  if (dataType == builder.getF32Type()) {
    gpuMemCopyFuncName = "mgpuMemCopy4DFloat";
  } else if (dataType == builder.getF16Type()) {
    gpuMemCopyFuncName = "mgpuMemCopy4DHalf";
  } else if (dataType == builder.getBF16Type()) {
    gpuMemCopyFuncName = "mgpuMemCopy4DBF16";
  }
  auto mgpuMemCopy4DFuncOp =
      FuncOp::create(builder.getUnknownLoc(), gpuMemCopyFuncName,
                     builder.getFunctionType({fourDimUnknownSizeMemRefType,
                                              fourDimUnknownSizeMemRefType,
                                              builder.getIntegerType(32)},
                                             {}));
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
  block->push_back(filterCpuToGpuCopyOp);
  block->push_back(inputCpuToGpuCopyOp);
  block->push_back(outputCpuToGpuCopyOp);

  // Emit memref_cast.
  auto filterGpuMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), filterGpuAllocOp.getResult(0), filterMemRefType);
  auto inputGpuMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), inputGpuAllocOp.getResult(0), inputMemRefType);
  auto outputGpuMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), outputGpuAllocOp.getResult(0), outputMemRefType);
  block->push_back(filterGpuMemRefCastOp);
  block->push_back(inputGpuMemRefCastOp);
  block->push_back(outputGpuMemRefCastOp);

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
  block->push_back(kernelCallOp);

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
  block->push_back(outputGpuToCpuCopyOp);

  // Emit verification logic.
  StringRef printMemRefFuncName;
  if (dataType == builder.getF32Type()) {
    printMemRefFuncName = "print_memref_f32";
  } else if (dataType == builder.getF16Type()) {
    printMemRefFuncName = "print_memref_f16";
  } else if (dataType == builder.getBF16Type()) {
    printMemRefFuncName = "print_memref_bf16";
  }
  auto unrankedMemRefType = UnrankedMemRefType::get(dataType, 0);
  auto printMemRefCastOp = builder.create<MemRefCastOp>(
      builder.getUnknownLoc(), resultCpuValue, unrankedMemRefType);
  auto printMemRefFuncOp =
      FuncOp::create(builder.getUnknownLoc(), printMemRefFuncName,
                     builder.getFunctionType({unrankedMemRefType}, {}));
  auto printMemRefCallOp =
      builder.create<CallOp>(builder.getUnknownLoc(), printMemRefFuncOp,
                             ValueRange{printMemRefCastOp});
  module.push_back(printMemRefFuncOp);
  block->push_back(printMemRefCastOp);
  block->push_back(printMemRefCallOp);

  // Emit GPU memory deallocation function calls.
  StringRef gpuMemDeallocFuncName;
  if (dataType == builder.getF32Type()) {
    gpuMemDeallocFuncName = "mgpuMemDealloc4DFloat";
  } else if (dataType == builder.getF16Type()) {
    gpuMemDeallocFuncName = "mgpuMemDealloc4DHalf";
  } else if (dataType == builder.getBF16Type()) {
    gpuMemDeallocFuncName = "mgpuMemDealloc4DBF16";
  }
  auto mgpuMemDealloc4DFuncOp = FuncOp::create(
      builder.getUnknownLoc(), gpuMemDeallocFuncName,
      builder.getFunctionType({fourDimUnknownSizeMemRefType}, {}));
  module.push_back(mgpuMemDealloc4DFuncOp);

  auto filterGpuDeallocOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mgpuMemDealloc4DFuncOp,
      ValueRange{filterMemRefCastOp});
  auto inputGpuDeallocOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mgpuMemDealloc4DFuncOp,
      ValueRange{inputMemRefCastOp});
  auto outputGpuDeallocOp = builder.create<CallOp>(
      builder.getUnknownLoc(), mgpuMemDealloc4DFuncOp,
      ValueRange{outputMemRefCastOp});
  block->push_back(filterGpuDeallocOp);
  block->push_back(inputGpuDeallocOp);
  block->push_back(outputGpuDeallocOp);

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

  auto cstOne = builder.create<ConstantIndexOp>(builder.getUnknownLoc(), 1);
  auto cstBlockSize =
      builder.create<ConstantIndexOp>(builder.getUnknownLoc(), blockSize);
  auto cstGridSize =
      builder.create<ConstantIndexOp>(builder.getUnknownLoc(), gridSize);
  block->push_back(cstOne);
  block->push_back(cstBlockSize);
  block->push_back(cstGridSize);

  auto gpuLaunchFuncOp = builder.create<gpu::LaunchFuncOp>(
      builder.getUnknownLoc(), theGpuFunc, cstGridSize, cstOne, cstOne,
      cstBlockSize, cstOne, cstOne,
      ValueRange{theFunc.getArgument(0), theFunc.getArgument(1),
                 theFunc.getArgument(2)});
  block->push_back(gpuLaunchFuncOp);

  auto returnOp = builder.create<ReturnOp>(builder.getUnknownLoc());
  block->push_back(returnOp);

  return success();
}

static LogicalResult runMLIRPasses(ModuleOp &module, mlir::PassPipelineCLParser &passPipeline, StringRef kernelName) {
  PassManager pm(module.getContext());
  applyPassManagerCLOptions(pm);

  if (loweringWithDefaultPipeline.getValue()) {
    // Use fixed lowering pipeline.

    // Passes for lowering MIOpen dialect.
    pm.addPass(mlir::miopen::createLowerMIOpenOpsStep1Pass());
    pm.addPass(mlir::miopen::createAffineTransformPass());
    pm.addPass(mlir::miopen::createAffixTuningParametersPass(
        blockSize, [&](int64_t computedBlockSize, int64_t computedGridSize) {
          // Use computed block size and grid size in case they are not
          // specified from command line.
          if (blockSize == 0)
            blockSize = computedBlockSize;
          if (gridSize == 0)
            gridSize = computedGridSize;
        }));
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
    // Use lowering pipeline specified at command line.
    if (failed(passPipeline.addToPipeline(pm)))
      return failure();
  }

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

  MLIRContext context;
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
  mlir::FloatType dataType = builder.getF32Type();
  if (tensorDataType == "f32") {
    dataType = builder.getF32Type();
  } else if (tensorDataType == "f16") {
    dataType = builder.getF16Type();
  } else if (tensorDataType == "bf16") {
    dataType = builder.getBF16Type();
  }

  // Populate the module.
  SmallString<128> kernelName;
  populateDefaults();
  if (failed(populateConvolutionLogic(
          arch.getValue(), num_cu.getValue(), operation.getValue(),
          inputLayout.getValue(), outputLayout.getValue(),
          filterLayout.getValue(), batchSize.getValue(),
          inputChannel.getValue(), inputHeight.getValue(),
          inputWidth.getValue(), outputChannel.getValue(),
          outputHeight.getValue(), outputWidth.getValue(),
          filterWidth.getValue(), filterHeight.getValue(),
          dilationHeight.getValue(), dilationWidth.getValue(),
          strideHeight.getValue(), strideWidth.getValue(),
          paddingHeight.getValue(), paddingWidth.getValue(), module, builder,
          kernelName, dataType, xdlopsV2.getValue()))) {
    llvm::errs() << "Module population failed.\n";
    exit(1);
  }

  // Apply passes.
  if (failed(runMLIRPasses(module, passPipeline, kernelName))) {
    llvm::errs() << "Lowering failed.\n";
    exit(1);
  }

  // populate host launch logic.
  if (useHostHarness.getValue()) {
    if (failed(
            populateKernelLaunchLogic(module, builder, context, kernelName))) {
      llvm::errs() << "Host kernel launch logic populated failed.\n";
      exit(1);
    }
  } else if (populateHostHarness.getValue()) {
    if (failed(populateHostHarnessLogic(module, builder, context, dataType)) ||
        failed(
            populateKernelLaunchLogic(module, builder, context, kernelName))) {
      llvm::errs() << "Host logic populated failed.\n";
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
