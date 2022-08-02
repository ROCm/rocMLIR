//===- miopen-gen.cpp - MLIR MIOpen Test Generator ------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for miopen-gen test generator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Conversion/MIOpenPasses.h"
#include "mlir/Conversion/MIOpenToGPU/MIOpenToGPU.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MIOpen/Generator/Conv2dGenerator.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/utility/IsaNameSplitter.h"
#include "mlir/Dialect/MIOpen/utility/builderUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "bf16convert.hpp"
#include <unordered_map>

#include <tuple>

using namespace llvm;
using namespace mlir;

static cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                          llvm::cl::desc("<input file>"),
                                          llvm::cl::init(""));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<std::string> testFuncName("func-under-test",
                                         cl::desc("Name of func to test"),
                                         cl::init(""));
static cl::alias aliasTestFuncName("fut", cl::aliasopt(testFuncName));

//////////////////////////////////////////////////////////////////////////////////////////////////////
//// MIOpen Convolution spec

static cl::opt<mlir::miopen::ConvOpType> operation(
    "operation", cl::desc("Convolution operation,"),
    cl::values(clEnumValN(miopen::ConvOpType::Fwd, "conv2d",
                          "Forward convolution"),
               clEnumValN(miopen::ConvOpType::BwdData, "conv2d_bwd_data",
                          "Backpropogate convolution data"),
               clEnumValN(miopen::ConvOpType::BwdWeight, "conv2d_bwd_weight",
                          "Backpropogate convolution weights")),
    cl::value_desc("convolution type"), cl::init(miopen::ConvOpType::Fwd));

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

static cl::opt<std::string>
    perfConfig("perf_config",
               cl::desc("performance config data used for tuning"),
               cl::value_desc("Serialized tuning parameters"), cl::init(""));

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

// conv-config
static cl::opt<std::string> populateConvConfig(
    "conv-config",
    cl::desc("Populate full config settings (overrides all specific settings)"),
    cl::value_desc("config settings matching the C-API"), cl::init(""));

// populate default values
static cl::opt<bool>
    populateDefaultValues("p", cl::desc("To populate default values"),
                          cl::value_desc("To populate default values"),
                          cl::init(false));

//////////////////////////////////////////////////////////////////////////
////  Host Generator options
//////////////////////////////////////////////////////////////////////////
////  * Host harness
////    * kernel options
////      * cmd-line def (see above)
////        * gpu gen
////        * cpu gen
////      * user defined (input file)
////    * verifier
////      * cpu gen
////      * gpu gen
////      * compare results
////    * print results
////      * optionally print inputs
////      * optionally print validation results
////    * profiling (TBD)
////      * plumb thru runner (TBD)
//////////////////////////////////////////////////////////////////////////

// generate host harness program.
static cl::opt<bool> readHostHarness("host", cl::desc("To use host harness"),
                                     cl::value_desc("To use host harness"),
                                     cl::init(false));

static cl::opt<bool> genHostHarness("host-harness",
                                    cl::desc("To use host harness"),
                                    cl::value_desc("To use host harness"),
                                    cl::init(false));

static cl::alias aliasGenHostHarness("ph", cl::aliasopt(genHostHarness));

// print results
static cl::opt<bool> printResults("print-results",
                                  cl::desc("To print result tensor"),
                                  cl::init(false));
static cl::alias aliasPrintResults("pr", cl::aliasopt(printResults));

static cl::opt<bool> printInputs("print-inputs",
                                 cl::desc("To print input tensors"),
                                 cl::init(false));
static cl::alias aliasPrintInputs("pi", cl::aliasopt(printInputs));

static cl::opt<bool>
    printValidationResults("print-validation-results",
                           cl::desc("To print result tensor for validation"),
                           cl::init(false));
static cl::alias
    aliasPrintValidationResults("pvr", cl::aliasopt(printValidationResults));

// populate host validation logic.
static cl::opt<std::string> genValidation(
    "verifier", cl::desc("Select verification from: none(default), cpu, gpu"),
    cl::cb<void, std::string>([](const std::string &v) {
      if (!v.empty())
        genHostHarness = true;
    }),
    cl::value_desc("Specify host validation logic"), cl::init(""));

static cl::opt<bool> genCPUValidation("pv", cl::Hidden, cl::init(false),
                                      cl::Optional,
                                      cl::cb<void, bool>([](bool v) {
                                        if (v) {
                                          genValidation = "cpu";
                                          genHostHarness = true;
                                        }
                                      }));

static cl::opt<bool> genCPPValidation("pv_with_cpp", cl::Hidden,
                                      cl::init(false), cl::Optional,
                                      cl::cb<void, bool>([](bool v) {
                                        if (v) {
                                          genValidation.setValue("cpp");
                                          genHostHarness.setValue(true);
                                        }
                                      }));

static cl::opt<bool> genMLIRValidation("pv_with_mlir", cl::Hidden,
                                       cl::init(false), cl::Optional,
                                       cl::cb<void, bool>([](bool v) {
                                         if (v) {
                                           genValidation.setValue("mlir");
                                           genHostHarness.setValue(true);
                                         }
                                       }));

static cl::opt<bool> genGPUValidation("pv_with_gpu", cl::Hidden,
                                      cl::init(false), cl::Optional,
                                      cl::cb<void, bool>([](bool v) {
                                        if (v) {
                                          genValidation = "gpu";
                                          genHostHarness = true;
                                        }
                                      }));

static cl::opt<bool> genCPUKernel("cpu-kernels",
                                  cl::desc("Generate CPU kernel for test"),
                                  cl::init(false), cl::Optional,
                                  cl::cb<void, bool>([](bool v) {
                                    if (v) {
                                      genHostHarness = true;
                                      printResults = true;
                                    }
                                  }));
static cl::alias aliasGenCPUKernel("prc", cl::aliasopt(genCPUKernel));

// Input data spec
static cl::opt<std::string> randomSeed(
    "rand",
    cl::desc(
        "A positive integer or zero indicates the seed of random data generator"
        "for convolution inputs, e.g. -rand 1. If not specifed, or 'fixed', "
        "use a fixed nonuniform test pattern. If 'none', use all 1s as the "
        "values. If 0, use time(0) as the seed."),
    cl::value_desc("seed"), cl::init("fixed"));

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

// Verification function options
static cl::opt<float> RMSThreshold("RMS_threshold",
                                   cl::desc("Threshold for RMS metric"),
                                   cl::value_desc("error"), cl::init(0.00003f));

static cl::opt<float> absDiffThreshold("absDiff_threshold",
                                       cl::desc("Threshold for absDiff metric"),
                                       cl::value_desc("error"),
                                       cl::init(100.0f));

static cl::opt<float> relDiffThreshold("relDiff_threshold",
                                       cl::desc("Threshold for relDiff metric"),
                                       cl::value_desc("error"),
                                       cl::init(100.0f));
static cl::opt<std::string> printVerifyResults(
    "print-verify-results",
    cl::desc("Choose when to print verbose debug information in the "
             "verification function:"
             "always: print debug info"
             "failure: print debug info if the test fails (default)"
             "off: do not print debug info"),
    cl::value_desc("info"), cl::init("failure"));
static cl::alias aliasPrintVerifyResults("p_verify",
                                         cl::aliasopt(printVerifyResults));

static cl::opt<int> deviceNum(
    "device",
    cl::desc("Device index on which to run the kernel (only with host code)"),
    cl::value_desc("Between 0 and number of GPUs on system. "
                   "Omission leaves current device intact."));
static cl::alias deviceShort("dev", cl::aliasopt(deviceNum));

////////////////////////////////////////////////////////////////////////////////
////  Struct KernelIF
////  - Detected/capture kernel interface
////////////////////////////////////////////////////////////////////////////////
struct KernelIF {
  func::FuncOp func;
  SmallVector<mlir::Type, 8> params;

  // CTOR w/ FuncOp
  KernelIF(func::FuncOp _f) : func(_f) {
    assert(func.getNumResults() == 0);
    for (auto &paramType : func.getFunctionType().getInputs())
      params.push_back(paramType);
  }
};

namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test

static void correctParameters() {
  std::string filterLayoutValue = filterLayout.getValue();
  std::string inputLayoutValue = inputLayout.getValue();
  std::string outputLayoutValue = outputLayout.getValue();
  // yxcgk not implement yet
  if (filterLayoutValue == "kcyx")
    filterLayout = "gkcyx";
  else if (filterLayoutValue == "kyxc")
    filterLayout = "gkyxc";
  else if (filterLayoutValue.size() == 4)
    filterLayout = "g" + filterLayoutValue;

  if (outputLayoutValue == "nkhw")
    outputLayout = "ngkhw";
  else if (outputLayoutValue == "nhwk")
    outputLayout = "nhwgk";
  else if (outputLayoutValue.size() == 4)
    outputLayout = "g" + outputLayoutValue;

  if (inputLayoutValue == "nchw")
    inputLayout = "ngchw";
  else if (inputLayoutValue == "nhwc")
    inputLayout = "nhwgc";
  else if (inputLayoutValue.size() == 4)
    inputLayout = "g" + inputLayoutValue;

  // we can use paddingHeight or paddingHeightLeft + paddingHeightRight
  // if use paddingHeight , paddingHeightLeft and paddingHeightRight =
  // paddingHeight if use paddingHeightLeft + paddingHeightRight , please
  // assigne value
  auto validatePadding = [](cl::opt<int>& combined, cl::opt<int>& left,
                            cl::opt<int>& right, StringRef name) {
    if (combined.getValue() > 0) {
      int combinedVal = combined.getValue();
      int leftVal = left.getValue();
      int rightVal = right.getValue();
      if (leftVal == 0 && rightVal == 0) {
        left = combinedVal;
        right = combinedVal;
      } else {
        if (leftVal != combinedVal || rightVal != combinedVal) {
          llvm::errs()
            << "you can't use both " << name << " and (" << name << "_l,"
            << name << "_r).\n";
        }
      }
    }
  };

  validatePadding(paddingHeight, paddingHeightLeft, paddingHeightRight,
                  "padding_h");
  validatePadding(paddingWidth, paddingWidthLeft, paddingWidthRight,
                  "padding_w");

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
  int in_right_pad_h = paddingHeightRight.getValue();
  int conv_stride_h = strideHeight.getValue();
  int conv_dilation_h = dilationHeight.getValue();
  int ho = getOutputDim(hi, y, in_left_pad_h, in_right_pad_h, conv_stride_h,
                        conv_dilation_h);
  int hi_minimum = 1 + (y - 1) * conv_dilation_h + (ho - 1) * conv_stride_h;
  int hi_specified = hi + in_left_pad_h + in_right_pad_h;
  // hi_minimum is the miminum number of input elements needed to correctly
  // apply the filter in the h direction, which is a function of the stride and
  // dilation parameters. If the specified input height is less than this value,
  // add extra padding on the right to allow the convolution to execute
  // successfully.
  if (hi_minimum > hi_specified)
    paddingHeightRight = in_right_pad_h + (hi_minimum - hi_specified);

  int wi = inputWidth.getValue();
  int x = filterWidth.getValue();
  int in_left_pad_w = paddingWidthLeft.getValue();
  int in_right_pad_w = paddingWidthRight.getValue();
  int conv_stride_w = strideWidth.getValue();
  int conv_dilation_w = dilationWidth.getValue();
  int wo = getOutputDim(wi, x, in_left_pad_w, in_right_pad_w, conv_stride_w,
                        conv_dilation_w);

  int wi_minimum = 1 + (x - 1) * conv_dilation_w + (wo - 1) * conv_stride_w;
  int wi_specified = wi + in_left_pad_w + in_right_pad_w;
  // wi_minimum is the miminum number of input elements needed to correctly
  // apply the filter in the w direction, which is a function of the stride and
  // dilation parameters. If the specified input height is less than this value,
  // add extra padding on the right to allow the convolution to execute
  // successfully.
  if (wi_minimum > wi_specified)
    paddingWidthRight = in_right_pad_w + (wi_minimum - wi_specified);
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
  arch = "amdgcn-amd-amdhsa:gfx900";

  if (populateDefaultValues) {
    if (!xdlopsV2.getValue()) {
      groupSize = 1;
      batchSize = 128;
      inputChannel = 8;
      outputChannel = 128;
      inputHeight = 32;
      inputWidth = 32;
      filterHeight = 3;
      filterWidth = 3;
      dilationHeight = 1;
      dilationWidth = 1;
      strideHeight = 1;
      strideWidth = 1;
      paddingHeightLeft = 0;
      paddingHeightRight = 0;
      paddingWidthLeft = 0;
      paddingWidthRight = 0;
    } else {
      groupSize = 1;
      batchSize = 128;
      inputChannel = 1024;
      outputChannel = 1024;
      inputHeight = 14;
      inputWidth = 14;
      filterHeight = 1;
      filterWidth = 1;
      dilationHeight = 1;
      dilationWidth = 1;
      strideHeight = 1;
      strideWidth = 1;
      paddingHeightLeft = 0;
      paddingHeightRight = 0;
      paddingWidthLeft = 0;
      paddingWidthRight = 0;
      num_cu = 120;
      arch = "amdgcn-amd-amdhsa:gfx908";
    }
  }

  if (xdlopsV2.getValue()) {
    num_cu = 120;
    arch = "amdgcn-amd-amdhsa:gfx908";
  }

  if (outputHeight.getNumOccurrences() == 0) {
    outputHeight = miopen::Conv2dGenerator::outputDim(
        inputHeight.getValue(), filterHeight.getValue(),
        paddingHeightLeft.getValue(), paddingHeightRight.getValue(),
        strideHeight.getValue(), dilationHeight.getValue());
  }
  if (outputWidth.getNumOccurrences() == 0) {
    outputWidth = miopen::Conv2dGenerator::outputDim(
        inputWidth.getValue(), filterWidth.getValue(),
        paddingWidthLeft.getValue(), paddingWidthRight.getValue(),
        strideWidth.getValue(), dilationWidth.getValue());
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

static func::FuncOp makeFuncDecl(ModuleOp module, StringRef funcName,
                                 TypeRange inputs, TypeRange results = {}) {
  func::FuncOp func = module.lookupSymbol<func::FuncOp>(funcName);
  if (!func) {
    OpBuilder builder(module.getContext());
    func = func::FuncOp::create(builder.getUnknownLoc(), funcName,
                                builder.getFunctionType(inputs, results));
    func.setSymVisibilityAttr(builder.getStringAttr("private"));
    module.push_back(func);
  }

  return func;
}

static mlir::Value makeNDMemRef(OpBuilder &b, mlir::Value var, uint32_t ndim) {
  auto context = b.getContext();
  auto oprType = var.getType().template cast<ShapedType>();
  if (!oprType.hasStaticShape())
    return mlir::Value();

  auto shape = oprType.getShape();
  auto loc = var.getLoc();

  if (shape.size() > ndim) {
    // Collapse last dims
    SmallVector<int64_t, 5> colShape;
    SmallVector<ReassociationExprs, 5> reassocs;
    uint32_t dim = 0;
    for (; dim < ndim - 1; ++dim) {
      colShape.push_back(shape[dim]);
      reassocs.push_back({getAffineDimExpr(dim, context)});
    }

    // Last dim
    uint64_t lastDim = 1;
    SmallVector<AffineExpr, 2> exprs;
    for (; dim < shape.size(); ++dim) {
      lastDim *= shape[dim];
      exprs.push_back(getAffineDimExpr(dim, context));
    }
    colShape.push_back(lastDim);
    reassocs.push_back(exprs);

    auto colType = MemRefType::get(colShape, oprType.getElementType());

    // Emit memref.collapse_shape
    var = b.create<memref::CollapseShapeOp>(loc, colType, var, reassocs);
  } else if (shape.size() < ndim) {
    // Expand last dims
    SmallVector<int64_t, 5> expShape;
    SmallVector<ReassociationExprs, 5> reassocs;
    uint32_t dim = 0;
    for (; dim < shape.size() - 1; ++dim) {
      expShape.push_back(shape[dim]);
      reassocs.push_back({getAffineDimExpr(dim, context)});
    }

    // Last dim
    expShape.push_back(shape[dim]);
    SmallVector<AffineExpr, 2> exprs;
    for (; dim < ndim; ++dim) {
      expShape.push_back(1);
      exprs.push_back(getAffineDimExpr(dim, context));
    }
    expShape.pop_back();
    reassocs.push_back(exprs);

    auto expType = MemRefType::get(expShape, oprType.getElementType());

    // Emit memref.collapse_shape
    var = b.create<memref::ExpandShapeOp>(loc, expType, var, reassocs);
  }

  return var;
}

static func::FuncOp createGPUWrapper(ModuleOp &module, const KernelIF &kernel) {
  auto context = module.getContext();
  OpBuilder b(context);
  auto loc = kernel.func->getLoc();

  // Create gpu wrapper function
  auto kfunc = kernel.func;
  std::string funcName = kfunc.getName().str() + "_gpu";
  auto gpuWrapperFuncType = b.getFunctionType(kernel.params, {});

  auto gpuWrapperFunc =
      func::FuncOp::create(loc, StringRef(funcName), gpuWrapperFuncType);
  module.push_back(gpuWrapperFunc);

  // Emit gpu convolution logic.
  auto block = gpuWrapperFunc.addEntryBlock();
  b.setInsertionPoint(block, block->begin());

  // Emit device selection
  if (deviceNum.getNumOccurrences() > 0)
    b.create<gpu::SetDefaultDeviceOp>(
        loc, b.create<arith::ConstantIntOp>(loc, deviceNum.getValue(),
                                            b.getIntegerType(32)));

  SmallVector<mlir::Value, 4> cpuMem;
  SmallVector<mlir::Value, 4> gpuMem;
  for (auto pair : llvm::enumerate(kernel.params)) {
    mlir::Value arg = block->getArgument(pair.index());
    cpuMem.push_back(arg);

    // Emit GPU memory allocation function calls.
    auto gpuAllocOp = b.create<gpu::AllocOp>(
        loc, arg.getType(), mlir::Type(), /*asyncDependencies=*/ValueRange{},
        /*dynamicSizes=*/ValueRange{}, /*symbolOperands=*/ValueRange{});
    mlir::Value gpuAlloc = gpuAllocOp.getResult(0);
    gpuMem.push_back(gpuAlloc);

    // Emit CPU->GPU memcpy function calls.
    b.create<gpu::MemcpyOp>(loc, TypeRange{}, ValueRange{gpuAlloc, arg});
  }

  // Emit kernel function call.
  auto wrappedCall = b.create<func::CallOp>(loc, kernel.func, gpuMem);
  wrappedCall->setAttr("wrapped_call", b.getUnitAttr());

  for (auto &pair : llvm::enumerate(kernel.params)) {
    uint32_t i = pair.index();
    b.create<gpu::MemcpyOp>(loc, TypeRange{}, ValueRange{cpuMem[i], gpuMem[i]});
    b.create<gpu::DeallocOp>(loc, TypeRange{}, ValueRange{gpuMem[i]});
  }

  b.create<func::ReturnOp>(loc, ValueRange{});

  return gpuWrapperFunc;
}

// Determine the range and seed for the random data generator
static std::tuple<short, short, int> getRandomTestData(int idx) {
  short min, max = min = 1;
  int seed = 1;

  int32_t idx_spec = -1;
  switch (randomSide.getValue()[0]) {
  case 'f':
    idx_spec = 0;
    break;
  case 'i':
    idx_spec = 1;
    break;
  case 'o':
    idx_spec = 2;
    break;
  case 'b':
  default:
    break;
  }

  if (randomSeed.getValue() != "none" && randomSeed.getValue() != "fixed") {
    if ((idx_spec >= 0) && (idx_spec != idx)) {
    } else if (randomDataType.getValue() == "int") {
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

static std::string getMemsetFuncName(mlir::Type dataType) {
  std::string memsetFuncName;
  if (dataType.isF32()) {
    memsetFuncName = "mcpuMemset5DFloatRand";
  } else if (dataType.isF16()) {
    memsetFuncName = "mcpuMemset5DHalfRand";
  } else if (dataType.isBF16()) {
    memsetFuncName = "mcpuMemset5DBF16Rand";
  } else if (dataType.isInteger(8)) {
    memsetFuncName = "mcpuMemset5DInt8Rand";
  } else if (dataType.isInteger(32)) {
    memsetFuncName = "mcpuMemset5DInt32Rand";
  }
  if (randomDataType == "float")
    memsetFuncName += "Float";
  else
    memsetFuncName += "Int";
  return memsetFuncName;
}

static func::FuncOp getMemsetFunc(ModuleOp module, mlir::Type elemType) {
  OpBuilder b(module.getContext());

  auto int16Type = b.getIntegerType(16);
  auto int32Type = b.getIntegerType(32);

  // Emit CPU memset function calls.
  std::string memsetFuncName = getMemsetFuncName(elemType);
  auto fiveDimUnknownSizeMemRefType =
      MemRefType::get({-1, -1, -1, -1, -1}, elemType);
  return makeFuncDecl(
      module, memsetFuncName,
      {fiveDimUnknownSizeMemRefType, int16Type, int16Type, int32Type});
}

static std::tuple<int64_t, int64_t, int64_t>
getConv2dBounds(miopen::ConvOpType dir,
                const miopen::Conv2dGenerator::Config &genConfig) {
  int64_t dim, dimH, dimW;
  char channel;
  StringRef layout;
  ArrayRef<int64_t> dimension;
  switch (dir) {
  case miopen::ConvOpType::Fwd:
    channel = 'c';
    dimension = genConfig.inputDimension;
    layout = genConfig.inputLayout;
    break;
  case miopen::ConvOpType::BwdData:
    channel = 'k';
    dimension = genConfig.outputDimension;
    layout = genConfig.outputLayout;
    break;
  case miopen::ConvOpType::BwdWeight:
    channel = 'n';
    dimension = genConfig.inputDimension;
    layout = genConfig.inputLayout;
    break;
  }
  for (const auto &t : llvm::zip(layout, dimension)) {
    char c(std::get<0>(t));
    if (c == channel)
      dim = std::get<1>(t);
    if (c == 'h')
      dimH = std::get<1>(t);
    if (c == 'w')
      dimW = std::get<1>(t);
  }
  return std::make_tuple(dim, dimH, dimW);
}

static void
createCPUConvWithMLIR(ModuleOp module, func::FuncOp &func,
                      const miopen::Conv2dGenerator::Config &genConfig) {
  OpBuilder b(module.getContext());

  Block *block = func.addEntryBlock();
  b.setInsertionPoint(block, block->begin());

  Location loc = b.getUnknownLoc();
  auto zeroI16Op = b.create<arith::ConstantIntOp>(loc, 0, b.getIntegerType(16));
  auto zeroI32Op = b.create<arith::ConstantIntOp>(loc, 0, b.getIntegerType(32));

  // Initialize the result tensor
  mlir::BlockArgument resultTensor;
  switch (genConfig.operation.getValue()) {
  case miopen::ConvOpType::Fwd:
    resultTensor = block->getArgument(2);
    break;
  case miopen::ConvOpType::BwdData:
    resultTensor = block->getArgument(1);
    break;
  case miopen::ConvOpType::BwdWeight:
    resultTensor = block->getArgument(0);
    break;
  }
  auto resultType = resultTensor.getType().template dyn_cast<MemRefType>();
  mlir::Type elemType = resultType.getElementType();
  MemRefType mr5DUnkType = MemRefType::get({-1, -1, -1, -1, -1}, elemType);
  auto memrefCastOp = b.create<memref::CastOp>(loc, mr5DUnkType, resultTensor);
  b.create<func::CallOp>(
      loc, getMemsetFunc(module, elemType),
      ValueRange{memrefCastOp, zeroI16Op, zeroI16Op, zeroI32Op});

  // Create affine maps
  AffineExpr heightExpr, widthExpr;
  AffineMap heightMap, widthMap;
  AffineExpr outputHeightExpr, outputWidthExpr;
  AffineMap outputHeightMap, outputWidthMap;

  switch (genConfig.operation.getValue()) {
  case miopen::ConvOpType::Fwd:
  case miopen::ConvOpType::BwdWeight:
    // d0 * stride + d1 * dilation - padding
    heightExpr = b.getAffineDimExpr(0) * genConfig.strideHeight +
                 b.getAffineDimExpr(1) * genConfig.dilationHeight -
                 genConfig.paddingHeightLeft;
    widthExpr = b.getAffineDimExpr(0) * genConfig.strideWidth +
                b.getAffineDimExpr(1) * genConfig.dilationWidth -
                genConfig.paddingWidthLeft;
    break;
  case miopen::ConvOpType::BwdData:
    // d0 + padding - d1 * dilation
    heightExpr = b.getAffineDimExpr(0) + genConfig.paddingHeightLeft -
                 b.getAffineDimExpr(1) * genConfig.dilationHeight;
    widthExpr = b.getAffineDimExpr(0) + genConfig.paddingWidthLeft -
                b.getAffineDimExpr(1) * genConfig.dilationWidth;
    break;
  }
  heightMap = AffineMap::get(2, 0, {heightExpr}, b.getContext());
  widthMap = AffineMap::get(2, 0, {widthExpr}, b.getContext());

  // Create extra maps for backward data
  if (genConfig.operation.getValue() == miopen::ConvOpType::BwdData) {
    // d0 / stride
    outputHeightExpr = b.getAffineDimExpr(0).floorDiv(genConfig.strideHeight);
    outputWidthExpr = b.getAffineDimExpr(0).floorDiv(genConfig.strideWidth);
    outputHeightMap = AffineMap::get(1, 0, {outputHeightExpr}, b.getContext());
    outputWidthMap = AffineMap::get(1, 0, {outputWidthExpr}, b.getContext());
  }

  // Create constraints for boundary checks
  SmallVector<AffineExpr, 6> exprs;
  SmallVector<bool, 6> eqFlags;
  IntegerSet condition;
  if (genConfig.operation.getValue() == miopen::ConvOpType::BwdData) {
    // out_h_tmp % stride_h == 0, out_w_tmp % stride_w == 0
    exprs.push_back(b.getAffineDimExpr(2) % genConfig.strideHeight);
    eqFlags.push_back(true);
    exprs.push_back(b.getAffineDimExpr(3) % genConfig.strideWidth);
    eqFlags.push_back(true);
  }
  // out_h_idx >= 0, out_h_idx < out_height, out_w_idx >= 0, out_w_idx <
  // out_width
  exprs.push_back(b.getAffineDimExpr(0));
  eqFlags.push_back(false);
  exprs.push_back(b.getAffineSymbolExpr(0) - b.getAffineDimExpr(0) - 1);
  eqFlags.push_back(false);
  exprs.push_back(b.getAffineDimExpr(1));
  eqFlags.push_back(false);
  exprs.push_back(b.getAffineSymbolExpr(1) - b.getAffineDimExpr(1) - 1);
  eqFlags.push_back(false);
  if (genConfig.operation.getValue() == miopen::ConvOpType::BwdData) {
    condition = IntegerSet::get(4, 2, exprs, eqFlags);
  } else {
    condition = IntegerSet::get(2, 2, exprs, eqFlags);
  }

  SmallVector<int64_t, 8> lowerBounds(8, 0);
  SmallVector<int64_t, 8> upperBounds;
  SmallVector<int64_t, 8> steps(8, 1);
  std::string loopIVs;

  int64_t dimX, dimH, dimW;
  int64_t out_h, out_w;
  std::tie(dimX, dimH, dimW) =
      getConv2dBounds(genConfig.operation.getValue(), genConfig);

  // Create the upper bounds
  switch (genConfig.operation.getValue()) {
  case miopen::ConvOpType::Fwd:
    llvm::copy(genConfig.outputDimension, std::back_inserter(upperBounds));
    upperBounds.push_back(dimX);
    upperBounds.push_back(genConfig.filterHeight);
    upperBounds.push_back(genConfig.filterWidth);
    loopIVs.append(genConfig.outputLayout);
    loopIVs.append("cyx");
    break;
  case miopen::ConvOpType::BwdData:
    llvm::copy(genConfig.inputDimension, std::back_inserter(upperBounds));
    upperBounds.push_back(dimX);
    upperBounds.push_back(genConfig.filterHeight);
    upperBounds.push_back(genConfig.filterWidth);
    loopIVs.append(genConfig.inputLayout);
    loopIVs.append("kyx");
    break;
  case miopen::ConvOpType::BwdWeight:
    std::tie(std::ignore, out_h, out_w) =
        getConv2dBounds(miopen::ConvOpType::BwdData, genConfig);
    llvm::copy(genConfig.filterDimension, std::back_inserter(upperBounds));
    upperBounds.push_back(dimX);
    upperBounds.push_back(out_h);
    upperBounds.push_back(out_w);
    loopIVs.append(genConfig.filterLayout);
    loopIVs.append("nhw");
    break;
  }

  auto createConv2dLoopNest = [&](OpBuilder b, Location loc, ValueRange ivs) {
    mlir::Value heightIdx, widthIdx;
    mlir::Value heightTempIdx, widthTempIdx;

    switch (genConfig.operation.getValue()) {
    case miopen::ConvOpType::Fwd:
      // in_h_idx = out_h_idx * stride_h + fil_h_idx * dilation_h - padding_h_l;
      // in_w_idx = out_w_idx * stride_w + fil_w_idx * dilation_w - padding_w_l;
      heightIdx = b.create<AffineApplyOp>(
          loc, heightMap,
          ValueRange{ivs[genConfig.outputLayout.find('h')], ivs[6]});
      widthIdx = b.create<AffineApplyOp>(
          loc, widthMap,
          ValueRange{ivs[genConfig.outputLayout.find('w')], ivs[7]});
      break;
    case miopen::ConvOpType::BwdData:
      // out_h_tmp = in_h_idx + padding_h_l - fil_h_idx * dilation_h;
      // out_w_tmp = in_w_idx + padding_w_l - fil_w_idx * dilation_w;
      heightTempIdx = b.create<AffineApplyOp>(
          loc, heightMap,
          ValueRange{ivs[genConfig.inputLayout.find('h')], ivs[6]});
      widthTempIdx = b.create<AffineApplyOp>(
          loc, widthMap,
          ValueRange{ivs[genConfig.inputLayout.find('w')], ivs[7]});
      // out_h_idx = out_h_tmp / stride_h;
      // out_w_idx = out_w_tmp / stride_w;
      heightIdx = b.create<AffineApplyOp>(loc, outputHeightMap,
                                          ValueRange{heightTempIdx});
      widthIdx = b.create<AffineApplyOp>(loc, outputWidthMap,
                                         ValueRange{widthTempIdx});
      break;
    case miopen::ConvOpType::BwdWeight:
      // in_h_idx = out_h_idx * stride_h + fil_h_idx * dilation_h - padding_h_l;
      // in_w_idx = out_w_idx * stride_w + fil_w_idx * dilation_w - padding_w_l;
      heightIdx = b.create<AffineApplyOp>(
          loc, heightMap,
          ValueRange{ivs[6], ivs[genConfig.filterLayout.find('y')]});
      widthIdx = b.create<AffineApplyOp>(
          loc, widthMap,
          ValueRange{ivs[7], ivs[genConfig.filterLayout.find('x')]});
      break;
    }

    enum TENSOR { FILTER = 0, INPUT = 1, OUTPUT = 2 };
    auto getIndices = [&](TENSOR tensor, SmallVectorImpl<mlir::Value> &result) {
      std::string layout;
      if (tensor == FILTER)
        layout = genConfig.filterLayout;
      else if (tensor == INPUT)
        layout = genConfig.inputLayout;
      else
        layout = genConfig.outputLayout;
      for (auto c : layout) {
        auto direction = genConfig.operation.getValue();
        if ((direction == miopen::ConvOpType::Fwd ||
             direction == miopen::ConvOpType::BwdWeight) &&
            tensor == INPUT) {
          if (c == 'h') {
            result.push_back(heightIdx);
            continue;
          } else if (c == 'w') {
            result.push_back(widthIdx);
            continue;
          }
        } else if (direction == miopen::ConvOpType::BwdData &&
                   tensor == OUTPUT) {
          if (c == 'h') {
            result.push_back(heightIdx);
            continue;
          } else if (c == 'w') {
            result.push_back(widthIdx);
            continue;
          }
        }
        result.push_back(ivs[loopIVs.find(c)]);
      }
      return;
    };

    // Generate boundary testing
    auto dimHeight = b.create<arith::ConstantIndexOp>(loc, dimH);
    auto dimWidth = b.create<arith::ConstantIndexOp>(loc, dimW);

    AffineIfOp ifOp;
    if (genConfig.operation.getValue() == miopen::ConvOpType::BwdData) {
      ifOp = b.create<mlir::AffineIfOp>(loc, condition,
                                        ValueRange{heightIdx, widthIdx,
                                                   heightTempIdx, widthTempIdx,
                                                   dimHeight, dimWidth},
                                        false);
    } else {
      ifOp = b.create<mlir::AffineIfOp>(
          loc, condition, ValueRange{heightIdx, widthIdx, dimHeight, dimWidth},
          false);
    }
    auto thenBody = ifOp.getThenBodyBuilder();

    // Perform MAC operation
    SmallVector<mlir::Value, 5> idx1, idx2;
    BlockArgument opd1, opd2, result;

    switch (genConfig.operation.getValue()) {
    case miopen::ConvOpType::Fwd:
      getIndices(FILTER, idx1);
      getIndices(INPUT, idx2);
      opd1 = block->getArgument(0);
      opd2 = block->getArgument(1);
      result = block->getArgument(2);
      break;
    case miopen::ConvOpType::BwdWeight:
      getIndices(OUTPUT, idx1);
      getIndices(INPUT, idx2);
      opd1 = block->getArgument(2);
      opd2 = block->getArgument(1);
      result = block->getArgument(0);
      break;
    case miopen::ConvOpType::BwdData:
      getIndices(FILTER, idx1);
      getIndices(OUTPUT, idx2);
      opd1 = block->getArgument(0);
      opd2 = block->getArgument(2);
      result = block->getArgument(1);
      break;
    }
    llvm::ArrayRef<mlir::Value> idxRef1(idx1.data(), idx1.size());
    auto loadOp1 =
        thenBody.create<memref::LoadOp>(loc, opd1, ValueRange{idxRef1});
    llvm::ArrayRef<mlir::Value> idxRef2(idx2.data(), idx2.size());
    auto loadOp2 =
        thenBody.create<memref::LoadOp>(loc, opd2, ValueRange{idxRef2});
    auto loadOutput = thenBody.create<memref::LoadOp>(
        loc, result, ValueRange{ivs[0], ivs[1], ivs[2], ivs[3], ivs[4]});
    if (elemType.isIntOrIndex()) {
      auto muliOp = thenBody.create<arith::MulIOp>(loc, loadOp1, loadOp2);
      auto extsiOp = thenBody.create<arith::ExtSIOp>(loc, elemType, muliOp);
      auto addiOp = thenBody.create<arith::AddIOp>(loc, loadOutput, extsiOp);
      auto storeOp = thenBody.create<memref::StoreOp>(
          loc, addiOp, result,
          ValueRange{ivs[0], ivs[1], ivs[2], ivs[3], ivs[4]});
    } else {
      auto mulfOp = thenBody.create<arith::MulFOp>(loc, loadOp1, loadOp2);
      auto addfOp = thenBody.create<arith::AddFOp>(loc, loadOutput, mulfOp);
      auto storeOp = thenBody.create<memref::StoreOp>(
          loc, addfOp, result,
          ValueRange{ivs[0], ivs[1], ivs[2], ivs[3], ivs[4]});
    }
  };

  // Generate the loop nest
  mlir::buildAffineLoopNest(b, loc, lowerBounds, upperBounds, steps,
                            createConv2dLoopNest);

  b.create<func::ReturnOp>(loc, ValueRange{});
  return;
}

static void
createCPUConvWithCPP(ModuleOp module, func::FuncOp &func,
                     const miopen::Conv2dGenerator::Config &genConfig) {
  OpBuilder b(module.getContext());

  Block *block = func.addEntryBlock();
  b.setInsertionPoint(block, block->begin());

  auto loc = b.getUnknownLoc();

  mlir::Type elemType = b.getF32Type();
  if (genConfig.dataTypeStr == "i8") {
    elemType = b.getI8Type();
  }

  auto filterType =
      block->getArgument(0).getType().template dyn_cast<MemRefType>();
  auto outputType =
      block->getArgument(2).getType().template dyn_cast<MemRefType>();
  // Emit memref_cast.
  // %a0 = memref_cast %arg0 : memref<128x8x3x3xf32> to memref<*xf32>
  // %a1 = memref_cast %arg1 : memref<128x8x32x32xf32> to memref<*xf32>
  // %a2 = memref_cast %arg2 : memref<128x128x30x30xf32> to memref<*xf32>
  auto unrankedMemRefType =
      UnrankedMemRefType::get(filterType.getElementType(), 0);

  auto unrankedMemRefOutputType =
      UnrankedMemRefType::get(outputType.getElementType(), 0);

  auto filterMemRefCastOp =
      b.create<memref::CastOp>(loc, unrankedMemRefType, block->getArgument(0));
  auto inputMemRefCastOp =
      b.create<memref::CastOp>(loc, unrankedMemRefType, block->getArgument(1));
  auto outputMemRefCastOp = b.create<memref::CastOp>(
      loc, unrankedMemRefOutputType, block->getArgument(2));

  // Emit ConstantOps to be used for strides, paddings and dilations
  auto intType = b.getIntegerType(32);

  auto strideHeightConstantOp =
      b.create<arith::ConstantIntOp>(loc, genConfig.strideHeight, intType);

  auto strideWidthConstantOp =
      b.create<arith::ConstantIntOp>(loc, genConfig.strideWidth, intType);

  auto paddingHeightLeftConstantOp =
      b.create<arith::ConstantIntOp>(loc, genConfig.paddingHeightLeft, intType);

  auto paddingHeightRightConstantOp = b.create<arith::ConstantIntOp>(
      loc, genConfig.paddingHeightRight, intType);

  auto paddingWidthLeftConstantOp =
      b.create<arith::ConstantIntOp>(loc, genConfig.paddingWidthLeft, intType);

  auto paddingWidthRightConstantOp =
      b.create<arith::ConstantIntOp>(loc, genConfig.paddingWidthRight, intType);

  auto dilationHeightConstantOp =
      b.create<arith::ConstantIntOp>(loc, genConfig.dilationHeight, intType);

  auto dilationWidthConstantOp =
      b.create<arith::ConstantIntOp>(loc, genConfig.dilationWidth, intType);

  // Emit ConstantIndex ops
  // %c_0 = constant 0 : index
  // %c_1 = constant 1 : index
  // %c_2 = constant 2 : index
  // %c_3 = constant 3 : index
  // %c_4 = constant 4 : index
  std::vector<arith::ConstantIndexOp> indexOpVec;
  for (int i = 0; i < 5; i++) {
    auto indexOp = b.create<arith::ConstantIndexOp>(loc, i);
    indexOpVec.push_back(indexOp);
  }

  auto charType = b.getIntegerType(8);
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
  auto kConstantOp = b.create<arith::ConstantIntOp>(loc, 'k', charType);
  auto cConstantOp = b.create<arith::ConstantIntOp>(loc, 'c', charType);
  auto yConstantOp = b.create<arith::ConstantIntOp>(loc, 'y', charType);
  auto xConstantOp = b.create<arith::ConstantIntOp>(loc, 'x', charType);
  auto nConstantOp = b.create<arith::ConstantIntOp>(loc, 'n', charType);
  auto hConstantOp = b.create<arith::ConstantIntOp>(loc, 'h', charType);
  auto wConstantOp = b.create<arith::ConstantIntOp>(loc, 'w', charType);
  auto gConstantOp = b.create<arith::ConstantIntOp>(loc, 'g', charType);

  // reduce precision if !xdlops
  bool hasXdlops =
      miopen::bitEnumContains(genConfig.features, miopen::GemmFeatures::xdlops);
  auto xdlopsConstantOp =
      b.create<arith::ConstantIntOp>(loc, hasXdlops, intType);

  std::unordered_map<char, arith::ConstantIntOp> layoutConstOps;
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
  auto filLayoutAllocOp = b.create<memref::AllocaOp>(loc, layoutMemRefType);
  auto inLayoutAllocOp = b.create<memref::AllocaOp>(loc, layoutMemRefType);
  auto outLayoutAllocOp = b.create<memref::AllocaOp>(loc, layoutMemRefType);

  // Store layouts into layoutAllocOp
  // store %k, %3[%c_0]: memref<5xi32>
  std::string fil_layout = genConfig.filterLayout;
  std::string in_layout = genConfig.inputLayout;
  std::string out_layout = genConfig.outputLayout;
  for (int i = 0; i < 5; i++) {
    b.create<memref::StoreOp>(loc, layoutConstOps[fil_layout[i]],
                              filLayoutAllocOp, ValueRange{indexOpVec[i]});
  }

  for (int i = 0; i < 5; i++) {
    b.create<memref::StoreOp>(loc, layoutConstOps[in_layout[i]],
                              inLayoutAllocOp, ValueRange{indexOpVec[i]});
  }

  for (int i = 0; i < 5; i++) {
    b.create<memref::StoreOp>(loc, layoutConstOps[out_layout[i]],
                              outLayoutAllocOp, ValueRange{indexOpVec[i]});
  }

  // Emit memref_cast
  // %6 = memref_cast %3 : memref<5xi8> to memref<*xi8>
  // %7 = memref_cast %4 : memref<5xi8> to memref<*xi8>
  // %8 = memref_cast %5 : memref<5xi8> to memref<*xi8>
  auto unrankedLayoutMemRefType = UnrankedMemRefType::get(charType, 0);
  auto filLayoutMemRefCastOp =
      b.create<memref::CastOp>(loc, unrankedLayoutMemRefType, filLayoutAllocOp);
  auto inLayoutMemRefCastOp =
      b.create<memref::CastOp>(loc, unrankedLayoutMemRefType, inLayoutAllocOp);
  auto outLayoutMemRefCastOp =
      b.create<memref::CastOp>(loc, unrankedLayoutMemRefType, outLayoutAllocOp);

  std::string mcpuFuncName;

  switch (genConfig.operation.getValue()) {
  case miopen::ConvOpType::Fwd:
    mcpuFuncName = "mcpuConv2d";
    break;
  case miopen::ConvOpType::BwdData:
    mcpuFuncName = "mcpuConv2dBwdData";
    break;
  case miopen::ConvOpType::BwdWeight:
    mcpuFuncName = "mcpuConv2dBwdWeight";
    break;
  }

  if (elemType.isF32()) {
    mcpuFuncName += "Float";
  } else if (elemType.isInteger(8)) {
    mcpuFuncName += "Int8";
  }

  // Emit cpu convolution function call op
  auto mcpuConv2dFuncOp = makeFuncDecl(
      module, mcpuFuncName,
      {unrankedMemRefType, unrankedMemRefType, unrankedMemRefOutputType,
       unrankedLayoutMemRefType, unrankedLayoutMemRefType,
       unrankedLayoutMemRefType, intType, intType, intType, intType, intType,
       intType, intType, intType, intType});

  b.create<func::CallOp>(
      loc, mcpuConv2dFuncOp,
      ValueRange{filterMemRefCastOp, inputMemRefCastOp, outputMemRefCastOp,
                 filLayoutMemRefCastOp, inLayoutMemRefCastOp,
                 outLayoutMemRefCastOp, strideHeightConstantOp,
                 strideWidthConstantOp, paddingHeightLeftConstantOp,
                 paddingHeightRightConstantOp, paddingWidthLeftConstantOp,
                 paddingWidthRightConstantOp, dilationHeightConstantOp,
                 dilationWidthConstantOp, xdlopsConstantOp});

  // Emit return op
  b.create<func::ReturnOp>(loc, ValueRange{});
  return;
}

static func::FuncOp
createCPUConvFunc(ModuleOp module,
                  const miopen::Conv2dGenerator::Config &genConfig) {
  assert(genConfig.operation.hasValue());
  std::string funcName =
      miopen::getNameForConvOpType(genConfig.operation.getValue()).str();

  funcName += "_cpu";
  func::FuncOp func = module.lookupSymbol<func::FuncOp>(funcName);
  if (func) // already exists
    return func;

  OpBuilder b(module.getContext());
  auto loc = b.getUnknownLoc();

  mlir::Type elemType = b.getF32Type();
  mlir::Type outputElemType = b.getF32Type();
  if (genConfig.dataTypeStr == "i8") {
    elemType = b.getI8Type();
    outputElemType = b.getIntegerType(32);
    assert(genConfig.operation.getValue() == miopen::ConvOpType::Fwd);
  }

  auto filterDimension = genConfig.filterDimension;
  auto inputDimension = genConfig.inputDimension;
  auto outputDimension = genConfig.outputDimension;

  auto filterType = MemRefType::get(filterDimension, elemType);
  auto inputType = MemRefType::get(inputDimension, elemType);
  auto outputType = MemRefType::get(outputDimension, outputElemType);

  // Create conv2d_host function
  miopen::Conv2dGenerator conv2dGenerator(genConfig);

  bool hasWorkspace = conv2dGenerator.hasWorkspace(b);
  mlir::Type workspaceArgType;
  if (hasWorkspace) {
    workspaceArgType = MemRefType::get(
        ArrayRef<int64_t>(filterDimension.begin(), filterDimension.end()),
        b.getF32Type());
  }

  SmallVector<mlir::Type, 3> funcArgTypes = {filterType, inputType, outputType};

  if (hasWorkspace) {
    funcArgTypes = {filterType, inputType, outputType, workspaceArgType};
  }

  func =
      func::FuncOp::create(loc, funcName, b.getFunctionType(funcArgTypes, {}));
  module.push_back(func);

  if (genValidation.getValue() == "mlir") { // -pv_with_mlir or -prc
    createCPUConvWithMLIR(module, func, genConfig);
  } else { // -pv_with_cpp
    createCPUConvWithCPP(module, func, genConfig);
  }

  return func;
}

std::string getTypeStr(const mlir::Type &type) {
  std::string typeName;
  llvm::raw_string_ostream nameStream(typeName);
  nameStream << type;
  return nameStream.str();
}

static func::FuncOp getMemcpyFuncDecl(ModuleOp &module,
                                      const mlir::Type &srcElemType,
                                      const mlir::Type &dstElemType) {
  OpBuilder b(module.getContext());

  // memcpy_<srcElemType>_<dstElemType>
  std::string funcName = "_memcpy_";
  funcName += getTypeStr(srcElemType);
  funcName += "_";
  funcName += getTypeStr(dstElemType);

  func::FuncOp func = module.lookupSymbol<func::FuncOp>(funcName);
  if (func) // already exists
    return func;

  auto loc = b.getUnknownLoc();

  auto rSrcType = MemRefType::get({-1}, srcElemType);
  auto rDstType = MemRefType::get({-1}, dstElemType);

  // clang-format off
  // func _memcpy_<srcElemType>_<dstElemType> (%arg0 : memref<?xf32>, %arg1 : memref<?xf16, %arg2 : index) {
  //   scf.for %i0 = %c0 to %arg2 step %c1 {
  //     %2 = load %arg0[%i0] : memref<?xf32>
  //     store %2, %arg1[%i0] : memref<?xf32>
  //   }
  // }
  // clang-format on

  // Emit function definition
  func = func::FuncOp::create(
      loc, funcName,
      b.getFunctionType({rSrcType, rDstType, b.getIndexType()}, {}));

  module.push_back(func);

  // Create a new block
  Block *block = func.addEntryBlock();
  b.setInsertionPoint(block, block->begin());

  auto src = block->getArgument(0);
  auto dst = block->getArgument(1);
  auto size = block->getArgument(2);

  auto cst0Op = b.create<arith::ConstantIndexOp>(loc, 0);
  auto cst1Op = b.create<arith::ConstantIndexOp>(loc, 1);

  auto loop0 = b.create<scf::ForOp>(loc, cst0Op, size, cst1Op);
  auto bt0 = OpBuilder::atBlockTerminator(loop0.getBody());
  auto iv0 = loop0.getInductionVar();

  mlir::Value loadOp = bt0.create<memref::LoadOp>(loc, src, ValueRange{iv0});
  if (srcElemType != dstElemType) {
    // insert conversion logic
    auto srcBitWidth = srcElemType.getIntOrFloatBitWidth();
    auto dstBitWidth = dstElemType.getIntOrFloatBitWidth();
    if (srcElemType.isIntOrIndex()) {
      assert(!dstElemType.isIntOrIndex());
      loadOp = bt0.create<arith::SIToFPOp>(loc, dstElemType, loadOp);
    } else {
      assert(srcElemType.isa<FloatType>());
      if (dstElemType.isIntOrIndex()) {
        loadOp = bt0.create<arith::FPToSIOp>(loc, dstElemType, loadOp);
      } else {
        if (srcBitWidth < dstBitWidth)
          loadOp = bt0.create<arith::ExtFOp>(loc, dstElemType, loadOp);
        else
          loadOp = bt0.create<arith::TruncFOp>(loc, dstElemType, loadOp);
      }
    }
  }

  bt0.create<memref::StoreOp>(loc, loadOp, dst, ValueRange{iv0});

  b.create<func::ReturnOp>(loc, ValueRange{});

  return func;
}

static void emitMemcpy(OpBuilder &b, mlir::Value src, mlir::Value dst) {
  auto module = b.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
  auto loc = b.getUnknownLoc();

  auto srcType = src.getType().template dyn_cast<MemRefType>();
  auto dstType = dst.getType().template dyn_cast<MemRefType>();

  auto srcElemType = srcType.getElementType();
  auto dstElemType = dstType.getElementType();

  auto memcpyFunc = getMemcpyFuncDecl(module, srcElemType, dstElemType);

  // Emit call to memcopy
  auto srcFlatType = MemRefType::get({-1}, srcElemType);
  auto srcFlat =
      b.create<memref::CastOp>(loc, srcFlatType, makeNDMemRef(b, src, 1));
  auto dstFlatType = MemRefType::get({-1}, dstElemType);
  auto dstFlat =
      b.create<memref::CastOp>(loc, dstFlatType, makeNDMemRef(b, dst, 1));

  auto cstSize =
      b.create<arith::ConstantIndexOp>(loc, srcType.getNumElements());
  b.create<func::CallOp>(loc, memcpyFunc,
                         ValueRange{srcFlat, dstFlat, cstSize});
}

static void emitPrintTensor(OpBuilder &b, mlir::Value var) {
  auto loc = b.getUnknownLoc();
  auto varType = var.getType().template dyn_cast<MemRefType>();
  auto elemType = varType.getElementType();
  auto floatType = b.getF32Type();

  // get print func
  mlir::Value pvar = var;
  if (elemType != floatType) {
    // make copy
    auto pvarType = MemRefType::get(varType.getShape(), floatType);
    pvar = b.create<memref::AllocOp>(loc, pvarType);
    emitMemcpy(b, var, pvar);
  }

  auto module = b.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
  auto unrankedMRType = UnrankedMemRefType::get(b.getF32Type(), 0);
  auto printFunc = makeFuncDecl(module, "printMemrefF32", {unrankedMRType});

  // Emit cast + call print
  auto printCast = b.create<memref::CastOp>(loc, unrankedMRType, pvar);
  b.create<func::CallOp>(loc, printFunc, ValueRange{printCast});

  if (pvar != var) {
    // dealloc pvar
    b.create<memref::DeallocOp>(loc, pvar);
  }
}

static func::FuncOp createVerifierFunc(ModuleOp &module, const KernelIF &kernel,
                                       mlir::MemRefType testType,
                                       mlir::MemRefType valType) {
  auto kfunc = kernel.func;
  std::string funcName = kfunc.getName().str() + "_verify";
  func::FuncOp func = module.lookupSymbol<func::FuncOp>(funcName);
  if (func) // already exists
    return func;

  OpBuilder b(module.getContext());
  auto loc = b.getUnknownLoc();
  auto floatType = b.getF32Type();
  auto charType = b.getIntegerType(8);

  // Emit verify_results function call
  func = func::FuncOp::create(loc, funcName,
                              b.getFunctionType({testType, valType}, {}));
  module.push_back(func);

  // Emit verification logic.
  // Create a new block
  Block *block = func.addEntryBlock();
  b.setInsertionPoint(block, block->begin());

  // obtain function arguments
  // arg0: test result
  // arg1: validation result
  auto arg0 = block->getArgument(0);
  auto arg1 = block->getArgument(1);
  // obtain element type
  auto valElemType = valType.getElementType();
  auto testOutType = testType.getElementType();

  // Emit constants for thresholds

  // clang-format off
  // %cst = arith.constant 9.99999974E-6 : f32
  // %cst_0 = arith.constant 1.000000e+02 : f32
  // %cst_1 = arith.constant 1.000000e+02 : f32
  // clang-format on

  auto getF32Val = [&](float val) -> mlir::Value {
    llvm::APFloat apVal(val);
    return b.create<arith::ConstantFloatOp>(loc, apVal, floatType);
  };
  // Thresholds for different metrics
  // RMS: 0.00003f for all data types
  // absDiff: 100.0f for all data types, i.e. the maxAbsDiff metric is disabled
  // relDiff 100.0f for f16, i.e. maxRelDiff metric is disabled for f16
  // datatypes
  //         0.000001f for other data types
  auto thr_RMS = getF32Val(RMSThreshold.getValue());
  auto thr_absDiff = getF32Val(absDiffThreshold.getValue());
  mlir::Value thr_relDiff;
  if (testOutType.isF16())
    thr_relDiff = getF32Val(relDiffThreshold.getValue());
  else
    thr_relDiff = getF32Val(0.000001f);
  char printDebug = 1;
  std::string printVerifyOption = printVerifyResults.getValue();
  if (printVerifyOption == "always") {
    printDebug = 2;
  } else if (printVerifyOption == "failure") {
    printDebug = 1;
  } else if (printVerifyOption == "off") {
    printDebug = 0;
  } else {
    llvm::errs() << "Unsupported print-verify-results option: "
                 << printVerifyOption;
    llvm::errs() << " (supported options: always, failure, off)\n";
    exit(1);
  }
  auto printDebugVal =
      b.create<arith::ConstantIntOp>(loc, printDebug, charType);

  // obtain function name of the verifier wrapper
  std::string verifyFuncName = "mcpuVerify5D";
  if (valElemType.isF32()) {
    verifyFuncName += "Float";
  } else if (valElemType.isInteger(32)) {
    verifyFuncName += "Int32";
  } else {
    llvm::errs() << "Unsupported type of validation function output: ";
    valElemType.dump();
    llvm::errs() << " (Only f32 and int32 are supported)\n";
    exit(1);
  }

  auto mr5DUnkType = MemRefType::get({-1, -1, -1, -1, -1}, valElemType);
  bool isTestAndValSameType =
      (testOutType.isInteger(32) || testOutType.isF32());

  mlir::Value testResult, valResult; // Values passed to the verify function
  mlir::Value testResultNew;         // Values used for type conversion
  if (!isTestAndValSameType) {
    // When gpu kernel output data type = f16 | bf16, type conversions
    // are required before calling the verify function

    // Cast test result to the same type as valid result

    // clang-format off
    // %0 = memref.alloc() : memref<1x1x64x112x112xf32>
    // %1 = memref.collapse_shape %arg0 [[0, 1, 2, 3, 4]] : memref<1x1x64x112x112xf16> into memref<802816xf16>
    // %2 = memref.cast %1 : memref<802816xf16> to memref<?xf16>
    // %3 = memref.collapse_shape %0 [[0, 1, 2, 3, 4]] : memref<1x1x64x112x112xf32> into memref<802816xf32>
    // %4 = memref.cast %3 : memref<802816xf32> to memref<?xf32>
    // %c802816 = arith.constant 802816 : index
    // call @_memcpy_f16_f32(%2, %4, %c802816) : (memref<?xf16>, memref<?xf32>, index) -> ()
    // %5 = memref.cast %0 : memref<1x1x64x112x112xf32> to memref<?x?x?x?x?xf32>
    // clang-format on

    testResultNew = b.create<memref::AllocOp>(loc, valType);
    emitMemcpy(b, arg0, testResultNew);
    testResult = b.create<memref::CastOp>(loc, mr5DUnkType, testResultNew);

    // Cast valid result down to the same type as test result and cast back
    //   For f16 and bf16 datatypes, gpu hardware outputs f32 results, which are
    //   truncated to f16/bf16 before returning from the gpu kernel
    //   To make the comparison fair, the truncation step is added manually to
    //   the validation results.

    // clang-format off
    // affine.for %arg2 = 0 to 1 {
    //   affine.for %arg3 = 0 to 1 {
    //     affine.for %arg4 = 0 to 64 {
    //       affine.for %arg5 = 0 to 112 {
    //         affine.for %arg6 = 0 to 112 {
    //           %7 = memref.load %arg1[%arg2, %arg3, %arg4, %arg5, %arg6] : memref<1x1x64x112x112xf32>
    //           %8 = arith.truncf %7 : f32 to f16
    //           %9 = arith.extf %8 : f16 to f32
    //           memref.store %9, %arg1[%arg2, %arg3, %arg4, %arg5, %arg6] : memref<1x1x64x112x112xf32>
    //         }
    //       }
    //     }
    //   }
    // }
    // clang-format on

    SmallVector<int64_t, 5> lowerBounds(5, 0);
    SmallVector<int64_t, 5> upperBounds;
    llvm::copy(valType.getShape(), std::back_inserter(upperBounds));
    SmallVector<int64_t, 5> steps(5, 1);

    mlir::buildAffineLoopNest(
        b, loc, lowerBounds, upperBounds, steps,
        [arg1, testOutType, valElemType](OpBuilder b, Location loc,
                                         ValueRange ivs) {
          auto valOrig = b.create<memref::LoadOp>(loc, arg1, ivs);
          auto valTruncated =
              b.create<arith::TruncFOp>(loc, testOutType, valOrig);
          auto valExt = b.create<arith::ExtFOp>(loc, valElemType, valTruncated);
          b.create<memref::StoreOp>(loc, valExt, arg1, ivs);
        });
  } else {
    testResult = b.create<memref::CastOp>(loc, mr5DUnkType, arg0);
  }

  // Prepare the validation result for the verify function
  valResult = b.create<memref::CastOp>(loc, mr5DUnkType, arg1);
  // Declare and call the wrapper verify function
  auto verifyFuncDecl = makeFuncDecl(
      module, verifyFuncName,
      {mr5DUnkType, mr5DUnkType, floatType, floatType, floatType, charType});
  b.create<func::CallOp>(loc, verifyFuncDecl,
                         ValueRange{testResult, valResult, thr_RMS, thr_absDiff,
                                    thr_relDiff, printDebugVal});

  if (!isTestAndValSameType) {
    // Deallocate the buffer for f32 version of the test results
    b.create<memref::DeallocOp>(loc, testResultNew);
  }

  b.create<func::ReturnOp>(loc, ValueRange{});

  return func;
}

static LogicalResult populateTensorFillLogic(mlir::OpBuilder &b,
                                             mlir::Location loc,
                                             mlir::Type elemType,
                                             mlir::Value lv5D) {
  auto lv5DType = lv5D.getType().template cast<mlir::MemRefType>();
  llvm::SmallVector<float, 3> pattern;
  if (elemType.isIntOrIndex())
    pattern = {1.0, -1.0, 2.0};
  else
    pattern = {0.5, -1, 0.75};
  // TODO(kdrewnia) Refactor this to create the constant vector up front
  // TODO(kdrewnia) Factor out the anti-bf16 pass from GPU lowering, apply
  // it here
  mlir::Type i16 = b.getIntegerType(16);
  mlir::Value constantsVec;
  if (elemType == b.getBF16Type()) {
    uint16_t init = 0;
    constantsVec = b.create<arith::ConstantOp>(
        loc, mlir::SplatElementsAttr::get(
                 mlir::VectorType::get(pattern.size(), i16), init));
  } else {
    constantsVec = mlir::miopen::createZeroConstantOp(
        b, loc, mlir::VectorType::get(pattern.size(), elemType));
  }
  for (auto v : llvm::enumerate(pattern)) {
    mlir::Value vOp;
    if (elemType == b.getBF16Type()) {
      llvm::APFloat fl(v.value());
      bool losesInfo = false;
      fl.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                 &losesInfo);
      llvm::APInt val = fl.bitcastToAPInt();
      vOp = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(i16, val));
    } else if (elemType.isIntOrIndex()) {
      vOp = mlir::miopen::createConstantIntOp(b, loc, elemType, elemType,
                                              static_cast<int64_t>(v.value()));
    } else {
      vOp = mlir::miopen::createConstantFloatOp(b, loc, elemType, elemType,
                                                v.value());
    }
    constantsVec = b.create<vector::InsertElementOp>(
                        loc, vOp, constantsVec,
                        b.create<arith::ConstantIndexOp>(loc, v.index()))
                       .getResult();
  }

  SmallVector<int64_t, 5> lowerBounds(5, 0);
  SmallVector<int64_t, 5> upperBounds;
  llvm::copy(lv5DType.getShape(), std::back_inserter(upperBounds));
  SmallVector<int64_t, 5> steps(5, 1);
  AffineExpr rowMajor = b.getAffineConstantExpr(0);
  for (uint32_t i = 0; i < 5; ++i) {
    rowMajor = b.getAffineDimExpr(i) + lv5DType.getDimSize(i) * rowMajor;
  }
  rowMajor = rowMajor % b.getAffineConstantExpr(pattern.size());
  AffineMap rowMajorMap = AffineMap::get(5, 0, {rowMajor}, b.getContext());

  mlir::buildAffineLoopNest(
      b, loc, lowerBounds, upperBounds, steps,
      [rowMajorMap, &constantsVec, lv5D, elemType](OpBuilder b, Location loc,
                                                   ValueRange ivs) {
        auto selectorOp = b.create<AffineApplyOp>(loc, rowMajorMap, ivs);
        mlir::Value toStore = b.create<vector::ExtractElementOp>(
                                   loc, constantsVec, selectorOp->getResult(0))
                                  .getResult();
        if (elemType == b.getBF16Type())
          toStore = b.create<arith::BitcastOp>(loc, b.getBF16Type(), toStore);
        b.create<memref::StoreOp>(loc, toStore, lv5D, ivs);
      });
  return success();
}

static LogicalResult
populateHostHarnessLogic(ModuleOp &module,
                         const SmallVector<KernelIF, 8> &kernels,
                         const SmallVector<KernelIF, 8> &roots,
                         const miopen::Conv2dGenerator::Config &genConfig) {

  auto context = module.getContext();
  OpBuilder b(context);
  auto loc = b.getUnknownLoc();

  // Construct main function.
  auto func = func::FuncOp::create(loc, "main", b.getFunctionType({}, {}));
  module.push_back(func);

  // Construct a new Block.
  Block *block = func.addEntryBlock();
  b.setInsertionPoint(block, block->begin());

  int32_t outIdx = -1;
  if (genConfig.operation.hasValue()) {
    switch (genConfig.operation.getValue()) {
    case miopen::ConvOpType::Fwd:
      outIdx = 2;
      break;
    case miopen::ConvOpType::BwdData:
      outIdx = 1;
      break;
    case miopen::ConvOpType::BwdWeight:
      outIdx = 0;
      break;
    }
  }

  llvm::SmallDenseMap<short, mlir::Value> i16vals;
  auto getI16Val = [&](short v) {
    if (i16vals.find(v) == i16vals.end()) {
      auto i16Type = b.getIntegerType(16);
      i16vals.try_emplace(v, b.create<arith::ConstantIntOp>(loc, v, i16Type));
    }
    return i16vals[v];
  };

  llvm::SmallDenseMap<int32_t, mlir::Value> i32vals;
  auto getI32Val = [&](int32_t v) {
    if (i32vals.find(v) == i32vals.end()) {
      auto i32Type = b.getIntegerType(32);
      i32vals.try_emplace(v, b.create<arith::ConstantIntOp>(loc, v, i32Type));
    }
    return i32vals[v];
  };
  auto floatType = b.getF32Type();

  auto validationType = genValidation.getValue();

  // Create all local variables for each kernel param
  // - assumes all kernels read the same memrefs
  if (roots.size() > 1) {
    // TODO: verify that all parameter lists match
  }
  auto root0 = *roots.begin();
  bool isCPUKernel = !root0.func->hasAttr("kernel");
  bool hasValidation = !validationType.empty() && !genCPUKernel.getValue();
  SmallVector<mlir::Value, 5> localVars;
  SmallVector<mlir::Value, 5> valVars;
  int32_t idx = 0;
  for (auto &paramType : root0.params) {
    auto paramMRType = paramType.template dyn_cast<MemRefType>();
    assert(paramMRType && "currently only supports memref types");
    auto elemType = paramMRType.getElementType();
    if (isCPUKernel) {
      assert(elemType.isF32() || elemType.isInteger(8) ||
             elemType.isInteger(32));
      if (genConfig.dataTypeStr == "f32")
        elemType = b.getF32Type();
      else if (genConfig.dataTypeStr == "f16")
        elemType = b.getF16Type();
      else if (genConfig.dataTypeStr == "bf16")
        elemType = b.getBF16Type();
      else if (genConfig.dataTypeStr == "i8") {
        elemType = b.getI8Type();
        if (idx == 2) {
          elemType = b.getIntegerType(32);
        }
      }
      paramMRType = MemRefType::get(paramMRType.getShape(), elemType);
    }
    auto mr5DUnkType = MemRefType::get({-1, -1, -1, -1, -1}, elemType);
    auto lvar = b.create<memref::AllocOp>(loc, paramMRType);
    localVars.push_back(lvar);

    auto lv5D = makeNDMemRef(b, lvar, 5);
    if (randomSeed.getValue() == "fixed") {
      if (failed(populateTensorFillLogic(b, loc, elemType, lv5D)))
        return failure();
    } else {
      auto lvU5D = b.create<memref::CastOp>(loc, mr5DUnkType, lv5D);

      short min, max;
      int seed = 1;
      std::tie(min, max, seed) = getRandomTestData(idx);

      b.create<func::CallOp>(
          loc, getMemsetFunc(module, elemType),
          ValueRange{lvU5D, getI16Val(min), getI16Val(max), getI32Val(seed)});
    }

    if (hasValidation ||
        (isCPUKernel && (elemType.isF16() || elemType.isBF16()))) {
      // Emit validation var
      mlir::Type valElemType = elemType;
      valElemType = floatType; // QY fix this later
      if (genConfig.dataTypeStr == "i8") {
        valElemType = elemType;
      }
      auto valType = MemRefType::get(paramMRType.getShape(), valElemType);
      auto vvar = b.create<memref::AllocOp>(loc, valType);
      valVars.push_back(vvar);

      emitMemcpy(b, lvar, vvar);
    }
    idx++;
  }

  // capture result index
  if (outIdx < 0) {
    outIdx = localVars.size() - 1;
  }

  // Call the roots.
  for (auto &root : roots) {
    // Is the root also a kernel?
    bool rootKernel =
        std::find_if(kernels.begin(), kernels.end(), [&](const KernelIF &k) {
          return k.func == root.func;
        }) != kernels.end();
    if (rootKernel) {
      b.create<func::CallOp>(loc, root.func, localVars);
    } else if (!valVars.empty()) {
      b.create<func::CallOp>(loc, root.func, valVars);
      if (!root.func->hasAttr("kernel")) {
        printValidationResults = true;
        printResults = false;
      }
    } else {
      b.create<func::CallOp>(loc, root.func, localVars);
      if (!root.func->hasAttr("kernel")) {
        printValidationResults = false;
        printResults = true;
      }
    }
  }

  // Wrap the kernels and gather them to substitute in calls.
  llvm::SmallDenseMap<func::FuncOp, func::FuncOp> wrappedFuncs;
  for (auto &kernel : kernels) {
    if (kernel.func->hasAttr("kernel")) {
      wrappedFuncs[kernel.func] = createGPUWrapper(module, kernel);
    } else {
      wrappedFuncs[kernel.func] = kernel.func;
    }
  }

  // Redirect calls to kernel functions to point at wrapped functions.
  module.walk([&](func::CallOp call) -> WalkResult {
    // Don't substitute the call inside the wrapper.
    if (call->hasAttr("wrapped_call")) {
      call->removeAttr("wrapped_call");
      return WalkResult::advance();
    }

    // If the callee matches a wrapped function, update the call.
    Operation *op = call;
    CallOpInterface callInt = dyn_cast<CallOpInterface>(op);
    Operation *callableFromInt = callInt.resolveCallable();
    func::FuncOp fop = dyn_cast<func::FuncOp>(*callableFromInt);
    if (wrappedFuncs.find(fop) != wrappedFuncs.end()) {
      call->setAttr("callee", FlatSymbolRefAttr::get(
                                  context, wrappedFuncs[fop].getSymName()));
    }
    return WalkResult::advance();
  });

  // Run validation
  bool hasXdlops =
      miopen::bitEnumContains(genConfig.features, miopen::GemmFeatures::xdlops);
  if (hasValidation) {
    if (validationType == "gpu" &&
        (hasXdlops || genConfig.dataTypeStr == "f16" ||
         genConfig.dataTypeStr == "bf16")) {
      // generate generic kernels
      miopen::Conv2dGenerator conv2dGenerator(genConfig);
      // use non-xdlops kernels to verify xdlops kernels
      if (hasXdlops)
        conv2dGenerator.flipXdlops();
      if (!hasXdlops || genConfig.dataTypeStr != "i8")
        // use f32 data type to verify non-f32 or xdlops f32 kernels
        // except that i8 xdlops is verified with i8 non-xdlops
        conv2dGenerator.setDataType("f32");

      int kernelStart = genConfig.kernelId;
      int kernelCount = conv2dGenerator.getKernelCount(b);
      if (kernelStart < 0) {
        kernelStart = 0;
      } else {
        kernelCount = kernelStart + 1;
      }
      // generate all sub-kernels, and get corresponding gemmId
      std::string kernelBaseName = genConfig.kernelBaseName;
      for (int i = kernelStart; i < kernelCount; ++i) {
        conv2dGenerator.setKernelName(kernelBaseName + "_" + std::to_string(i));
        if (failed(conv2dGenerator.genConvModule(module, i, true,
                                                 /*ignoreTuning=*/true))) {
          llvm::errs() << "Module population failed.\n";
          exit(1);
        }
        KernelIF kernel(conv2dGenerator.getKernelFunc());
        miopen::Conv2dGenerator::Config newConfig = conv2dGenerator.getConfig();
        auto kernelWrapperFunc = createGPUWrapper(module, kernel);

        // Decide whether to trim the last workspace argument to the verifier
        // GPU kernel.
        miopen::Conv2dGenerator originalConv2dGenerator(genConfig);
        bool originalHasWorkspace = originalConv2dGenerator.hasWorkspace(b);
        bool verifierHasWorkspace = conv2dGenerator.hasWorkspace(b);
        if (originalHasWorkspace && !verifierHasWorkspace) {
          valVars.resize(valVars.size() - 1);
        }

        b.create<func::CallOp>(loc, kernelWrapperFunc, valVars);
      }
      conv2dGenerator.setKernelName(kernelBaseName);
    } else { // -pv_with_cpp or -pv_with_mlir (-pv)
      // Emit call to host_<conv>
      auto cpuConvFunc = createCPUConvFunc(module, genConfig);
      b.create<func::CallOp>(loc, cpuConvFunc, valVars);
    }

    // Emit call to verifier
    mlir::Value testResult = localVars[outIdx];
    mlir::Value valResult = valVars[outIdx];

    auto testType = testResult.getType().dyn_cast<MemRefType>();
    auto valType = valResult.getType().dyn_cast<MemRefType>();
    auto verifierFunc = createVerifierFunc(module, root0, testType, valType);
    b.create<func::CallOp>(loc, verifierFunc,
                           ValueRange{testResult, valResult});
  }

  // Print and cleanup validation vars
  for (auto &vvar : valVars) {
    // print vvar
    if ((vvar == valVars[outIdx]) && printValidationResults.getValue())
      emitPrintTensor(b, vvar);
    // dealloc vvar
    b.create<memref::DeallocOp>(loc, vvar);
  }

  // Print and cleanup
  for (auto &lvar : localVars) {
    // print lvar
    bool printp = printInputs.getValue();
    if (lvar == localVars[outIdx])
      printp = printResults.getValue();
    if (printp)
      emitPrintTensor(b, lvar);
    // dealloc lvar
    b.create<memref::DeallocOp>(loc, lvar);
  }

  b.create<func::ReturnOp>(loc, ValueRange{});

  return success();
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
#ifdef MLIR_INCLUDE_TESTS
  test::registerTestDialect(registry);
#endif
  MLIRContext context(registry);
  context.loadDialect<miopen::MIOpenDialect, func::FuncDialect, scf::SCFDialect,
                      AffineDialect, memref::MemRefDialect, math::MathDialect,
                      arith::ArithmeticDialect, vector::VectorDialect,
                      gpu::GPUDialect>();

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv,
                              "MLIR MIOpen Dialect host generation\n");

  OpBuilder builder(&context);
  ModuleOp module;

  verifyLayout();
  correctParameters();
  populateDefaults();

  miopen::Conv2dGenerator conv2dGenerator;

  SmallVector<KernelIF, 8> kernels;

  auto testFuncNameVal = testFuncName.getValue();
  bool hasUserKernel = !testFuncNameVal.empty();

  std::string errorMessage;
  auto inputFilenameStr = inputFilename.getValue();
  if (inputFilenameStr.size()) {
    // Set up the input file.
    auto file = openInputFile(inputFilename, &errorMessage);
    if (!file) {
      llvm::errs() << errorMessage << "\n";
      exit(1);
    }

    // Parse the input file.
    SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
    OwningOpRef<ModuleOp> moduleRef =
        parseSourceFile<ModuleOp>(sourceMgr, &context);
    if (!moduleRef) {
      llvm::errs() << "Parse host harness " << inputFilename << " failed.\n";
      exit(1);
    }
    module = moduleRef.release();

    module.walk([&](func::FuncOp func) -> WalkResult {
      if (func->hasAttr("kernel")) {
        hasUserKernel = true;
      }
      return WalkResult::advance();
    });
  } else {
    // Construct a new ModuleOp.
    module = ModuleOp::create(builder.getUnknownLoc());
  }

  if (!hasUserKernel) {
    auto convConfig = populateConvConfig.getValue();

    if (convConfig.empty() && failed(detectMissingArguments())) {
      exit(1);
    }

    // Scenario 1: We use conv config to initialize everything
    if (!convConfig.empty()) {
      if (failed(conv2dGenerator.parseConvConfig(convConfig.c_str()))) {
        llvm::errs() << "Module population failed.\n";
        exit(1);
      }
      // Scenario 2: We use cl::opt to initialize everything
    } else {
      std::string chip, triple, chipFeatures;
      IsaNameSplitter splitter(arch.getValue());
      auto status = splitter.parseIsaName(chip, triple, chipFeatures);
      if (status.failed()) {
        exit(1);
      }

      miopen::GemmFeatures enabledFeatures = miopen::GemmFeatures::none;
      if (xdlopsV2.getValue())
        enabledFeatures = enabledFeatures | miopen::GemmFeatures::xdlops;
      conv2dGenerator = miopen::Conv2dGenerator(
          chip, triple, chipFeatures, perfConfig.getValue(), num_cu.getValue(),
          enabledFeatures, operation.getValue(), tensorDataType.getValue(),
          dilationHeight.getValue(), dilationWidth.getValue(),
          strideHeight.getValue(), strideWidth.getValue(),
          paddingHeightLeft.getValue(), paddingHeightRight.getValue(),
          paddingWidthLeft.getValue(), paddingWidthRight.getValue(),
          filterLayout.getValue(), inputLayout.getValue(),
          outputLayout.getValue());

      status = conv2dGenerator.parseConvDims(
          batchSize, groupSize, inputChannel, inputHeight, inputWidth,
          outputChannel, outputHeight, outputWidth, filterHeight, filterWidth);
      if (failed(status)) {
        llvm::errs() << "Could not parse convolution dimensions\n";
        exit(1);
      }
    }

    // TODO: Extract isApplicable check to be its own component
    if (failed(conv2dGenerator.isApplicable())) {
      llvm::errs() << "Convolution configuration not applicable\n";
      exit(1);
    }
  }

  const auto &genConfig = conv2dGenerator.getConfig();

  if (!hasUserKernel) {
    if (genCPUKernel.getValue()) {
      (void)createCPUConvFunc(module, genConfig);
    } else {
      // Populate the module.
      int kernelStart = genConfig.kernelId;
      int kernelCount = conv2dGenerator.getKernelCount(builder);
      if (kernelStart < 0) {
        kernelStart = 0;
      } else {
        kernelCount = kernelStart + 1;
      }
      // generate all sub-kernels, and get corresponding gemmId
      std::string kernelBaseName = genConfig.kernelBaseName;
      for (int i = kernelStart; i < kernelCount; ++i) {
        conv2dGenerator.setKernelName(kernelBaseName + "_" + std::to_string(i));
        if (failed(conv2dGenerator.genConvModule(module, i))) {
          llvm::errs() << "Module population failed.\n";
          exit(1);
        }
      }
      conv2dGenerator.setKernelName(kernelBaseName);
    }
  }

  // Compute set of call-graph root nodes;  they're the ones we need to
  // call from main().  Start with all nodes, then erase the ones that
  // have edges to them.  Use SetVector because we want to preserve the
  // order to match an older implementation.
  CallGraph cg(module);
  mlir::SetVector<CallGraphNode *> roots(cg.begin(), cg.end());
  for (auto &node : roots)
    for (auto &edge : *node)
      roots.remove(edge.getTarget());

  // Make KernelIFs for the roots, to pass to populateHostHarnessLogic().
  SmallVector<KernelIF, 8> rootIFs;
  for (auto node : roots) {
    func::FuncOp func =
        dyn_cast<func::FuncOp>(node->getCallableRegion()->getParentOp());
    rootIFs.emplace_back(func);
  }

  if (testFuncNameVal.empty()) {
    module.walk([&](func::FuncOp func) -> WalkResult {
      if (func->hasAttr("kernel")) {
        kernels.emplace_back(func);
      }
      return WalkResult::advance();
    });
  } else {
    auto func = module.lookupSymbol<func::FuncOp>(testFuncName);
    assert(func);
    rootIFs.clear();
    kernels.emplace_back(func);
    rootIFs.emplace_back(func);
  }

  // populate host logic.
  if (genHostHarness.getValue()) {
    if (failed(populateHostHarnessLogic(module, kernels, rootIFs, genConfig))) {
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

  module.print(output->os(), OpPrintingFlags().printGenericOpForm());
  output->keep();
  return 0;
}
