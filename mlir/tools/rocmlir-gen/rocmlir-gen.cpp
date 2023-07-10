//===- rocmlir-gen.cpp - MLIR Rock Test Generator ------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for rocmlir-gen test generator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Conversion/RocMLIRPasses.h"
#include "mlir/Conversion/RockToGPU/RockToGPU.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/Generator/Conv2dGenerator.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/ExecutionEngine/RocmDeviceName.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/InitRocMLIRDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "bf16convert.hpp"
#include <unordered_map>

#include <tuple>

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init(""));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    testFuncName("func-under-test", llvm::cl::desc("Name of func to test"),
                 llvm::cl::init(""));
static llvm::cl::alias aliasTestFuncName("fut",
                                         llvm::cl::aliasopt(testFuncName));

//////////////////////////////////////////////////////////////////////////////////////////////////////
//// Rock Convolution spec

static llvm::cl::opt<rock::KernelType> operation(
    "operation", llvm::cl::desc("Convolution operation,"),
    llvm::cl::values(
        clEnumValN(rock::KernelType::Conv2D, "conv2d", "Forward convolution"),
        clEnumValN(rock::KernelType::Conv2DBwdData, "conv2d_bwd_data",
                   "Backpropogate convolution data"),
        clEnumValN(rock::KernelType::Conv2DBwdWeight, "conv2d_bwd_weight",
                   "Backpropogate convolution weights"),
        clEnumValN(rock::KernelType::Gemm, "gemm", "Matrix multiplication")),
    llvm::cl::value_desc("kernel type"),
    llvm::cl::init(rock::KernelType::Conv2D));

static llvm::cl::opt<std::string> arch(
    "arch",
    llvm::cl::desc("amdgpu architecture, eg: gfx803, gfx900, gfx906, gfx908"),
    llvm::cl::value_desc("GFX architecture string"), llvm::cl::init(""));

static llvm::cl::opt<int> num_cu(
    "num_cu",
    llvm::cl::desc("Number of compute units, valid combinations include: "
                   "gfx803(36/64), gfx900(56/64), "
                   "gfx906(60/64), gfx908(120)"),
    llvm::cl::value_desc("compute unit value"), llvm::cl::init(64));

static llvm::cl::opt<std::string> perfConfig(
    "perf_config", llvm::cl::desc("performance config data used for tuning"),
    llvm::cl::value_desc("Serialized tuning parameters"), llvm::cl::init(""));

static llvm::cl::opt<std::string>
    filterLayout("fil_layout", llvm::cl::desc("Filter layout"),
                 llvm::cl::value_desc("layout string"),
                 llvm::cl::init("gkcyx"));

static llvm::cl::opt<std::string>
    inputLayout("in_layout", llvm::cl::desc("Input layout"),
                llvm::cl::value_desc("layout string"), llvm::cl::init("ngchw"));

static llvm::cl::opt<std::string>
    outputLayout("out_layout", llvm::cl::desc("Output layout"),
                 llvm::cl::value_desc("layout string"),
                 llvm::cl::init("ngkhw"));

static llvm::cl::opt<int64_t> groupSize("groupsize",
                                        llvm::cl::desc("Group size"),
                                        llvm::cl::value_desc("dimension value"),
                                        llvm::cl::init(1));
static llvm::cl::alias groupSizeShort("g",
                                      llvm::cl::desc("alias for -groupsize"),
                                      llvm::cl::aliasopt(groupSize));

// N
static llvm::cl::opt<int64_t> batchSize("batchsize",
                                        llvm::cl::desc("Batch size"),
                                        llvm::cl::value_desc("dimension value"),
                                        llvm::cl::init(-1));

// C
static llvm::cl::opt<int64_t>
    inputChannel("in_channels", llvm::cl::desc("Input channels"),
                 llvm::cl::value_desc("dimension value"), llvm::cl::init(-1));

// Hi
static llvm::cl::opt<int64_t>
    inputHeight("in_h", llvm::cl::desc("Input height"),
                llvm::cl::value_desc("dimension value"), llvm::cl::init(-1));

// Wi
static llvm::cl::opt<int64_t>
    inputWidth("in_w", llvm::cl::desc("Input width"),
               llvm::cl::value_desc("dimension value"), llvm::cl::init(-1));

// K
static llvm::cl::opt<int64_t>
    outputChannel("out_channels", llvm::cl::desc("Output channels"),
                  llvm::cl::value_desc("dimension value"), llvm::cl::init(-1));

// Y
static llvm::cl::opt<int64_t>
    filterWidth("fil_w", llvm::cl::desc("Filter width"),
                llvm::cl::value_desc("dimension value"), llvm::cl::init(-1));

// X
static llvm::cl::opt<int64_t>
    filterHeight("fil_h", llvm::cl::desc("Filter height"),
                 llvm::cl::value_desc("dimension value"), llvm::cl::init(-1));

// Ho
static llvm::cl::opt<int64_t> outputHeight(
    "out_h", llvm::cl::desc("Output height"),
    llvm::cl::value_desc("ouput dimension value, does not need to set."),
    llvm::cl::init(-1));

// Wo
static llvm::cl::opt<int64_t> outputWidth(
    "out_w", llvm::cl::desc("Output width"),
    llvm::cl::value_desc("ouput dimension value, does not need to set."),
    llvm::cl::init(-1));

// dilation height
static llvm::cl::opt<int>
    dilationHeight("dilation_h", llvm::cl::desc("Dilation height"),
                   llvm::cl::value_desc("attribute value"), llvm::cl::init(1));

// dilation width
static llvm::cl::opt<int> dilationWidth("dilation_w",
                                        llvm::cl::desc("Dilation width"),
                                        llvm::cl::value_desc("attribute value"),
                                        llvm::cl::init(1));

// stride height
static llvm::cl::opt<int> strideHeight("conv_stride_h",
                                       llvm::cl::desc("Stride height"),
                                       llvm::cl::value_desc("attribute value"),
                                       llvm::cl::init(1));

// stride width
static llvm::cl::opt<int> strideWidth("conv_stride_w",
                                      llvm::cl::desc("Stride width"),
                                      llvm::cl::value_desc("attribute value"),
                                      llvm::cl::init(1));

// padding height
static llvm::cl::opt<int> paddingHeight("padding_h",
                                        llvm::cl::desc("Padding height"),
                                        llvm::cl::value_desc("attribute value"),
                                        llvm::cl::init(0));

static llvm::cl::opt<int>
    paddingHeightLeft("padding_h_l", llvm::cl::desc("Padding height Left"),
                      llvm::cl::value_desc("attribute value"),
                      llvm::cl::init(0));

static llvm::cl::opt<int>
    paddingHeightRight("padding_h_r", llvm::cl::desc("Padding height Right"),
                       llvm::cl::value_desc("attribute value"),
                       llvm::cl::init(0));
// padding width
static llvm::cl::opt<int> paddingWidth("padding_w",
                                       llvm::cl::desc("Padding width"),
                                       llvm::cl::value_desc("attribute value"),
                                       llvm::cl::init(0));

static llvm::cl::opt<int>
    paddingWidthLeft("padding_w_l", llvm::cl::desc("Padding width Left"),
                     llvm::cl::value_desc("attribute value"),
                     llvm::cl::init(0));

static llvm::cl::opt<int>
    paddingWidthRight("padding_w_r", llvm::cl::desc("Padding width Right"),
                      llvm::cl::value_desc("attribute value"),
                      llvm::cl::init(0));

/// Matrix options
static llvm::cl::opt<int64_t> gemmM("m",
                                    llvm::cl::desc("M dimennsion of gemm()"),
                                    llvm::cl::value_desc("positive integer"),
                                    llvm::cl::init(-1));

static llvm::cl::opt<int64_t> gemmK("k",
                                    llvm::cl::desc("K dimennsion of gemm()"),
                                    llvm::cl::value_desc("positive integer"),
                                    llvm::cl::init(-1));

static llvm::cl::opt<int64_t> gemmN("n",
                                    llvm::cl::desc("N dimennsion of gemm()"),
                                    llvm::cl::value_desc("positive integer"),
                                    llvm::cl::init(-1));

static llvm::cl::opt<bool>
    transposeA("transA",
               llvm::cl::desc("whether matrix A is GxMxK (default) or GxKxM"),
               llvm::cl::init(false));

static llvm::cl::opt<bool>
    transposeB("transB",
               llvm::cl::desc("whether matrix B is GxKxN (default) or GxNxK"),
               llvm::cl::init(false));

static llvm::cl::opt<bool>
    transposeC("transC",
               llvm::cl::desc("whether matrix C is GxMxN (default) or GxNxM"),
               llvm::cl::init(false));

static llvm::cl::opt<rock::StoreMethod> storeMethod(
    "store-method", llvm::cl::desc("storage method for gemm"),
    llvm::cl::values(
        clEnumValN(rock::StoreMethod::Set, "set", "set results in C (default)"),
        clEnumValN(rock::StoreMethod::AtomicAdd, "atomic_add",
                   "atomically add results to values in matrix C")),
    llvm::cl::init(rock::StoreMethod::Set));

// A toggle to control whether a feature should be added to the feature list
enum class FeatureToggle : uint32_t { infer, on, off };

// use the toggle on each feature
// mfma
static llvm::cl::opt<FeatureToggle> mfmaFeature(
    "mfma", llvm::cl::desc("toggle feature mfma"),
    llvm::cl::values(clEnumValN(FeatureToggle::infer, "infer",
                                "use the default value provided by the chip"),
                     clEnumValN(FeatureToggle::on, "on",
                                "force mfma into the feature list"),
                     clEnumValN(FeatureToggle::off, "off",
                                "remove mfma from the feature list")),
    llvm::cl::init(FeatureToggle::infer));

// wmma
static llvm::cl::opt<FeatureToggle> wmmaFeature(
    "wmma", llvm::cl::desc("toggle feature wmma"),
    llvm::cl::values(clEnumValN(FeatureToggle::infer, "infer",
                                "use the default value provided by the chip"),
                     clEnumValN(FeatureToggle::on, "on",
                                "force wmma into the feature list"),
                     clEnumValN(FeatureToggle::off, "off",
                                "remove wmma from the feature list")),
    llvm::cl::init(FeatureToggle::infer));

// dot
static llvm::cl::opt<FeatureToggle> dotFeature(
    "dot", llvm::cl::desc("toggle feature dot"),
    llvm::cl::values(clEnumValN(FeatureToggle::infer, "infer",
                                "use the default value provided by the chip"),
                     clEnumValN(FeatureToggle::on, "on",
                                "force dot into the feature list"),
                     clEnumValN(FeatureToggle::off, "off",
                                "remove dot from the feature list")),
    llvm::cl::init(FeatureToggle::infer));

// atomicAdd
static llvm::cl::opt<FeatureToggle> atomicAddFeature(
    "atomic_add", llvm::cl::desc("toggle feature atomic_add"),
    llvm::cl::values(clEnumValN(FeatureToggle::infer, "infer",
                                "use the default value provided by the chip"),
                     clEnumValN(FeatureToggle::on, "on",
                                "force atomic_add into the feature list"),
                     clEnumValN(FeatureToggle::off, "off",
                                "remove atomic_add from the feature list")),
    llvm::cl::init(FeatureToggle::infer));

// atomicFmaxF32
static llvm::cl::opt<FeatureToggle> atomicFMaxF32Feature(
    "atomic_fmax_f32", llvm::cl::desc("toggle feature atomic_fmax_f32"),
    llvm::cl::values(clEnumValN(FeatureToggle::infer, "infer",
                                "use the default value provided by the chip"),
                     clEnumValN(FeatureToggle::on, "on",
                                "force atomic_add into the feature list"),
                     clEnumValN(FeatureToggle::off, "off",
                                "remove atomic_add from the feature list")),
    llvm::cl::init(FeatureToggle::infer));

static llvm::cl::opt<std::string>
    filterDataType("fil_dtype",
                   llvm::cl::desc("Data type for filter tensor or matrix A"),
                   llvm::cl::init("f32"));
static llvm::cl::alias filTypeAliasA("a_dtype",
                                     llvm::cl::aliasopt(filterDataType));
static llvm::cl::alias filTypeAliasShortF("tf",
                                          llvm::cl::aliasopt(filterDataType));
static llvm::cl::alias filTypeAliasShortA("ta",
                                          llvm::cl::aliasopt(filterDataType));

static llvm::cl::opt<std::string>
    inputDataType("in_dtype",
                  llvm::cl::desc("Data type for input tensor or matrix B"),
                  llvm::cl::init("f32"));
static llvm::cl::alias inTypeAliasB("b_dtype",
                                    llvm::cl::aliasopt(inputDataType));
static llvm::cl::alias inTypeAliasShortI("ti",
                                         llvm::cl::aliasopt(inputDataType));
static llvm::cl::alias inTypeAliasShortB("tb",
                                         llvm::cl::aliasopt(inputDataType));

// Note that this is defaulted to blank so we can implement `-t` easily
// and know if we should use default i8 input -> i32 output behavior.
static llvm::cl::opt<std::string>
    outputDataType("out_dtype",
                   llvm::cl::desc("Data type for output tensor or matrix C"),
                   llvm::cl::init(""));
static llvm::cl::alias outTypeAliasC("c_dtype",
                                     llvm::cl::aliasopt(outputDataType));
static llvm::cl::alias outTypeAliasLongOut("out_datatype",
                                           llvm::cl::aliasopt(outputDataType));
static llvm::cl::alias outTypeAliasShortO("to",
                                          llvm::cl::aliasopt(outputDataType));
static llvm::cl::alias outTypeAliasShortC("tc",
                                          llvm::cl::aliasopt(outputDataType));

// Convenience setter for when you need all the data types the same or when you
// want the default (32-bit output) behavior for i8 or 8-bit floats. Also allows
// [a]_[b] syntax for mixed-type operations.
static llvm::cl::opt<std::string> dataTypeAlias(
    "t",
    llvm::cl::desc("Data type selector. Extends i8 to i32 output and 8-bit "
                   "floats to f32 output"),
    llvm::cl::value_desc("Type or Type_Type for mixed-type kernels."),
    llvm::cl::cb<void, std::string>([](std::string v) {
      StringRef val(v);
      if (val.contains("_")) {
        StringRef filter, input;
        std::tie(filter, input) = val.split("_");
        filterDataType = filter.str();
        inputDataType = input.str();
      } else {
        filterDataType = v;
        inputDataType = v;
      }

      if (outputDataType.getNumOccurrences() == 0 || outputDataType.empty()) {
        if (val == "i8")
          outputDataType = "i32";
        else if (val.starts_with("f8") || val.starts_with("fp8") ||
                 val.starts_with("bf8"))
          outputDataType = "f32";
        else
          outputDataType = v;
      }
    }));
llvm::cl::alias dataTypeAliasLong("dtype", llvm::cl::aliasopt(dataTypeAlias));

// conv-config
static llvm::cl::opt<std::string> populateConvConfig(
    "conv-config",
    llvm::cl::desc(
        "Populate full config settings (overrides all specific settings)"),
    llvm::cl::value_desc("config settings matching the C-API"),
    llvm::cl::init(""));

// populate default values
static llvm::cl::opt<bool>
    populateDefaultValues("p", llvm::cl::desc("To populate default values"),
                          llvm::cl::value_desc("To populate default values"),
                          llvm::cl::init(false));

static llvm::cl::opt<bool>
    emitTuningSpace("emit-tuning-space", llvm::cl::desc("Tune a Gemm kernel"),
                    llvm::cl::value_desc("To tune a Gemm kernel"),
                    llvm::cl::init(false));

static llvm::cl::opt<bool> emitTuningKey(
    "emit-tuning-key",
    llvm::cl::desc(
        "Prints out the struct of the problem to be tuned for inspection."),
    llvm::cl::value_desc(
        "String formatted fields of the problem which is going to be tuned."),
    llvm::cl::init(false));

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
static llvm::cl::opt<bool>
    genHostHarness("host-harness", llvm::cl::desc("To use host harness"),
                   llvm::cl::value_desc("To use host harness"),
                   llvm::cl::init(false));

static llvm::cl::alias aliasGenHostHarness("ph",
                                           llvm::cl::aliasopt(genHostHarness));

// print results
static llvm::cl::opt<bool>
    printResults("print-results", llvm::cl::desc("To print result tensor"),
                 llvm::cl::init(false));
static llvm::cl::alias aliasPrintResults("pr",
                                         llvm::cl::aliasopt(printResults));

static llvm::cl::opt<bool> printInputs("print-inputs",
                                       llvm::cl::desc("To print input tensors"),
                                       llvm::cl::init(false));
static llvm::cl::alias aliasPrintInputs("pi", llvm::cl::aliasopt(printInputs));

static llvm::cl::opt<bool> printValidationResults(
    "print-validation-results",
    llvm::cl::desc("To print result tensor for validation"),
    llvm::cl::init(false));
static llvm::cl::alias
    aliasPrintValidationResults("pvr",
                                llvm::cl::aliasopt(printValidationResults));

// populate host validation logic.
static llvm::cl::opt<std::string> genValidation(
    "verifier",
    llvm::cl::desc(
        "Select verification from: none(default), cpu, gpu, cpp, mlir, clone"),
    llvm::cl::cb<void, std::string>([](const std::string &v) {
      if (!v.empty())
        genHostHarness = true;
    }),
    llvm::cl::value_desc("Specify host validation logic"), llvm::cl::init(""));

static llvm::cl::opt<bool>
    genCPUValidation("pv", llvm::cl::Hidden, llvm::cl::init(false),
                     llvm::cl::Optional, llvm::cl::cb<void, bool>([](bool v) {
                       if (v) {
                         genValidation = "mlir";
                         genHostHarness = true;
                       }
                     }));

static llvm::cl::opt<bool>
    genCPPValidation("pv_with_cpp", llvm::cl::Hidden, llvm::cl::init(false),
                     llvm::cl::Optional, llvm::cl::cb<void, bool>([](bool v) {
                       if (v) {
                         genValidation = "cpp";
                         genHostHarness = true;
                       }
                     }));

static llvm::cl::opt<bool>
    genMLIRValidation("pv_with_mlir", llvm::cl::Hidden, llvm::cl::init(false),
                      llvm::cl::Optional, llvm::cl::cb<void, bool>([](bool v) {
                        if (v) {
                          genValidation = "mlir";
                          genHostHarness = true;
                        }
                      }));

static llvm::cl::opt<bool>
    genGPUValidation("pv_with_gpu", llvm::cl::Hidden, llvm::cl::init(false),
                     llvm::cl::Optional, llvm::cl::cb<void, bool>([](bool v) {
                       if (v) {
                         genValidation = "gpu";
                         genHostHarness = true;
                       }
                     }));

static llvm::cl::opt<bool> genVerifierKeepPerfConfig(
    "verifier-keep-perf-config", llvm::cl::init(false),
    llvm::cl::desc(
        "whether to clear perf config on verification with GPU kernels"));

static llvm::cl::opt<bool>
    genCPUKernel("cpu-kernels", llvm::cl::desc("Generate CPU kernel for test"),
                 llvm::cl::init(false), llvm::cl::Optional,
                 llvm::cl::cb<void, bool>([](bool v) {
                   if (v) {
                     genValidation = "mlir";
                     genHostHarness = true;
                     printResults = true;
                   }
                 }));
static llvm::cl::alias aliasGenCPUKernel("prc",
                                         llvm::cl::aliasopt(genCPUKernel));

// Input data spec
static llvm::cl::opt<std::string> randomSeed(
    "rand",
    llvm::cl::desc(
        "A positive integer or zero indicates the seed of random data generator"
        "for convolution inputs, e.g. -rand 1. If not specifed, or 'fixed', "
        "use a fixed nonuniform test pattern. If 'none', use all 1s as the "
        "values. If 0, use time(0) as the seed."),
    llvm::cl::value_desc("seed"), llvm::cl::init("fixed"));

static llvm::cl::opt<std::string> randomDataType(
    "rand_type",
    llvm::cl::desc("To specify data type for random number generator,"
                   "e.g. -rand_type float, -rand_type int (default)."),
    llvm::cl::value_desc("type"), llvm::cl::init("int"));

static llvm::cl::opt<std::string> randomSide(
    "rand_side",
    llvm::cl::desc(
        "To populate random numbers to a specified tensor: "
        "For conv2d, -rand_side filter or -rand_side input; "
        "For conv2d_bwd_data, -rand_side filter or -rand_side output; "
        "For conv2d_bwd_weight, -rand_side input or -rand_side output. "
        "By default, populate random numbers to both tensors."),
    llvm::cl::value_desc("tensor"), llvm::cl::init("both"));

// float random inputs range
static llvm::cl::opt<int>
    randMin("rand_min", llvm::cl::desc("lower bound for float random input"),
            llvm::cl::value_desc("range"), llvm::cl::init(0));

static llvm::cl::opt<int>
    randMax("rand_max", llvm::cl::desc("upper bound for float random input"),
            llvm::cl::value_desc("range"), llvm::cl::init(1));

// Verification function options
static llvm::cl::opt<float>
    RMSThreshold("RMS_threshold", llvm::cl::desc("Threshold for RMS metric"),
                 llvm::cl::value_desc("error"), llvm::cl::init(0.00003f));

static llvm::cl::opt<float>
    absDiffThreshold("absDiff_threshold",
                     llvm::cl::desc("Threshold for absDiff metric"),
                     llvm::cl::value_desc("error"), llvm::cl::init(100.0f));

static llvm::cl::opt<float>
    relDiffThreshold("relDiff_threshold",
                     llvm::cl::desc("Threshold for relDiff metric"),
                     llvm::cl::value_desc("error"), llvm::cl::init(0.000001f));

// A toggle to control what to print in the verification function
enum class VerificationPrintToggle : char {
  Always = 3,
  Failure = 2,
  Summary = 1,
  Off = 0
};

static llvm::cl::opt<VerificationPrintToggle> printVerifyResults(
    "print-verify-results",
    llvm::cl::desc("Choose when to print verbose debug information in the "
                   "verification function:"),
    llvm::cl::values(
        clEnumValN(VerificationPrintToggle::Always, "always",
                   "always print debug info"),
        clEnumValN(VerificationPrintToggle::Failure, "failure",
                   "print elem-wise diff + summary only if the test fails"),
        clEnumValN(VerificationPrintToggle::Summary, "summary",
                   "print summary info only if the test fails"),
        clEnumValN(VerificationPrintToggle::Off, "off",
                   "do not print debug info")),
    llvm::cl::init(VerificationPrintToggle::Summary));
static llvm::cl::alias
    aliasPrintVerifyResults("p_verify", llvm::cl::aliasopt(printVerifyResults));

static llvm::cl::opt<int> deviceNum(
    "device",
    llvm::cl::desc(
        "Device index on which to run the kernel (only with host code)"),
    llvm::cl::value_desc("Between 0 and number of GPUs on system. "
                         "Omission leaves current device intact."));
static llvm::cl::alias deviceShort("dev", llvm::cl::aliasopt(deviceNum));

static llvm::cl::opt<int> kernelRepeats(
    "kernel-repeats",
    llvm::cl::desc("Number of times to repeat the kernel invocation"),
    llvm::cl::value_desc("positive integer"), llvm::cl::init(1));

////////////////////////////////////////////////////////////////////////////////
////  Struct KernelIF
////  - Detected/capture kernel interface
////////////////////////////////////////////////////////////////////////////////
struct KernelIF {
  func::FuncOp func;
  SmallVector<Type, 8> params;
  SmallVector<int32_t, 2> outIndices;

  // CTOR w/ FuncOp
  KernelIF(func::FuncOp _f) : func(_f) {
    assert(func.getNumResults() == 0);
    llvm::SmallDenseSet<Value> outs;
    auto walker = [&](memref::CopyOp copy) { outs.insert(copy.getTarget()); };
    func.walk(walker);
    size_t argCount = func.getArguments().size();
    for (size_t i = 0; i < argCount; i++) {
      params.push_back(func.getArgument(i).getType());
      if (outs.contains(func.getArgument(i))) {
        outIndices.push_back(i);
      }
    }
  }
};

struct GenParams {
  std::optional<rock::KernelType> operation = std::nullopt;
  SmallVector<Type, 3> types;
  rock::GemmFeatures features = rock::GemmFeatures::none;
  std::optional<const rock::Conv2dGenerator::Config *> convConfig =
      std::nullopt;
  StringRef arch;
  StringRef perfConfig;
};

namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test

static void correctConvParameters() {
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

  auto validatePadding = [](llvm::cl::opt<int> &combined,
                            llvm::cl::opt<int> &left, llvm::cl::opt<int> &right,
                            StringRef name) {
    if (combined.getValue() > 0) {
      int combinedVal = combined.getValue();
      int leftVal = left.getValue();
      int rightVal = right.getValue();
      if (leftVal == 0 && rightVal == 0) {
        left = combinedVal;
        right = combinedVal;
      } else {
        if (leftVal != combinedVal || rightVal != combinedVal) {
          llvm::errs() << "you can't use both " << name << " and (" << name
                       << "_l," << name << "_r).\n";
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

static void verifyConvLayout() {
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
  bool isGemm = operation == rock::KernelType::Gemm;
  // Default f32 if we passed no `-t` arguments at all.
  if (outputDataType.empty())
    outputDataType = "f32";
  if (populateDefaultValues) {
    if (isGemm) {
      gemmM = 1024;
      gemmK = 769;
      gemmN = 512;
      groupSize = 1;
    }
    if (mfmaFeature != FeatureToggle::on) {
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
    }
  }

  if (!isGemm && outputHeight.getNumOccurrences() == 0) {
    outputHeight = rock::Conv2dGenerator::outputDim(
        inputHeight.getValue(), filterHeight.getValue(),
        paddingHeightLeft.getValue(), paddingHeightRight.getValue(),
        strideHeight.getValue(), dilationHeight.getValue());
  }
  if (!isGemm && outputWidth.getNumOccurrences() == 0) {
    outputWidth = rock::Conv2dGenerator::outputDim(
        inputWidth.getValue(), filterWidth.getValue(),
        paddingWidthLeft.getValue(), paddingWidthRight.getValue(),
        strideWidth.getValue(), dilationWidth.getValue());
  }
}

static LogicalResult detectMissingArguments() {
  const static std::vector<const llvm::cl::opt<int64_t> *> requiredConvArgs = {
      &groupSize,  &batchSize,     &inputChannel, &inputHeight,
      &inputWidth, &outputChannel, &filterWidth,  &filterHeight};
  const static std::vector<const llvm::cl::opt<int64_t> *> requiredGemmArgs = {
      &groupSize, &gemmM, &gemmK, &gemmN};
  for (auto *arg : ((operation == rock::KernelType::Gemm) ? requiredGemmArgs
                                                          : requiredConvArgs)) {
    if (arg->getValue() <= 0) {
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

static Value makeNDMemRef(OpBuilder &b, Value var, uint32_t ndim) {
  MLIRContext *context = b.getContext();
  auto oprType = var.getType().template cast<ShapedType>();
  if (!oprType.hasStaticShape())
    return Value();

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
  } else if (!shape.empty() && shape.size() < ndim) {
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

static func::FuncOp createGPUWrapper(ModuleOp module, const KernelIF &kernel) {
  MLIRContext *context = module.getContext();
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
  Block *block = gpuWrapperFunc.addEntryBlock();
  b.setInsertionPoint(block, block->begin());

  // Emit device selection
  if (deviceNum.getNumOccurrences() > 0)
    b.create<gpu::SetDefaultDeviceOp>(
        loc, b.create<arith::ConstantIntOp>(loc, deviceNum.getValue(),
                                            b.getIntegerType(32)));

  SmallVector<Value, 4> cpuMem;
  SmallVector<Value, 4> gpuMem;
  for (auto pair : llvm::enumerate(kernel.params)) {
    Value arg = block->getArgument(pair.index());
    cpuMem.push_back(arg);

    // Emit GPU memory allocation function calls.
    auto gpuAllocOp = b.create<gpu::AllocOp>(
        loc, arg.getType(), Type(), /*asyncDependencies=*/ValueRange{},
        /*dynamicSizes=*/ValueRange{}, /*symbolOperands=*/ValueRange{});
    Value gpuAlloc = gpuAllocOp.getResult(0);
    gpuMem.push_back(gpuAlloc);

    // Emit CPU->GPU memcpy function calls.
    b.create<gpu::MemcpyOp>(loc, TypeRange{}, ValueRange{gpuAlloc, arg});
  }

  // Emit kernel function call, repeating it if needed.
  // We assume that the repeated atomic add usages in a wrw kernel will not
  // substantially impact performance as the result becomes large
  auto emitWrappedCall = [&kernel, &gpuMem](OpBuilder &b, Location loc,
                                            Value ignoredIv,
                                            ValueRange noArgs) {
    auto wrappedCall = b.create<func::CallOp>(loc, kernel.func, gpuMem);
    wrappedCall->setAttr("wrapped_call", b.getUnitAttr());
    if (ignoredIv) { // we're creating an actual loop
      b.create<scf::YieldOp>(loc);
    }
  };
  if (kernelRepeats > 1) {
    Value zeroOp = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
    Value kernelRepeatsOp =
        b.createOrFold<arith::ConstantIndexOp>(loc, kernelRepeats);
    Value step = b.createOrFold<arith::ConstantIndexOp>(loc, 1);
    b.create<scf::ForOp>(loc, zeroOp, kernelRepeatsOp, step,
                         /*args=*/std::nullopt, emitWrappedCall);
  } else {
    emitWrappedCall(b, loc, nullptr, {});
  }

  for (auto pair : llvm::enumerate(kernel.params)) {
    uint32_t i = pair.index();
    b.create<gpu::MemcpyOp>(loc, TypeRange{}, ValueRange{cpuMem[i], gpuMem[i]});
    b.create<gpu::DeallocOp>(loc, TypeRange{}, ValueRange{gpuMem[i]});
  }

  b.create<func::ReturnOp>(loc, ValueRange{});

  return gpuWrapperFunc;
}

// Map data type string to MLIR type
static Type typeFromString(StringRef name, MLIRContext *ctx) {
  std::optional<Type> result =
      llvm::StringSwitch<std::optional<Type>>(name)
          .Case("f32", Float32Type::get(ctx))
          .Case("f16", Float16Type::get(ctx))
          .Case("bf16", BFloat16Type::get(ctx))
          .Case("i8", IntegerType::get(ctx, 8))
          .Case("i32", IntegerType::get(ctx, 32))
          .Cases("bf8", "f8E5M2FNUZ", Float8E5M2FNUZType::get(ctx))
          .Cases("fp8", "f8E4M3FNUZ", Float8E4M3FNUZType::get(ctx))
          .Default(std::nullopt);
  if (!result) {
    llvm::errs() << "Unknown data type: " << name << "\n";
    exit(1);
  }
  return *result;
}

// Determine the range and seed for the random data generator
static int getRandomSeed() {
  std::string rseed = randomSeed;
  if (rseed[0] >= '0' and rseed[0] <= '9')
    return std::stoi(rseed);
  return -1;
}

static std::tuple<short, short> getRandomTestData(int idx) {
  short min = 1, max = 1;

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

  if (randomSeed != "none" && randomSeed != "fixed") {
    if ((idx_spec >= 0) && (idx_spec != idx)) {
    } else if (randomDataType.getValue() == "int") {
      // generate random integer in [-5, 5)
      min = -5;
      max = 5;
    } else {
      // generate random floats in [rand_min, rand_max)
      min = randMin.getValue();
      max = randMax.getValue();
    }
  }
  return std::make_tuple(min, max);
}

llvm::SmallVector<float, 3> getTensorInitPattern(Type elemType) {
  llvm::SmallVector<float, 3> pattern;
  if (randomSeed == "none") {
    float fixedVal = 1.0f;
    if (randomDataType == "float")
      // Clamp the fixed rondam float by 0.1 to avoid infs in some f16 tests
      fixedVal *= 0.1f;
    pattern = {static_cast<float>(fixedVal)};
  } else if (randomSeed == "fixed") {
    if (elemType.isIntOrIndex())
      pattern = {1.0, -1.0, 2.0};
    else
      pattern = {0.5, -1, 0.75};
  } else {
    llvm_unreachable("We shouldn't be here for random values");
  }
  return pattern;
}

static LogicalResult populateTensorFillLogic(OpBuilder &b, Location loc,
                                             ArrayRef<float> pattern,
                                             Type elemType, Value toFill) {
  // TODO(kdrewnia) Refactor this to create the constant vector up front
  // TODO(kdrewnia) Factor out the anti-bf16 pass from GPU lowering, apply
  // it here
  Type i16 = b.getIntegerType(16);
  Value constantsVec;
  if (elemType == b.getBF16Type()) {
    uint16_t init = 0;
    constantsVec = b.create<arith::ConstantOp>(
        loc,
        SplatElementsAttr::get(VectorType::get(pattern.size(), i16), init));
  } else {
    constantsVec = rock::createZeroConstantOp(
        b, loc, VectorType::get(pattern.size(), elemType));
  }
  for (auto v : llvm::enumerate(pattern)) {
    Value vOp;
    if (elemType == b.getBF16Type()) {
      llvm::APFloat fl(v.value());
      bool losesInfo = false;
      fl.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                 &losesInfo);
      llvm::APInt val = fl.bitcastToAPInt();
      vOp = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(i16, val));
    } else if (elemType.isIntOrIndex()) {
      vOp = rock::createConstantIntOp(b, loc, elemType, elemType,
                                      static_cast<int64_t>(v.value()));
    } else {
      vOp = rock::createConstantFloatOp(b, loc, elemType, elemType, v.value());
    }
    constantsVec = b.create<vector::InsertElementOp>(
                        loc, vOp, constantsVec,
                        b.create<arith::ConstantIndexOp>(loc, v.index()))
                       .getResult();
  }

  Value toFillFlat = makeNDMemRef(b, toFill, 1);
  MemRefType flatType = toFillFlat.getType().cast<MemRefType>();
  SmallVector<int64_t, 1> lowerBounds;
  SmallVector<int64_t, 1> upperBounds;
  SmallVector<int64_t, 1> steps;
  AffineMap rowMajorMap = AffineMap::getConstantMap(0, b.getContext());
  if (!flatType.getShape().empty()) {
    AffineExpr rowMajor = b.getAffineDimExpr(0);
    rowMajor = rowMajor % b.getAffineConstantExpr(pattern.size());
    rowMajorMap = AffineMap::get(1, 0, {rowMajor}, b.getContext());
    lowerBounds.push_back(0);
    upperBounds.push_back(flatType.getNumElements());
    steps.push_back(1);
  }

  affine::buildAffineLoopNest(
      b, loc, lowerBounds, upperBounds, steps,
      [rowMajorMap, &constantsVec, toFillFlat,
       elemType](OpBuilder &b, Location loc, ValueRange ivs) {
        auto selectorOp =
            b.create<affine::AffineApplyOp>(loc, rowMajorMap, ivs);
        Value toStore = b.create<vector::ExtractElementOp>(
                             loc, constantsVec, selectorOp->getResult(0))
                            .getResult();
        if (elemType == b.getBF16Type())
          toStore = b.create<arith::BitcastOp>(loc, b.getBF16Type(), toStore);
        b.create<memref::StoreOp>(loc, toStore, toFillFlat, ivs);
      });
  return success();
}

static LogicalResult populateRandomTensorFillLogic(OpBuilder &b, Location loc,
                                                   ModuleOp module,
                                                   Type elemType, Value toFill,
                                                   int idx) {

  llvm::SmallDenseMap<short, Value> i16vals;
  auto getI16Val = [&](short v) {
    if (i16vals.find(v) == i16vals.end()) {
      auto i16Type = b.getIntegerType(16);
      i16vals.try_emplace(
          v, b.createOrFold<arith::ConstantIntOp>(loc, v, i16Type));
    }
    return i16vals[v];
  };

  Value toFillFlat = makeNDMemRef(b, toFill, 1);
  auto flatType = toFillFlat.getType().cast<MemRefType>();

  bool isRandFloat = (randomDataType == "float");
  func::FuncOp randFunc;
  Type i16 = b.getI16Type();
  Type f32 = b.getF32Type();
  if (isRandFloat)
    randFunc = makeFuncDecl(module, "randomFloatValue", {i16, i16}, {f32});
  else
    randFunc = makeFuncDecl(module, "randomIntegerValue", {i16, i16}, {f32});

  short min, max;
  std::tie(min, max) = getRandomTestData(idx);
  Value minConst = getI16Val(min), maxConst = getI16Val(max);

  SmallVector<int64_t, 1> lowerBounds;
  SmallVector<int64_t, 1> upperBounds;
  SmallVector<int64_t, 1> steps;

  if (!flatType.getShape().empty()) {
    lowerBounds.push_back(0);
    upperBounds.push_back(flatType.getNumElements());
    steps.push_back(1);
  }

  affine::buildAffineLoopNest(
      b, loc, lowerBounds, upperBounds, steps,
      [elemType, randFunc, toFillFlat, minConst,
       maxConst](OpBuilder &b, Location loc, ValueRange ivs) {
        auto randFloatCall = b.create<func::CallOp>(
            loc, randFunc, ValueRange{minConst, maxConst});
        Value randFloat = randFloatCall.getResult(0);
        Value randVal;
        if (elemType.isIntOrIndex())
          randVal = b.create<arith::FPToSIOp>(loc, elemType, randFloat);
        else if (!elemType.isF32())
          randVal = b.create<arith::TruncFOp>(loc, elemType, randFloat);
        else
          randVal = randFloat;

        b.create<memref::StoreOp>(loc, randVal, toFillFlat, ivs);
      });

  return success();
}

static std::tuple<int64_t, int64_t, int64_t>
getConv2dBounds(rock::ConvOpType dir,
                const rock::Conv2dGenerator::Config &genConfig) {
  int64_t dim, dimH, dimW;
  char channel;
  StringRef layout;
  ArrayRef<int64_t> dimension;
  switch (dir) {
  case rock::ConvOpType::Fwd:
    channel = 'c';
    dimension = genConfig.inputDimension;
    layout = genConfig.inputLayout;
    break;
  case rock::ConvOpType::BwdData:
    channel = 'k';
    dimension = genConfig.outputDimension;
    layout = genConfig.outputLayout;
    break;
  case rock::ConvOpType::BwdWeight:
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
                      const rock::Conv2dGenerator::Config &genConfig) {
  OpBuilder b(module.getContext());

  Block *block = func.addEntryBlock();
  b.setInsertionPoint(block, block->begin());

  Location loc = b.getUnknownLoc();

  // Initialize the result tensor
  BlockArgument resultTensor;
  switch (genConfig.operation.value()) {
  case rock::ConvOpType::Fwd:
    resultTensor = block->getArgument(2);
    break;
  case rock::ConvOpType::BwdData:
    resultTensor = block->getArgument(1);
    break;
  case rock::ConvOpType::BwdWeight:
    resultTensor = block->getArgument(0);
    break;
  }
  auto resultType = resultTensor.getType().template dyn_cast<MemRefType>();
  Type elemType = resultType.getElementType();
  SmallVector<float, 1> zeroPattern = {0.0};
  if (failed(
          populateTensorFillLogic(b, loc, zeroPattern, elemType, resultTensor)))
    llvm_unreachable("Tensor fill logic population shouldn't fail");

  // Create affine maps
  AffineExpr heightExpr, widthExpr;
  AffineMap heightMap, widthMap;
  AffineExpr outputHeightExpr, outputWidthExpr;
  AffineMap outputHeightMap, outputWidthMap;

  switch (genConfig.operation.value()) {
  case rock::ConvOpType::Fwd:
  case rock::ConvOpType::BwdWeight:
    // d0 * stride + d1 * dilation - padding
    heightExpr = b.getAffineDimExpr(0) * genConfig.strideHeight +
                 b.getAffineDimExpr(1) * genConfig.dilationHeight -
                 genConfig.paddingHeightLeft;
    widthExpr = b.getAffineDimExpr(0) * genConfig.strideWidth +
                b.getAffineDimExpr(1) * genConfig.dilationWidth -
                genConfig.paddingWidthLeft;
    break;
  case rock::ConvOpType::BwdData:
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
  if (genConfig.operation.value() == rock::ConvOpType::BwdData) {
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
  if (genConfig.operation.value() == rock::ConvOpType::BwdData) {
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
  if (genConfig.operation.value() == rock::ConvOpType::BwdData) {
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
      getConv2dBounds(genConfig.operation.value(), genConfig);

  // Create the upper bounds
  switch (genConfig.operation.value()) {
  case rock::ConvOpType::Fwd:
    llvm::copy(genConfig.outputDimension, std::back_inserter(upperBounds));
    upperBounds.push_back(dimX);
    upperBounds.push_back(genConfig.filterHeight);
    upperBounds.push_back(genConfig.filterWidth);
    loopIVs.append(genConfig.outputLayout);
    loopIVs.append("cyx");
    break;
  case rock::ConvOpType::BwdData:
    llvm::copy(genConfig.inputDimension, std::back_inserter(upperBounds));
    upperBounds.push_back(dimX);
    upperBounds.push_back(genConfig.filterHeight);
    upperBounds.push_back(genConfig.filterWidth);
    loopIVs.append(genConfig.inputLayout);
    loopIVs.append("kyx");
    break;
  case rock::ConvOpType::BwdWeight:
    std::tie(std::ignore, out_h, out_w) =
        getConv2dBounds(rock::ConvOpType::BwdData, genConfig);
    llvm::copy(genConfig.filterDimension, std::back_inserter(upperBounds));
    upperBounds.push_back(dimX);
    upperBounds.push_back(out_h);
    upperBounds.push_back(out_w);
    loopIVs.append(genConfig.filterLayout);
    loopIVs.append("nhw");
    break;
  }

  auto createConv2dLoopNest = [&](OpBuilder &b, Location loc, ValueRange ivs) {
    Value heightIdx, widthIdx;
    Value heightTempIdx, widthTempIdx;

    switch (genConfig.operation.value()) {
    case rock::ConvOpType::Fwd:
      // in_h_idx = out_h_idx * stride_h + fil_h_idx * dilation_h - padding_h_l;
      // in_w_idx = out_w_idx * stride_w + fil_w_idx * dilation_w - padding_w_l;
      heightIdx = b.create<affine::AffineApplyOp>(
          loc, heightMap,
          ValueRange{ivs[genConfig.outputLayout.find('h')], ivs[6]});
      widthIdx = b.create<affine::AffineApplyOp>(
          loc, widthMap,
          ValueRange{ivs[genConfig.outputLayout.find('w')], ivs[7]});
      break;
    case rock::ConvOpType::BwdData:
      // out_h_tmp = in_h_idx + padding_h_l - fil_h_idx * dilation_h;
      // out_w_tmp = in_w_idx + padding_w_l - fil_w_idx * dilation_w;
      heightTempIdx = b.create<affine::AffineApplyOp>(
          loc, heightMap,
          ValueRange{ivs[genConfig.inputLayout.find('h')], ivs[6]});
      widthTempIdx = b.create<affine::AffineApplyOp>(
          loc, widthMap,
          ValueRange{ivs[genConfig.inputLayout.find('w')], ivs[7]});
      // out_h_idx = out_h_tmp / stride_h;
      // out_w_idx = out_w_tmp / stride_w;
      heightIdx = b.create<affine::AffineApplyOp>(loc, outputHeightMap,
                                                  ValueRange{heightTempIdx});
      widthIdx = b.create<affine::AffineApplyOp>(loc, outputWidthMap,
                                                 ValueRange{widthTempIdx});
      break;
    case rock::ConvOpType::BwdWeight:
      // in_h_idx = out_h_idx * stride_h + fil_h_idx * dilation_h - padding_h_l;
      // in_w_idx = out_w_idx * stride_w + fil_w_idx * dilation_w - padding_w_l;
      heightIdx = b.create<affine::AffineApplyOp>(
          loc, heightMap,
          ValueRange{ivs[6], ivs[genConfig.filterLayout.find('y')]});
      widthIdx = b.create<affine::AffineApplyOp>(
          loc, widthMap,
          ValueRange{ivs[7], ivs[genConfig.filterLayout.find('x')]});
      break;
    }

    enum TENSOR { FILTER = 0, INPUT = 1, OUTPUT = 2 };
    auto getIndices = [&](TENSOR tensor, SmallVectorImpl<Value> &result) {
      std::string layout;
      if (tensor == FILTER)
        layout = genConfig.filterLayout;
      else if (tensor == INPUT)
        layout = genConfig.inputLayout;
      else
        layout = genConfig.outputLayout;
      for (auto c : layout) {
        auto direction = genConfig.operation.value();
        if ((direction == rock::ConvOpType::Fwd ||
             direction == rock::ConvOpType::BwdWeight) &&
            tensor == INPUT) {
          if (c == 'h') {
            result.push_back(heightIdx);
            continue;
          } else if (c == 'w') {
            result.push_back(widthIdx);
            continue;
          }
        } else if (direction == rock::ConvOpType::BwdData && tensor == OUTPUT) {
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

    affine::AffineIfOp ifOp;
    if (genConfig.operation.value() == rock::ConvOpType::BwdData) {
      ifOp = b.create<affine::AffineIfOp>(
          loc, condition,
          ValueRange{heightIdx, widthIdx, heightTempIdx, widthTempIdx,
                     dimHeight, dimWidth},
          false);
    } else {
      ifOp = b.create<affine::AffineIfOp>(
          loc, condition, ValueRange{heightIdx, widthIdx, dimHeight, dimWidth},
          false);
    }
    auto thenBody = ifOp.getThenBodyBuilder();

    // Perform MAC operation
    SmallVector<Value, 5> idx1, idx2;
    BlockArgument opd1, opd2, result;

    switch (genConfig.operation.value()) {
    case rock::ConvOpType::Fwd:
      getIndices(FILTER, idx1);
      getIndices(INPUT, idx2);
      opd1 = block->getArgument(0);
      opd2 = block->getArgument(1);
      result = block->getArgument(2);
      break;
    case rock::ConvOpType::BwdWeight:
      getIndices(OUTPUT, idx1);
      getIndices(INPUT, idx2);
      opd1 = block->getArgument(2);
      opd2 = block->getArgument(1);
      result = block->getArgument(0);
      break;
    case rock::ConvOpType::BwdData:
      getIndices(FILTER, idx1);
      getIndices(OUTPUT, idx2);
      opd1 = block->getArgument(0);
      opd2 = block->getArgument(2);
      result = block->getArgument(1);
      break;
    }
    llvm::ArrayRef<Value> idxRef1(idx1.data(), idx1.size());
    auto loadOp1 =
        thenBody.create<memref::LoadOp>(loc, opd1, ValueRange{idxRef1});
    llvm::ArrayRef<Value> idxRef2(idx2.data(), idx2.size());
    auto loadOp2 =
        thenBody.create<memref::LoadOp>(loc, opd2, ValueRange{idxRef2});
    auto loadOutput = thenBody.create<memref::LoadOp>(
        loc, result, ValueRange{ivs[0], ivs[1], ivs[2], ivs[3], ivs[4]});
    if (elemType.isIntOrIndex()) {
      auto muliOp = thenBody.create<arith::MulIOp>(loc, loadOp1, loadOp2);
      auto extsiOp = thenBody.create<arith::ExtSIOp>(loc, elemType, muliOp);
      auto addiOp = thenBody.create<arith::AddIOp>(loc, loadOutput, extsiOp);
      thenBody.create<memref::StoreOp>(
          loc, addiOp, result,
          ValueRange{ivs[0], ivs[1], ivs[2], ivs[3], ivs[4]});
    } else {
      auto mulfOp = thenBody.create<arith::MulFOp>(loc, loadOp1, loadOp2);
      auto addfOp = thenBody.create<arith::AddFOp>(loc, loadOutput, mulfOp);
      thenBody.create<memref::StoreOp>(
          loc, addfOp, result,
          ValueRange{ivs[0], ivs[1], ivs[2], ivs[3], ivs[4]});
    }
  };

  // Generate the loop nest
  affine::buildAffineLoopNest(b, loc, lowerBounds, upperBounds, steps,
                              createConv2dLoopNest);

  b.create<func::ReturnOp>(loc, ValueRange{});
  return;
}

static void
createCPUConvWithCPP(ModuleOp module, func::FuncOp &func,
                     const rock::Conv2dGenerator::Config &genConfig) {
  OpBuilder b(module.getContext());

  Block *block = func.addEntryBlock();
  b.setInsertionPoint(block, block->begin());

  auto loc = b.getUnknownLoc();

  Type elemType = b.getF32Type();
  if (genConfig.inputDataTypeStr == "i8") {
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
  bool hasAccel = rock::isAccel(genConfig.features);
  auto accelConstantOp = b.create<arith::ConstantIntOp>(loc, hasAccel, intType);

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

  switch (genConfig.operation.value()) {
  case rock::ConvOpType::Fwd:
    mcpuFuncName = "mcpuConv2d";
    break;
  case rock::ConvOpType::BwdData:
    mcpuFuncName = "mcpuConv2dBwdData";
    break;
  case rock::ConvOpType::BwdWeight:
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
                 dilationWidthConstantOp, accelConstantOp});

  // Emit return op
  b.create<func::ReturnOp>(loc, ValueRange{});
  return;
}

static func::FuncOp
createCPUConvFunc(ModuleOp module,
                  const rock::Conv2dGenerator::Config &genConfig) {
  assert(genConfig.operation.has_value());
  std::string funcName =
      rock::getNameForConvOpType(genConfig.operation.value()).str();

  funcName += "_cpu";
  func::FuncOp func = module.lookupSymbol<func::FuncOp>(funcName);
  if (func) // already exists
    return func;

  OpBuilder b(module.getContext());
  auto loc = b.getUnknownLoc();

  Type elemType = b.getF32Type();
  Type outputElemType = b.getF32Type();
  if (genConfig.inputDataTypeStr == "i8") {
    elemType = b.getI8Type();
    // Compute the output in int64_t to detect overflow
    outputElemType = b.getIntegerType(64);
    assert(genConfig.operation.value() == rock::ConvOpType::Fwd);
  }

  auto filterDimension = genConfig.filterDimension;
  auto inputDimension = genConfig.inputDimension;
  auto outputDimension = genConfig.outputDimension;

  auto filterType = MemRefType::get(filterDimension, elemType);
  auto inputType = MemRefType::get(inputDimension, elemType);
  auto outputType = MemRefType::get(outputDimension, outputElemType);

  // Create conv2d_host function
  rock::Conv2dGenerator conv2dGenerator(genConfig);

  bool hasWorkspace = false;
  if (failed(conv2dGenerator.hasWorkspace(b, hasWorkspace))) {
    assert(genConfig.operation.value() == rock::ConvOpType::Fwd);
  }
  Type workspaceArgType;
  if (hasWorkspace) {
    workspaceArgType = MemRefType::get(
        ArrayRef<int64_t>(filterDimension.begin(), filterDimension.end()),
        b.getF32Type());
  }

  SmallVector<Type, 3> funcArgTypes = {filterType, inputType, outputType};

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

static void getGemmTypes(ArrayRef<Type> elemTypes,
                         SmallVectorImpl<Type> &result, bool isCpuVerifier) {
  Type cElemType = elemTypes[2];
  if (elemTypes[0].isInteger(8)) {
    // Verify in int64_t to detect overflow
    if (isCpuVerifier)
      cElemType = IntegerType::get(cElemType.getContext(), 64);
  }

  SmallVector<int64_t> aDims = {groupSize, transposeA ? gemmK : gemmM,
                                transposeA ? gemmM : gemmK},
                       bDims = {groupSize, transposeB ? gemmN : gemmK,
                                transposeB ? gemmK : gemmN},
                       cDims = {groupSize, transposeC ? gemmN : gemmM,
                                transposeC ? gemmM : gemmN};

  MemRefType aType = MemRefType::get(aDims, elemTypes[0]),
             bType = MemRefType::get(bDims, elemTypes[1]),
             cType = MemRefType::get(cDims, cElemType);
  result.push_back(aType);
  result.push_back(bType);
  result.push_back(cType);
}

static func::FuncOp createGpuGemmKernel(ModuleOp module,
                                        const GenParams &params,
                                        bool isVerifier = false) {
  MLIRContext *ctx = module.getContext();
  Location loc = module->getLoc();
  OpBuilder b(ctx);

  // Set mhal.arch on module to make compilation pipeline work
  StringAttr archAttr = b.getStringAttr(params.arch);
  if (!module->hasAttr("mhal.arch"))
    module->setAttr("mhal.arch", archAttr);

  SmallVector<Type, 3> argTypes;
  getGemmTypes(params.types, argTypes,
               /*isCpuVerifier=*/false);
  constexpr StringLiteral kernelName("rock_gemm");
  constexpr StringLiteral kernelNameVerifier("rock_gemm_ver");

  SmallVector<NamedAttribute, 2> funcAttrs = {
      b.getNamedAttr("kernel", b.getUnitAttr()),
      b.getNamedAttr("mhal.arch", archAttr)};
  auto func =
      b.create<func::FuncOp>(loc, isVerifier ? kernelNameVerifier : kernelName,
                             b.getFunctionType(argTypes, {}), funcAttrs);

  Block *block = func.addEntryBlock();
  b.setInsertionPointToStart(block);
  Value aVal = block->getArgument(0), bVal = block->getArgument(1),
        cVal = block->getArgument(2);
  IntegerAttr numCuAttr = nullptr;
  if (num_cu.getNumOccurrences() > 0)
    numCuAttr = b.getI32IntegerAttr(num_cu);
  auto gemm = b.create<rock::GemmOp>(
      loc, /*resultTypes=*/TypeRange{}, aVal, bVal, cVal, transposeA,
      transposeB, transposeC, archAttr.getValue(), numCuAttr, params.features,
      storeMethod,
      /*blockSize=*/nullptr, /*gridSize=*/nullptr, /*params=*/nullptr);
  if (!params.perfConfig.empty())
    gemm->setAttr("perf_config", b.getStringAttr(params.perfConfig));
  b.create<func::ReturnOp>(loc);

  module.push_back(func);
  return func;
}

static func::FuncOp createCpuGemmKernelWithMlir(ModuleOp module,
                                                const GenParams &params) {
  MLIRContext *ctx = module.getContext();
  OpBuilder b(ctx);
  Location loc = module->getLoc();

  SmallVector<Type, 3> cpuTypes = params.types;
  // Raise floats to f32
  for (Type &type : cpuTypes) {
    if (type.isa<FloatType>())
      type = b.getF32Type();
  }

  SmallVector<Type, 3> argTypes;
  getGemmTypes(cpuTypes, argTypes, /*isCpuVerifier=*/true);

  constexpr llvm::StringLiteral cpuKernName("host_naive_gemm");
  auto func =
      b.create<func::FuncOp>(loc, cpuKernName, b.getFunctionType(argTypes, {}));

  Block *block = func.addEntryBlock();
  b.setInsertionPointToStart(block);

  Value aVal = block->getArgument(0), bVal = block->getArgument(1),
        cVal = block->getArgument(2);
  auto cType = argTypes[2].cast<MemRefType>();
  Value zeroOut = rock::createZeroConstantOp(b, loc, cType.getElementType());

  b.create<linalg::FillOp>(loc, zeroOut, cVal);

  AffineExpr g = b.getAffineDimExpr(0), m = b.getAffineDimExpr(1),
             n = b.getAffineDimExpr(2), k = b.getAffineDimExpr(3);
  AffineMap aMap = AffineMap::get(
                4, 0, {g, transposeA ? k : m, transposeA ? m : k}, ctx),
            bMap = AffineMap::get(
                4, 0, {g, transposeB ? n : k, transposeB ? k : n}, ctx),
            cMap = AffineMap::get(
                4, 0, {g, transposeC ? n : m, transposeC ? m : n}, ctx);
  b.create<linalg::GenericOp>(
      loc, ValueRange{aVal, bVal}, ValueRange{cVal},
      ArrayRef<AffineMap>{aMap, bMap, cMap},
      ArrayRef<utils::IteratorType>{
          utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::parallel, utils::IteratorType::reduction},
      /*doc=*/"", /*library_call=*/"",
      [](OpBuilder &builder, Location loc, ValueRange elems) {
        Value a = elems[0], b = elems[1], c = elems[2];
        Type cType = c.getType();
        if (cType.isa<IntegerType>()) {
          Value aExt = rock::createTypeConversionOp(builder, loc, a, cType);
          Value bExt = rock::createTypeConversionOp(builder, loc, b, cType);
          Value mul = builder.create<arith::MulIOp>(loc, aExt, bExt);
          Value add = builder.create<arith::AddIOp>(loc, mul, c);
          builder.create<linalg::YieldOp>(loc, add);
        } else {
          Value mul = builder.create<arith::MulFOp>(loc, a, b);
          Value add = builder.create<arith::AddFOp>(loc, mul, c);
          builder.create<linalg::YieldOp>(loc, add);
        }
      });
  b.create<func::ReturnOp>(loc);
  module.push_back(func);
  return func;
}

static func::FuncOp getMemcpyFuncDecl(ModuleOp module, const MemRefType srcType,
                                      const MemRefType destType) {
  OpBuilder b(module.getContext());

  assert(srcType.getRank() <= 1 && "Memcopy takes 1-D sources");
  assert(destType.getRank() <= 1 && "Memcopy takes 1-D destinations");

  Type srcElemType = srcType.getElementType();
  Type dstElemType = destType.getElementType();
  // memcpy_<srcElemType>_<dstElemType>_(srcSize|any)
  std::string funcName = "_memcpy_";
  llvm::raw_string_ostream funcNameStr(funcName);
  funcNameStr << srcElemType << "_" << dstElemType << "_";

  int64_t numElements = -1;
  if (srcType.hasStaticShape())
    numElements = srcType.getNumElements();

  if (numElements != -1)
    funcNameStr << numElements;
  else
    funcNameStr << "any";

  if ((numElements == -1 && !destType.hasStaticShape()) ||
      numElements != destType.getNumElements())
    assert(0 && "Called for an uneven memcpy");

  func::FuncOp func = module.lookupSymbol<func::FuncOp>(funcName);
  if (func) // already exists
    return func;

  Location loc = b.getUnknownLoc();

  // clang-format off
  // func _memcpy_<srcElemType>_<dstElemType>_<size> (%arg0 : memref<sizexf32>, %arg1 : memref<sizexf16>) {
  //   %size = memref.dim %arg0(%c0)
  //   scf.for %i0 = %c0 to %size step %c1 {
  //     %2 = load %arg0[%i0] : memref<?xf32>
  //     store %2, %arg1[%i0] : memref<?xf32>
  //   }
  // }
  // clang-format on

  // Emit function definition
  func = func::FuncOp::create(loc, funcName,
                              b.getFunctionType({srcType, destType}, {}));

  module.push_back(func);

  // Create a new block
  Block *block = func.addEntryBlock();
  b.setInsertionPoint(block, block->begin());

  auto src = block->getArgument(0);
  auto dst = block->getArgument(1);

  auto insertConversionLogic = [&](OpBuilder &opBuilder, Value loadOp) {
    Value newLoadOp = loadOp;
    if (srcElemType != dstElemType) {
      // insert conversion logic
      auto srcBitWidth = srcElemType.getIntOrFloatBitWidth();
      auto dstBitWidth = dstElemType.getIntOrFloatBitWidth();
      if (srcElemType.isIntOrIndex()) {
        if (dstElemType.isIntOrIndex()) {
          if (srcBitWidth < dstBitWidth)
            newLoadOp =
                opBuilder.create<arith::ExtSIOp>(loc, dstElemType, loadOp);
          else
            newLoadOp =
                opBuilder.create<arith::TruncIOp>(loc, dstElemType, loadOp);
        } else {
          assert(dstElemType.isa<FloatType>());
          newLoadOp =
              opBuilder.create<arith::SIToFPOp>(loc, dstElemType, loadOp);
        }
      } else {
        assert(srcElemType.isa<FloatType>());
        if (dstElemType.isIntOrIndex()) {
          newLoadOp =
              opBuilder.create<arith::FPToSIOp>(loc, dstElemType, loadOp);
        } else {
          if (srcBitWidth < dstBitWidth)
            newLoadOp =
                opBuilder.create<arith::ExtFOp>(loc, dstElemType, loadOp);
          else
            newLoadOp =
                opBuilder.create<arith::TruncFOp>(loc, dstElemType, loadOp);
        }
      }
    }
    return newLoadOp;
  };

  if (srcType.getRank() == 1) {
    auto cst0Op = b.create<arith::ConstantIndexOp>(loc, 0);
    auto cst1Op = b.create<arith::ConstantIndexOp>(loc, 1);
    auto size = b.create<memref::DimOp>(loc, src, cst0Op);

    auto loop0 = b.create<scf::ForOp>(loc, cst0Op, size, cst1Op);
    auto bt0 = OpBuilder::atBlockTerminator(loop0.getBody());
    auto iv0 = loop0.getInductionVar();

    Value loadOp = bt0.create<memref::LoadOp>(loc, src, ValueRange{iv0});
    loadOp = insertConversionLogic(bt0, loadOp);
    bt0.create<memref::StoreOp>(loc, loadOp, dst, ValueRange{iv0});
  } else {
    Value loadOp = b.create<memref::LoadOp>(loc, src, ValueRange{});
    loadOp = insertConversionLogic(b, loadOp);
    b.create<memref::StoreOp>(loc, loadOp, dst, ValueRange{});
  }

  b.create<func::ReturnOp>(loc, ValueRange{});

  return func;
}

static void emitMemcpy(OpBuilder &b, Value src, Value dst) {
  auto module = b.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
  auto loc = b.getUnknownLoc();

  Value srcFlat = makeNDMemRef(b, src, 1);
  Value dstFlat = makeNDMemRef(b, dst, 1);
  auto srcFlatType = srcFlat.getType().cast<MemRefType>();
  auto dstFlatType = dstFlat.getType().cast<MemRefType>();
  auto memcpyFunc = getMemcpyFuncDecl(module, srcFlatType, dstFlatType);

  b.create<func::CallOp>(loc, memcpyFunc, ValueRange{srcFlat, dstFlat});
}

static void emitPrintTensor(OpBuilder &b, Value var) {
  auto loc = b.getUnknownLoc();
  auto varType = var.getType().template dyn_cast<MemRefType>();
  auto elemType = varType.getElementType();
  auto floatType = b.getF32Type();
  auto int32Type = b.getIntegerType(32);

  // get print func
  std::string printFuncName = "printMemrefF32";
  Value pvar = var;
  Type tensorType = floatType;
  if (elemType.isIntOrIndex()) {
    printFuncName = "printMemrefI32";
    tensorType = int32Type;
    if (elemType != int32Type) {
      // make copy
      auto pvarType = MemRefType::get(varType.getShape(), int32Type);
      pvar = b.create<memref::AllocOp>(loc, pvarType);
      emitMemcpy(b, var, pvar);
    }
  } else if (elemType != floatType) {
    // make copy
    auto pvarType = MemRefType::get(varType.getShape(), floatType);
    pvar = b.create<memref::AllocOp>(loc, pvarType);
    emitMemcpy(b, var, pvar);
  }

  auto module = b.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
  auto unrankedMRType = UnrankedMemRefType::get(tensorType, 0);
  auto printFunc = makeFuncDecl(module, printFuncName, {unrankedMRType});

  // Emit cast + call print
  auto printCast = b.create<memref::CastOp>(loc, unrankedMRType, pvar);
  b.create<func::CallOp>(loc, printFunc, ValueRange{printCast});

  if (pvar != var) {
    // dealloc pvar
    b.create<memref::DeallocOp>(loc, pvar);
  }
}

static void checkRandomInputsE2E() {
  if (randomSeed != "none" && randomSeed != "fixed" &&
      randomDataType == "float") {
    int min = randMin.getValue();
    int max = randMax.getValue();
    if (min < 0 && max > 0) {
      llvm::errs() << "WARNING: E2E tests with float random inputs within ";
      llvm::errs() << "WARNING: E2E tests may fail with both positive and ";
      llvm::errs() << "negative float random inputs\n";
      llvm::errs() << "         Try range [1, 3] by setting ";
      llvm::errs() << "\"-rand_min 1 -rand_max 3\"\n";
    }
  }
}

static func::FuncOp createVerifierFunc(ModuleOp module, const KernelIF &kernel,
                                       MemRefType testType, MemRefType valType,
                                       std::string funcName) {
  checkRandomInputsE2E();
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
  auto test = block->getArgument(0);
  auto val = block->getArgument(1);
  // obtain element type
  auto testElemType = testType.getElementType();
  auto valElemType = valType.getElementType();

  // Flatten the arguments to 1D for passing to the verification function
  // %test_flat = memref.collapse_shape %arg0 ...
  // %val_flat = memref.collapse_shape %arg1 ...
  Value testFlat = makeNDMemRef(b, test, 1);
  Value valFlat = makeNDMemRef(b, val, 1);
  auto valFlatType = valFlat.getType().cast<MemRefType>();
  // Emit constants for thresholds

  // clang-format off
  // %cst = arith.constant 9.99999974E-6 : f32
  // %cst_0 = arith.constant 1.000000e+02 : f32
  // %cst_1 = arith.constant 1.000000e+02 : f32
  // clang-format on

  auto getF32Val = [&](float val) -> Value {
    llvm::APFloat apVal(val);
    return b.create<arith::ConstantFloatOp>(loc, apVal, floatType);
  };
  // Thresholds for different metrics
  // RMS: 0.00003f for all data types
  // absDiff: 100.0f for all data types, i.e. the maxAbsDiff metric is disabled
  // relDiff 100.0f for f16, i.e. maxRelDiff metric is disabled for f16
  // datatypes
  //         0.000001f for other data types
  char printDebug = static_cast<char>(printVerifyResults.getValue());

  auto printDebugVal =
      b.create<arith::ConstantIntOp>(loc, printDebug, charType);

  // obtain function name of the verifier wrapper
  std::string verifyFuncName = "mcpuVerify";
  if (valElemType.isF32()) {
    verifyFuncName += "Float";
  } else if (valElemType.isInteger(8) || valElemType.isInteger(32) ||
             valElemType.isInteger(64)) {
    verifyFuncName +=
        "Int" + std::to_string(testElemType.getIntOrFloatBitWidth()) + "Int" +
        std::to_string(valElemType.getIntOrFloatBitWidth());
  } else {
    llvm::errs() << "Unsupported type of validation function output: ";
    llvm::errs() << " (Only f32, int32 and int64 are supported)\n";
    exit(1);
  }

  auto mr1DUnkTestType =
      MemRefType::get({mlir::ShapedType::kDynamic}, testElemType);
  auto mr1DUnkValType =
      MemRefType::get({mlir::ShapedType::kDynamic}, valElemType);

  bool isTestAndValSameType =
      (testElemType.isIntOrIndex() || testElemType.isF32());

  Value testResult, valResult; // Values passed to the verify function
  Value testResultNew;         // Values used for type conversion
  if (!isTestAndValSameType) {
    // When gpu kernel output data type = f16 | bf16, type conversions
    // are required before calling the verify function

    // Cast test result to the same type as valid result

    // clang-format off
    // %0 = memref.alloc() : memref<802816xf32>
    // call @_memcpy_f16_f32_802816(%test_flat, %0) : (memref<802816xf16>, memref<802816xf32>) -> ()
    // %5 = memref.cast %0 : memref<802816xf32> to memref<?x?x?x?x?xf32>
    // clang-format on

    testResultNew = b.create<memref::AllocOp>(loc, valFlatType);
    emitMemcpy(b, testFlat, testResultNew);
    testResult = b.create<memref::CastOp>(loc, mr1DUnkValType, testResultNew);
    mr1DUnkTestType = mr1DUnkValType;

    // Cast valid result down to the same type as test result and cast back
    //   For f16 and bf16 datatypes, gpu hardware outputs f32 results, which are
    //   truncated to f16/bf16 before returning from the gpu kernel
    //   To make the comparison fair, the truncation step is added manually to
    //   the validation results.

    // clang-format off
    // affine.for %arg2 = 0 to 802816 {
    //   %7 = memref.load %arg1[%arg2] : memref<802816xf32>
    //   %8 = arith.truncf %7 : f32 to f16
    //   %9 = arith.extf %8 : f16 to f32
    //   memref.store %9, %arg1[%arg2] : memref<802816xf32>
    // }
    // clang-format on

    SmallVector<int64_t, 1> lowerBounds(1, 0);
    SmallVector<int64_t, 1> upperBounds = {valFlatType.getNumElements()};
    SmallVector<int64_t, 1> steps(1, 1);

    if (testElemType != valElemType) {
      affine::buildAffineLoopNest(
          b, loc, lowerBounds, upperBounds, steps,
          [valFlat, testElemType, valElemType](OpBuilder &b, Location loc,
                                               ValueRange ivs) {
            auto valOrig = b.create<memref::LoadOp>(loc, valFlat, ivs);
            auto valTruncated =
                b.create<arith::TruncFOp>(loc, testElemType, valOrig);
            auto valExt =
                b.create<arith::ExtFOp>(loc, valElemType, valTruncated);
            b.create<memref::StoreOp>(loc, valExt, valFlat, ivs);
          });
    }
  } else {
    testResult = b.create<memref::CastOp>(loc, mr1DUnkTestType, testFlat);
  }

  // Prepare the validation result for the verify function
  valResult = b.create<memref::CastOp>(loc, mr1DUnkValType, valFlat);
  // Declare and call the wrapper verify function
  func::FuncOp verifyFuncDecl;

  if (testElemType.isa<FloatType>()) {
    auto thr_RMS = getF32Val(RMSThreshold.getValue());
    auto thr_absDiff = getF32Val(absDiffThreshold.getValue());
    Value thr_relDiff = getF32Val(relDiffThreshold.getValue());
    if (testElemType.isF16())
      thr_relDiff = getF32Val(100.0f);

    verifyFuncDecl = makeFuncDecl(module, verifyFuncName,
                                  {mr1DUnkTestType, mr1DUnkValType, floatType,
                                   floatType, floatType, charType});
    b.create<func::CallOp>(loc, verifyFuncDecl,
                           ValueRange{testResult, valResult, thr_RMS,
                                      thr_absDiff, thr_relDiff, printDebugVal});
  } else {
    verifyFuncDecl = makeFuncDecl(module, verifyFuncName,
                                  {mr1DUnkTestType, mr1DUnkValType, charType});
    b.create<func::CallOp>(loc, verifyFuncDecl,
                           ValueRange{testResult, valResult, printDebugVal});
  }

  if (!isTestAndValSameType) {
    // Deallocate the buffer for f32 version of the test results
    b.create<memref::DeallocOp>(loc, testResultNew);
  }

  b.create<func::ReturnOp>(loc, ValueRange{});

  return func;
}

// If the fut expects certain args (mostly output buffers),
// this will populate the linalg.fill calls to do those based
// on the presense of rock::PrefillAttr. This is to mimic the
// requirement on the kernel launcher to do the same for the
// expected funtionality.
void insertPrefills(func::FuncOp fut) {
  SmallVector<ModuleOp, 1> innerModules;
  fut->getParentOfType<ModuleOp>().walk(
      [&](ModuleOp module) { innerModules.push_back(module); });
  innerModules.push_back(fut->getParentOfType<ModuleOp>());
  fut.walk([&](mhal::LaunchOp launchOp) {
    Location loc = launchOp->getLoc();
    DenseMap<int, Attribute> argInitValues;
    StringRef callee = launchOp.getCallee();
    for (ModuleOp module : innerModules) {
      if (func::FuncOp calleeFunc = module.lookupSymbol<func::FuncOp>(callee)) {
        size_t argCount = calleeFunc.getArguments().size();
        for (size_t i = 0; i < argCount; i++) {
          if (Attribute initAttr =
                  calleeFunc.getArgAttr(i, rock::PrefillAttr::getMnemonic())) {
            argInitValues[i] = initAttr;
          }
        }
      }
    }
    {
      OpBuilder builder(launchOp);
      OpBuilder::InsertionGuard guard(builder);
      for (auto argIdxAndValueAttr : argInitValues) {
        int argIdx = argIdxAndValueAttr.first;
        auto valueAttr = argIdxAndValueAttr.second;
        auto fillValue =
            builder.create<arith::ConstantOp>(loc, cast<TypedAttr>(valueAttr));
        Value originalArg = launchOp.getArgOperands()[argIdx];
        builder.create<linalg::FillOp>(loc, ValueRange{fillValue},
                                       ValueRange{originalArg});
      }
    }
  });
}

// Convert the mhal.launch/mhal.await pattern back to func.call.
void undoAsyncLaunchPass(Operation *cloneFunc) {
  SymbolTableCollection symbolTable;
  auto walker = [&](Operation *op) {
    OpBuilder builder(op);
    if (auto launch = dyn_cast<mhal::LaunchOp>(op)) {
      SymbolRefAttr calleeAttr = launch->getAttrOfType<SymbolRefAttr>("callee");
      CallOpInterface callInt = dyn_cast<CallOpInterface>(op);
      assert(callInt);
      auto operands = callInt.getArgOperands();
      auto call = builder.create<func::CallOp>(op->getLoc(), calleeAttr,
                                               TypeRange{}, operands);
      call->moveBefore(op);
      op->dropAllUses();
      op->erase();
      return WalkResult::interrupt();
    }
    if (auto launch = dyn_cast<mhal::AwaitOp>(op)) {
      op->erase();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };
  while (cloneFunc->walk(walker).wasInterrupted()) {
  }
}

static void insertValidationCalls(const GenParams &genParams, OpBuilder &b,
                                  ModuleOp module,
                                  SmallVectorImpl<Value> &valVars,
                                  SmallVectorImpl<Value> &localVars,
                                  ArrayRef<int32_t> outIndices, Operation *func,
                                  KernelIF &root0) {
  auto validationType = genValidation.getValue();
  auto loc = b.getUnknownLoc();
  bool hasAccel = rock::isAccel(genParams.features);
  bool heuristicValidation =
      !genVerifierKeepPerfConfig && !genParams.perfConfig.empty();
  bool isSmallFloatIn = false;
  if (!genParams.types.empty())
    if (auto ftype = genParams.types[0].dyn_cast<FloatType>())
      isSmallFloatIn = ftype.getWidth() < 32;
  bool gpuValidation = validationType == "gpu" &&
                       ((hasAccel || isSmallFloatIn) || heuristicValidation);
  if (gpuValidation) {
    if (genParams.convConfig.has_value()) { // conv GPU validation
      // generate generic kernels
      const auto &genConfig = **genParams.convConfig;
      rock::Conv2dGenerator conv2dGenerator(genConfig);
      if (heuristicValidation || hasAccel)
        conv2dGenerator.setPerfConfig("");
      // use non-accel kernels to verify accel kernels except when
      // verifying a tuning case
      if (hasAccel)
        conv2dGenerator.flipAccel();
      if (!((hasAccel || heuristicValidation) &&
            genConfig.inputDataTypeStr == "i8"))
        // use f32 data type to verify non-f32 or xdlops f32 kernels
        // except that i8 xdlops or tuned is verified with i8 non-xdlops.
        conv2dGenerator.setDataTypes("f32");

      int kernelStart = genConfig.kernelId;
      int kernelCount = 0;
      if (failed(conv2dGenerator.getKernelCount(b, kernelCount))) {
        llvm::errs() << "Getting kernel count failed.\n";
        exit(1);
      }
      if (kernelStart < 0) {
        kernelStart = 0;
      } else {
        kernelCount = kernelStart + 1;
      }
      // generate all sub-kernels, and get corresponding gemmId
      std::string kernelBaseName = genConfig.kernelBaseName;
      llvm::errs() << kernelBaseName << "\n";
      for (int i = kernelStart; i < kernelCount; ++i) {
        conv2dGenerator.setKernelName(kernelBaseName + "_" + std::to_string(i));
        if (failed(conv2dGenerator.genConvModule(module, i, true,
                                                 /*ignoreTuning=*/true))) {
          llvm::errs() << "Module population failed.\n";
          exit(1);
        }
        KernelIF kernel(conv2dGenerator.getKernelFunc());
        rock::Conv2dGenerator::Config newConfig = conv2dGenerator.getConfig();
        auto kernelWrapperFunc = createGPUWrapper(module, kernel);

        // Decide whether to trim the last workspace argument to the verifier
        // GPU kernel.
        rock::Conv2dGenerator originalConv2dGenerator(genConfig);
        bool originalHasWorkspace = false, verifierHasWorkspace = false;
        if (failed(originalConv2dGenerator.hasWorkspace(
                b, originalHasWorkspace))) {
          llvm::errs() << "Getting workspace failed.\n";
          exit(1);
        }
        if (failed(conv2dGenerator.hasWorkspace(b, verifierHasWorkspace))) {
          llvm::errs() << "Getting workspace failed.\n";
          exit(1);
        }
        if (originalHasWorkspace && !verifierHasWorkspace) {
          valVars.resize(valVars.size() - 1);
        }

        b.create<func::CallOp>(loc, kernelWrapperFunc, valVars);
      }
      conv2dGenerator.setKernelName(kernelBaseName);
    } else { // gemm GPU validation
      GenParams newParams = genParams;

      if (heuristicValidation || hasAccel)
        newParams.perfConfig = "";
      if (hasAccel)
        newParams.features = bitEnumClear(genParams.features,
                                          mlir::rock::GemmFeatures::mfma |
                                              mlir::rock::GemmFeatures::wmma);

      if (!((heuristicValidation || hasAccel) &&
            genParams.types[0].isInteger(8)))
        // use f32 data type to verify non-f32 or xdlops f32 kernels
        // except that i8 xdops is verified with i8 non-xdolps and tuned i8 is
        // verified with itself in heuristic mode.
        newParams.types = SmallVector<Type>(3, b.getF32Type());

      KernelIF kernel(
          createGpuGemmKernel(module, newParams, /*is_verifier=*/true));
      auto kernelWrapperFunc = createGPUWrapper(module, kernel);
      b.create<func::CallOp>(loc, kernelWrapperFunc, valVars);
    }
  } else if (validationType != "clone") { // -pv_with_cpp or -pv_with_mlir (-pv)
    // Emit call to host_<conv>
    if (genParams.convConfig.has_value()) {
      const auto &genConfig = **genParams.convConfig;
      auto cpuConvFunc = createCPUConvFunc(module, genConfig);
      b.create<func::CallOp>(loc, cpuConvFunc, valVars);
    } else if (genParams.operation == rock::KernelType::Gemm) {
      // Emit call to host gemm
      if (validationType == "cpp") {
        llvm::errs() << "External gemm validator is not available\n";
        exit(1);
      }
      auto cpuGemmFunc = createCpuGemmKernelWithMlir(module, genParams);
      b.create<func::CallOp>(loc, cpuGemmFunc, valVars);
    } else {
      llvm::errs()
          << "Validation generation requested, but no operation specified\n";
      exit(1);
    }
  } else { // clone
    // Clone the kernel-calling function.  xmir-runner will call the appropriate
    // binary kernel from the mhal.launch ops;  here, we'll replace those with
    // func.call which will get the MLIR kernel.  No redirection of callees
    // needed.
    auto *cloneFunc = func->clone();
    insertPrefills(static_cast<func::FuncOp>(func));
    undoAsyncLaunchPass(cloneFunc);
    SymbolOpInterface cloneFuncOp = dyn_cast<SymbolOpInterface>(cloneFunc);
    SmallString<128> nameBuffer(cloneFuncOp.getName());
    nameBuffer += "_cloned";
    cloneFuncOp.setName(nameBuffer);
    cloneFunc->removeAttr("kernel");
    SymbolTable symbolTable(module);
    symbolTable.insert(cloneFunc);
    b.create<func::CallOp>(loc, SymbolRefAttr::get(cloneFunc), TypeRange{},
                           valVars);
  }

  // Emit call to verifier
  for (int32_t outIdx : outIndices) {
    Value testResult = localVars[outIdx];
    Value valResult = valVars[outIdx];
    auto testType = testResult.getType().dyn_cast<MemRefType>();
    auto valType = valResult.getType().dyn_cast<MemRefType>();
    std::string funcName =
        root0.func.getName().str() + "_verify" + std::to_string(outIdx);
    auto verifierFunc =
        createVerifierFunc(module, root0, testType, valType, funcName);
    b.create<func::CallOp>(loc, verifierFunc,
                           ValueRange{testResult, valResult});
  }
}

static LogicalResult populateHostHarnessLogic(
    ModuleOp module, const SmallVector<KernelIF, 8> &kernels,
    const SmallVector<KernelIF, 8> &roots, const GenParams &genParams) {
  MLIRContext *context = module.getContext();
  OpBuilder b(context);
  Location loc = b.getUnknownLoc();

  // Construct main function.
  auto func = func::FuncOp::create(loc, "main", b.getFunctionType({}, {}));
  module.push_back(func);
  Block *block = func.addEntryBlock();
  b.setInsertionPoint(block, block->begin());

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
  bool hasCloneValidation = hasValidation && (validationType == "clone");
  bool hasAccel = rock::isAccel(genParams.features);
  bool heuristicValidation =
      !genVerifierKeepPerfConfig && !genParams.perfConfig.empty();
  bool isSmallFloatIn = false;
  bool isFp8 = false;

  if (!genParams.types.empty()) {
    if (auto ftype = genParams.types[0].dyn_cast<FloatType>()) {
      isSmallFloatIn = ftype.getWidth() < 32;
      isFp8 = ftype.isFloat8E4M3FNUZ() || ftype.isFloat8E5M2FNUZ();
    }
  }
  bool gpuValidation = validationType == "gpu" &&
                       ((hasAccel || isSmallFloatIn) || heuristicValidation);

  bool isRandom = (randomSeed != "fixed" && randomSeed != "none");
  if (isRandom && isFp8) {
    llvm::errs() << "WARNING: Random values not supported for fp8, defaulting "
                    "to -rand fixed\n";
    randomSeed = "fixed";
    isRandom = false;
  }

  if (isRandom) {
    auto seedFunc = makeFuncDecl(module, "seedRandomValues", {b.getI32Type()});
    int seed = getRandomSeed();
    Value seedConst = b.create<arith::ConstantIntOp>(loc, seed, b.getI32Type());
    b.create<func::CallOp>(loc, seedFunc, seedConst);
  }

  SmallVector<int32_t, 2> outIndices;
  if (genParams.operation.has_value()) {
    switch (genParams.operation.value()) {
    case rock::KernelType::Conv2D:
    case rock::KernelType::Gemm:
      outIndices.push_back(2);
      break;
    case rock::KernelType::Conv2DBwdData:
      outIndices.push_back(1);
      break;
    case rock::KernelType::Conv2DBwdWeight:
      outIndices.push_back(0);
      break;
    }
  } else {
    outIndices = root0.outIndices;
  }

  SmallVector<Value, 5> localVars;
  SmallVector<Value, 5> valVars;
  for (auto [idx, paramType] : llvm::enumerate(root0.params)) {
    auto paramMRType = paramType.dyn_cast<MemRefType>();
    assert(paramMRType && "currently only supports memref types");
    Type elemType = paramMRType.getElementType();
    if (isCPUKernel) { // -prc
      assert(elemType.isF32() || elemType.isInteger(8) ||
             elemType.isInteger(32) || elemType.isInteger(64));
      if (genParams.operation.has_value()) {
        if (idx < genParams.types.size())
          elemType = genParams.types[idx];
        if (elemType.isa<IntegerType>() && llvm::is_contained(outIndices, idx))
          elemType = b.getIntegerType(64);
        paramMRType = MemRefType::get(paramMRType.getShape(), elemType);
      }
    }
    auto lvar = b.create<memref::AllocOp>(loc, paramMRType);
    localVars.push_back(lvar);
    if (!isRandom) {
      SmallVector<float, 3> initPattern = getTensorInitPattern(elemType);
      if (failed(populateTensorFillLogic(b, loc, initPattern, elemType, lvar)))
        return failure();
    } else {
      if (failed(populateRandomTensorFillLogic(b, loc, module, elemType, lvar,
                                               idx)))
        return failure();
    }

    if (hasValidation || (isCPUKernel && (isSmallFloatIn))) {
      // Emit validation var
      Type valElemType = floatType;
      if (genParams.operation.has_value() && elemType.isa<IntegerType>()) {
        valElemType = elemType;
        if (!gpuValidation && idx == 2)
          //-pv_with_mlir, -pv_with_cpp, or -pv_with_gpu && non-accel
          // validate in int64_t to detect overflow
          valElemType = b.getIntegerType(64);
      } else if ((genValidation == "clone") || elemType.isInteger(8) ||
                 elemType.isInteger(32)) {
        valElemType = elemType;
      }
      auto valType = MemRefType::get(paramMRType.getShape(), valElemType);
      auto vvar = b.create<memref::AllocOp>(loc, valType);
      valVars.push_back(vvar);

      emitMemcpy(b, lvar, vvar);
    }
  }

  // capture result index
  if (outIndices.empty()) {
    outIndices.push_back(localVars.size() - 1);
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
    // Clone-style validation wants to validate each root function.
    // Non-clone validation validates at end;  the roots are related kernels.
    if (hasCloneValidation)
      insertValidationCalls(genParams, b, module, valVars, localVars,
                            outIndices, root.func, root0);
  }

  // Run validation
  if (hasValidation && !hasCloneValidation)
    insertValidationCalls(genParams, b, module, valVars, localVars, outIndices,
                          func, root0);

  // Print and cleanup validation vars
  for (auto &vvar : valVars) {
    // print vvar
    for (int32_t outIdx : outIndices) {
      if ((vvar == valVars[outIdx]) && printValidationResults.getValue()) {
        emitPrintTensor(b, vvar);
      }
    }
  }

  // Print and cleanup
  for (auto &lvar : localVars) {
    // print lvar
    bool printp = printInputs.getValue();
    for (int32_t outIdx : outIndices) {
      if (lvar == localVars[outIdx])
        printp = printResults.getValue();
      if (printp)
        emitPrintTensor(b, lvar);
    }
  }

  for (auto &vvar : valVars) {
    b.create<memref::DeallocOp>(loc, vvar);
  }
  for (auto &lvar : localVars) {
    b.create<memref::DeallocOp>(loc, lvar);
  }

  b.create<func::ReturnOp>(loc, ValueRange{});

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
  module.walk([&](CallOpInterface callOp) -> WalkResult {
    // Don't substitute the call inside the wrapper.
    if (callOp->hasAttr("wrapped_call")) {
      callOp->removeAttr("wrapped_call");
      return WalkResult::advance();
    }

    // If the callee matches a wrapped function, update the call.
    Operation *callable = callOp.resolveCallable();
    if (callable) {
      func::FuncOp fop = dyn_cast<func::FuncOp>(*callable);
      if (wrappedFuncs.find(fop) != wrappedFuncs.end()) {
        callOp->setAttr("callee", FlatSymbolRefAttr::get(
                                      context, wrappedFuncs[fop].getSymName()));
      }
    }
    return WalkResult::advance();
  });

  return success();
}

static ModuleOp readTestFile(std::string inputFilenameStr, bool &hasUserKernel,
                             MLIRContext *context) {
  std::string errorMessage;

  // Set up the input file.
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  // Parse the input file.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  OwningOpRef<ModuleOp> moduleRef =
      parseSourceFile<ModuleOp>(sourceMgr, context);
  if (!moduleRef) {
    llvm::errs() << "Parse host harness " << inputFilename << " failed.\n";
    exit(1);
  }
  ModuleOp module = moduleRef.release();

  if (!perfConfig.empty()) {
    WalkResult findGemmOp =
        module->walk([&](rock::RockGemmWrapperInterface gemmOp) -> WalkResult {
          OpBuilder b(gemmOp.getContext());
          gemmOp->setAttr("perf_config", b.getStringAttr(perfConfig));
          return WalkResult::interrupt();
        });
    if (!findGemmOp.wasInterrupted()) {
      llvm::errs() << "Cannot find a Gemm kernel for perf_config\n";
      exit(1);
    }
  }

  module.walk([&](func::FuncOp func) -> WalkResult {
    if (func->hasAttr("kernel")) {
      hasUserKernel = true;
    }
    return WalkResult::advance();
  });

  return module;
}

static ModuleOp generateKernel(MLIRContext *context, GenParams &genParams,
                               ModuleOp module) {
  OpBuilder builder(context);
  static rock::Conv2dGenerator conv2dGenerator;

  bool isGemm = operation == rock::KernelType::Gemm;
  auto convConfig = populateConvConfig.getValue();

  if (!convConfig.empty() && isGemm) {
    llvm::errs() << "Cannot use --conv-config with gemm operations\n";
    exit(1);
  }

  if (convConfig.empty() && failed(detectMissingArguments())) {
    exit(1);
  }

  // Scenario 1: We use conv config to initialize everything
  if (!convConfig.empty()) {
    if (failed(conv2dGenerator.parseConvConfig(builder, convConfig.c_str()))) {
      llvm::errs() << "Module population failed.\n";
      exit(1);
    }
    genParams.types.push_back(conv2dGenerator.getFilterDataType(builder));
    genParams.types.push_back(conv2dGenerator.getInputDataType(builder));
    genParams.types.push_back(conv2dGenerator.getOutputDataType(builder));
    const auto *convConfig = &conv2dGenerator.getConfig();
    genParams.convConfig = convConfig;
    genParams.features = convConfig->features;
    genParams.operation =
        rock::kernelTypeFromConvOpType(convConfig->operation.value());
    genParams.perfConfig = convConfig->perfConfig;
  } else {
    // Scenario 2: We use llvm::cl::opt to initialize everything
    if (arch.getValue().empty()) {
      llvm::errs() << "--arch is not set\n";
      exit(1);
    }

    RocmDeviceName targetInfo;
    if (failed(targetInfo.parse(arch.getValue()))) {
      llvm::errs() << "Invalid architecture name: " << arch << "\n";
      exit(1);
    }
    std::string triple = targetInfo.getTriple().str();
    std::string chip = targetInfo.getChip().str();
    std::string chipFeatures = targetInfo.getFeaturesForBackend();
    SmallString<64> canonicalArch;
    targetInfo.getFullName(canonicalArch);
    arch = canonicalArch.str().str();

    LogicalResult status = success();

    Type elemType = typeFromString(inputDataType.getValue(), context);
    rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
    rock::GemmFeatures enabledFeatures = archInfo.getDefaultFeatures(elemType);
    // toggle feature list according to llvm::cl::opt inputs
    if (mfmaFeature != FeatureToggle::infer)
      enabledFeatures = bitEnumSet(enabledFeatures, rock::GemmFeatures::mfma,
                                   mfmaFeature == FeatureToggle::on);
    if (dotFeature != FeatureToggle::infer)
      enabledFeatures = bitEnumSet(enabledFeatures, rock::GemmFeatures::dot,
                                   dotFeature == FeatureToggle::on);
    if (atomicAddFeature != FeatureToggle::infer)
      enabledFeatures =
          bitEnumSet(enabledFeatures, rock::GemmFeatures::atomic_add,
                     atomicAddFeature == FeatureToggle::on);
    if (atomicFMaxF32Feature != FeatureToggle::infer)
      enabledFeatures =
          bitEnumSet(enabledFeatures, rock::GemmFeatures::atomic_fmax_f32,
                     atomicFMaxF32Feature == FeatureToggle::on);

    if (wmmaFeature != FeatureToggle::infer)
      enabledFeatures = bitEnumSet(enabledFeatures, rock::GemmFeatures::wmma,
                                   wmmaFeature == FeatureToggle::on);

    genParams.operation = operation;
    genParams.features = enabledFeatures;
    genParams.arch = arch;
    genParams.perfConfig = perfConfig;
    if (isGemm) {
      for (const auto &arg :
           {filterDataType.getValue(), inputDataType.getValue(),
            outputDataType.getValue()})
        genParams.types.push_back(typeFromString(arg, context));
      genParams.convConfig = std::nullopt;
      (void)createGpuGemmKernel(module, genParams);
    } else {
      conv2dGenerator = rock::Conv2dGenerator(
          arch, chip, triple, chipFeatures, perfConfig.getValue(),
          num_cu.getValue(), enabledFeatures,
          rock::convOpTypeFromKernelType(operation.getValue()),
          filterDataType.getValue(), inputDataType.getValue(),
          outputDataType.getValue(), dilationHeight.getValue(),
          dilationWidth.getValue(), strideHeight.getValue(),
          strideWidth.getValue(), paddingHeightLeft.getValue(),
          paddingHeightRight.getValue(), paddingWidthLeft.getValue(),
          paddingWidthRight.getValue(), filterLayout.getValue(),
          inputLayout.getValue(), outputLayout.getValue());

      status = conv2dGenerator.parseConvDims(
          batchSize, groupSize, inputChannel, inputHeight, inputWidth,
          outputChannel, outputHeight, outputWidth, filterHeight, filterWidth);
      if (failed(status)) {
        llvm::errs() << "Could not parse convolution dimensions\n";
        exit(1);
      }

      genParams.types.push_back(conv2dGenerator.getFilterDataType(builder));
      genParams.types.push_back(conv2dGenerator.getInputDataType(builder));
      genParams.types.push_back(conv2dGenerator.getOutputDataType(builder));
      genParams.convConfig = &conv2dGenerator.getConfig();
    }
  }

  // TODO: Extract isApplicable check to be its own component
  if (!isGemm &&
      failed(conv2dGenerator.isApplicable(/* checkChip = */ false))) {
    llvm::errs() << "Convolution configuration does not have valid dimension\n";
    exit(1);
  }

  if (genParams.convConfig.has_value()) {
    const auto &genConfig = **genParams.convConfig;
    if (genCPUKernel.getValue()) {
      (void)createCPUConvFunc(module, genConfig);
    } else {
      // Populate the module.
      int kernelStart = genConfig.kernelId;
      int kernelCount = 0;
      if (failed(conv2dGenerator.getKernelCount(builder, kernelCount))) {
        llvm::errs() << "Getting kernel count failed.\n";
        exit(1);
      }
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

  return module;
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerRocMLIRDialects(registry);
  // Parse pass names in main to ensure static initialization completed.
  mlir::registerMLIRContextCLOptions();
  MLIRContext context(registry);
  context.loadDialect<rock::RockDialect, func::FuncDialect, scf::SCFDialect,
                      affine::AffineDialect, memref::MemRefDialect,
                      math::MathDialect, arith::ArithDialect,
                      vector::VectorDialect, gpu::GPUDialect,
                      linalg::LinalgDialect>();

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR Rock Dialect host generation\n");

  if (operation != rock::KernelType::Gemm) {
    verifyConvLayout();
    correctConvParameters();
  }
  populateDefaults();

  bool hasUserKernel = !testFuncName.empty();

  ModuleOp module;
  GenParams genParams;

  if (!inputFilename.empty()) {
    module = readTestFile(inputFilename.getValue(), hasUserKernel, &context);
  } else {
    if (genValidation == "clone") {
      llvm::errs()
          << "Clone validation is not compatible with kernel generation.\n";
      exit(1);
    }
    OpBuilder builder(&context);
    module = ModuleOp::create(builder.getUnknownLoc());
  }

  if (!hasUserKernel) {
    module = generateKernel(&context, genParams, module);
  }

  if (emitTuningSpace) {
    auto tunableParams = rock::createTunableParamSpace(module);
    std::string perfConfig;
    for (auto param : tunableParams->tuningRange) {
      param.getPerfConfigStr(perfConfig);
      llvm::outs() << perfConfig << "\n";
    }
    delete tunableParams;
    return 0;
  }

  if (emitTuningKey) {
    llvm::outs() << rock::getTuningProblemStr(module) << "\n";
    return 0;
  }

  SmallVector<KernelIF, 8> kernels;
  SmallVector<KernelIF, 8> rootIFs;

  if (testFuncName.empty()) {
    // Compute set of call-graph root nodes;  they're the ones we need to
    // call from main().  Start with all nodes, then erase the ones that
    // have edges to them.  Use SetVector because we want to preserve the
    // order to match an older implementation.
    CallGraph cg(module);
    SetVector<CallGraphNode *> roots(cg.begin(), cg.end());
    for (auto &node : roots) {
      for (auto &edge : *node)
        roots.remove(edge.getTarget());
      func::FuncOp func =
          dyn_cast<func::FuncOp>(node->getCallableRegion()->getParentOp());
      if (func->hasAttr("original_func"))
        roots.remove(node);
      if (func->getParentOp() && func->getParentOp()->getParentOp())
        roots.remove(node);
    }

    for (auto *node : roots) {
      func::FuncOp func =
          dyn_cast<func::FuncOp>(node->getCallableRegion()->getParentOp());
      rootIFs.emplace_back(func);
    }
    module.walk([&](func::FuncOp func) -> WalkResult {
      if (func->hasAttr("kernel")) {
        kernels.emplace_back(func);
      }
      return WalkResult::advance();
    });
  } else {
    auto func = module.lookupSymbol<func::FuncOp>(testFuncName);
    assert(func && "does -fut point to the wrong function?");
    kernels.emplace_back(func); // +++pf: should it be a kernel?
    rootIFs.emplace_back(func);
  }

  // populate host logic.
  if (genHostHarness.getValue()) {
    if (failed(populateHostHarnessLogic(module, kernels, rootIFs, genParams))) {
      llvm::errs() << "Host logic populated failed.\n";
      exit(1);
    }
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
