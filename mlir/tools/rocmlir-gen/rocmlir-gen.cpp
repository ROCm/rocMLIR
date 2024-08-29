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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/Generator/ConvGenerator.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/ExecutionEngine/RocmDeviceName.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AsmState.h"
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
#include "mlir/InitRocMLIRCLOptions.h"
#include "mlir/InitRocMLIRDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

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
        clEnumValN(rock::KernelType::Conv, "conv", "Forward convolution"),
        clEnumValN(rock::KernelType::ConvBwdData, "conv_bwd_data",
                   "Backpropogate convolution data"),
        clEnumValN(rock::KernelType::ConvBwdWeight, "conv_bwd_weight",
                   "Backpropogate convolution weights"),
        clEnumValN(rock::KernelType::Gemm, "gemm", "Matrix multiplication"),
        clEnumValN(rock::KernelType::Attention, "attention",
                   "Attention operation of transformer models")),
    llvm::cl::value_desc("kernel type"),
    llvm::cl::init(rock::KernelType::Conv));

static llvm::cl::opt<std::string> arch(
    "arch",
    llvm::cl::desc("amdgpu architecture, eg: gfx803, gfx900, gfx906, gfx908"),
    llvm::cl::value_desc("GFX architecture string"), llvm::cl::init(""));

static llvm::cl::opt<int> num_cu(
    "num_cu",
    llvm::cl::desc("Number of compute units, valid combinations include: "
                   "gfx803(36/64), gfx900(56/64), "
                   "gfx906(60/64), gfx908(120)"),
    llvm::cl::value_desc("compute unit value"), llvm::cl::init(0));

static llvm::cl::opt<bool> reverse_grid(
    "reverse_grid",
    llvm::cl::desc(
        "Indicates whether to reverse the workgroup indices in the kernel"),
    llvm::cl::value_desc("boolean"), llvm::cl::init(false));

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

// Di
static llvm::cl::opt<int64_t>
    inputDepth("in_d", llvm::cl::desc("Input depth"),
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

// Z
static llvm::cl::opt<int64_t>
    filterDepth("fil_d", llvm::cl::desc("Filter depth"),
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

// Do
static llvm::cl::opt<int64_t> outputDepth(
    "out_d", llvm::cl::desc("Output depth"),
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

// dilation depth
static llvm::cl::opt<int> dilationDepth("dilation_d",
                                        llvm::cl::desc("Dilation depth"),
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

// stride depth
static llvm::cl::opt<int> strideDepth("conv_stride_d",
                                      llvm::cl::desc("Stride depth"),
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

// padding depth
static llvm::cl::opt<int> paddingDepth("padding_d",
                                       llvm::cl::desc("Padding depth"),
                                       llvm::cl::value_desc("attribute value"),
                                       llvm::cl::init(0));

static llvm::cl::opt<int>
    paddingDepthLeft("padding_d_l", llvm::cl::desc("Padding depth Left"),
                     llvm::cl::value_desc("attribute value"),
                     llvm::cl::init(0));

static llvm::cl::opt<int>
    paddingDepthRight("padding_d_r", llvm::cl::desc("Padding depth Right"),
                      llvm::cl::value_desc("attribute value"),
                      llvm::cl::init(0));

/// Matrix options
static llvm::cl::opt<int64_t> gemmM("m",
                                    llvm::cl::desc("M dimension of gemm()"),
                                    llvm::cl::value_desc("positive integer"),
                                    llvm::cl::init(-1));

static llvm::cl::opt<int64_t> gemmK("k",
                                    llvm::cl::desc("K dimension of gemm()"),
                                    llvm::cl::value_desc("positive integer"),
                                    llvm::cl::init(-1));

static llvm::cl::opt<int64_t> gemmN("n",
                                    llvm::cl::desc("N dimension of gemm()"),
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
        else if (filterDataType == inputDataType)
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

static llvm::cl::opt<bool> emitSplitKSelectionLikelihood(
    "emit-split-k-selection-likelihood",
    llvm::cl::desc(
        "Print SplitK selection likelihood for the specified kernel"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> emitModuleFusabilityForPerfConfig(
    "emit-module-fusibility-for",
    llvm::cl::desc("Print whether module is fusible given a perf config"),
    llvm::cl::init(""));

static llvm::cl::opt<rock::TuningParamSetKind> emitTuningSpace(
    "emit-tuning-space",
    llvm::cl::desc("Print a tuning space for the specified kernel"),
    llvm::cl::values(
        clEnumValN(rock::TuningParamSetKind::Quick, "quick",
                   "Quick tuning space"),
        clEnumValN(rock::TuningParamSetKind::Full, "full",
                   "Full tuning space, excluding known-bad configurations"),
        clEnumValN(
            rock::TuningParamSetKind::Exhaustive, "exhaustive",
            "All tuning space combinations, including inapplicable ones")),
    llvm::cl::value_desc("tuning space kind to emit"),
    llvm::cl::init(rock::TuningParamSetKind::Full));

static llvm::cl::opt<bool> emitTuningKey(
    "emit-tuning-key",
    llvm::cl::desc(
        "Prints out the struct of the problem to be tuned for inspection."),
    llvm::cl::value_desc(
        "String formatted fields of the problem which is going to be tuned."),
    llvm::cl::init(false));

// Attention related args
// ----------------------

static llvm::cl::opt<int64_t> sequenceLengthQ(
    "seq_len_q", llvm::cl::desc("sequence length of Q in attention()"),
    llvm::cl::value_desc("positive integer"), llvm::cl::init(-1));

static llvm::cl::opt<int64_t> sequenceLengthK(
    "seq_len_k", llvm::cl::desc("sequence length of K in attention()"),
    llvm::cl::value_desc("positive integer"), llvm::cl::init(-1));

static llvm::cl::opt<int64_t>
    headDimQK("head_dim_qk",
              llvm::cl::desc("head dimension of Q,K in attention()"),
              llvm::cl::value_desc("positive integer"), llvm::cl::init(-1));

static llvm::cl::opt<int64_t>
    headDimV("head_dim_v", llvm::cl::desc("head dimension of v in attention()"),
             llvm::cl::value_desc("positive integer"), llvm::cl::init(-1));

static llvm::cl::opt<bool>
    hasAttnScale("with-attn-scale",
                 llvm::cl::desc("Generate an attention kernel that is using a "
                                "scaling input for the first gemm"),
                 llvm::cl::init(false));

static llvm::cl::opt<bool> hasAttnBias(
    "with-attn-bias",
    llvm::cl::desc("Generate an attention kernel that is using a bias"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> transposeQ(
    "transQ",
    llvm::cl::desc("whether matrix Q of attention op is "
                   "Gxseq_len_qxhead_qk (default) or Gxhead_qkxseq_len_q"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> transposeK(
    "transK",
    llvm::cl::desc("whether matrix K of attention op is "
                   "Gxseq_len_kxhead_qk (default) or Gxheadxseq_len_q"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> transposeV(
    "transV",
    llvm::cl::desc("whether matrix V of attention op is "
                   "Gxseq_len_kxhead_v (default) or Gxhead_vxseq_len_k"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> transposeO(
    "transO",
    llvm::cl::desc("whether matrix O of attention op is "
                   "Gxseq_len_qxhead_v (default) or Gxhead_vxseq_len_q"),
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

// generate clone harness program.
static llvm::cl::opt<bool> genCloneHarness(
    "clone-harness", llvm::cl::desc("To generate clone harness"),
    llvm::cl::value_desc("To generate clone harness"), llvm::cl::init(false));

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
        "Select verification from: none(default), cpu, gpu, mlir, clone"),
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
        "For conv, -rand_side filter or -rand_side input; "
        "For conv_bwd_data, -rand_side filter or -rand_side output; "
        "For conv_bwd_weight, -rand_side input or -rand_side output. "
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
                 llvm::cl::value_desc("error"));

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

static llvm::cl::opt<bool> applyBufferizationPipeline(
    "apply-bufferization-pipeline",
    llvm::cl::desc("apply bufferization pipeline defined in rock dialect"),
    llvm::cl::init(true));

// TODO[split-K]: remove after integrating with MIGraphX
static llvm::cl::opt<bool> disableSplitKForTuning(
    "disable-split-k-for-tuning",
    llvm::cl::desc("disable split-K GEMM scheme for tuning"),
    llvm::cl::init(false));

enum class F8TypesChoice : int { Arch = 0, Nanoo = 1, OCP = 2 };

static llvm::cl::opt<F8TypesChoice> forceF8Types(
    "force-f8-types",
    llvm::cl::desc("use OCP F8 types;  otherwise, use old F8 types"),
    llvm::cl::values(clEnumValN(F8TypesChoice::Arch, "arch",
                                "usual F8 types for architecture"),
                     clEnumValN(F8TypesChoice::Nanoo, "nanoo",
                                "older 'NANOO' or 'FNUZ' types"),
                     clEnumValN(F8TypesChoice::Nanoo, "fnuz",
                                "older 'NANOO' or 'FNUZ' types"),
                     clEnumValN(F8TypesChoice::OCP, "ocp",
                                "'OCP' or 'OFP8' types")),
    llvm::cl::init(F8TypesChoice::Arch));

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

// This helper struct defines the argument ordering for
// quantized attention operator.
struct AttentionQuantizedArgIndex {
  static const size_t q = 0;
  static const size_t k = 1;
  static const size_t v = 2;
  static const size_t quantBias = 3;
  static const size_t quantScale = 4;
  static const size_t scale = 5;
  static const size_t bias = 6;
};

struct GenParams {
  std::optional<rock::KernelType> operation = std::nullopt;
  SmallVector<Type, 5> types;
  rock::GemmFeatures features = rock::GemmFeatures::none;
  std::optional<const rock::ConvGenerator::Config *> convConfig = std::nullopt;
  StringRef arch;
  StringRef perfConfig;
};

namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test

static void correctConvParameters() {
  std::string filterLayoutValue = filterLayout.getValue();

  // yxcgk not implement yet
  if (filterLayoutValue.find('g') == std::string::npos &&
      (filterLayoutValue.substr(0, 2) == "kc" ||
       (filterLayoutValue[0] == 'k' && filterLayoutValue.back() == 'c') ||
       filterLayoutValue.substr(filterLayoutValue.size() - 2) == "ck"))
    filterLayout = "g" + filterLayoutValue;

  auto addGToLayout = [&](std::string ch,
                          std::string &layoutValue) -> std::string {
    std::string layout;
    if (layoutValue.find('g') == std::string::npos) {
      if (layoutValue.substr(0, 2) == "n" + ch)
        layout = "ng" + ch + layoutValue.substr(2);
      else if (layoutValue[0] == 'n' && layoutValue.back() == ch[0])
        layout = layoutValue.substr(0, layoutValue.size() - 1) + "g" + ch;
      else
        layout = "g" + layoutValue;
    } else
      layout = layoutValue;
    return layout;
  };

  inputLayout = addGToLayout("c", inputLayout.getValue());
  outputLayout = addGToLayout("k", outputLayout.getValue());

  // +++pf:  update old key names.
  std::replace(filterLayout.getValue().begin(), filterLayout.getValue().end(),
               'y', '0');
  std::replace(filterLayout.getValue().begin(), filterLayout.getValue().end(),
               'x', '1');
  std::replace(inputLayout.getValue().begin(), inputLayout.getValue().end(),
               'h', '0');
  std::replace(inputLayout.getValue().begin(), inputLayout.getValue().end(),
               'w', '1');
  std::replace(outputLayout.getValue().begin(), outputLayout.getValue().end(),
               'h', '0');
  std::replace(outputLayout.getValue().begin(), outputLayout.getValue().end(),
               'w', '1');

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
  validatePadding(paddingDepth, paddingDepthLeft, paddingDepthRight,
                  "padding_d");

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
  // hi_minimum is the mininum number of input elements needed to correctly
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

  int di = inputDepth.getValue();
  int z = filterDepth.getValue();
  int in_left_pad_d = paddingDepthLeft.getValue();
  int in_right_pad_d = paddingDepthRight.getValue();
  int conv_stride_d = strideDepth.getValue();
  int conv_dilation_d = dilationDepth.getValue();
  int d_o = getOutputDim(di, z, in_left_pad_d, in_right_pad_d, conv_stride_d,
                         conv_dilation_d);

  int di_minimum = 1 + (z - 1) * conv_dilation_d + (d_o - 1) * conv_stride_d;
  int di_specified = di + in_left_pad_d + in_right_pad_d;
  // di_minimum is the miminum number of input elements needed to correctly
  // apply the filter in the d direction, which is a function of the stride and
  // dilation parameters. If the specified input height is less than this value,
  // add extra padding on the right to allow the convolution to execute
  // successfully.
  if (di_minimum > di_specified)
    paddingDepthRight = in_right_pad_d + (di_minimum - di_specified);
}

static void verifyConvLayout() {
  std::string filterLayoutValue = filterLayout.getValue();
  std::string inputLayoutValue = inputLayout.getValue();

  if (filterLayoutValue.find("yx") == std::string::npos &&
      filterLayoutValue.find("xy") == std::string::npos &&
      filterLayoutValue.find("01") == std::string::npos &&
      filterLayoutValue.find("10") == std::string::npos) {
    llvm::errs() << "Unsupported filter layout: disjointed yx!\n";
    exit(1);
  }

  if (inputLayoutValue.find("hw") == std::string::npos &&
      inputLayoutValue.find("wh") == std::string::npos &&
      inputLayoutValue.find("01") == std::string::npos &&
      inputLayoutValue.find("10") == std::string::npos) {

    llvm::errs() << "Unsupported input layout: disjointed hw!\n";
    exit(1);
  }
}

static void populateDefaults() {
  const bool isGemm = operation == rock::KernelType::Gemm;
  const bool isAttention = operation == rock::KernelType::Attention;
  const bool isConv = !(isGemm || isAttention);
  // Default f32 if we passed no `-t` arguments at all.
  if (outputDataType.empty()) {
    if (filterDataType != inputDataType) {
      llvm::errs() << "Missing output type for mixed input types\n";
      exit(1);
    }

    outputDataType = "f32";
  }
  if (populateDefaultValues) {
    if (isGemm) {
      groupSize = 1;
      gemmM = 1024;
      gemmK = 769;
      gemmN = 512;
    }
    if (isAttention) {
      groupSize = 1;
      sequenceLengthQ = 1024;
      sequenceLengthK = 1024;
      headDimQK = 32;
      headDimV = 32;
    }
    if (isConv) {
      if (mfmaFeature != FeatureToggle::on) {
        groupSize = 1;
        batchSize = 128;
        inputChannel = 8;
        outputChannel = 128;
        inputHeight = 32;
        inputWidth = 32;
        inputDepth = 1;
        filterHeight = 3;
        filterWidth = 3;
        filterDepth = 1;
        dilationHeight = 1;
        dilationWidth = 1;
        dilationDepth = 1;
        strideHeight = 1;
        strideWidth = 1;
        strideDepth = 1;
        paddingHeightLeft = 0;
        paddingHeightRight = 0;
        paddingWidthLeft = 0;
        paddingWidthRight = 0;
        paddingDepthLeft = 0;
        paddingDepthRight = 0;
      } else {
        groupSize = 1;
        batchSize = 128;
        inputChannel = 1024;
        outputChannel = 1024;
        inputHeight = 14;
        inputWidth = 14;
        inputDepth = 1;
        filterHeight = 1;
        filterWidth = 1;
        filterDepth = 1;
        dilationHeight = 1;
        dilationWidth = 1;
        dilationDepth = 1;
        strideHeight = 1;
        strideWidth = 1;
        strideDepth = 1;
        paddingHeightLeft = 0;
        paddingHeightRight = 0;
        paddingWidthLeft = 0;
        paddingWidthRight = 0;
        paddingDepthLeft = 0;
        paddingDepthRight = 0;
      }
    }
  }

  if (isConv && outputHeight.getNumOccurrences() == 0) {
    outputHeight = rock::ConvGenerator::outputDim(
        inputHeight.getValue(), filterHeight.getValue(),
        paddingHeightLeft.getValue(), paddingHeightRight.getValue(),
        strideHeight.getValue(), dilationHeight.getValue());
  }
  if (isConv && outputWidth.getNumOccurrences() == 0) {
    outputWidth = rock::ConvGenerator::outputDim(
        inputWidth.getValue(), filterWidth.getValue(),
        paddingWidthLeft.getValue(), paddingWidthRight.getValue(),
        strideWidth.getValue(), dilationWidth.getValue());
  }
  if (isConv && outputDepth.getNumOccurrences() == 0) {
    outputDepth = rock::ConvGenerator::outputDim(
        inputDepth.getValue(), filterDepth.getValue(),
        paddingDepthLeft.getValue(), paddingDepthRight.getValue(),
        strideDepth.getValue(), dilationDepth.getValue());
  }
}

auto getRequiredArgs(std::optional<rock::KernelType> kernelType) {
  using RequiredArgsType = std::vector<const llvm::cl::opt<int64_t> *>;
  switch (kernelType.value()) {
  case rock::KernelType::Gemm: {
    const static RequiredArgsType requiredGemmArgs = {&groupSize, &gemmM,
                                                      &gemmK, &gemmN};
    return requiredGemmArgs;
  }
  case rock::KernelType::Attention: {
    const static RequiredArgsType requiredAttenArgs = {
        &groupSize, &sequenceLengthQ, &sequenceLengthK, &headDimQK, &headDimV};
    return requiredAttenArgs;
  }
  default: {
    const static RequiredArgsType requiredConvArgs = {
        &groupSize,  &batchSize,     &inputChannel, &inputHeight,
        &inputWidth, &outputChannel, &filterWidth,  &filterHeight};
    return requiredConvArgs;
  }
  };
}

static LogicalResult detectMissingArguments() {
  const static auto requiredArgs = getRequiredArgs(operation);
  for (auto *arg : requiredArgs) {
    if (arg->getValue() <= 0) {
      llvm::errs() << "Value for: " << arg->ArgStr << " not specified\n";
      return failure();
    }
  }

  if (operation == rock::KernelType::Attention) {
    if (dataTypeAlias.getValue().empty()) {
      llvm::errs() << "Type of the Attention operation is not specified\n";
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
  auto oprType = cast<ShapedType>(var.getType());
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

llvm::SmallString<32> archChip() {
  RocmDeviceName targetInfo;
  if (failed(targetInfo.parse(arch.getValue()))) {
    llvm::errs() << "Invalid architecture name: " << arch << "\n";
    exit(1);
  }
  return targetInfo.getChip();
}

// Map data type string to MLIR type
static Type typeFromString(StringRef name, MLIRContext *ctx) {
  if (name == "fp8") {
    switch (forceF8Types) {
    case F8TypesChoice::Arch:
      // f8E4M3FN for navi4, f8E4M3FNUZ for everyone else
      if (archChip().substr(0, 5) == "gfx12")
        name = "f8E4M3FN";
      else
        name = "f8E4M3FNUZ";
      break;
    case F8TypesChoice::Nanoo:
      name = "f8E4M3FNUZ";
      break;
    case F8TypesChoice::OCP:
      name = "f8E4M3FN";
      break;
    }
  } else if (name == "bf8") {
    switch (forceF8Types) {
    case F8TypesChoice::Arch:
      // f8E5M2 for navi4, f8E5M2FNUZ for everyone else
      if (archChip().substr(0, 5) == "gfx12")
        name = "f8E5M2";
      else
        name = "f8E5M2FNUZ";
      break;
    case F8TypesChoice::Nanoo:
      name = "f8E5M2FNUZ";
      break;
    case F8TypesChoice::OCP:
      name = "f8E5M2";
      break;
    }
  }
  std::optional<Type> result =
      llvm::StringSwitch<std::optional<Type>>(name)
          .Case("f32", Float32Type::get(ctx))
          .Case("fp32", Float32Type::get(ctx))
          .Case("f16", Float16Type::get(ctx))
          .Case("fp16", Float16Type::get(ctx))
          .Case("bf16", BFloat16Type::get(ctx))
          .Case("i8", IntegerType::get(ctx, 8))
          .Case("i32", IntegerType::get(ctx, 32))
          .Case("f8E5M2", Float8E5M2Type::get(ctx))
          .Case("f8E4M3FN", Float8E4M3FNType::get(ctx))
          .Case("f8E5M2FNUZ", Float8E5M2FNUZType::get(ctx))
          .Case("f8E4M3FNUZ", Float8E4M3FNUZType::get(ctx))
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
  MemRefType flatType = cast<MemRefType>(toFillFlat.getType());
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
  auto flatType = cast<MemRefType>(toFillFlat.getType());

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

struct ConvTensorDimInfo {
  unsigned nonImg1Dim;
  int64_t nonImg1Len;
  unsigned nonImg2Dim;
  int64_t nonImg2Len;
  unsigned gDim;
  int64_t gLen;
  SmallVector<unsigned, 4> imageDims;
  SmallVector<int64_t, 4> imageLens;
};

/// Given the layout string for some tensor (ex ngc01 or gk012c), the tensor
/// shape of the value whose layout has that form, and the identifiers ('n',
/// 'c', or 'k') for the two non-image dimensions expected in the layout, return
/// the positions and lengths of those two non-image dimensions and the image
/// dimensions (in order).
static ConvTensorDimInfo parseConvTensorLayout(StringRef layout,
                                               ArrayRef<int64_t> shape,
                                               char nonImg1Sym,
                                               char nonImg2Sym) {
  // The two non-image dimensions and the group.
  unsigned nImageDims = shape.size() - 3;
  // Neither value is particularly special, excetpt that I used -2 because -1 is
  // the dynamic dimension indicator and we might need that later.
  SmallVector<unsigned, 4> imageDims(nImageDims, 0xdeadbeef);
  SmallVector<int64_t, 4> imageLens(nImageDims, -2);
  unsigned nonImg1Dim, nonImg2Dim, gDim = 0xdeadbeef;
  int64_t nonImg1Len, nonImg2Len, gLen = -2;

  for (auto [pos, dim, len] : llvm::enumerate(layout, shape)) {
    if (dim == nonImg1Sym) {
      nonImg1Dim = pos;
      nonImg1Len = len;
    } else if (dim == nonImg2Sym) {
      nonImg2Dim = pos;
      nonImg2Len = len;
    } else if (dim >= '0' && dim <= '9') {
      size_t dimIdx = dim - '0';
      if (dimIdx >= nImageDims) {
        llvm::errs() << "Dimension value '" << dimIdx << "' too large\n";
        exit(1);
      }
      imageDims[dimIdx] = pos;
      imageLens[dimIdx] = len;
    } else if (dim == 'g') {
      gDim = pos;
      gLen = len;
    } else {
      llvm::errs() << "Unknown layout key '" << dim << "'\n";
      exit(1);
    }
  }
  return ConvTensorDimInfo{nonImg1Dim, nonImg1Len, nonImg2Dim, nonImg2Len,
                           gDim,       gLen,       imageDims,  imageLens};
}

static SmallVector<Value> arrangeByConvLayout(const ConvTensorDimInfo &layout,
                                              Value nonImg1, Value nonImg2,
                                              Value g, ValueRange image) {
  SmallVector<Value> result;
  result.resize_for_overwrite(image.size() + 3);
  result[layout.nonImg1Dim] = nonImg1;
  result[layout.nonImg2Dim] = nonImg2;
  result[layout.gDim] = g;
  for (auto [idx, value] : llvm::zip(layout.imageDims, image))
    result[idx] = value;
  return result;
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
          assert(isa<FloatType>(dstElemType));
          newLoadOp =
              opBuilder.create<arith::SIToFPOp>(loc, dstElemType, loadOp);
        }
      } else {
        assert(isa<FloatType>(srcElemType));
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
  auto srcFlatType = cast<MemRefType>(srcFlat.getType());
  auto dstFlatType = cast<MemRefType>(dstFlat.getType());

  if (srcFlatType == dstFlatType) {
    b.create<memref::CopyOp>(loc, srcFlat, dstFlat);
  } else {
    auto memcpyFunc = getMemcpyFuncDecl(module, srcFlatType, dstFlatType);
    b.create<func::CallOp>(loc, memcpyFunc, ValueRange{srcFlat, dstFlat});
  }
}

// If the ref is float and not F32, make an F32 buffer and copy into it.
// Used when a CPU kernel will have parameters that it can't handle natively.
static Value ensureFloatIsF32(OpBuilder &b, Location loc, Value ref,
                              Type floatType) {
  auto refType = dyn_cast<MemRefType>(ref.getType());
  Type refElemType = refType.getElementType();
  if (!isa<FloatType>(refElemType) || refElemType.isF32())
    return ref;
  Value refFlat = makeNDMemRef(b, ref, 1);
  auto f32NewType = MemRefType::get(refType.getShape(), floatType);
  auto refNew = b.create<memref::AllocOp>(loc, f32NewType);
  Value refNewFlat = makeNDMemRef(b, refNew, 1);
  emitMemcpy(b, refFlat, refNewFlat);
  return refNew;
}

static AffineMap getLinearIndexingMap(OpBuilder &b, ArrayRef<int64_t> lengths) {
  size_t n = lengths.size();
  AffineExpr res = b.getAffineConstantExpr(0);
  for (auto [i, len] : llvm::enumerate(lengths)) {
    res = res * b.getAffineConstantExpr(len) + b.getAffineDimExpr(i);
  }
  return AffineMap::get(n, 0, res, b.getContext());
}

static void
createCPUConvWithMLIR(ModuleOp module, func::FuncOp func,
                      const rock::ConvGenerator::Config &genConfig) {
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
  auto resultType = dyn_cast<MemRefType>(resultTensor.getType());
  Type elemType = resultType.getElementType();
  SmallVector<float, 1> zeroPattern = {0.0};
  if (failed(
          populateTensorFillLogic(b, loc, zeroPattern, elemType, resultTensor)))
    llvm_unreachable("Tensor fill logic population shouldn't fail");

  // Create affine maps
  size_t nImageDims = genConfig.strideDims.size();
  SmallVector<AffineMap, 3> inputImageDimMaps;
  // Extra maps used for backward data.
  SmallVector<AffineMap> imageDimMaps2(nImageDims, AffineMap{});
  inputImageDimMaps.resize_for_overwrite(nImageDims);
  for (auto [stride, dilation, paddingLeft, map, map2] :
       llvm::zip(genConfig.strideDims, genConfig.dilationDims,
                 genConfig.paddingLeftDims, inputImageDimMaps, imageDimMaps2)) {
    switch (genConfig.operation.value()) {
    case rock::ConvOpType::Fwd:
    case rock::ConvOpType::BwdWeight:
      // d0 * stride + d1 * dilation - padding
      map = AffineMap::get(2, 0,
                           b.getAffineDimExpr(0) * stride +
                               b.getAffineDimExpr(1) * dilation - paddingLeft);
      break;
    case rock::ConvOpType::BwdData:
      // d0 + padding - d1 * dilation
      map = AffineMap::get(2, 0,
                           b.getAffineDimExpr(0) + paddingLeft -
                               b.getAffineDimExpr(1) * dilation);
      // d0 / stride
      map2 = AffineMap::get(1, 0, b.getAffineDimExpr(0).floorDiv(stride));
      break;
    }
  }

  ConvTensorDimInfo filterInfo = parseConvTensorLayout(
                        genConfig.filterLayout, genConfig.filterDimension, 'k',
                        'c'),
                    inputInfo = parseConvTensorLayout(genConfig.inputLayout,
                                                      genConfig.inputDimension,
                                                      'n', 'c'),
                    outputInfo = parseConvTensorLayout(
                        genConfig.outputLayout, genConfig.outputDimension, 'n',
                        'k');

  // Create constraints for boundary checks
  SmallVector<AffineExpr, 6> exprs;
  SmallVector<bool, 6> eqFlags;
  bool isBwdData = genConfig.operation.value() == rock::ConvOpType::BwdData;
  for (size_t i = 0; i < nImageDims; ++i) {
    size_t inputDIdx = i;
    if (isBwdData) {
      // out_D_tmp % stride_D == 0 for all D
      exprs.push_back(b.getAffineDimExpr(2 * i) % genConfig.strideDims[i]);
      eqFlags.push_back(true);
      inputDIdx = 2 * i + 1;
    }
    // input_D_idx >= 0, input_D_idx < input_D for all D
    exprs.push_back(b.getAffineDimExpr(inputDIdx));
    eqFlags.push_back(false);
    int64_t upperBound =
        isBwdData ? outputInfo.imageLens[i] : inputInfo.imageLens[i];
    exprs.push_back(upperBound - b.getAffineDimExpr(inputDIdx) - 1);
    eqFlags.push_back(false);
  }
  IntegerSet condition =
      IntegerSet::get(nImageDims * (isBwdData ? 2 : 1), 0, exprs, eqFlags);

  SmallVector<int64_t, 8> lowerBounds(2 * nImageDims + 4, 0);
  SmallVector<int64_t, 8> upperBounds;
  SmallVector<int64_t, 8> steps(2 * nImageDims + 4, 1);

  // Create the upper bounds
  switch (genConfig.operation.value()) {
  case rock::ConvOpType::Fwd:
    upperBounds.append(genConfig.outputDimension);
    upperBounds.push_back(filterInfo.nonImg2Len); // input channels 'c'
    upperBounds.append(filterInfo.imageLens);
    break;
  case rock::ConvOpType::BwdData:
    upperBounds.append(genConfig.inputDimension);
    upperBounds.push_back(filterInfo.nonImg1Len); // output channels 'k'
    upperBounds.append(filterInfo.imageLens);
    break;
  case rock::ConvOpType::BwdWeight:
    upperBounds.append(genConfig.filterDimension);
    upperBounds.push_back(outputInfo.nonImg1Len); // batch size 'n'
    upperBounds.append(outputInfo.imageLens);
    break;
  }

  Value opd1, opd2, result;
  AffineMap opd1Map, opd2Map, resultStoreMap;
  ConvTensorDimInfo resultInfo;

  switch (genConfig.operation.value()) {
  case rock::ConvOpType::Fwd:
    opd1 = block->getArgument(0);
    opd2 = block->getArgument(1);
    result = block->getArgument(2);
    opd1Map = getLinearIndexingMap(b, genConfig.filterDimension);
    opd2Map = getLinearIndexingMap(b, genConfig.inputDimension);
    resultStoreMap = getLinearIndexingMap(b, genConfig.outputDimension);
    resultInfo = outputInfo;
    break;
  case rock::ConvOpType::BwdWeight:
    opd1 = block->getArgument(2);
    opd2 = block->getArgument(1);
    result = block->getArgument(0);
    opd1Map = getLinearIndexingMap(b, genConfig.outputDimension);
    opd2Map = getLinearIndexingMap(b, genConfig.inputDimension);
    resultStoreMap = getLinearIndexingMap(b, genConfig.filterDimension);
    resultInfo = filterInfo;
    break;
  case rock::ConvOpType::BwdData:
    opd1 = block->getArgument(0);
    opd2 = block->getArgument(2);
    result = block->getArgument(1);
    opd1Map = getLinearIndexingMap(b, genConfig.filterDimension);
    opd2Map = getLinearIndexingMap(b, genConfig.outputDimension);
    resultStoreMap = getLinearIndexingMap(b, genConfig.inputDimension);
    resultInfo = inputInfo;
    break;
  }

  auto floatType = b.getF32Type();

  opd1 = ensureFloatIsF32(b, loc, opd1, floatType);
  opd2 = ensureFloatIsF32(b, loc, opd2, floatType);
  result = ensureFloatIsF32(b, loc, result, floatType);

  auto createConvLoopNest = [&](OpBuilder &b, Location loc, ValueRange ivs) {
    Value resultNonImg1 = ivs[resultInfo.nonImg1Dim];
    Value resultNonImg2 = ivs[resultInfo.nonImg2Dim];
    Value g = ivs[resultInfo.gDim];
    Value reductionNonImg = ivs[nImageDims + 3];
    // Drop result coordinates and the reduction channels.
    ValueRange reductionImage = ivs.drop_front(nImageDims + 3 + 1);
    SmallVector<Value> resultImage = llvm::map_to_vector(
        resultInfo.imageDims, [&](unsigned i) { return ivs[i]; });

    // Note: for backward data, this is the 'output' tensor
    SmallVector<Value> inputImageComputed;
    inputImageComputed.resize_for_overwrite(nImageDims);
    SmallVector<Value> conditionArgs;
    conditionArgs.reserve(isBwdData ? 2 * nImageDims : nImageDims);

    for (auto [resultD, reduceD, dMap, dMap2, applied] :
         llvm::zip(resultImage, reductionImage, inputImageDimMaps,
                   imageDimMaps2, inputImageComputed)) {
      switch (genConfig.operation.value()) {
      case rock::ConvOpType::Fwd:
        // in_D_idx = out_D_idx * stride_D + fil_D_idx * dilation_D -
        // padding_D_l;
        applied = b.create<affine::AffineApplyOp>(loc, dMap,
                                                  ValueRange{resultD, reduceD});
        break;
      case rock::ConvOpType::BwdData: {
        // out_D_tmp = in_D_idx + padding_D_l - fil_D_idx * dilation_D;
        Value tmpIdx = b.create<affine::AffineApplyOp>(
            loc, dMap, ValueRange{resultD, reduceD});
        conditionArgs.push_back(tmpIdx);
        // out_D_idx = out_D_tmp / stride_D;
        applied =
            b.create<affine::AffineApplyOp>(loc, dMap2, ValueRange{tmpIdx});
        break;
      }
      case rock::ConvOpType::BwdWeight:
        // in_D_idx = out_D_idx * stride_h + fil_D_idx * dilation_h -
        // padding_D_l;
        applied = b.create<affine::AffineApplyOp>(loc, dMap,
                                                  ValueRange{reduceD, resultD});
        break;
      }
      conditionArgs.push_back(applied);
    }

    affine::AffineIfOp ifOp =
        b.create<affine::AffineIfOp>(loc, condition, conditionArgs, false);
    auto thenBody = ifOp.getThenBodyBuilder();

    // Perform MAC operation
    SmallVector<Value> idx1, idx2;
    switch (genConfig.operation.value()) {
    case rock::ConvOpType::Fwd:
      // K, C, G, fil_h, fil_w, ...
      idx1 = arrangeByConvLayout(filterInfo, resultNonImg2, reductionNonImg, g,
                                 reductionImage);
      // N, C, G, in_h, in_w, ...
      idx2 = arrangeByConvLayout(inputInfo, resultNonImg1, reductionNonImg, g,
                                 inputImageComputed);
      break;
    case rock::ConvOpType::BwdWeight:
      // N, K, G, out_h, out_w, ...
      idx1 = arrangeByConvLayout(outputInfo, reductionNonImg, resultNonImg1, g,
                                 reductionImage);
      // N, C, G, in_h, in_w, ...
      idx2 = arrangeByConvLayout(inputInfo, reductionNonImg, resultNonImg2, g,
                                 inputImageComputed);
      break;
    case rock::ConvOpType::BwdData:
      // K, C, G, fil_h, fil_w, ...
      idx1 = arrangeByConvLayout(filterInfo, reductionNonImg, resultNonImg2, g,
                                 reductionImage);
      // N, K, G, out_h (stored as in_h), out_w (stored as in_w), ...
      idx2 = arrangeByConvLayout(outputInfo, resultNonImg1, reductionNonImg, g,
                                 inputImageComputed);
      break;
    }

    auto loadOp1 =
        thenBody.create<affine::AffineLoadOp>(loc, opd1, opd1Map, idx1);
    auto loadOp2 =
        thenBody.create<affine::AffineLoadOp>(loc, opd2, opd2Map, idx2);
    size_t nIVs = genConfig.inputDimension.size();
    auto loadOutput = thenBody.create<affine::AffineLoadOp>(
        loc, result, resultStoreMap, ivs.take_front(nIVs));
    if (elemType.isIntOrIndex()) {
      auto muliOp = thenBody.create<arith::MulIOp>(loc, loadOp1, loadOp2);
      auto extsiOp = thenBody.create<arith::ExtSIOp>(loc, elemType, muliOp);
      auto addiOp = thenBody.create<arith::AddIOp>(loc, loadOutput, extsiOp);
      thenBody.create<affine::AffineStoreOp>(
          loc, addiOp, result, resultStoreMap, ivs.take_front(nIVs));
    } else {
      auto mulfOp = thenBody.create<arith::MulFOp>(loc, loadOp1, loadOp2);
      auto addfOp = thenBody.create<arith::AddFOp>(loc, loadOutput, mulfOp);
      thenBody.create<affine::AffineStoreOp>(
          loc, addfOp, result, resultStoreMap, ivs.take_front(nIVs));
    }
  };

  // Generate the loop nest
  affine::buildAffineLoopNest(b, loc, lowerBounds, upperBounds, steps,
                              createConvLoopNest);

  if (!isa<BlockArgument>(opd1))
    b.create<memref::DeallocOp>(loc, opd1);
  if (!isa<BlockArgument>(opd2))
    b.create<memref::DeallocOp>(loc, opd2);
  if (!isa<BlockArgument>(result)) {
    BlockArgument resultBlockArg;
    switch (genConfig.operation.value()) {
    case rock::ConvOpType::Fwd:
      resultBlockArg = block->getArgument(2);
      break;
    case rock::ConvOpType::BwdWeight:
      resultBlockArg = block->getArgument(0);
      break;
    case rock::ConvOpType::BwdData:
      resultBlockArg = block->getArgument(1);
      break;
    }

    Value resultFlat = makeNDMemRef(b, result, 1);
    emitMemcpy(b, resultFlat, resultBlockArg);
    b.create<memref::DeallocOp>(loc, result);
  }

  b.create<func::ReturnOp>(loc, ValueRange{});
}

static func::FuncOp
createCPUConvFunc(ModuleOp module,
                  const rock::ConvGenerator::Config &genConfig) {
  assert(genConfig.operation.has_value());
  std::string funcName =
      rock::getNameForConvOpType(genConfig.operation.value()).str();

  funcName += "_cpu";
  func::FuncOp func = module.lookupSymbol<func::FuncOp>(funcName);
  if (func) // already exists
    return func;

  OpBuilder b(module.getContext());
  auto loc = b.getUnknownLoc();

  Type elemType =
      typeFromString(genConfig.inputDataTypeStr, module.getContext());
  Type outputElemType =
      typeFromString(genConfig.outputDataTypeStr, module.getContext());

  if (genConfig.inputDataTypeStr == "i8") {
    elemType = b.getI8Type();
    // Compute the output in int64_t to detect overflow
    outputElemType = b.getIntegerType(64);
    assert(genConfig.operation.value() == rock::ConvOpType::Fwd);
  }

  int64_t filterElems = computeProduct(genConfig.filterDimension);
  int64_t inputElems = computeProduct(genConfig.inputDimension);
  int64_t outputElems = computeProduct(genConfig.outputDimension);

  auto filterType = MemRefType::get(filterElems, elemType);
  auto inputType = MemRefType::get(inputElems, elemType);
  auto outputType = MemRefType::get(outputElems, outputElemType);

  // Create conv_host function
  rock::ConvGenerator convGenerator(genConfig);

  bool hasWorkspace = false;
  if (failed(convGenerator.hasWorkspace(b, hasWorkspace))) {
    assert(genConfig.operation.value() == rock::ConvOpType::Fwd);
  }
  Type workspaceArgType;
  if (hasWorkspace) {
    workspaceArgType = MemRefType::get(filterElems, b.getF32Type());
  }
  SmallVector<Type, 4> funcArgTypes = {filterType, inputType, outputType};
  if (hasWorkspace) {
    funcArgTypes.push_back(workspaceArgType);
  }

  func =
      func::FuncOp::create(loc, funcName, b.getFunctionType(funcArgTypes, {}));
  module.push_back(func);

  createCPUConvWithMLIR(module, func, genConfig);
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
  SmallVector<Type, 3> flatTypes =
      llvm::map_to_vector(argTypes, rock::getFlattenedType);
  auto func =
      b.create<func::FuncOp>(loc, isVerifier ? kernelNameVerifier : kernelName,
                             b.getFunctionType(flatTypes, {}), funcAttrs);
  if (reverse_grid) {
    func->setAttr(rock::ReverseGridAttrAttr::getMnemonic(), b.getUnitAttr());
  }

  constexpr StringLiteral gName = "g", mName = "m", kName = "k", nName = "n";
  SmallVector<SmallVector<StringRef>> allArgNames;
  allArgNames.emplace_back(SmallVector<StringRef>{
      gName, transposeA ? kName : mName, transposeA ? mName : kName});
  allArgNames.emplace_back(SmallVector<StringRef>{
      gName, transposeB ? nName : kName, transposeB ? kName : nName});
  allArgNames.emplace_back(SmallVector<StringRef>{
      gName, transposeC ? nName : mName, transposeC ? mName : nName});

  Block *block = func.addEntryBlock();
  b.setInsertionPointToStart(block);
  SmallVector<Value, 3> expandedArgs;
  rock::expandFlatFunctionArguments(b, func, allArgNames, argTypes,
                                    expandedArgs);

  Value aVal = expandedArgs[0], bVal = expandedArgs[1], cVal = expandedArgs[2];

  IntegerAttr numCUAttr =
      (num_cu.getNumOccurrences() > 0 ? b.getI32IntegerAttr(num_cu) : nullptr);
  auto gemm = b.create<rock::GemmOp>(
      loc, /*resultTypes=*/TypeRange{}, aVal, bVal, cVal, transposeA,
      transposeB, transposeC, archAttr.getValue(), numCUAttr, params.features,
      storeMethod,
      /*blockSize=*/nullptr, /*gridSize=*/nullptr, /*params=*/nullptr);

  if (!params.perfConfig.empty())
    gemm->setAttr("perf_config", b.getStringAttr(params.perfConfig));

  b.create<func::ReturnOp>(loc);

  // TODO[split-K]: remove after integrating split-K into MIGraphX
  if (!disableSplitKForTuning)
    func->setAttr(rock::EnableSplitKForTuningAttr::getMnemonic(),
                  b.getUnitAttr());

  module.push_back(func);
  return func;
}

static void getAttentionTypes(SmallVectorImpl<Type> &result,
                              ArrayRef<Type> elemTypes) {
  SmallVector<int64_t> qDims{groupSize, sequenceLengthQ, headDimQK};
  SmallVector<int64_t> transposedQDims{groupSize, headDimQK, sequenceLengthQ};
  SmallVector<int64_t> kDims{groupSize, sequenceLengthK, headDimQK};
  SmallVector<int64_t> transposedKDims{groupSize, headDimQK, sequenceLengthK};
  SmallVector<int64_t> vDims{groupSize, sequenceLengthK, headDimV};
  SmallVector<int64_t> transposedVDims{groupSize, headDimV, sequenceLengthK};
  SmallVector<int64_t> oDims{groupSize, sequenceLengthQ, headDimV};
  SmallVector<int64_t> transposedODims{groupSize, headDimV, sequenceLengthQ};
  bool isQuantized =
      elemTypes[0] == IntegerType::get(elemTypes[0].getContext(), 8);

  MemRefType qType = MemRefType::get(transposeQ ? transposedQDims : qDims,
                                     elemTypes[0]),
             kType = MemRefType::get(transposeK ? kDims : transposedKDims,
                                     elemTypes[1]),
             vType = MemRefType::get(transposeV ? transposedVDims : vDims,
                                     elemTypes[2]);

  result.push_back(qType);
  result.push_back(kType);
  result.push_back(vType);
  unsigned optionalArgsCounter{3};
  if (isQuantized) {
    // quant bias is to be broadcasted
    SmallVector<int64_t> quantBiasDims{1, 1, 1};
    MemRefType qbType =
        MemRefType::get(quantBiasDims, elemTypes[optionalArgsCounter++]);
    result.push_back(qbType);
    // quant scale is to be broadcasted
    SmallVector<int64_t> quantScaleDims{1, 1, 1};
    MemRefType qsType =
        MemRefType::get(quantScaleDims, elemTypes[optionalArgsCounter++]);
    result.push_back(qsType);
  }
  if (hasAttnScale) {
    SmallVector<int64_t> scaleDims{groupSize, sequenceLengthQ, sequenceLengthK};
    MemRefType sType =
        MemRefType::get(scaleDims, elemTypes[optionalArgsCounter++]);
    result.push_back(sType);
  }
  if (hasAttnBias) {
    SmallVector<int64_t> biasDims{groupSize, sequenceLengthQ, sequenceLengthK};
    MemRefType bType =
        MemRefType::get(biasDims, elemTypes[optionalArgsCounter++]);
    result.push_back(bType);
  }
  MemRefType outType =
      MemRefType::get(transposeO ? transposedODims : oDims, elemTypes.back());
  result.push_back(outType);
}

static void
getAttentionDimNames(SmallVectorImpl<SmallVector<StringRef>> &result,
                     ArrayRef<Type> elementTypes) {
  result.reserve(elementTypes.size());
  constexpr StringLiteral gName = "g", seqQName = "seq_q", seqKName = "seq_k",
                          headQKName = "head_qk", headVName = "head_v";
  if (transposeQ)
    result.emplace_back(SmallVector<StringRef>{gName, headQKName, seqQName});
  else
    result.emplace_back(SmallVector<StringRef>{gName, seqQName, headQKName});
  if (transposeK)
    result.emplace_back(SmallVector<StringRef>{gName, headQKName, seqKName});
  else
    result.emplace_back(SmallVector<StringRef>{gName, seqKName, headQKName});
  if (transposeV)
    result.emplace_back(SmallVector<StringRef>{gName, headVName, seqKName});
  else
    result.emplace_back(SmallVector<StringRef>{gName, seqKName, headVName});
  bool isQuantized = elementTypes[0].isInteger(8);
  if (isQuantized) {
    result.emplace_back(SmallVector<StringRef>{gName, seqQName, seqKName});
    result.emplace_back(SmallVector<StringRef>{gName, seqQName, seqKName});
  }
  if (hasAttnScale)
    result.emplace_back(SmallVector<StringRef>{gName, seqQName, seqKName});
  if (hasAttnBias)
    result.emplace_back(SmallVector<StringRef>{gName, seqQName, seqKName});

  if (transposeO)
    result.emplace_back(SmallVector<StringRef>{gName, headVName, seqQName});
  else
    result.emplace_back(SmallVector<StringRef>{gName, seqQName, headVName});
}

template <typename TosaOp, typename... Args>
static TosaOp createOpAndInfer(OpBuilder &builder, Location loc, Type elemType,
                               Args &&...args) {
  auto op =
      builder.create<TosaOp>(loc, UnrankedTensorType::get(elemType), args...);
  InferShapedTypeOpInterface shapeInterface =
      cast<InferShapedTypeOpInterface>(op.getOperation());
  SmallVector<ShapedTypeComponents> returnShape;
  LogicalResult shapeInferenceStatus = shapeInterface.inferReturnTypeComponents(
      op.getContext(), op.getLoc(), op->getOperands(), op->getAttrDictionary(),
      op->getPropertiesStorage(), op->getRegions(), returnShape);
  assert(shapeInferenceStatus.succeeded());
  Type newOutTy = RankedTensorType::get({returnShape[0].getDims()}, elemType);
  auto result = op->getResult(0);
  result.setType(newOutTy);
  return op;
}

Value addTensorArgToBlock(OpBuilder &builder, Location loc,
                          Block *preSoftmaxElemwiseBlock, Value funcArg) {
  ShapedType funcArgType = cast<ShapedType>(funcArg.getType());
  Value funcArgMemRef = preSoftmaxElemwiseBlock->addArgument(
      MemRefType::get(funcArgType.getShape(), funcArgType.getElementType()),
      loc);
  Value funcArgTensor = rock::getAsTensor(builder, loc, funcArgMemRef);
  return funcArgTensor;
}

static func::FuncOp createGpuAttentionKernel(ModuleOp module,
                                             const GenParams &params) {
  MLIRContext *ctx = module.getContext();
  Location loc = module->getLoc();
  OpBuilder builder(ctx);

  // Set mhal.arch on module to make compilation pipeline work
  StringAttr archAttr = builder.getStringAttr(params.arch);
  if (!module->hasAttr("mhal.arch"))
    module->setAttr("mhal.arch", archAttr);

  SmallVector<Type, 5> argTypes;
  getAttentionTypes(argTypes, params.types);
  bool isQuantized = params.types[0] == IntegerType::get(ctx, 8);
  SmallVector<Type, 5> flatArgTypes =
      llvm::map_to_vector(argTypes, rock::getFlattenedType);

  SmallVector<NamedAttribute, 2> funcAttrs = {
      builder.getNamedAttr("kernel", builder.getUnitAttr()),
      builder.getNamedAttr("mhal.arch", archAttr)};

  constexpr StringLiteral kernelName("rock_attention");
  auto func = builder.create<func::FuncOp>(
      loc, kernelName, builder.getFunctionType(flatArgTypes, {}), funcAttrs);
  if (reverse_grid) {
    func->setAttr(rock::ReverseGridAttrAttr::getMnemonic(),
                  builder.getUnitAttr());
  }

  Block *block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);

  SmallVector<Value> unflattenedArgs;
  SmallVector<SmallVector<StringRef>> allNames;
  getAttentionDimNames(allNames, params.types);
  rock::expandFlatFunctionArguments(builder, func, allNames, argTypes,
                                    unflattenedArgs);

  Value queries = unflattenedArgs[0];
  Value keys = unflattenedArgs[1];
  Value values = unflattenedArgs[2];

  Value quantBias;
  Value quantScale;
  Value scale;
  Value bias;
  Value output;

  SmallVector<Value> elemwiseInputs;
  unsigned optionalArgsCounter = 3;
  if (isQuantized) {
    quantBias = unflattenedArgs[optionalArgsCounter++];
    elemwiseInputs.push_back(quantBias);
    quantScale = unflattenedArgs[optionalArgsCounter++];
    elemwiseInputs.push_back(quantScale);
  }
  if (hasAttnScale) {
    scale = unflattenedArgs[optionalArgsCounter++];
    elemwiseInputs.push_back(scale);
  }
  if (hasAttnBias) {
    bias = unflattenedArgs[optionalArgsCounter++];
    elemwiseInputs.push_back(bias);
  }
  output = unflattenedArgs[optionalArgsCounter];

  IntegerAttr numCUAttr =
      (num_cu.getNumOccurrences() > 0 ? builder.getI32IntegerAttr(num_cu)
                                      : nullptr);
  auto attention = builder.create<rock::AttentionOp>(
      loc, TypeRange{}, queries, keys, values, elemwiseInputs, output,
      transposeQ, transposeK, transposeV, transposeO, archAttr, params.features,
      numCUAttr,
      /*params0=*/nullptr, /*params1=*/nullptr, /*firstGemmIdx=*/0);
  {
    Block *preSoftmaxElemwiseBlock =
        &attention.getPreSoftmaxBody().emplaceBlock();
    PatternRewriter::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(preSoftmaxElemwiseBlock);
    ShapedType qType = cast<ShapedType>(queries.getType());
    ArrayRef<int64_t> qShape = qType.getShape();
    Type qkElemType = qType.getElementType();
    if (isQuantized) {
      qkElemType = IntegerType::get(ctx, 32);
    }
    MemRefType qkMemRefType = MemRefType::get(
        {qShape[0], sequenceLengthQ, sequenceLengthK}, qkElemType);
    Value qkMemRef = preSoftmaxElemwiseBlock->addArgument(qkMemRefType, loc);
    Value qkTensor = rock::getAsTensor(builder, loc, qkMemRef);
    if (isQuantized) {
      Value quantBiasI8 =
          addTensorArgToBlock(builder, loc, preSoftmaxElemwiseBlock, quantBias);
      Value quantScaleF16 = addTensorArgToBlock(
          builder, loc, preSoftmaxElemwiseBlock, quantScale);
      Value quantBiasI32 = createOpAndInfer<tosa::CastOp>(
          builder, loc, IntegerType::get(ctx, 32), quantBiasI8);
      qkTensor = createOpAndInfer<tosa::SubOp>(
          builder, loc, IntegerType::get(ctx, 32), qkTensor, quantBiasI32);
      qkTensor = createOpAndInfer<tosa::CastOp>(
          builder, loc, Float16Type::get(ctx), qkTensor);
      qkTensor =
          createOpAndInfer<tosa::MulOp>(builder, loc, Float16Type::get(ctx),
                                        qkTensor, quantScaleF16, /*shift=*/0);
    }
    if (hasAttnScale) {
      Value scaleTensor =
          addTensorArgToBlock(builder, loc, preSoftmaxElemwiseBlock, scale);
      qkTensor = createOpAndInfer<tosa::MulOp>(
          builder, loc,
          cast<ShapedType>(scaleTensor.getType()).getElementType(), qkTensor,
          scaleTensor, /*shift=*/0);
    }
    if (hasAttnBias) {
      Value biasTensor =
          addTensorArgToBlock(builder, loc, preSoftmaxElemwiseBlock, bias);
      qkTensor = createOpAndInfer<tosa::AddOp>(
          builder, loc, cast<ShapedType>(biasTensor.getType()).getElementType(),
          qkTensor, biasTensor);
    }
    MemRefType resMemRefType =
        MemRefType::get({qShape[0], sequenceLengthQ, sequenceLengthK},
                        cast<ShapedType>(qkTensor.getType()).getElementType());
    Value resMemref =
        builder.create<bufferization::ToMemrefOp>(loc, resMemRefType, qkTensor);
    Value outMemref = preSoftmaxElemwiseBlock->addArgument(resMemRefType, loc);
    builder.create<memref::CopyOp>(loc, resMemref, outMemref);
    builder.create<rock::YieldOp>(loc);
  }

  if (!params.perfConfig.empty())
    attention->setAttr("perf_config", builder.getStringAttr(params.perfConfig));

  builder.create<func::ReturnOp>(loc);
  module.push_back(func);
  return func;
}

static func::FuncOp createCpuGemmKernelWithMlir(ModuleOp module,
                                                const GenParams &params) {
  MLIRContext *ctx = module.getContext();
  OpBuilder b(ctx);
  Location loc = module->getLoc();

  auto cpuTypes = params.types;
  SmallVector<Type, 3> argTypes;
  getGemmTypes(cpuTypes, argTypes, /*isCpuVerifier=*/true);
  SmallVector<Type> flatArgTypes =
      llvm::map_to_vector(argTypes, rock::getFlattenedType);

  constexpr llvm::StringLiteral cpuKernName("host_naive_gemm");
  auto func = b.create<func::FuncOp>(loc, cpuKernName,
                                     b.getFunctionType(flatArgTypes, {}));
  module.push_back(func);

  Block *block = func.addEntryBlock();
  b.setInsertionPointToStart(block);

  Value aVal = block->getArgument(0), bVal = block->getArgument(1),
        cVal = block->getArgument(2);

  auto floatType = b.getF32Type();

  aVal = ensureFloatIsF32(b, loc, aVal, floatType);
  bVal = ensureFloatIsF32(b, loc, bVal, floatType);
  cVal = ensureFloatIsF32(b, loc, cVal, floatType);

  auto cType = cast<MemRefType>(cVal.getType());
  Value zeroOut = rock::createZeroConstantOp(b, loc, cType.getElementType());

  b.create<linalg::FillOp>(loc, zeroOut, cVal);

  auto expandArg = [&loc, &b](Value arg, Type rawLogicalType) -> Value {
    auto logicalType = cast<MemRefType>(rawLogicalType);
    // Replicate the effect of ensureFloatIsF32()
    if (isa<FloatType>(logicalType.getElementType()))
      logicalType = cast<MemRefType>(
          logicalType.clone(Float32Type::get(arg.getContext())));
    ArrayRef<int64_t> logicalShape = logicalType.getShape();
    ReassociationIndices allDims = llvm::to_vector(
        llvm::iota_range<int64_t>(0, logicalShape.size(), false));
    return b.create<memref::ExpandShapeOp>(loc, logicalType, arg, allDims);
  };
  AffineExpr g = b.getAffineDimExpr(0), m = b.getAffineDimExpr(1),
             n = b.getAffineDimExpr(2), k = b.getAffineDimExpr(3);
  AffineMap aMap = AffineMap::get(
                4, 0, {g, transposeA ? k : m, transposeA ? m : k}, ctx),
            bMap = AffineMap::get(
                4, 0, {g, transposeB ? n : k, transposeB ? k : n}, ctx),
            cMap = AffineMap::get(
                4, 0, {g, transposeC ? n : m, transposeC ? m : n}, ctx);
  Value aExpVal = expandArg(aVal, argTypes[0]),
        bExpVal = expandArg(bVal, argTypes[1]),
        cExpVal = expandArg(cVal, argTypes[2]);
  b.create<linalg::GenericOp>(
      loc, ValueRange{aExpVal, bExpVal}, ValueRange{cExpVal},
      ArrayRef<AffineMap>{aMap, bMap, cMap},
      ArrayRef<utils::IteratorType>{
          utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::parallel, utils::IteratorType::reduction},
      /*doc=*/"", /*library_call=*/"",
      [](OpBuilder &builder, Location loc, ValueRange elems) {
        Value a = elems[0], b = elems[1], c = elems[2];
        Type cType = c.getType();
        if (isa<IntegerType>(cType)) {
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

  if (!isa<BlockArgument>(aVal))
    b.create<memref::DeallocOp>(loc, aVal);
  if (!isa<BlockArgument>(bVal))
    b.create<memref::DeallocOp>(loc, bVal);
  if (!isa<BlockArgument>(cVal)) {
    BlockArgument resultBlockArg = block->getArgument(2);
    Value resultFlat = makeNDMemRef(b, cVal, 1);
    emitMemcpy(b, resultFlat, resultBlockArg);
    b.create<memref::DeallocOp>(loc, cVal);
  }

  b.create<func::ReturnOp>(loc);
  return func;
}

static Value transposeMatrix(OpBuilder &builder, Location loc, Value src,
                             ArrayRef<int64_t> perm) {
  auto elemType = cast<RankedTensorType>(src.getType()).getElementType();
  auto permutationAttr = DenseIntElementsAttr::get(
      RankedTensorType::get({(int64_t)perm.size()}, builder.getI64Type()),
      perm);
  Value permutationValue =
      builder.create<arith::ConstantOp>(loc, permutationAttr);
  return createOpAndInfer<tosa::TransposeOp>(builder, loc, elemType, src,
                                             permutationValue);
}

static func::FuncOp createCpuAttentionKernelWithMlir(ModuleOp module,
                                                     const GenParams &params) {
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);
  Location loc = module->getLoc();

  bool isQuantized = params.types[0] == IntegerType::get(ctx, 8);
  SmallVector<Type, 5> argTypes;
  getAttentionTypes(argTypes, params.types);
  SmallVector<Type, 5> flatArgTypes =
      llvm::map_to_vector(argTypes, rock::getFlattenedType);

  constexpr llvm::StringLiteral cpuKernName("host_naive_attention");
  auto func = builder.create<func::FuncOp>(
      loc, cpuKernName, builder.getFunctionType(flatArgTypes, {}));

  Block *block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);

  auto getTensorForBlockArg = [&builder, &loc, &block,
                               &argTypes](unsigned blockArgIndex,
                                          bool isWritable = false) {
    constexpr bool isRestrict{true};
    Value flatTensor = builder.create<bufferization::ToTensorOp>(
        loc, block->getArgument(blockArgIndex), isRestrict, isWritable);
    ArrayRef<int64_t> origShape =
        cast<ShapedType>(argTypes[blockArgIndex]).getShape();

    Value reshapedTensor;
    if (origShape.size() == 2) {
      SmallVector<int64_t, 3> expShape(origShape.size() + 1, 0);
      expShape[0] = 1;
      llvm::copy(origShape, expShape.begin() + 1);
      reshapedTensor =
          builder.create<tosa::ReshapeOp>(loc, flatTensor, expShape);
    } else {
      reshapedTensor =
          builder.create<tosa::ReshapeOp>(loc, flatTensor, origShape);
    }
    return reshapedTensor;
  };

  auto queriesTensor = getTensorForBlockArg(0);
  if (transposeQ) {
    queriesTensor = transposeMatrix(builder, loc, queriesTensor, {0, 2, 1});
  }
  auto keysTensor = getTensorForBlockArg(1);
  if (transposeK) {
    keysTensor = transposeMatrix(builder, loc, keysTensor, {0, 2, 1});
  }
  auto valuesTensor = getTensorForBlockArg(2);
  if (transposeV) {
    valuesTensor = transposeMatrix(builder, loc, valuesTensor, {0, 2, 1});
  }
  Type firstGemmOutElemType = params.types[0];
  if (isQuantized) {
    firstGemmOutElemType = IntegerType::get(ctx, 32);
  }
  Value qkTensor = createOpAndInfer<tosa::MatMulOp>(
      builder, loc, firstGemmOutElemType, queriesTensor, keysTensor);
  unsigned optionalArgsCounter = 3;
  if (isQuantized) {
    auto quantBiasI8 = getTensorForBlockArg(optionalArgsCounter++);
    Value quantBiasI32 = createOpAndInfer<tosa::CastOp>(
        builder, loc, IntegerType::get(ctx, 32), quantBiasI8);
    qkTensor = createOpAndInfer<tosa::SubOp>(
        builder, loc, IntegerType::get(ctx, 32), qkTensor, quantBiasI32);
    qkTensor = createOpAndInfer<tosa::CastOp>(builder, loc,
                                              Float16Type::get(ctx), qkTensor);
    auto quantScaleF16 = getTensorForBlockArg(optionalArgsCounter++);
    qkTensor =
        createOpAndInfer<tosa::MulOp>(builder, loc, Float16Type::get(ctx),
                                      qkTensor, quantScaleF16, /*shift=*/0);
  }
  if (hasAttnScale) {
    auto scaleTensor = getTensorForBlockArg(optionalArgsCounter++);
    qkTensor = createOpAndInfer<tosa::MulOp>(
        builder, loc, cast<ShapedType>(scaleTensor.getType()).getElementType(),
        qkTensor, scaleTensor, /*shift=*/0);
  }

  if (hasAttnBias) {
    auto biasTensor = getTensorForBlockArg(optionalArgsCounter++);
    qkTensor = createOpAndInfer<tosa::AddOp>(
        builder, loc, cast<ShapedType>(biasTensor.getType()).getElementType(),
        qkTensor, biasTensor);
  }

  constexpr int64_t reductionAxis = 2;
  auto qkMaxs = createOpAndInfer<tosa::ReduceMaxOp>(
      builder, loc, cast<ShapedType>(qkTensor.getType()).getElementType(),
      qkTensor, reductionAxis);
  auto normilizedQkTensor = createOpAndInfer<tosa::SubOp>(
      builder, loc, cast<ShapedType>(qkTensor.getType()).getElementType(),
      qkTensor, qkMaxs);
  auto expsTensor = createOpAndInfer<tosa::ExpOp>(
      builder, loc,
      cast<ShapedType>(normilizedQkTensor.getType()).getElementType(),
      normilizedQkTensor);
  auto expsSums = createOpAndInfer<tosa::ReduceSumOp>(
      builder, loc, cast<ShapedType>(expsTensor.getType()).getElementType(),
      expsTensor, reductionAxis);
  auto invExpsSums = createOpAndInfer<tosa::ReciprocalOp>(
      builder, loc, cast<ShapedType>(expsSums.getType()).getElementType(),
      expsSums);
  Value softmaxTensor = createOpAndInfer<tosa::MulOp>(
      builder, loc, cast<ShapedType>(expsSums.getType()).getElementType(),
      expsTensor, invExpsSums, /*shift=*/0);
#ifdef ROCK_DEBUG_ATTENTION_REMOVE_SOFTMAX
  softmaxTensor = qkTensor;
#endif
  Value resultTensor = createOpAndInfer<tosa::MatMulOp>(
      builder, loc, cast<ShapedType>(softmaxTensor.getType()).getElementType(),
      softmaxTensor, valuesTensor);
  if (transposeO) {
    resultTensor = transposeMatrix(builder, loc, resultTensor, {0, 2, 1});
  }

  Value output = block->getArguments().back();
  auto outputType = cast<MemRefType>(output.getType());
  auto flatResultTensor =
      builder.create<tosa::ReshapeOp>(loc, resultTensor, outputType.getShape());

  auto flatResultMemref = builder.create<bufferization::ToMemrefOp>(
      loc, outputType, flatResultTensor);

  builder.create<memref::CopyOp>(loc, flatResultMemref, output);

  builder.create<func::ReturnOp>(loc);
  module.push_back(func);
  return func;
}

static void emitPrintTensor(OpBuilder &b, Value var) {
  auto loc = b.getUnknownLoc();
  auto varType = dyn_cast<MemRefType>(var.getType());
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
  auto valFlatType = cast<MemRefType>(valFlat.getType());
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
  if (isa<FloatType>(valElemType)) {
    // f16, bf16, fp8, bf8 will be converted to f32 by wrapper.
    verifyFuncName += "Float";
  } else if (valElemType.isInteger(8) || valElemType.isInteger(32) ||
             valElemType.isInteger(64)) {
    verifyFuncName +=
        "Int" + std::to_string(testElemType.getIntOrFloatBitWidth()) + "Int" +
        std::to_string(valElemType.getIntOrFloatBitWidth());
  } else {
    llvm_unreachable("There's a valElemType not accounted for");
  }

  auto mr1DUnkTestType =
      MemRefType::get({mlir::ShapedType::kDynamic}, testElemType);
  auto mr1DUnkValType =
      MemRefType::get({mlir::ShapedType::kDynamic}, valElemType);
  auto mr1DUnkF32Type =
      MemRefType::get({mlir::ShapedType::kDynamic}, floatType);

  bool isTestAndValSameType =
      (testElemType.isIntOrIndex() || testElemType.isF32());

  Value testResult, valResult;       // Values passed to the verify function
  Value testResultNew, valResultNew; // Values used for type conversion
  if (!isTestAndValSameType) {
    // When gpu kernel output data type = f16 | bf16, type conversions
    // are required before calling the verify function

    // Cast test result to the same type as valid result

    // clang-format off
    // %0 = memref.alloc() : memref<802816xf32>
    // call @_memcpy_f16_f32_802816(%test_flat, %0) : (memref<802816xf16>, memref<802816xf32>) -> ()
    // %5 = memref.cast %0 : memref<802816xf32> to memref<?x?x?x?x?xf32>
    // clang-format on

    auto f32FlatType = MemRefType::get(valFlatType.getShape(), floatType);
    testResultNew = b.create<memref::AllocOp>(loc, f32FlatType);
    emitMemcpy(b, testFlat, testResultNew);
    testResult = b.create<memref::CastOp>(loc, mr1DUnkF32Type, testResultNew);
    mr1DUnkTestType = mr1DUnkF32Type;
    if (!valElemType.isF32()) {
      valResultNew = b.create<memref::AllocOp>(loc, f32FlatType);
      emitMemcpy(b, valFlat, valResultNew);
      valResult = b.create<memref::CastOp>(loc, mr1DUnkF32Type, valResultNew);
      mr1DUnkValType = mr1DUnkF32Type;
    } else {
      valResult = b.create<memref::CastOp>(loc, mr1DUnkValType, valFlat);
    }

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
            Value valOrig = b.create<affine::AffineLoadOp>(loc, valFlat, ivs);
            Value valTruncated =
                b.create<arith::TruncFOp>(loc, testElemType, valOrig);
            Value valExt =
                b.create<arith::ExtFOp>(loc, valElemType, valTruncated);
            b.create<affine::AffineStoreOp>(loc, valExt, valFlat, ivs);
          });
    }
  } else {
    testResult = b.create<memref::CastOp>(loc, mr1DUnkTestType, testFlat);
    valResult = b.create<memref::CastOp>(loc, mr1DUnkValType, valFlat);
  }

  // Prepare the validation result for the verify function
  // Declare and call the wrapper verify function
  func::FuncOp verifyFuncDecl;

  if (isa<FloatType>(testElemType)) {
    constexpr float defaultRMSThreshold(0.00003f);
    constexpr float defaultRMSThresholdFP16(0.001f);
    float RMSThresholdValue =
        testElemType.isF16() ? defaultRMSThresholdFP16 : defaultRMSThreshold;
    if (RMSThreshold)
      RMSThresholdValue = RMSThreshold.getValue();
    Value thr_RMS = getF32Val(RMSThresholdValue);
    Value thr_absDiff = getF32Val(absDiffThreshold.getValue());
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
    if (!valElemType.isF32())
      b.create<memref::DeallocOp>(loc, valResultNew);
  }

  b.create<func::ReturnOp>(loc, ValueRange{});

  return func;
}

// If the fut expects certain args (mostly output buffers),
// this will populate the linalg.fill calls to do those based
// on the presense of mhal::PrefillAttr. This is to mimic the
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
  if (!genParams.types.empty()) {
    FloatType ftype, itype;
    if ((ftype = dyn_cast<FloatType>(genParams.types[0])) &&
        (itype = dyn_cast<FloatType>(genParams.types[1])))
      isSmallFloatIn = ftype.getWidth() < 32 && itype.getWidth() < 32;
  }
  bool gpuValidation = validationType == "gpu" &&
                       ((hasAccel || isSmallFloatIn) || heuristicValidation);
  if (gpuValidation) {
    if (genParams.convConfig.has_value()) { // conv GPU validation
      // generate generic kernels
      const auto &genConfig = **genParams.convConfig;
      rock::ConvGenerator convGenerator(genConfig);
      if (heuristicValidation || hasAccel)
        convGenerator.setPerfConfig("");
      // use non-accel kernels to verify accel kernels except when
      // verifying a tuning case
      if (hasAccel)
        convGenerator.flipAccel();
      if (!((hasAccel || heuristicValidation) &&
            genConfig.inputDataTypeStr == "i8"))
        // use f32 data type to verify non-f32 or xdlops f32 kernels
        // except that i8 xdlops or tuned is verified with i8 non-xdlops.
        convGenerator.setDataTypes("f32");

      int kernelStart = genConfig.kernelId;
      int kernelCount = 0;
      if (failed(convGenerator.getKernelCount(b, kernelCount))) {
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
        convGenerator.setKernelName(kernelBaseName + "_" + std::to_string(i));
        if (failed(convGenerator.genConvModule(module, i, true,
                                               /*ignoreTuning=*/true))) {
          llvm::errs() << "Module population failed.\n";
          exit(1);
        }
        KernelIF kernel(convGenerator.getKernelFunc());
        auto kernelWrapperFunc = createGPUWrapper(module, kernel);

        // Decide whether to trim the last workspace argument to the verifier
        // GPU kernel.
        rock::ConvGenerator originalConvGenerator(genConfig);
        bool originalHasWorkspace = false, verifierHasWorkspace = false;
        if (failed(
                originalConvGenerator.hasWorkspace(b, originalHasWorkspace))) {
          llvm::errs() << "Getting workspace failed.\n";
          exit(1);
        }
        if (failed(convGenerator.hasWorkspace(b, verifierHasWorkspace))) {
          llvm::errs() << "Getting workspace failed.\n";
          exit(1);
        }
        if (originalHasWorkspace && !verifierHasWorkspace) {
          valVars.resize(valVars.size() - 1);
        }

        b.create<func::CallOp>(loc, kernelWrapperFunc, valVars);
      }
      convGenerator.setKernelName(kernelBaseName);
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
          createGpuGemmKernel(module, newParams, /*isVerifier=*/true));
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
    } else if (genParams.operation == rock::KernelType::Attention) {
      if (validationType == "cpp") {
        llvm::errs() << "External attention validator is not available\n";
        exit(1);
      }
      auto cpuAttentionFunc =
          createCpuAttentionKernelWithMlir(module, genParams);
      b.create<func::CallOp>(loc, cpuAttentionFunc, valVars);
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
    auto testType = dyn_cast<MemRefType>(testResult.getType());
    auto valType = dyn_cast<MemRefType>(valResult.getType());
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
  if (!genParams.types.empty()) {
    FloatType ftype, itype;
    if ((ftype = dyn_cast<FloatType>(genParams.types[0])) &&
        (itype = dyn_cast<FloatType>(genParams.types[1])))
      isSmallFloatIn = ftype.getWidth() < 32 && itype.getWidth() < 32;
  }
  bool gpuValidation = validationType == "gpu" &&
                       ((hasAccel || isSmallFloatIn) || heuristicValidation);
  bool isRandom = (randomSeed != "fixed" && randomSeed != "none");

  if (isRandom) {
    auto seedFunc = makeFuncDecl(module, "seedRandomValues", {b.getI32Type()});
    int seed = getRandomSeed();
    Value seedConst = b.create<arith::ConstantIntOp>(loc, seed, b.getI32Type());
    b.create<func::CallOp>(loc, seedFunc, seedConst);
  }

  SmallVector<int32_t, 2> outIndices;
  if (genParams.operation.has_value()) {
    switch (genParams.operation.value()) {
    case rock::KernelType::Conv:
    case rock::KernelType::Gemm:
      outIndices.push_back(2);
      break;
    case rock::KernelType::ConvBwdData:
      outIndices.push_back(1);
      break;
    case rock::KernelType::ConvBwdWeight:
      outIndices.push_back(0);
      break;
    case rock::KernelType::Attention:
      int32_t optionalArgsCounter{3};
      if (hasAttnScale)
        ++optionalArgsCounter;
      if (hasAttnBias)
        ++optionalArgsCounter;
      outIndices.push_back(optionalArgsCounter);
    }
  } else {
    outIndices = root0.outIndices;
  }

  SmallVector<Value, 5> localVars;
  SmallVector<Value, 5> valVars;
  for (auto [idx, paramType] : llvm::enumerate(root0.params)) {
    auto paramMRType = dyn_cast<MemRefType>(paramType);
    assert(paramMRType && "currently only supports memref types");
    Type elemType = paramMRType.getElementType();
    bool isSmallFloat =
        isa<FloatType>(elemType) && elemType.getIntOrFloatBitWidth() < 32;
    if (isCPUKernel) { // -prc
      if (genParams.operation.has_value()) {
        if (idx < genParams.types.size())
          elemType = genParams.types[idx];
        if (isa<IntegerType>(elemType) && llvm::is_contained(outIndices, idx))
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

    if (hasValidation || (isCPUKernel && isSmallFloat)) {
      // Emit validation var
      Type valElemType = floatType;
      if (genParams.operation.has_value() && isa<IntegerType>(elemType)) {
        valElemType = elemType;
        if (!gpuValidation && idx == 2)
          //-pv_with_mlir, -pv_with_cpp, or -pv_with_gpu && non-accel
          // validate in int64_t to detect overflow
          valElemType = b.getIntegerType(64);
      } else if ((genValidation == "clone") || elemType.isInteger(8) ||
                 elemType.isInteger(32)) {
        valElemType = elemType;
      } else if (!gpuValidation && isSmallFloat &&
                 genParams.operation.has_value()) {
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

static OwningOpRef<ModuleOp> readTestFile(std::string inputFilenameStr,
                                          bool &hasUserKernel,
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
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, context);
  if (!module) {
    llvm::errs() << "Parse host harness " << inputFilename << " failed.\n";
    exit(1);
  }

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

  module->walk([&](func::FuncOp func) -> WalkResult {
    if (func->hasAttr("kernel")) {
      hasUserKernel = true;
    }
    return WalkResult::advance();
  });

  return module;
}

static void generateKernel(MLIRContext *context, GenParams &genParams,
                           ModuleOp module) {
  OpBuilder builder(context);
  static rock::ConvGenerator convGenerator; // genParams keeps pointer to config

  const bool isGemm = operation == rock::KernelType::Gemm;
  const bool isAttention = operation == rock::KernelType::Attention;
  const bool isConv = !(isGemm || isAttention);
  auto convConfigStr = populateConvConfig.getValue();

  if (!convConfigStr.empty() && !isConv) {
    llvm::errs() << "Cannot use --conv-config with gemm/attention operations\n";
    exit(1);
  }

  if (convConfigStr.empty() && failed(detectMissingArguments())) {
    exit(1);
  }

  // Scenario 1: We use conv config to initialize everything
  if (!convConfigStr.empty()) {
    if (failed(convGenerator.parseConvConfig(builder, convConfigStr.c_str()))) {
      llvm::errs() << "Module population failed.\n";
      exit(1);
    }
    genParams.types.push_back(convGenerator.getFilterDataType(builder));
    genParams.types.push_back(convGenerator.getInputDataType(builder));
    genParams.types.push_back(convGenerator.getOutputDataType(builder));
    const auto *convConfig = &convGenerator.getConfig();
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

    Type filterElemType = typeFromString(filterDataType.getValue(), context);
    Type inputElemType = typeFromString(inputDataType.getValue(), context);
    Type elemType = inputElemType;
    rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
    rock::GemmFeatures enabledFeatures = archInfo.getDefaultFeatures(elemType);
    // toggle feature list according to llvm::cl::opt inputs
    if (mfmaFeature == FeatureToggle::infer) {
      // Disable acceleration for mixed types
      if (filterElemType.getIntOrFloatBitWidth() !=
          inputElemType.getIntOrFloatBitWidth()) {
        enabledFeatures =
            bitEnumClear(enabledFeatures, rock::GemmFeatures::mfma);
      }
    } else
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

    if (wmmaFeature == FeatureToggle::infer) {
      // Disable acceleration for mixed types
      if (filterElemType.getIntOrFloatBitWidth() !=
          inputElemType.getIntOrFloatBitWidth()) {
        enabledFeatures =
            bitEnumClear(enabledFeatures, rock::GemmFeatures::wmma);
      }
    } else
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
    } else if (isAttention) {
      auto elemType = typeFromString(inputDataType.getValue(), context);
      // We only support first-gemm i8 version of attention
      // This will be changed when we support both gemms of i8.
      if (elemType == IntegerType::get(context, 8)) {
        constexpr size_t maxNumArgs{7};
        genParams.types.resize(maxNumArgs);
        genParams.types[AttentionQuantizedArgIndex::q] =
            IntegerType::get(context, 8);
        genParams.types[AttentionQuantizedArgIndex::k] =
            IntegerType::get(context, 8);
        genParams.types[AttentionQuantizedArgIndex::v] =
            Float16Type::get(context);
        genParams.types[AttentionQuantizedArgIndex::quantBias] =
            IntegerType::get(context, 8);
        genParams.types[AttentionQuantizedArgIndex::quantScale] =
            Float16Type::get(context);
        genParams.types[AttentionQuantizedArgIndex::scale] =
            Float16Type::get(context);
        genParams.types[AttentionQuantizedArgIndex::bias] =
            Float16Type::get(context);
      } else {
        constexpr size_t maxNumArgs{5};
        // Note: In the current implementation, all operands have the same type.
        // This behaviour enforced by `-t`. See, detectMissingArguments()
        for (size_t argIdx{0}; argIdx < maxNumArgs; ++argIdx) {
          genParams.types.push_back(elemType);
        }
      }
      genParams.convConfig = std::nullopt;
      (void)createGpuAttentionKernel(module, genParams);
    } else {
      int nDims = filterLayout.getValue().size() - 3; // +++pf: magic number.
      SmallVector<int, 4> dilations;
      SmallVector<int, 4> strides;
      SmallVector<int, 4> paddingLeft;
      SmallVector<int, 4> paddingRight;

      // +++pf: needs generalising, coupled with command-line options.
      dilations.push_back(dilationHeight.getValue());
      strides.push_back(strideHeight.getValue());
      paddingLeft.push_back(paddingHeightLeft.getValue());
      paddingRight.push_back(paddingHeightRight.getValue());

      if (nDims > 1) {
        dilations.push_back(dilationWidth.getValue());
        strides.push_back(strideWidth.getValue());
        paddingLeft.push_back(paddingWidthLeft.getValue());
        paddingRight.push_back(paddingWidthRight.getValue());
      }
      if (nDims > 2) {
        dilations.push_back(dilationDepth.getValue());
        strides.push_back(strideDepth.getValue());
        paddingLeft.push_back(paddingDepthLeft.getValue());
        paddingRight.push_back(paddingDepthRight.getValue());
      }

      convGenerator = rock::ConvGenerator(
          arch, chip, triple, chipFeatures, perfConfig.getValue(),
          num_cu.getNumOccurrences() ? std::optional<int>(num_cu.getValue())
                                     : std::nullopt,
          reverse_grid, enabledFeatures,
          rock::convOpTypeFromKernelType(operation.getValue()),
          filterDataType.getValue(), inputDataType.getValue(),
          outputDataType.getValue(), dilations, strides, paddingLeft,
          paddingRight, filterLayout.getValue(), inputLayout.getValue(),
          outputLayout.getValue());

      SmallVector<int64_t> inDims{inputHeight, inputWidth};
      if (nDims > 2) {
        if (inputDepth < 1)
          inputDepth = 1;
        inDims.push_back(inputDepth);
      }
      SmallVector<int64_t> outDims{outputHeight, outputWidth};
      if (nDims > 2) {
        if (outputDepth < 1)
          outputDepth = 1;
        outDims.push_back(outputDepth);
      }
      SmallVector<int64_t> filDims{filterHeight, filterWidth};
      if (nDims > 2) {
        if (filterDepth < 1)
          filterDepth = 1;
        filDims.push_back(filterDepth);
      }

      status =
          convGenerator.parseConvDims(batchSize, groupSize, inputChannel,
                                      inDims, outputChannel, outDims, filDims);
      if (failed(status)) {
        llvm::errs() << "Could not parse convolution dimensions\n";
        exit(1);
      }

      genParams.types.push_back(convGenerator.getFilterDataType(builder));
      genParams.types.push_back(convGenerator.getInputDataType(builder));
      genParams.types.push_back(convGenerator.getOutputDataType(builder));
      genParams.convConfig = &convGenerator.getConfig();
    }
  }

  // TODO: Extract isApplicable check to be its own component
  if (isConv && failed(convGenerator.isApplicable(/* checkChip = */ false))) {
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
      if (failed(convGenerator.getKernelCount(builder, kernelCount))) {
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
        convGenerator.setKernelName(kernelBaseName + "_" + std::to_string(i));
        if (failed(convGenerator.genConvModule(module, i))) {
          llvm::errs() << "Module population failed.\n";
          exit(1);
        }
      }
      convGenerator.setKernelName(kernelBaseName);
    }
  }
}

static void populateCloneHarnessLogic(ModuleOp module) {
  if (arch.getValue().empty()) {
    llvm::errs() << "--arch is not set\n";
    exit(1);
  }
  func::FuncOp originalFunc = module.lookupSymbol<func::FuncOp>(testFuncName);
  assert(originalFunc && "does -fut point to the wrong function?");

  MLIRContext *context = module.getContext();
  OpBuilder b(context);

  originalFunc->removeAttr("kernel");
  StringAttr archAttr = b.getStringAttr(arch);
  if (originalFunc->hasAttr("arch"))
    originalFunc->setAttr("arch", archAttr);
  auto readAttr =
      b.getNamedAttr(func::FuncOp::getReadAccessAttrName(), b.getUnitAttr());
  auto writeAttr =
      b.getNamedAttr(func::FuncOp::getWriteAccessAttrName(), b.getUnitAttr());
  for (size_t index = 0; index < originalFunc.getArguments().size(); index++)
    originalFunc.setArgAttrs(index, readAttr);
  for (size_t index = 0; index < originalFunc.getNumResults(); index++)
    originalFunc.setResultAttrs(index, writeAttr);
  auto loc = originalFunc->getLoc();
  auto wrapperFunc = func::FuncOp::create(loc, testFuncName + "_wrapper",
                                          originalFunc.getFunctionType());
  Block *block = wrapperFunc.addEntryBlock();
  b.setInsertionPointToStart(block);
  auto launchOp = b.create<mhal::LaunchOp>(loc, originalFunc, ValueRange{},
                                           block->getArguments());
  auto results = launchOp->getResults();
  b.create<mhal::AwaitOp>(loc, results.front());
  b.create<func::ReturnOp>(loc, ValueRange{results.drop_front()});
  module.push_back(wrapperFunc);

  auto xmoduleOp = ModuleOp::create(loc, "__xmodule_");
  xmoduleOp->setAttr("mhal.arch", archAttr);
  xmoduleOp->setAttr("mhal.module", b.getUnitAttr());
  auto *cloneFunc = originalFunc->clone();
  auto cloneFuncOp = dyn_cast<func::FuncOp>(cloneFunc);
  cloneFuncOp->setAttr("kernel", b.getUnitAttr());
  cloneFuncOp->setAttr("original_func", SymbolRefAttr::get(originalFunc));
  xmoduleOp.push_back(cloneFuncOp);
  module.push_back(xmoduleOp);
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerRocMLIRDialects(registry);
  // Parse pass names in main to ensure static initialization completed.
  mlir::registerMLIRCLOptions();
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  // LLVM dialect is temporary for the freeze trick.
  context.loadDialect<rock::RockDialect, func::FuncDialect, scf::SCFDialect,
                      affine::AffineDialect, memref::MemRefDialect,
                      math::MathDialect, arith::ArithDialect,
                      vector::VectorDialect, gpu::GPUDialect,
                      linalg::LinalgDialect, mhal::MHALDialect,
                      bufferization::BufferizationDialect, tosa::TosaDialect>();

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR Rock Dialect host generation\n");

  if (!arch.getValue().empty()) {
    bool archPrefersOCP = (archChip().substr(0, 5) == "gfx12");
    std::map<F8TypesChoice,std::string>f8e4m3TypeNames{
        {F8TypesChoice::Arch, archPrefersOCP ? "f8E4M3FN" : "f8E4M3FNUZ"},
        {F8TypesChoice::Nanoo, "f8E4M3FNUZ"},
        {F8TypesChoice::OCP, "f8E4M3FN"}};
    std::map<F8TypesChoice,std::string>f8e5m2TypeNames{
        {F8TypesChoice::Arch, archPrefersOCP ? "f8E5M2" : "f8E5M2FNUZ"},
        {F8TypesChoice::Nanoo, "f8E5M2FNUZ"},
        {F8TypesChoice::OCP, "f8E5M2"}};

    auto canonicaliseF8Type = [&](std::string name) {
                                if (name == "fp8")
                                  return f8e4m3TypeNames[forceF8Types.getValue()];
                                if (name == "bf8")
                                  return f8e5m2TypeNames[forceF8Types.getValue()];
                                return std::string(name);
                              };

    filterDataType = canonicaliseF8Type(filterDataType);
    inputDataType = canonicaliseF8Type(inputDataType);
    outputDataType = canonicaliseF8Type(outputDataType);
  }

  if (operation != rock::KernelType::Gemm) {
    verifyConvLayout();
    correctConvParameters();
  }
  populateDefaults();

  bool hasUserKernel = !testFuncName.empty();

  OwningOpRef<ModuleOp> module;
  GenParams genParams;

  if (!inputFilename.empty()) {
    module = readTestFile(inputFilename.getValue(), hasUserKernel, &context);
  } else {
    if (genValidation == "clone") {
      llvm::errs()
          << "Clone validation is not compatible with kernel generation.\n";
      exit(1);
    }
    module = ModuleOp::create(UnknownLoc::get(&context));
  }

  if (genCloneHarness.getValue()) {
    populateCloneHarnessLogic(*module);
  } else if (!hasUserKernel) {
    generateKernel(&context, genParams, *module);
  }

  if (emitSplitKSelectionLikelihood) {
    module->walk([](rock::RockGemmWrapperInterface gemmOp) {
      const int32_t numCU = rock::lookupArchInfo(gemmOp.getArch()).minNumCU;
      const rock::GemmSize gemmSize = gemmOp.getGemmSize();
      const auto likelihood = rock::isSplitKFaster(
          gemmSize.g, gemmSize.m, gemmSize.n, gemmSize.k, numCU);
      switch (likelihood) {
      case RocmlirSplitKSelectionLikelihood::always: {
        llvm::outs() << "always\n";
        break;
      }
      case RocmlirSplitKSelectionLikelihood::maybe: {
        llvm::outs() << "maybe\n";
        break;
      }
      case RocmlirSplitKSelectionLikelihood::never: {
        llvm::outs() << "never\n";
        break;
      }
      }
    });
    return 0;
  }

  if (!emitModuleFusabilityForPerfConfig.empty()) {
    llvm::outs() << "fusible:"
                 << rock::isModuleFusible(module.get(),
                                          emitModuleFusabilityForPerfConfig)
                 << "\n";
    return 0;
  }

  if (emitTuningSpace.getNumOccurrences() > 0) {
    std::unique_ptr<rock::TuningParamSet> tunableParams(
        rock::createTunableParamSpace(*module, emitTuningSpace));
    SmallString<64> perfConfig;
    for (auto param : tunableParams->tuningRange) {
      param.getPerfConfigStr(perfConfig);
      llvm::outs() << perfConfig << "\n";
      perfConfig.clear();
    }
    return 0;
  }

  if (emitTuningKey) {
    SmallString<2048> tuningKey;
    if (failed(rock::getTuningProblemStr(*module, tuningKey))) {
      llvm::errs() << "Failed to get tuning key for module: " << *module
                   << "\n";
      return EXIT_FAILURE;
    }
    llvm::outs() << tuningKey << "\n";
    return 0;
  }

  SmallVector<KernelIF, 8> kernels;
  SmallVector<KernelIF, 8> rootIFs;

  if (testFuncName.empty()) {
    // Compute set of call-graph root nodes;  they're the ones we need to
    // call from main().  Start with all nodes, then erase the ones that
    // have edges to them.  Use SetVector because we want to preserve the
    // order to match an older implementation.
    CallGraph cg(*module);
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
    module->walk([&](func::FuncOp func) -> WalkResult {
      if (func->hasAttr("kernel")) {
        kernels.emplace_back(func);
      }
      return WalkResult::advance();
    });
  } else if (!genCloneHarness.getValue()) {
    auto func = module->lookupSymbol<func::FuncOp>(testFuncName);
    assert(func && "does -fut point to the wrong function?");
    kernels.emplace_back(func); // +++pf: should it be a kernel?
    rootIFs.emplace_back(func);
  }

  // populate host logic.
  if (genHostHarness.getValue()) {
    if (failed(
            populateHostHarnessLogic(*module, kernels, rootIFs, genParams))) {
      llvm::errs() << "Host logic populated failed.\n";
      exit(1);
    }
  }

  // Running the bufferization pipeline when rocmlir-gen is actually
  // generating a kernel.
  if (applyBufferizationPipeline.getValue() && !hasUserKernel) {
    PassManager pm(module.get()->getName(), PassManager::Nesting::Implicit);

    rock::BufferizeOptions bufferizeOptions;
    bufferizeOptions.disableRock = true;
    rock::buildBufferizePipeline(pm, bufferizeOptions);

    if (failed(pm.run(*module))) {
      llvm::errs() << "failed to apply rocm bufferize pipeline.\n";
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

  module->print(output->os());
  output->keep();
  return 0;
}
