//===- ConvertToMIOpenCPP.cpp - MLIR to MIOpen C++ conversion -------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR MIOpen dialect and C++.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpenOps/MIOpenCPP.h"
#include "mlir/Dialect/MIOpenOps/MIOpenOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Translation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

namespace {

static constexpr StringLiteral kVarName[3] = {"weight", "input", "output"};

static constexpr int kConv2DTensorDimension = 4;

static constexpr StringLiteral kCppPreamblePart1 = R"(
#include "common_header.hpp"
)";

static constexpr StringLiteral kCppPreamblePart2 = R"(
#include "float_types.h"

extern "C" __global__
)";

static constexpr StringLiteral kCppPreamblePart3 = R"(
        (const FLOAT* const __restrict__ p_in_global,
        const FLOAT* const __restrict__ p_wei_global,
        FLOAT* const __restrict__ p_out_global)
{
    using namespace ck;

    constexpr index_t ConvStrideH = CK_PARAM_PROBLEM_CONV_STRIDE_H;
    constexpr index_t ConvStrideW = CK_PARAM_PROBLEM_CONV_STRIDE_W;

    constexpr index_t ConvDilationH = CK_PARAM_PROBLEM_CONV_DILATION_H;
    constexpr index_t ConvDilationW = CK_PARAM_PROBLEM_CONV_DILATION_W;

    constexpr index_t InLeftPadH = CK_PARAM_PROBLEM_IN_LEFT_PAD_H;
    constexpr index_t InLeftPadW = CK_PARAM_PROBLEM_IN_LEFT_PAD_W;

    constexpr index_t InRightPadH = CK_PARAM_PROBLEM_IN_RIGHT_PAD_H;
    constexpr index_t InRightPadW = CK_PARAM_PROBLEM_IN_RIGHT_PAD_W;

    constexpr index_t BlockSize = CK_PARAM_TUNABLE_BLOCK_SIZE;
    constexpr index_t GridSize  = CK_PARAM_DEPENDENT_GRID_SIZE;

    constexpr index_t GemmMPerBlock = CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK;
    constexpr index_t GemmNPerBlock = CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK;
    constexpr index_t GemmKPerBlock = CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK;

)";

static constexpr StringLiteral kCppInterlude = R"(
    using ConvStrides   = Sequence<ConvStrideH, ConvStrideW>;
    using ConvDilations = Sequence<ConvDilationH, ConvDilationW>;

    using InLeftPads  = Sequence<InLeftPadH, InLeftPadW>;
    using InRightPads = Sequence<InRightPadH, InRightPadW>;

    // read and calculate tuning parameter
    constexpr index_t GemmMPerThreadSubC = CK_PARAM_TUNABLE_GEMM_M_PER_THREAD_SUB_C;
    constexpr index_t GemmNPerThreadSubC = CK_PARAM_TUNABLE_GEMM_N_PER_THREAD_SUB_C;
    constexpr index_t GemmMLevel0Cluster = CK_PARAM_TUNABLE_GEMM_M_LEVEL0_CLUSTER;
    constexpr index_t GemmNLevel0Cluster = CK_PARAM_TUNABLE_GEMM_N_LEVEL0_CLUSTER;
    constexpr index_t GemmMLevel1Cluster = CK_PARAM_TUNABLE_GEMM_M_LEVEL1_CLUSTER;
    constexpr index_t GemmNLevel1Cluster = CK_PARAM_TUNABLE_GEMM_N_LEVEL1_CLUSTER;
    constexpr index_t GemmKPerThreadLoop = 1;

    constexpr index_t GemmThreadGemmDataPerReadM = GemmMPerThreadSubC;
    constexpr index_t GemmThreadGemmDataPerReadN = GemmNPerThreadSubC;

    // A matrix
    constexpr index_t GemmABlockCopyClusterLengths_GemmK =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K;

    constexpr index_t GemmABlockCopyClusterLengths_GemmM =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M;

    constexpr index_t GemmABlockCopyThreadSliceLengths_GemmK =
        GemmKPerBlock / GemmABlockCopyClusterLengths_GemmK;

    constexpr index_t GemmABlockCopyThreadSliceLengths_GemmM =
        GemmMPerBlock / GemmABlockCopyClusterLengths_GemmM;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM =
        Sequence<GemmABlockCopyThreadSliceLengths_GemmK, GemmABlockCopyThreadSliceLengths_GemmM>;

    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM =
        Sequence<GemmABlockCopyClusterLengths_GemmK, GemmABlockCopyClusterLengths_GemmM>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_K;

    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M;

    // B matrix
    constexpr index_t GemmBBlockCopyClusterLengths_GemmK =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K;

    constexpr index_t GemmBBlockCopyClusterLengths_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N;

    constexpr index_t GemmBBlockCopyThreadSliceLengths_GemmK =
        GemmKPerBlock / GemmBBlockCopyClusterLengths_GemmK;

    constexpr index_t GemmBBlockCopyThreadSliceLengths_GemmN =
        GemmNPerBlock / GemmBBlockCopyClusterLengths_GemmN;
    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN =
        Sequence<GemmBBlockCopyThreadSliceLengths_GemmK, GemmBBlockCopyThreadSliceLengths_GemmN>;

    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN =
        Sequence<GemmBBlockCopyClusterLengths_GemmK, GemmBBlockCopyClusterLengths_GemmN>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_N;

    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N;

    // C matrix
    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 =
        CK_PARAM_TUNABLE_GEMM_C_THREAD_COPY_DST_DATA_PER_WRITE_GEMM_N1;
)";

static constexpr StringLiteral kCppEpiloguePart1 = R"(
        <GridSize,
        BlockSize,
        FLOAT,
        FLOAT_ACCUM,
)";

static constexpr StringLiteral kCppEpiloguePart2 =R"(
        ConvStrides,
        ConvDilations,
        InLeftPads,
        InRightPads,
        GemmMPerBlock,
        GemmNPerBlock,
        GemmKPerBlock,
        GemmMPerThreadSubC,
        GemmNPerThreadSubC,
        GemmMLevel0Cluster,
        GemmNLevel0Cluster,
        GemmMLevel1Cluster,
        GemmNLevel1Cluster,
        GemmKPerThreadLoop,
        GemmThreadGemmDataPerReadM,
        GemmThreadGemmDataPerReadN,
        GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
        GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
        GemmABlockCopySrcDataPerRead_GemmK,
        GemmABlockCopyDstDataPerWrite_GemmM,
        GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
        GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
        GemmBBlockCopySrcDataPerRead_GemmN,
        GemmBBlockCopyDstDataPerWrite_GemmN,
        GemmCThreadCopyDstDataPerWrite_GemmN1>{};

    gridwise_conv.Run(p_in_global, p_wei_global, p_out_global);
}
)";
 
void EmitCppPreamble(llvm::raw_ostream &output, llvm::StringRef layoutStr) {
  output << kCppPreamblePart1;
// Between Preamble Part 1 and Part 2:
// #include "gridwise_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw.hpp"
  output << R"(#include "gridwise_convolution_implicit_gemm_v4r4_)";

  // Change to fixed "mlir".
  //output << layoutStr << R"(.hpp")";
  output << "mlir" << R"(.hpp")";

  output << kCppPreamblePart2;
// Between Preamble Part 2 and Par 3:
//    __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void gridwise_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw(
  output << R"(
    __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void gridwise_convolution_implicit_gemm_v4r4_)";
  // Change to fixed "mlir".
  //output << layoutStr;
  output << "mlir";

  output << kCppPreamblePart3;
}

void EmitCppInterlude(llvm::raw_ostream &output) {
  output << kCppInterlude;
}

void EmitCppEpilogue(llvm::raw_ostream &output, llvm::StringRef layoutStr, llvm::SmallVector<std::string, 3> tensorDescs) {
// Before Part1:
//    constexpr auto gridwise_conv = GridwiseConvolutionImplicitGemm_v4r4_nchw_kcyx_nkhw
  output << R"(
    constexpr auto gridwise_conv = GridwiseConvolutionImplicitGemm_v4r4_)";

  // Change to fixed "mlir".
  //output << layoutStr;
  output << "mlir";

  output << kCppEpiloguePart1;
// Between Part1 and Part2:
//        decltype(in_nchw_desc),
//        decltype(wei_kcyx_desc),
//        decltype(out_nkhw_desc),
  output << "        decltype(" << tensorDescs[1] << "),\n";
  output << "        decltype(" << tensorDescs[0] << "),\n";
  output << "        decltype(" << tensorDescs[2] << "),\n";
  output << kCppEpiloguePart2;
}

static constexpr StringLiteral kHeaderPreamblePart1 = R"(
#ifndef CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_HPP
#define CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm.hpp"

namespace ck {

// GemmM = K
// GemmN = N * Ho * Wo
// GemmK = C * Y * X
template <index_t GridSize,
          index_t BlockSize,
          typename Float,
          typename AccFloat,
          typename InGlobalDesc,
          typename WeiGlobalDesc,
          typename OutGlobalDesc,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads,
          index_t GemmMPerBlock,
          index_t GemmNPerBlock,
          index_t GemmKPerBlock,
          index_t GemmMPerThreadSubC,
          index_t GemmNPerThreadSubC,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t GemmKPerThreadLoop,
          index_t GemmThreadGemmDataPerReadM,
          index_t GemmThreadGemmDataPerReadN,
          typename GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
          typename GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
          index_t GemmABlockCopySrcDataPerRead_GemmK,
          index_t GemmABlockCopyDstDataPerWrite_GemmM,
          typename GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
          typename GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
          index_t GemmBBlockCopySrcDataPerRead_GemmN,
          index_t GemmBBlockCopyDstDataPerWrite_GemmN,
          index_t GemmCThreadCopyDstDataPerWrite_GemmN1>
)";

static constexpr StringLiteral kHeaderPreamblePart2 = R"(
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {
)";

static constexpr StringLiteral kHeaderPreamblePart3 = R"(
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];
)";

static constexpr StringLiteral kHeaderEpiloguePart1 = R"(
        // GEMM
        constexpr auto gridwise_gemm =
            GridwiseGemmTransposedANormalBNormalC_v1<GridSize,
                                                     BlockSize,
                                                     Float,
                                                     AccFloat,
)";

static constexpr StringLiteral kHeaderEpiloguePart2 = R"(
                                                     InMemoryDataOperation::none,
                                                     GemmMPerBlock,
                                                     GemmNPerBlock,
                                                     GemmKPerBlock,
                                                     GemmMPerThreadSubC,
                                                     GemmNPerThreadSubC,
                                                     GemmMLevel0Cluster,
                                                     GemmNLevel0Cluster,
                                                     GemmMLevel1Cluster,
                                                     GemmNLevel1Cluster,
                                                     GemmKPerThreadLoop,
                                                     GemmThreadGemmDataPerReadM,
                                                     GemmThreadGemmDataPerReadN,
                                                     GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
                                                     GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
                                                     Sequence<1, 0>,
                                                     Sequence<1, 0>,
                                                     0,
                                                     GemmABlockCopySrcDataPerRead_GemmK,
                                                     GemmABlockCopyDstDataPerWrite_GemmM,
                                                     GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
                                                     GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
                                                     Sequence<0, 1>,
                                                     Sequence<0, 1>,
                                                     1,
                                                     GemmBBlockCopySrcDataPerRead_GemmN,
                                                     GemmBBlockCopyDstDataPerWrite_GemmN,
                                                     Sequence<0, 1, 2, 3>,
                                                     3,
                                                     GemmCThreadCopyDstDataPerWrite_GemmN1>{};

        gridwise_gemm.Run(p_wei_global, p_in_global, p_out_global);
    }
};

} // namespace ck
#endif
)";

void EmitHeaderPreamble(llvm::raw_ostream &output, llvm::StringRef layoutStr, llvm::SmallVector<std::string, 3> &tensorDescs) {
  output << kHeaderPreamblePart1;
  output << R"(
struct GridwiseConvolutionImplicitGemm_v4r4_)";

  // Change to fixed "mlir".
  //output << layoutStr;
  output << "mlir";

  output << kHeaderPreamblePart2;
  output << kHeaderPreamblePart3;
  output << '\n';

  output << R"(
         constexpr auto )" << tensorDescs[0] << " = WeiGlobalDesc{};";
  output << R"(
          constexpr auto )" << tensorDescs[1] << " = InGlobalDesc{};";
  output << R"(
          constexpr auto )" << tensorDescs[2] << " = OutGlobalDesc{};";
  output << '\n';
}

void EmitHeaderEpilogue(llvm::raw_ostream &output, llvm::SmallDenseMap<int64_t, std::string> &args) {
  output << kHeaderEpiloguePart1;
// Between Part1 and Part2 emit:
//                                                   decltype(wei_e_k_global_desc),
//                                                   decltype(in_e_b_global_desc),
//                                                   decltype(out_k_b_global_desc),
  for (unsigned i = 0; i < args.size(); ++i) {
    output << R"(
                                                     decltype()" << args[i] << "),";
  }
  output << kHeaderEpiloguePart2;
}

void EmitLayoutString(llvm::raw_ostream &output, llvm::ArrayRef<mlir::Attribute> &layoutArrayAttr, llvm::StringRef prefix, llvm::StringRef suffix, llvm::StringRef delimiter = "") {
  for (int i = 0; i < kConv2DTensorDimension; ++i) {
    auto attr = layoutArrayAttr[i];
    if (auto strAttr = attr.dyn_cast<StringAttr>()) {
      output << prefix << strAttr.getValue() << suffix;
    }
    if (i < kConv2DTensorDimension - 1) {
      output << delimiter;
    }
  }
}

void EmitHeaderDimensionLengths(llvm::raw_ostream &output, llvm::ArrayRef<mlir::Attribute> &layoutArrayAttr, llvm::StringRef tensorDesc) {
  for (int i = 0; i < kConv2DTensorDimension; ++i) {
    auto attr = layoutArrayAttr[i];
    if (auto strAttr = attr.dyn_cast<StringAttr>()) {
      output << "        constexpr index_t " << strAttr.getValue() << " = " << tensorDesc << ".GetLengths()[" << i << "];\n";
    }
  }
}

void EmitDimensionVariables(llvm::raw_ostream &output, llvm::ArrayRef<mlir::Attribute> &layoutArrayAttr) {
  for (int i = 0; i < kConv2DTensorDimension; ++i) {
    auto attr = layoutArrayAttr[i];
    if (auto strAttr = attr.dyn_cast<StringAttr>()) {
      output << "    constexpr index_t " << strAttr.getValue() << " = CK_PARAM_PROBLEM_";

      switch (llvm::toUpper(strAttr.getValue()[0])) {
          case 'H':
          case 'W':
            output << llvm::toUpper(strAttr.getValue()[0]);
            // XXX: fix this. 
            if (strAttr.getValue().size() > 1)
              output << llvm::toUpper(strAttr.getValue()[1]);
            break;
          default:
            output << llvm::toUpper(strAttr.getValue()[0]);
      }
      output << ";\n";
    }
  }
}

void EmitStrideVariables(llvm::raw_ostream &output, llvm::ArrayRef<mlir::Attribute> &layoutArrayAttr) {
  for (int i = kConv2DTensorDimension - 1; i >= 0; --i) {
    auto attr = layoutArrayAttr[i];
    if (auto strAttr = attr.dyn_cast<StringAttr>()) {
      output << "    constexpr index_t stride_" << strAttr.getValue() << " = ";

      if (i == kConv2DTensorDimension - 1) {
        output << "1;\n";
      } else {
        auto prevAttr = layoutArrayAttr[i + 1];
        if (auto strPrevAttr = prevAttr.dyn_cast<StringAttr>()) {
          output << strPrevAttr.getValue() << " * stride_" << strPrevAttr.getValue() << ";\n";
        }
      }
    }
  }
}

template<typename T>
void EmitInterleaveArrayAttrWithSeparator(llvm::raw_ostream &os, mlir::ArrayAttr &arrayAttr, const StringRef &separator) {
  if (arrayAttr) {
    interleave(arrayAttr, os, [&](Attribute attr) {
      if (auto typedAttr = attr.dyn_cast<T>())
        os << typedAttr.getValue();
    }, separator);
  }
}

template<typename T>
void EmitInterleaveCommaArrayAttr(llvm::raw_ostream &os, mlir::ArrayAttr &arrayAttr) {
  EmitInterleaveArrayAttrWithSeparator<T>(os, arrayAttr, ", ");
}

void ObtainModuleInfo(ModuleOp &m, std::string &layoutStr, llvm::SmallVector<std::string, 3> &tensorDescs) {
  // (TBD verifiying logic) The Module could contain multiple FuncOp, and inside each FuncOp there
  // should be exactly:
  // - 3 input arguments
  // - 1 result.
  //
  // - 0 conv2d op.
  // - 5 transform ops (1 for filter, 3 for input, 1 for output).
  // - 1 gridwise gemm op.

  // Enumerate FuncOp instances inside the ModuleOp.
  for (auto f : m.getOps<FuncOp>()) {
    int srcLayoutAttrCtr = 0;
    llvm::raw_string_ostream los(layoutStr);

    // First iteration. Construct tensor descriptor names.
    f.walk([&srcLayoutAttrCtr, &tensorDescs, &los](miopen::TransformOp op) {
      // get source_layout attribute.
      auto srcLayoutAttr = op.getAttrOfType<ArrayAttr>("source_layout");
      if (srcLayoutAttr) {
        auto srcLayout = srcLayoutAttr.getValue();

        // Prepare tensor descriptor variable name.
        std::string desc{};
        llvm::raw_string_ostream os(desc);
        os << kVarName[srcLayoutAttrCtr++] << "_";
        EmitLayoutString(os, srcLayout, "", "", "_");
        os << "_desc";
        os.flush();
        tensorDescs.push_back(desc);

        // Prepare layout string.
        if (srcLayoutAttrCtr != 1)
          los << "_";
        EmitLayoutString(los, srcLayout, "", "");
      }
    });
    los.flush();
  }
}

}

std::unique_ptr<llvm::StringRef> mlir::translateModuleToMIOpenHeader(ModuleOp m) {
  std::string resultStr;
  llvm::raw_string_ostream output(resultStr);

  // Enumerate FuncOp instances inside the ModuleOp.
  for (auto f : m.getOps<FuncOp>()) {
    std::string layoutStr;
    llvm::SmallVector<std::string, 3> tensorDescs;
    llvm::SmallDenseMap<int64_t, std::string> gridwiseGemmArguments;

    // Obtain critical information from ModuleOp.
    ObtainModuleInfo(m, layoutStr, tensorDescs);

    int srcLayoutAttrCtr = 0;

    // Start emitting.
    EmitHeaderPreamble(output, layoutStr, tensorDescs);

    // First iteration. Output source dimensions.
    f.walk([&output, &srcLayoutAttrCtr, &tensorDescs](miopen::TransformOp op) {
      // get source_layout attribute.
      auto srcLayoutAttr = op.getAttrOfType<ArrayAttr>("source_layout");
      if (srcLayoutAttr) {
        auto srcLayout = srcLayoutAttr.getValue();
        output << "\n        // ";
        EmitLayoutString(output, srcLayout, "", "", ", ");
        output << '\n';

        EmitHeaderDimensionLengths(output, srcLayout, tensorDescs[srcLayoutAttrCtr++]);
      }
    });
    output << '\n';
 
    srcLayoutAttrCtr = 0;
    // Second iteration. Output the rest.
    f.walk([&output, &srcLayoutAttrCtr, &tensorDescs, &gridwiseGemmArguments](miopen::TransformOp op) {
      // get source_layout attribute.
      auto srcLayoutAttr = op.getAttrOfType<ArrayAttr>("source_layout");

      // get layout attribute.
      auto layoutAttr = op.getAttrOfType<ArrayAttr>("layout");
      std::string inputTensorName;
      std::string outputTensorName;
      std::string operationSpec;
      std::string srcDimSpec;
      std::string dstDimSpec;
      llvm::raw_string_ostream ins(inputTensorName);
      llvm::raw_string_ostream outs(outputTensorName);
      llvm::raw_string_ostream ops(operationSpec);
      llvm::raw_string_ostream srcs(srcDimSpec);
      llvm::raw_string_ostream dsts(dstDimSpec);

      // determine input and output tensor name.
      auto immLayoutAttr = op.getAttrOfType<ArrayAttr>("intermediate_layout");
      auto outputLayoutAttr = op.getAttrOfType<ArrayAttr>("output_layout");
      if (srcLayoutAttr) {
        inputTensorName = tensorDescs[srcLayoutAttrCtr];
        outs << kVarName[srcLayoutAttrCtr] << "_";

        srcLayoutAttrCtr++;
      } else {
        // get intermediate_layout attribute.
        if (immLayoutAttr) {
          ins << kVarName[srcLayoutAttrCtr - 1] << "_";
          EmitInterleaveArrayAttrWithSeparator<StringAttr>(ins, immLayoutAttr, "_");
          ins << "_desc";
          ins.flush();

          outs << kVarName[srcLayoutAttrCtr - 1] << "_";
        }
      }
      EmitInterleaveArrayAttrWithSeparator<StringAttr>(outs, outputLayoutAttr, "_");
      outs << "_desc";
      outs.flush();

      // determine gridwise GEMM arguments.
      auto gridwiseGemmArgPosAttr = op.getAttrOfType<IntegerAttr>("gridwise_gemm_argument_position");
      if (gridwiseGemmArgPosAttr) {
        gridwiseGemmArguments[gridwiseGemmArgPosAttr.getInt()] = outputTensorName;
      }  

      ops << "            make_tuple(";
      srcs << "            make_tuple(";
      dsts << "            make_tuple(";

      // XXX see if we can get better than this.
      int convDilationCtr = 0;

      for (auto layoutSpec = layoutAttr.begin(); layoutSpec != layoutAttr.end(); ) {
        if (auto layoutSpecDict = layoutSpec->dyn_cast<DictionaryAttr>()) {
          auto srcNames = layoutSpecDict.get("source_names").dyn_cast<ArrayAttr>();
          auto dstNames = layoutSpecDict.get("names").dyn_cast<ArrayAttr>();
          auto srcDims = layoutSpecDict.get("source_dimensions").dyn_cast<ArrayAttr>();
          auto dstDims = layoutSpecDict.get("dimensions").dyn_cast<ArrayAttr>();

          if (auto transform = layoutSpecDict.get("transformation").dyn_cast<StringAttr>()) {
            if (transform.getValue() == "PassThrough") {
              ops << transform.getValue() << "<";
              EmitInterleaveCommaArrayAttr<StringAttr>(ops, srcNames);
              ops << ">{}";
            } else if (transform.getValue() == "Merge") {
              ops << transform.getValue() << "<"
                  << "Sequence<";
              EmitInterleaveCommaArrayAttr<StringAttr>(ops, srcNames);
              ops << ">" << ">{}";
            } else if (transform.getValue() == "Pad") {
              ops << transform.getValue() << "<"
                  << "Sequence<";
              EmitInterleaveCommaArrayAttr<StringAttr>(ops, srcNames);
              ops << ">, InLeftPads, InRightPads" << ">{}";
            } else if (transform.getValue() == "Embed") {
              ops << transform.getValue() << "<"
                  << "Sequence<";
              EmitInterleaveCommaArrayAttr<StringAttr>(ops, dstNames);
              if (convDilationCtr == 0) {
                ops << ">, Sequence<ConvDilationH, ConvDilationH, 0>>{}";
                convDilationCtr++;
              } else {
                ops << ">, Sequence<ConvDilationW, ConvDilationW, 0>>{}";
              }
            }
            srcs << "Sequence<";
            EmitInterleaveCommaArrayAttr<IntegerAttr>(srcs, srcDims);
            srcs << ">{}";
            dsts << "Sequence<";
            EmitInterleaveCommaArrayAttr<IntegerAttr>(dsts, dstDims);
            dsts << ">{}";
          }
        }

        ++layoutSpec;
        if (layoutSpec != layoutAttr.end()) {
          ops << ", ";
          srcs << ", ";
          dsts << ", ";
        }
      }
      ops << "),\n";
      ops.flush();
      srcs << "),\n";
      srcs.flush();
      dsts << ")";
      dsts.flush();

      output << "        constexpr auto " << outputTensorName << " = transform_tensor_descriptor(\n";
      output << "            " << inputTensorName << ",\n";
      output << operationSpec << srcDimSpec << dstDimSpec;
      output << ");\n\n";
    });

    // TBD get tuning parameters.
    //f.walk([&output](miopen::GridwiseGemmOp op) {
    //  // get op name.
    //  //output << "op name: " << op.getOperationName() << "\n";
    //  //op.dump();
    //});

    EmitHeaderEpilogue(output, gridwiseGemmArguments);
  }

  output.flush();
  return std::make_unique<llvm::StringRef>(resultStr);
}

std::unique_ptr<llvm::StringRef> mlir::translateModuleToMIOpenCpp(ModuleOp m) {
  std::string resultStr;
  llvm::raw_string_ostream output(resultStr);

  // Enumerate FuncOp instances inside the ModuleOp.
  for (auto f : m.getOps<FuncOp>()) {
    std::string layoutStr;
    llvm::SmallVector<std::string, 3> tensorDescs;

    // Obtain critical information from ModuleOp.
    ObtainModuleInfo(m, layoutStr, tensorDescs);

    int srcLayoutAttrCtr = 0;

    // Start emitting.
    EmitCppPreamble(output, layoutStr);

    f.walk([&output, &srcLayoutAttrCtr, &tensorDescs](miopen::TransformOp op) {
      // get source_layout attribute.
      auto srcLayoutAttr = op.getAttrOfType<ArrayAttr>("source_layout");
      if (srcLayoutAttr) {
        auto srcLayout = srcLayoutAttr.getValue();
        output << "    // ";
        EmitLayoutString(output, srcLayout, "", "", ",");
        output << '\n';

        EmitDimensionVariables(output, srcLayout);
        output << '\n';
        EmitStrideVariables(output, srcLayout);

        output << "    constexpr auto " << tensorDescs[srcLayoutAttrCtr++];
        output << " = make_native_tensor_descriptor(Sequence<";
        EmitLayoutString(output, srcLayout, "", "", ", ");
        output << ">{}, Sequence<";
        EmitLayoutString(output, srcLayout, "stride_", "", ", ");
        output << ">{});\n\n";
      }
    });

    EmitCppInterlude(output);

    // TBD get tuning parameters.
    //f.walk([&output](miopen::GridwiseGemmOp op) {
    //  // get op name.
    //  //output << "op name: " << op.getOperationName() << "\n";
    //  //op.dump();
    //});

    EmitCppEpilogue(output, layoutStr, tensorDescs);
  }

  output.flush();
  return std::make_unique<llvm::StringRef>(resultStr);
}

static TranslateFromMLIRRegistration
    toCpp("mlir-to-miopen-cpp", [](ModuleOp module, llvm::raw_ostream &output) {
      auto sourceCode = mlir::translateModuleToMIOpenCpp(module);
      if (!sourceCode)
        return failure();

      output << *sourceCode;
      return success();
    });

static TranslateFromMLIRRegistration
    toHeader("mlir-to-miopen-hpp", [](ModuleOp module, llvm::raw_ostream &output) {
      auto sourceCode = mlir::translateModuleToMIOpenHeader(module);
      if (!sourceCode)
        return failure();

      output << *sourceCode;
      return success();
    });

