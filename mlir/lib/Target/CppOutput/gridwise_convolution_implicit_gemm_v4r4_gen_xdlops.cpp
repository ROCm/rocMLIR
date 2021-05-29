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

#include "mlir/Dialect/MIOpen/GridwiseConvCppOutputHelper.h"
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/MIOpenCPP.h"
#include "mlir/Translation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

namespace {
// result string to keep C++ source / header / flags emission.

static constexpr StringLiteral kCppPreamblePart1 = R"(
#include "common_header.hpp"
#include "ConstantTensorDescriptor_deprecated.hpp"
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

    // read params: tunable params
    constexpr index_t BlockSize = CK_PARAM_TUNABLE_BLOCK_SIZE;

    constexpr index_t GemmMPerBlock = CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK;
    constexpr index_t GemmNPerBlock = CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK;
    constexpr index_t GemmKPerBlock = CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK;

    // read params: dependent params
    constexpr index_t GridSize = CK_PARAM_DEPENDENT_GRID_SIZE;

    constexpr index_t LeftPadH = CK_PARAM_PROBLEM_LEFT_PAD_H;
    constexpr index_t LeftPadW = CK_PARAM_PROBLEM_LEFT_PAD_W;

    constexpr index_t RightPadH = CK_PARAM_PROBLEM_RIGHT_PAD_H;
    constexpr index_t RightPadW = CK_PARAM_PROBLEM_RIGHT_PAD_W;

    using InLeftPads  = Sequence<LeftPadH, LeftPadW>;
    using InRightPads = Sequence<RightPadH, RightPadW>;

)";

static constexpr StringLiteral kCppInterlude = R"(
    using ConvStrides   = Sequence<ConvStrideH, ConvStrideW>;
    using ConvDilations = Sequence<ConvDilationH, ConvDilationW>;

    constexpr index_t GemmBBlockCopyClusterLengths_GemmK =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K;
    constexpr index_t GemmBBlockCopyClusterLengths_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N;

    constexpr index_t GemmBBlockCopyThreadSliceLengths_GemmK =
        GemmKPerBlock / GemmBBlockCopyClusterLengths_GemmK;
    constexpr index_t GemmBBlockCopyThreadSliceLengths_GemmN =
        GemmNPerBlock / GemmBBlockCopyClusterLengths_GemmN;

    constexpr index_t GemmABlockCopyClusterLengths_GemmK =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K;
    constexpr index_t GemmABlockCopyClusterLengths_GemmM =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M;

    constexpr index_t GemmABlockCopyThreadSliceLengths_GemmK =
        GemmKPerBlock / GemmABlockCopyClusterLengths_GemmK;
    constexpr index_t GemmABlockCopyThreadSliceLengths_GemmM =
        GemmMPerBlock / GemmABlockCopyClusterLengths_GemmM;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN =
        Sequence<GemmBBlockCopyThreadSliceLengths_GemmK, GemmBBlockCopyThreadSliceLengths_GemmN>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN =
        Sequence<GemmBBlockCopyClusterLengths_GemmK, GemmBBlockCopyClusterLengths_GemmN>;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM =
        Sequence<GemmABlockCopyThreadSliceLengths_GemmK, GemmABlockCopyThreadSliceLengths_GemmM>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM =
        Sequence<GemmABlockCopyClusterLengths_GemmK, GemmABlockCopyClusterLengths_GemmM>;

    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N;
    constexpr index_t GemmABlockCopySrcDataPerRead_GemmM =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM;
    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM;

    constexpr auto GemmMPerWave                   = CK_PARAM_GEMM_M_PER_WAVE;
    constexpr auto GemmNPerWave                   = CK_PARAM_GEMM_N_PER_WAVE;
    constexpr index_t ThreadGemmDataPerRead_GemmM = 1;
    constexpr index_t ThreadGemmDataPerRead_GemmN = 1;
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
            GemmMPerWave,
            GemmNPerWave,
            ThreadGemmDataPerRead_GemmM,
            ThreadGemmDataPerRead_GemmN,
            GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
            GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
            GemmABlockCopySrcDataPerRead_GemmK,
            GemmABlockCopySrcDataPerRead_GemmM,
            GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
            GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
            GemmBBlockCopySrcDataPerRead_GemmN,
            GemmBBlockCopyDstDataPerWrite_GemmN>{};

    gridwise_conv.Run(p_in_global, p_wei_global, p_out_global);
}
)";

void EmitCppPreamble(llvm::raw_ostream &output, llvm::StringRef layoutStr) {
  output << kCppPreamblePart1;
// Between Preamble Part 1 and Part 2:
// #include "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_nchw_kcyx_nkhw_lds_double_buffer.hpp"
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
  output << "            decltype(" << tensorDescs[1] << "),\n";
  output << "            decltype(" << tensorDescs[0] << "),\n";
  output << "            decltype(" << tensorDescs[2] << "),\n";
  output << kCppEpiloguePart2;
}

static constexpr StringLiteral kHeaderPreamblePart1 = R"(
#ifndef CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_HPP
#define CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "blockwise_generic_tensor_slice_copy.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"
#include "gridwise_gemm_xdlops.hpp"
#include "convolution_common.hpp"
#include "implicitgemm_params.hpp"

namespace ck {

// B = merge(N, Ho, Wo)
template <index_t GridSize,
          index_t BlockSize,
          class Float,
          class AccDataType,
          class InGlobalDesc,
          class WeiGlobalDesc,
          class OutGlobalDesc,
          class ConvStrides,
          class ConvDilations,
          class InLeftPads,
          class InRightPads,
          index_t GemmMPerBlock,
          index_t GemmNPerBlock,
          index_t GemmKPerBlock,
          index_t GemmMPerWave,
          index_t GemmNPerWave,
          index_t GemmThreadGemmDataPerReadM,
          index_t GemmThreadGemmDataPerReadN,
          class GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
          class GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
          index_t GemmABlockCopySrcDataPerRead_GemmK,
          index_t GemmABlockCopySrcDataPerRead_GemmM,
          class GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
          class GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
          index_t GemmBBlockCopySrcDataPerRead_GemmN,
          index_t GemmBBlockCopyDstDataPerWrite_GemmN>
)";

static constexpr StringLiteral kHeaderPreamblePart2 = R"(
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {
)";

static constexpr StringLiteral kHeaderPreamblePart3 = R"(
        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

)";

static constexpr StringLiteral kHeaderEpiloguePart1 = R"(
        // GEMM
        constexpr auto gridwise_gemm = GridwiseGemmTransposedANormalBNormalCXdlops_v1<
            GridSize,
            BlockSize,
            Float,
            AccDataType,
)";

static constexpr StringLiteral kHeaderEpiloguePart2 = R"(
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmMPerWave,
            GemmNPerWave,
            GemmThreadGemmDataPerReadM,
            GemmThreadGemmDataPerReadN,
            GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
            GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
            Sequence<1, 0>,
            Sequence<1, 0>,
            Sequence<0, 1>,
)";

static constexpr StringLiteral kHeaderEpiloguePart3 = R"(
            GemmABlockCopySrcDataPerRead_GemmK,
            GemmABlockCopySrcDataPerRead_GemmM,
            GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
            GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
            Sequence<0, 1>,
            Sequence<0, 1>,
            Sequence<0, 1>,
)";

static constexpr StringLiteral kHeaderEpiloguePart4 = R"(
            GemmBBlockCopySrcDataPerRead_GemmN,
            GemmBBlockCopyDstDataPerWrite_GemmN,
            InMemoryDataOperation::Set>{};

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

void EmitHeaderEpilogue(llvm::raw_ostream &output, llvm::SmallDenseMap<int64_t, std::string> &args, bool filterGemmKVectorizable, bool inputGemmKVectorizable) {
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

// Between Part2 and Part3 emit which dimension the vectorization takes place for filter tensor.
// kcyx, kyxc, yxkc, ckyx: 0
// yxck, cyxk: 1
  if (filterGemmKVectorizable) {
    output << "            0,";
  } else {
    output << "            1,";
  }
  output << kHeaderEpiloguePart3;
// Between Part3 and Part4 emit which dimension the vectorization takes place for input tensor.
// nhwc, hwnc: 0
// chwn, hwcn: 1
// nchw, cnhw: non-vectorizable for now, set to 0, with vectorization width to 1.
  if (inputGemmKVectorizable) {
    output << "            0,";
  } else {
    output << "            1,";
  }
  output << kHeaderEpiloguePart4;
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

template<typename T>
void EmitInterleaveAsteriskArrayAttr(llvm::raw_ostream &os, mlir::ArrayAttr &arrayAttr) {
  EmitInterleaveArrayAttrWithSeparator<T>(os, arrayAttr, " * ");
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
      // Determine if this is the lowest layer.
      auto lowestLayerAttr = op->getAttrOfType<BoolAttr>("lowest_layer");
      if (lowestLayerAttr) {
        // get lower_layer_layout attribute.
        auto srcLayoutAttr = op->getAttrOfType<ArrayAttr>("lower_layer_layout");
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

} // anontmous namespace

std::string mlir::translateModuleFromMIOpenToHeaderXDLOPS(ModuleOp m) {
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
      // Determine if this is the lowest layer.
      auto lowestLayerAttr = op->getAttrOfType<BoolAttr>("lowest_layer");
      if (lowestLayerAttr) {
        // get lower_layer_layout attribute.
        auto srcLayoutAttr = op->getAttrOfType<ArrayAttr>("lower_layer_layout");
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
    f.walk([&output, &srcLayoutAttrCtr, &tensorDescs,
            &gridwiseGemmArguments](miopen::TransformOp op) {
      // get lower_layer_layout attribute.
      auto srcLayoutAttr = op->getAttrOfType<ArrayAttr>("lower_layer_layout");

      // get layout attribute.
      auto layoutAttr = op->getAttrOfType<ArrayAttr>("layout");
      std::string inputTensorName;
      std::string transformedInputTensorName;
      std::string outputTensorName;
      std::string operationSpec;
      std::string srcDimSpec;
      std::string dstDimSpec;
      llvm::raw_string_ostream ins(inputTensorName);
      llvm::raw_string_ostream pins(transformedInputTensorName);
      llvm::raw_string_ostream outs(outputTensorName);
      llvm::raw_string_ostream ops(operationSpec);
      llvm::raw_string_ostream srcs(srcDimSpec);
      llvm::raw_string_ostream dsts(dstDimSpec);

      // determine input and output tensor name.
      auto immLayoutAttr = op->getAttrOfType<ArrayAttr>("lower_layer_layout");
      auto outputLayoutAttr =
          op->getAttrOfType<ArrayAttr>("upper_layer_layout");
      if (srcLayoutAttr && srcLayoutAttrCtr == 0) {
        inputTensorName = tensorDescs[srcLayoutAttrCtr];
        outs << kVarName[srcLayoutAttrCtr] << "_";

        srcLayoutAttrCtr++;
      } else {
        // get lower_layer_layout attribute.
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
      auto gridwiseGemmArgPosAttr =
          op->getAttrOfType<IntegerAttr>("gridwise_gemm_argument_position");
      if (gridwiseGemmArgPosAttr) {
        gridwiseGemmArguments[gridwiseGemmArgPosAttr.getInt()] = outputTensorName;
      }

      ops << "            make_tuple(";
      srcs << "            make_tuple(";
      dsts << "            make_tuple(";

      // XXX see if we can get better than this.
      int convDilationCtr = 0;
      bool hasUnfoldTransform = false;

      for (auto layoutSpec = layoutAttr.begin(); layoutSpec != layoutAttr.end(); ) {
        if (auto layoutSpecDict = layoutSpec->dyn_cast<DictionaryAttr>()) {
          auto srcNames =
              layoutSpecDict.get("lower_layer_names").dyn_cast<ArrayAttr>();
          auto dstNames =
              layoutSpecDict.get("upper_layer_names").dyn_cast<ArrayAttr>();
          auto srcDims = layoutSpecDict.get("lower_layer_dimensions")
                             .dyn_cast<ArrayAttr>();
          auto dstDims = layoutSpecDict.get("upper_layer_dimensions")
                             .dyn_cast<ArrayAttr>();

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
                  << inputTensorName << ".GetLengths()[" << srcDims.getValue()[0].dyn_cast<IntegerAttr>().getInt() << "], "
                  << "Sequence<";
              EmitInterleaveCommaArrayAttr<StringAttr>(ops, dstNames);
              if (convDilationCtr == 0) {
                ops << ">, Sequence<ConvDilationH, ConvDilationH, 0>>{}";
                convDilationCtr++;
              } else {
                ops << ">, Sequence<ConvDilationW, ConvDilationW, 0>>{}";
              }
            } else if (transform.getValue() == "Unfold") {
              hasUnfoldTransform = true;
              ops << "PassThrough<";
              EmitInterleaveAsteriskArrayAttr<StringAttr>(ops, srcNames);
              ops << ">{}";
            }
            srcs << "Sequence<";
            if (transform.getValue() == "Unfold") {
              pins << "unfold_tensor_descriptor(" << inputTensorName << ", "
                   << "Number<" << srcDims.getValue()[0].dyn_cast<IntegerAttr>().getInt() << ">{}, "
                   << "Number<" << srcDims.getValue()[srcDims.size() - 1].dyn_cast<IntegerAttr>().getInt() << ">{})";
              pins.flush();
              srcs << srcDims.getValue()[0].dyn_cast<IntegerAttr>().getInt();
            } else {
              if (hasUnfoldTransform) {
                // XXX see if we can do better than this.
                if (srcDims.getValue()[0].dyn_cast<IntegerAttr>().getInt() == 0) {
                  srcs << "0";
                } else {
                  srcs << "1";
                }
              } else {
                EmitInterleaveCommaArrayAttr<IntegerAttr>(srcs, srcDims);
              }
            }
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
      if (hasUnfoldTransform) {
        output << "            " << transformedInputTensorName << ",\n";
      } else {
        output << "            " << inputTensorName << ",\n";
      }
      output << operationSpec << srcDimSpec << dstDimSpec;
      output << ");\n\n";
    });

    bool filterGemmKVectorizable = false, inputGemmKVectorizable = false;
    f.walk([&filterGemmKVectorizable,
            &inputGemmKVectorizable](miopen::GridwiseGemmOp op) {
      auto filterLayoutAttr = op->getAttrOfType<ArrayAttr>("filter_layout");
      auto inputLayoutAttr = op->getAttrOfType<ArrayAttr>("input_layout");

      size_t dimKF, dimNI, dimCI;
      for (size_t i = 0; i < 5; ++i) {
        auto filterDim = filterLayoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();

        // Since XDLOPS cpp backend only supports forward pass so not all
        // variables are used
        if (filterDim.str() == "k") {
          dimKF = i;
        }

        auto inputDim = inputLayoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();
        if (inputDim.str() == "ni") {
          dimNI = i;
        } else if (inputDim.str() == "ci") {
          dimCI = i;
        }
      }

      // Filter tensor.
      // Find the fastest changing dimension.
      if (dimKF == 4) {
        // When K is the fastest changing dimension,
        // gemmM dimension is vectorizable.
        // vectorization width depending on length of K.

        // gemmK dimension non-vectorizable.
        filterGemmKVectorizable = false;
      } else {
        // gemmK dimension vectorizable,
        // depending on which among C, Y, X be the fastest changing dimension.
        filterGemmKVectorizable = true;
        // gemmM dimension non-vectorizable.
      }

      // Input tensor.
      // Find the fastest changing dimension.
      if (dimNI == 4) {
        // When N is the fastest changing dimension,
        // gemmN dimension is vectorizable.
        // vectorization width depending on length of N.

        // gemmK dimension non-vectorizable.
        inputGemmKVectorizable = false;
      } else if (dimCI == 4) {
        // When C is the fastest changing dimension,
        // gemmK dimension vectorizable.
        // vectorization width depending on length of C.
        inputGemmKVectorizable = true;

        // gemmN dimension non-vectorizable.
      }
    });

    EmitHeaderEpilogue(output, gridwiseGemmArguments, filterGemmKVectorizable, inputGemmKVectorizable);
  }

  output.flush();
  return resultStr;
}

std::string mlir::translateModuleFromMIOpenToCppXDLOPS(ModuleOp m) {
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
      // Determine if this is the lowest layer.
      auto lowestLayerAttr = op->getAttrOfType<BoolAttr>("lowest_layer");
      if (lowestLayerAttr) {
        // get lower_layer_layout attribute.
        auto srcLayoutAttr = op->getAttrOfType<ArrayAttr>("lower_layer_layout");
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

    EmitCppEpilogue(output, layoutStr, tensorDescs);
  }

  output.flush();
  return resultStr;
}

std::string mlir::translateModuleFromMIOpenToCFlagsXDLOPS(ModuleOp m) {
  std::string resultStr;
  llvm::raw_string_ostream output(resultStr);

  for (auto f : m.getOps<FuncOp>()) {
    f.walk([&output](miopen::GridwiseGemmOp op) {
      ConvolutionContext ctx = populateConvContext(op);
      std::map<std::string, int> parameters;

      // Filter
      parameters["CK_PARAM_PROBLEM_K"] = ctx.dimIndexVal["k"].second;
      parameters["CK_PARAM_PROBLEM_C"] = ctx.dimIndexVal["c"].second;
      parameters["CK_PARAM_PROBLEM_Y"] = ctx.dimIndexVal["y"].second;
      parameters["CK_PARAM_PROBLEM_X"] = ctx.dimIndexVal["x"].second;
      // Input
      parameters["CK_PARAM_PROBLEM_N"] = ctx.dimIndexVal["ni"].second;
      parameters["CK_PARAM_PROBLEM_HI"] = ctx.dimIndexVal["hi"].second;
      parameters["CK_PARAM_PROBLEM_WI"] = ctx.dimIndexVal["wi"].second;
      // Output
      parameters["CK_PARAM_PROBLEM_HO"] = ctx.dimIndexVal["ho"].second;
      parameters["CK_PARAM_PROBLEM_WO"] = ctx.dimIndexVal["wo"].second;
      // Stride
      parameters["CK_PARAM_PROBLEM_CONV_STRIDE_H"] = ctx.strideVal[0];
      parameters["CK_PARAM_PROBLEM_CONV_STRIDE_W"] = ctx.strideVal[1];
      // Dilation
      parameters["CK_PARAM_PROBLEM_CONV_DILATION_H"] = ctx.dilationVal[0];
      parameters["CK_PARAM_PROBLEM_CONV_DILATION_W"] = ctx.dilationVal[1];
      // Padding
      parameters["CK_PARAM_PROBLEM_LEFT_PAD_H"] = ctx.paddingVal[0];
      parameters["CK_PARAM_PROBLEM_LEFT_PAD_W"] = ctx.paddingVal[1];
      parameters["CK_PARAM_PROBLEM_RIGHT_PAD_H"] = ctx.paddingVal[2];
      parameters["CK_PARAM_PROBLEM_RIGHT_PAD_W"] = ctx.paddingVal[3];

      PopulateParamsXDL populateParams;
      InitParamsXDL validParams;
      DerivedParams gemmADerivedParam;
      DerivedParams gemmBDerivedParam;
      int64_t blockSize = 0;
      int64_t gridSize = 0;
      (void)populateParams.paramsFromCtx(ctx, 0, validParams, gemmADerivedParam,
                                         gemmBDerivedParam, blockSize,
                                         gridSize);

      parameters["CK_PARAM_TUNABLE_BLOCK_SIZE"] = blockSize;
      parameters["CK_PARAM_DEPENDENT_GRID_SIZE"] = gridSize;

      // parameters truly tunable.
      parameters["CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK"] =
          validParams.gemmMPerBlock;
      parameters["CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK"] =
          validParams.gemmNPerBlock;
      parameters["CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK"] =
          validParams.gemmKPerBlock;
      parameters["CK_PARAM_GEMM_M_PER_WAVE"] = validParams.gemmMPerWave;
      parameters["CK_PARAM_GEMM_N_PER_WAVE"] = validParams.gemmNPerWave;

      parameters["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K"] =
          gemmADerivedParam.clusterLenGemmPos1;
      parameters["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M"] =
          gemmADerivedParam.clusterLenGemmPos2;
      parameters["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM"] =
          gemmADerivedParam.srcDataPerRead;
      parameters
          ["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M"] =
              gemmADerivedParam.dstDataPerWrite;

      parameters["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K"] =
          gemmBDerivedParam.clusterLenGemmPos1;
      parameters["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N"] =
          gemmBDerivedParam.clusterLenGemmPos2;
      parameters["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM"] =
          gemmBDerivedParam.srcDataPerRead;
      parameters
          ["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N"] =
              gemmBDerivedParam.dstDataPerWrite;

      // TBD: be able to set data type.
      parameters["MIOPEN_USE_FP32"] = 1;
      parameters["MIOPEN_USE_FP16"] = 0;
      parameters["MIOPEN_USE_BFP16"] = 0;

      parameters["CK_PARAM_PROBLEM_CONV_DIRECTION_FORWARD"] = 1;
      parameters["CK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_DATA"] = 0;
      parameters["CK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_WEIGHT"] = 0;

      // Emit code-gen related parameters.
      parameters["CK_USE_AMD_XDLOPS"] = 1;
      parameters["CK_USE_AMD_XDLOPS_INLINE_ASM"] = 1;
      parameters["__HIP_PLATFORM_HCC__"] = 1;

      auto printParams = [&parameters](llvm::raw_ostream &os) {
        for (auto kv : parameters) {
          os << " -D" << kv.first << "=" << kv.second;
        }
      };

      // Print out the tunable parameters.
      printParams(output);

      output << " -std=c++14";
    });
  }

  output.flush();
  return resultStr;
}
