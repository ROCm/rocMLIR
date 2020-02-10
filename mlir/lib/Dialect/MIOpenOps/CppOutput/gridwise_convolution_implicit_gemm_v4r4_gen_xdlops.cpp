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
#include "llvm/Support/YAMLTraits.h"

using namespace mlir;
using namespace llvm::yaml;

LLVM_YAML_IS_STRING_MAP(int)

namespace {

struct TunableParameters {
  void init() {
    params["CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK"] = 128;
    params["CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK"] = 128;
    params["CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK"] = 8;
    params["CK_PARAM_TUNABLE_BLOCK_SIZE"] = 256;
    params["CK_PARAM_GEMM_M_PER_WAVE"] = 128;
    params["CK_PARAM_GEMM_N_PER_WAVE"] = 128;
    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K"] = 4;
    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N"] = 4;
    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K"] = 2;
    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M"] = 4;
    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_N"] = 1;
    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_K"] = 1;
    params["CK_PARAM_TUNABLE_GEMM_C_THREAD_COPY_DATA_PER_ACCESS_N"] = 1;
    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N"] = 1;
    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M"] = 1;
  }
  void print(llvm::raw_ostream &os) {
    for (auto &kv : params) {
      os << " -D" << kv.first << "=" << kv.second;
    }
  }
  void printYAML(llvm::raw_ostream &os) {
    Output xout(os, nullptr, 0);
    xout << params;
    os.flush();
  }
  void readYAML(std::string &str) {
    params.clear();
    Input yin(str);
    yin >> params;
  }
  std::map<std::string, int> params;
};

static constexpr StringLiteral kVarName[3] = {"weight", "input", "output"};

static constexpr int kConv2DTensorDimension = 4;

static constexpr StringLiteral kCppPreamblePart1 = R"(
#include "common_header.hpp"
#include "ConstantTensorDescriptor_deprecated.hpp"
)";

static constexpr StringLiteral kCppPreamblePart2 = R"(
#include "float_types.h"

extern "C" __global__
)";

static constexpr StringLiteral kCppPreamblePart3 = R"(
        const FLOAT* const __restrict__ p_in_global,
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

    constexpr index_t GemmNPerBlock = CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK;
    constexpr index_t GemmMPerBlock = CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK;
    constexpr index_t GemmKPerBlock = CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK;

    // read params: dependent params
    constexpr index_t GridSize = CK_PARAM_DEPENDENT_GRID_SIZE;

    constexpr index_t LeftPadH = CK_PARAM_PROBLEM_LEFT_PAD_H;
    constexpr index_t LeftPadW = CK_PARAM_PROBLEM_LEFT_PAD_W;

    constexpr index_t RightPadH = CK_PARAM_PROBLEM_RIGHT_PAD_H;
    constexpr index_t RightPadW = CK_PARAM_PROBLEM_RIGHT_PAD_W;

    using LeftPads  = Sequence<LeftPadH, LeftPadW>;
    using RightPads = Sequence<RightPadH, RightPadW>;
)";

static constexpr StringLiteral kCppInterlude = R"(
    constexpr auto dir  = ImplicitGemmDirection::ForwardData;
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

    using GemmBBlockCopyThreadClusterArrangeOrder = Sequence<0, 1>; // [E, B]
    using GemmBBlockCopySrcAccessOrder            = Sequence<0, 1>; // [E, B]
    using GemmBBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, B]

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM =
        Sequence<GemmABlockCopyThreadSliceLengths_GemmK, GemmABlockCopyThreadSliceLengths_GemmM>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM =
        Sequence<GemmABlockCopyClusterLengths_GemmK, GemmABlockCopyClusterLengths_GemmM>;

    using GemmABlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using GemmABlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using GemmABlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N;
    constexpr index_t GemmABlockCopySrcDataPerRead_GemmM =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_N;
    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_K;
    constexpr index_t GemmCThreadCopyDataPerAccess_GemmN =
        CK_PARAM_TUNABLE_GEMM_C_THREAD_COPY_DATA_PER_ACCESS_N;

    constexpr auto GemmMPerWave                  = CK_PARAM_GEMM_M_PER_WAVE;
    constexpr auto GemmNPerWave                  = CK_PARAM_GEMM_N_PER_WAVE;
    constexpr auto GemmMWaves                    = GemmMPerBlock / GemmMPerWave;
    constexpr auto GemmNWaves                    = GemmNPerBlock / GemmNPerWave;
    constexpr index_t GemmThreadGemmDataPerReadM = 1;
    constexpr index_t GemmThreadGemmDataPerReadN = 1;
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
            LeftPads,
            RightPads,
            GemmNPerBlock,
            GemmMPerBlock,
            GemmKPerBlock,
            GemmMPerWave,
            GemmNPerWave,
            GemmMWaves,
            GemmNWaves,
            GemmThreadGemmDataPerReadM,
            GemmThreadGemmDataPerReadN,
            GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
            GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
            GemmBBlockCopyThreadClusterArrangeOrder,
            GemmBBlockCopySrcAccessOrder,
            GemmBBlockCopyDstAccessOrder,
            GemmBBlockCopySrcDataPerRead_GemmN,
            GemmBBlockCopyDstDataPerWrite_GemmN,
            GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
            GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
            GemmABlockCopyThreadClusterArrangeOrder,
            GemmABlockCopySrcAccessOrder,
            GemmABlockCopyDstAccessOrder,
            GemmABlockCopySrcDataPerRead_GemmK,
            GemmABlockCopySrcDataPerRead_GemmM,
            GemmCThreadCopyDataPerAccess_GemmN,
            dir>{};

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
          class LeftPads,
          class RightPads,
          index_t GemmNPerBlock,
          index_t GemmMPerBlock,
          index_t GemmKPerBlock,
          index_t GemmMPerWave,
          index_t GemmNPerWave,
          index_t GemmMWaves,
          index_t GemmNWaves,
          index_t GemmThreadGemmDataPerReadM,
          index_t GemmThreadGemmDataPerReadN,
          class GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
          class GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
          class GemmBBlockCopyThreadClusterArrangeOrder,
          class GemmBBlockCopySrcAccessOrder,
          class GemmBBlockCopyDstAccessOrder,
          index_t GemmBBlockCopySrcDataPerRead_GemmN,
          index_t GemmBBlockCopyDstDataPerWrite_GemmN,
          class GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
          class GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
          class GemmABlockCopyThreadClusterArrangeOrder,
          class GemmABlockCopySrcAccessOrder,
          class GemmABlockCopyDstAccessOrder,
          index_t GemmABlockCopySrcDataPerRead_GemmK,
          index_t GemmABlockCopySrcDataPerRead_GemmM,
          index_t GemmCThreadCopyDataPerAccess_GemmN,
          ImplicitGemmDirection conv_dir>
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
        constexpr auto gridwise_gemm = GridwiseGemmTransposedANormalBNormalC_v1_xdlops<
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
            GemmMWaves,
            GemmNWaves,
            GemmThreadGemmDataPerReadM,
            GemmThreadGemmDataPerReadN,
            GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
            GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
            GemmABlockCopyThreadClusterArrangeOrder,
            GemmABlockCopySrcAccessOrder,
            GemmABlockCopyDstAccessOrder,
            GemmABlockCopySrcDataPerRead_GemmK,
            GemmABlockCopySrcDataPerRead_GemmM,
            GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
            GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
            GemmBBlockCopyThreadClusterArrangeOrder,
            GemmBBlockCopySrcAccessOrder,
            GemmBBlockCopyDstAccessOrder,
            GemmBBlockCopySrcDataPerRead_GemmN,
            GemmBBlockCopyDstDataPerWrite_GemmN,
            GemmCThreadCopyDataPerAccess_GemmN>{};

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

} // anontmous namespace

std::unique_ptr<llvm::StringRef> mlir::translateModuleToMIOpenHeaderXDLOPS(ModuleOp m) {
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

    EmitHeaderEpilogue(output, gridwiseGemmArguments);
  }

  output.flush();
  return std::make_unique<llvm::StringRef>(resultStr);
}

std::unique_ptr<llvm::StringRef> mlir::translateModuleToMIOpenCppXDLOPS(ModuleOp m) {
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

    EmitCppEpilogue(output, layoutStr, tensorDescs);
  }

  output.flush();
  return std::make_unique<llvm::StringRef>(resultStr);
}

std::unique_ptr<llvm::StringRef> mlir::translateModuleToMIOpenCFlagsXDLOPS(ModuleOp m) {
  std::string resultStr;
  resultStr.reserve(4096);
  llvm::raw_string_ostream output(resultStr);

  for (auto f : m.getOps<FuncOp>()) {
    f.walk([&output](miopen::GridwiseGemmOp op) {
      // Emit flags immediately determined from convolution configs.
      auto inputLayoutAttr = op.getAttrOfType<ArrayAttr>("input_layout");
      auto inputDimensionAttr = op.getAttrOfType<ArrayAttr>("input_dimension");
      auto outputLayoutAttr = op.getAttrOfType<ArrayAttr>("output_layout");
      auto outputDimensionAttr = op.getAttrOfType<ArrayAttr>("output_dimension");
      auto filterLayoutAttr = op.getAttrOfType<ArrayAttr>("filter_layout");
      auto filterDimensionAttr = op.getAttrOfType<ArrayAttr>("filter_dimension");

      int64_t n = 0, k = 0, ho = 0, wo = 0;

      for (size_t i = 0; i < 4; ++i) {
        auto filterDim = filterLayoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();
        auto inputDim = inputLayoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();
        auto outputDim = outputLayoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();

        if (filterDim.str() == "k") {
          k = filterDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_K=" << k;
        } else if (filterDim.str() == "c") {
          output << " -DCK_PARAM_PROBLEM_C=" << filterDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getValue();
        } else if (filterDim.str() == "y") {
          output << " -DCK_PARAM_PROBLEM_Y=" << filterDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getValue();
        } else if (filterDim.str() == "x") {
          output << " -DCK_PARAM_PROBLEM_X=" << filterDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getValue();
        }

        if (inputDim.str() == "ni") {
          n = inputDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_N=" << n;
        } else if (inputDim.str() == "hi") {
          output << " -DCK_PARAM_PROBLEM_HI=" << inputDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getValue();
        } else if (inputDim.str() == "wi") {
          output << " -DCK_PARAM_PROBLEM_WI=" << inputDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getValue();
        }

        if (outputDim.str() == "ho") {
          ho = outputDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_HO=" << ho;
        } else if (outputDim.str() == "wo") {
          wo = outputDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_WO=" << wo;
        }
      }

      auto strideAttr = op.getAttrOfType<ArrayAttr>("strides");
      auto dilationAttr = op.getAttrOfType<ArrayAttr>("dilations");
      auto paddingAttr = op.getAttrOfType<ArrayAttr>("padding");
      output << " -DCK_PARAM_PROBLEM_CONV_STRIDE_H=" << strideAttr.getValue()[0].dyn_cast<IntegerAttr>().getValue();
      output << " -DCK_PARAM_PROBLEM_CONV_STRIDE_W=" << strideAttr.getValue()[1].dyn_cast<IntegerAttr>().getValue();
      output << " -DCK_PARAM_PROBLEM_CONV_DILATION_H=" << dilationAttr.getValue()[0].dyn_cast<IntegerAttr>().getValue();
      output << " -DCK_PARAM_PROBLEM_CONV_DILATION_W=" << dilationAttr.getValue()[1].dyn_cast<IntegerAttr>().getValue();
      output << " -DCK_PARAM_PROBLEM_IN_LEFT_PAD_H=" << paddingAttr.getValue()[0].dyn_cast<IntegerAttr>().getValue();
      output << " -DCK_PARAM_PROBLEM_IN_LEFT_PAD_W=" << paddingAttr.getValue()[1].dyn_cast<IntegerAttr>().getValue();
      output << " -DCK_PARAM_PROBLEM_IN_RIGHT_PAD_H=" << paddingAttr.getValue()[0].dyn_cast<IntegerAttr>().getValue();
      output << " -DCK_PARAM_PROBLEM_IN_RIGHT_PAD_W=" << paddingAttr.getValue()[1].dyn_cast<IntegerAttr>().getValue();

      // TBD: ditinguish between:
      //      - parameters truly need to be tuned.
      //      - parameters deducible via transformations.
      //      - parameters which have heuristic-based values.
      //      - parameters related to code generation.

      // TBD: be able to set data type.
      output << " -DMIOPEN_USE_FP32=1 -DMIOPEN_USE_FP16=0";

      // TBD: be able to set convolution direction.
      output << " -DCK_PARAM_PROBLEM_DIRECTION=0";
      output << " -DCK_PARAM_PROBLEM_CONV_DIRECTION_FORWARD=1";
      output << " -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_DATA=0";
      output << " -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_WEIGHT=0";

      // TBD: be able to set group convolution counts.
      output << " -DCK_PARAM_PROBLEM_CONV_GROUP_COUNTS=1";

      int64_t gemmMPerBlock = 128;
      int64_t gemmNPerBlock = 128;
      int64_t gemmKPerBlock = 8;
      int64_t gemmM = k;
      int64_t gemmN = n * ho * wo;
      int64_t gridSize = (gemmM / gemmMPerBlock) * (gemmN / gemmNPerBlock);

      output << " -DCK_PARAM_DEPENDENT_GRID_SIZE=" << gridSize;

      TunableParameters params;
      params.init();
      params.print(output);

      // Emit code-gen related parameters.
      output << " -DCK_PARAM_KPACK_LENGTH=1";
      output << " -DCK_USE_AMD_XDLOPS=1";
      output << " -DCK_USE_AMD_XDLOPS_INLINE_ASM=1";
      output << " -std=c++14";
      output << " -D__HIP_PLATFORM_HCC__=1";
    });
  }

  output.flush();
  return std::make_unique<llvm::StringRef>(resultStr);
}
