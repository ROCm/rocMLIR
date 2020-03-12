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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
// result string to keep C++ source / header / flags emission.
std::string resultStr;

class TunableParameters : public TunableParametersBase {
public:
  TunableParameters() : TunableParametersBase("gridwise_convolution_implicit_gemm_v4r4.yaml") {}

  virtual void customInit() override {
    // parameters truly tunable.
    params["CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK"] = 128;
    params["CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK"] = 128;
    params["CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK"] = 8;
    params["CK_PARAM_TUNABLE_GEMM_M_PER_THREAD"] = 4;
    params["CK_PARAM_TUNABLE_GEMM_N_PER_THREAD"] = 4;
    params["CK_PARAM_TUNABLE_BLOCK_SIZE"] = 256;

    // parameters fixed.
    params["CK_PARAM_TUNABLE_GEMM_M_LEVEL0_CLUSTER"] = 4;
    params["CK_PARAM_TUNABLE_GEMM_N_LEVEL0_CLUSTER"] = 4;
    params["CK_PARAM_TUNABLE_GEMM_M_LEVEL1_CLUSTER"] = 4;
    params["CK_PARAM_TUNABLE_GEMM_N_LEVEL1_CLUSTER"] = 4;

    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M"] = 1;
    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N"] = 1;

    // parameters vary per data layout.
    // specify the most conservative parameters first.
    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K"] = 2;
    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M"] = 128;
    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM"] = 1;

    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K"] = 2;
    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N"] = 128;
    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM"] = 1;

    params["CK_PARAM_TUNABLE_GEMM_C_THREAD_COPY_DST_DATA_PER_WRITE_GEMM_N1"] = 1;
  }
};

static constexpr StringLiteral kVarName[3] = {"weight", "input", "output"};
static constexpr StringLiteral kVarArgName[3] = {"p_wei_global", "p_in_global",
                                                 "p_out_global"};

static constexpr int kConv2DTensorDimension = 4;

static constexpr StringLiteral kCppPreamblePart1 = R"(
#include "common_header.hpp"
)";

static constexpr StringLiteral kCppPreamblePart2 = R"(
#include "float_types.h"

extern "C" __global__
)";

static constexpr StringLiteral kCppPreamblePart3Format = R"(
        (const FLOAT* const __restrict__ %s,
        const FLOAT* const __restrict__ %s,
        FLOAT* const __restrict__ %s)
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

static constexpr StringLiteral kCppInterludeFormat = R"(
    using ConvStrides   = Sequence<ConvStrideH, ConvStrideW>;
    using ConvDilations = Sequence<ConvDilationH, ConvDilationW>;

    using InLeftPads  = Sequence<InLeftPadH, InLeftPadW>;
    using InRightPads = Sequence<InRightPadH, InRightPadW>;

    // read and calculate tuning parameter
    constexpr index_t GemmMPerThreadSubC = CK_PARAM_TUNABLE_GEMM_M_PER_THREAD;
    constexpr index_t GemmNPerThreadSubC = CK_PARAM_TUNABLE_GEMM_N_PER_THREAD;
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

    constexpr index_t GemmABlockCopySrcDataPerRead_%s =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_%s;

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
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM;

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

static constexpr StringLiteral kCppEpiloguePart2Format = R"(
        ConvStrides,
        ConvDilations,
        InLeftPads,
        InRightPads,
        GemmMPerBlock,
        GemmNPerBlock,
        GemmKPerBlock,
        GemmMPerThreadSubC,
        GemmNPerThreadSubC,
        GemmKPerThreadLoop,
        GemmMLevel0Cluster,
        GemmNLevel0Cluster,
        GemmMLevel1Cluster,
        GemmNLevel1Cluster,
        GemmThreadGemmDataPerReadM,
        GemmThreadGemmDataPerReadN,
        GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
        GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
        GemmABlockCopySrcDataPerRead_%s,
        GemmABlockCopyDstDataPerWrite_GemmM,
        GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
        GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
        GemmBBlockCopySrcDataPerRead_GemmN,
        GemmBBlockCopyDstDataPerWrite_GemmN,
        GemmCThreadCopyDstDataPerWrite_GemmN1>{};

    gridwise_conv.Run(p_in_global, p_wei_global, p_out_global);
}
)";

static constexpr StringLiteral kGemmNameABlockCopySrcDataPerRead[] = {
    "GemmK", // Conv2DOpType
    "GemmM", // Conv2DBwdDataOpType
};

void EmitCppPreamble(llvm::raw_ostream &output, miopen::ConvOpType opType) {
  output << kCppPreamblePart1;
// Between Preamble Part 1 and Part 2:
// #include "gridwise_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw.hpp"
  if (opType == miopen::ConvOpType::Conv2DOpType) {
    output << R"(#include "gridwise_convolution_implicit_gemm_v4r4_)";
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    output
        << R"(#include "gridwise_convolution_backward_data_implicit_gemm_v1r1_)";
  }

  // Change to fixed "mlir".
  output << "mlir" << R"(.hpp")";

  output << kCppPreamblePart2;
// Between Preamble Part 2 and Par 3:
//    __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void gridwise_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw(
  if (opType == miopen::ConvOpType::Conv2DOpType) {
    output << R"(
    __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void gridwise_convolution_implicit_gemm_v4r4_)";
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    output << R"(
    __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void gridwise_convolution_backward_data_implicit_gemm_v1r1_)";
  }
  // Change to fixed "mlir".
  output << "mlir";

  std::string argPInGlobal(kVarArgName[1]);
  std::string argPOutGlobal(kVarArgName[2]);
  std::string argPWeiGlobal(kVarArgName[0]);
  if (opType == miopen::ConvOpType::Conv2DOpType) {
    output << llvm::format(kCppPreamblePart3Format.data(), argPInGlobal.c_str(),
                           argPWeiGlobal.c_str(), argPOutGlobal.c_str());
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    output << llvm::format(kCppPreamblePart3Format.data(),
                           argPOutGlobal.c_str(), argPWeiGlobal.c_str(),
                           argPInGlobal.c_str());
  }
}

void EmitCppInterlude(llvm::raw_ostream &output, miopen::ConvOpType opType) {
  std::string gemmNameABlockCopySrcDataPerRead;
  if (opType == miopen::ConvOpType::Conv2DOpType) {
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[0];
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[1];
  }
  output << llvm::format(kCppInterludeFormat.data(),
                         gemmNameABlockCopySrcDataPerRead.c_str(),
                         gemmNameABlockCopySrcDataPerRead.c_str());
}

void EmitCppEpilogue(llvm::raw_ostream &output,
                     llvm::SmallVector<std::string, 3> tensorDescs,
                     miopen::ConvOpType opType) {
  // Before Part1:
  //    constexpr auto gridwise_conv =
  //    GridwiseConvolutionImplicitGemm_v4r4_nchw_kcyx_nkhw
  if (opType == miopen::ConvOpType::Conv2DOpType) {
    output << R"(
    constexpr auto gridwise_conv = GridwiseConvolutionImplicitGemm_v4r4_)";
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    output << R"(
    constexpr auto gridwise_conv = GridwiseConvolutionBackwardDataImplicitGemm_v1r1_)";
  }

  // Change to fixed "mlir".
  output << "mlir";

  output << kCppEpiloguePart1;
  // Between Part1 and Part2:
  //        decltype(in_nchw_desc),
  //        decltype(wei_kcyx_desc),
  //        decltype(out_nkhw_desc),
  output << "        decltype(" << tensorDescs[1] << "),\n";
  output << "        decltype(" << tensorDescs[0] << "),\n";
  output << "        decltype(" << tensorDescs[2] << "),";

  std::string gemmNameABlockCopySrcDataPerRead;
  if (opType == miopen::ConvOpType::Conv2DOpType) {
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[0];
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[1];
  }
  output << llvm::format(kCppEpiloguePart2Format.data(),
                         gemmNameABlockCopySrcDataPerRead.c_str(),
                         gemmNameABlockCopySrcDataPerRead.c_str());
}

static constexpr StringLiteral kHeaderPreamblePart1Format = R"(
#ifndef CK_GRIDWISE_CONVOLUTION_%s_HPP
#define CK_GRIDWISE_CONVOLUTION_%s_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm.hpp"

namespace ck {

// GemmM = %s
// GemmN = N * Ho * Wo
// GemmK = %s
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
          index_t GemmKPerThreadLoop,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t GemmThreadGemmDataPerReadM,
          index_t GemmThreadGemmDataPerReadN,
          typename GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
          typename GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
          index_t GemmABlockCopySrcDataPerRead_%s,
          index_t GemmABlockCopyDstDataPerWrite_GemmM,
          typename GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
          typename GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
          index_t GemmBBlockCopySrcDataPerRead_GemmN,
          index_t GemmBBlockCopyDstDataPerWrite_GemmN,
          index_t GemmCThreadCopyDstDataPerWrite_GemmN1>
)";

static constexpr StringLiteral kHeaderPreamblePart2Forward = R"(
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {
)";

static constexpr StringLiteral kHeaderPreamblePart2BwdData = R"(
{
    __device__ void Run(Float* __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        const Float* const __restrict__ p_out_global) const
    {
)";

static constexpr StringLiteral kHeaderPreamblePart3 = R"(
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];
)";

static constexpr StringLiteral kHeaderEpilogueInMemOp = R"(
        // \todo there are more combinations of Y, ConvDilationH and ConvStrideH that don't need
        // atomic, find out all of them
        constexpr bool not_need_atomic = (ConvStrideH >= ConvDilationH * (Y - 1) + 1) and
                                         (ConvStrideW >= ConvDilationW * (X - 1) + 1);
        constexpr auto in_memory_op = 
            not_need_atomic ? InMemoryDataOperation::Set : In MemoryOperation::AtomicAdd;
)";

static constexpr StringLiteral kHeaderEpiloguePart1 = R"(
        // GEM
        constexpr auto gridwise_gemm =
            GridwiseGemmTransposedANormalBNormalC_v1<GridSize,
                                                     BlockSize,
                                                     Float,
                                                     AccFloat,)";

static constexpr StringLiteral kHeaderEpiloguePart2 = R"(
                                                     %s,
                                                     GemmMPerBlock,
                                                     GemmNPerBlock,
                                                     GemmKPerBlock,
                                                     GemmMPerThreadSubC,
                                                     GemmNPerThreadSubC,
                                                     GemmKPerThreadLoop,
                                                     GemmMLevel0Cluster,
                                                     GemmNLevel0Cluster,
                                                     GemmMLevel1Cluster,
                                                     GemmNLevel1Cluster,
                                                     GemmThreadGemmDataPerReadM,
                                                     GemmThreadGemmDataPerReadN,
                                                     GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
                                                     GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
                                                     Sequence<1, 0>,
                                                     Sequence<1, 0>,
)";

static constexpr StringLiteral kHeaderEpiloguePart3Format = R"(
                                                     GemmABlockCopySrcDataPerRead_%s,
                                                     GemmABlockCopyDstDataPerWrite_GemmM,
                                                     GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
                                                     GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
                                                     %s,
                                                     %s,
)";

static constexpr StringLiteral kHeaderEpiloguePart4 = R"(
                                                     GemmBBlockCopySrcDataPerRead_GemmN,
                                                     GemmBBlockCopyDstDataPerWrite_GemmN,
                                                     Sequence<0, 1, 2, 3>,
                                                     3,
                                                     GemmCThreadCopyDstDataPerWrite_GemmN1>{};

        gridwise_gemm.Run(%s, %s, %s);
    }
};

} // namespace ck
#endif
)";

void EmitHeaderPreamble(llvm::raw_ostream &output,
                        llvm::SmallVector<std::string, 3> &tensorDescs,
                        miopen::ConvOpType opType) {
  std::string headerIncludeGuard;
  std::string commentGemmM;
  std::string commentGemmK;
  std::string gemmNameABlockCopySrcDataPerRead;
  if (opType == miopen::ConvOpType::Conv2DOpType) {
    headerIncludeGuard = "IMPLICIT_GEMM_V4R4";
    commentGemmM = "K";
    commentGemmK = "C * Y * X";
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[0];
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    headerIncludeGuard = "BACKWARD_DATA_IMPLICIT_GEMM_V1R1";
    commentGemmM = "C * Y * X";
    commentGemmK = "K";
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[1];
  }
  output << llvm::format(kHeaderPreamblePart1Format.data(),
                         headerIncludeGuard.c_str(), headerIncludeGuard.c_str(),
                         commentGemmM.c_str(), commentGemmK.c_str(),
                         gemmNameABlockCopySrcDataPerRead.c_str());

  if (opType == miopen::ConvOpType::Conv2DOpType) {
    output << R"(struct GridwiseConvolutionImplicitGemm_v4r4_)";
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    output << R"(struct GridwiseConvolutionBackwardDataImplicitGemm_v1r1_)";
  }

  // Change to fixed "mlir".
  output << "mlir";

  if (opType == miopen::ConvOpType::Conv2DOpType) {
    output << kHeaderPreamblePart2Forward;
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    output << kHeaderPreamblePart2BwdData;
  }
  output << kHeaderPreamblePart3;

  output << R"(
        constexpr auto )"
         << tensorDescs[0] << " = WeiGlobalDesc{};";
  output << R"(
        constexpr auto )"
         << tensorDescs[1] << " = InGlobalDesc{};";
  output << R"(
        constexpr auto )"
         << tensorDescs[2] << " = OutGlobalDesc{};";
  output << '\n';
}

void EmitHeaderEpilogue(llvm::raw_ostream &output,
                        llvm::SmallDenseMap<int64_t, std::string> &args,
                        bool input1GemmKVectorizable,
                        bool input2GemmKVectorizable,
                        miopen::ConvOpType opType) {
  if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    output << kHeaderEpilogueInMemOp;
  }

  output << kHeaderEpiloguePart1;
// Between Part1 and Part2 emit:
//                                                   decltype(wei_e_k_global_desc),
//                                                   decltype(in_e_b_global_desc),
//                                                   decltype(out_k_b_global_desc),
  for (unsigned i = 0; i < args.size(); ++i) {
    output << R"(
                                                     decltype()" << args[i] << "),";
  }
  std::string inMemOp;
  if (opType == miopen::ConvOpType::Conv2DOpType) {
    inMemOp = "InMemoryDataOperation::Set";
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    inMemOp = "in_memory_op";
  }
  output << llvm::format(kHeaderEpiloguePart2.data(), inMemOp.c_str());

  // Between Part2 and Part3 emit which dimension the vectorization takes place
  // for filter tensor. kcyx, kyxc, yxkc, ckyx: 0 yxck, cyxk: 1
  if (input1GemmKVectorizable) {
    output << "                                                     0,";
  } else {
    output << "                                                     1,";
  }

  std::string gemmNameABlockCopySrcDataPerRead;
  std::string gemmHeaderEpiloguePart3Sequence;
  if (opType == miopen::ConvOpType::Conv2DOpType) {
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[0];
    gemmHeaderEpiloguePart3Sequence = "Sequence<0, 1>";
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[1];
    gemmHeaderEpiloguePart3Sequence = "Sequence<1, 0>";
  }
  output << llvm::format(kHeaderEpiloguePart3Format.data(),
                         gemmNameABlockCopySrcDataPerRead.c_str(),
                         gemmHeaderEpiloguePart3Sequence.c_str(),
                         gemmHeaderEpiloguePart3Sequence.c_str());
  // Between Part3 and Part4 emit which dimension the vectorization takes place
  // for input tensor. nhwc, hwnc: 0 chwn, hwcn: 1 nchw, cnhw: non-vectorizable
  // for now, set to 0, with vectorization width to 1.
  if (input2GemmKVectorizable) {
    output << "                                                     0,";
  } else {
    output << "                                                     1,";
  }

  std::string argPInGlobal(kVarArgName[1]);
  std::string argPOutGlobal(kVarArgName[2]);
  std::string argPWeiGlobal(kVarArgName[0]);
  if (opType == miopen::ConvOpType::Conv2DOpType) {
    output << llvm::format(kHeaderEpiloguePart4.data(), argPWeiGlobal.c_str(),
                           argPInGlobal.c_str(), argPOutGlobal.c_str());
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    output << llvm::format(kHeaderEpiloguePart4.data(), argPWeiGlobal.c_str(),
                           argPOutGlobal.c_str(), argPInGlobal.c_str());
  }
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

void ObtainConvDirection(FuncOp &f, miopen::ConvOpType &opType) {
  f.walk([&opType](miopen::GridwiseGemmOp op) {
    auto kernel_algorithm = op.getAttrOfType<StringAttr>("kernel_algorithm");
    if (kernel_algorithm.getValue().find(StringRef("backward_data")) !=
        StringRef::npos) {
      opType = miopen::ConvOpType::Conv2DBwdDataOpType;
    } else if (kernel_algorithm.getValue().find(StringRef("backward_weight")) !=
               StringRef::npos) {
      opType = miopen::ConvOpType::Conv2DBwdWeightOpType;
    } else {
      opType = miopen::ConvOpType::Conv2DOpType;
    }
  });
}

void ObtainModuleInfo(ModuleOp &m,
                      llvm::SmallVector<std::string, 3> &tensorDescs,
                      miopen::ConvOpType &opType) {
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

    // First iteration. Construct tensor descriptor names.
    f.walk([&srcLayoutAttrCtr, &tensorDescs](miopen::TransformOp op) {
      // get source_layout attribute.
      auto srcLayoutAttr = op.getAttrOfType<ArrayAttr>("source_layout");
      if (srcLayoutAttr) {
        auto srcLayout = srcLayoutAttr.getValue();

        // Prepare tensor descriptor variable name.
        std::string desc{kVarName[srcLayoutAttrCtr++]};
        llvm::raw_string_ostream os(desc);
        os << "_";
        EmitLayoutString(os, srcLayout, "", "", "_");
        os << "_desc";
        os.flush();
        tensorDescs.push_back(desc);
      }
    });

    // Second iteration. Determine convolution direction.
    ObtainConvDirection(f, opType);
  }
}

void obtainHeaderVectorizableArgs(bool &input1GemmKVectorizable,
                                  bool &input2GemmKVectorizable, FuncOp &f,
                                  miopen::ConvOpType opType) {
  f.walk([&input1GemmKVectorizable, &input2GemmKVectorizable,
          opType](miopen::GridwiseGemmOp op) {
    auto filterLayoutAttr = op.getAttrOfType<ArrayAttr>("filter_layout");
    auto inputLayoutAttr = op.getAttrOfType<ArrayAttr>("input_layout");
    auto outputLayoutAttr = op.getAttrOfType<ArrayAttr>("output_layout");

    if (opType == miopen::ConvOpType::Conv2DOpType) {
      // Only needs to determine filter/input
      size_t dimKF, dimCI;
      for (size_t i = 0; i < 4; ++i) {
        auto filterDim =
            filterLayoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();
        if (filterDim.str() == "k") {
          dimKF = i;
        }
        auto inputDim =
            inputLayoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();
        if (inputDim.str() == "ci") {
          dimCI = i;
        }
      }

      // For filter tensor:
      // When K is not the fastest changing dimension,
      // gemmK dimension is vectorizable, gemmM is not, and vice versa.
      // Vectorization width depending on which among C, Y, X be the fastest
      // changing dimension.
      if (dimKF == 3) {
        input1GemmKVectorizable = false;
      } else {
        input1GemmKVectorizable = true;
      }

      // For input tensor.
      // When C is the fastest changing dimension,
      // gemmK dimension is vectorizable, gemmN is not, and vice versa.
      // Vectorization width depending on length of C.
      if (dimCI == 3) {
        input2GemmKVectorizable = true;
      } else {
        input2GemmKVectorizable = false;
      }
    } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
      // Only needs to determine filter/output
      size_t dimKF, dimKO;
      for (size_t i = 0; i < 4; ++i) {
        auto filterDim =
            filterLayoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();
        if (filterDim.str() == "k") {
          dimKF = i;
        }
        auto outputDim =
            outputLayoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();
        if (outputDim.str() == "ko") {
          dimKO = i;
        }
      }

      // For filter tensor:
      // When K is the fastest changing dimension(3),
      // gemmK dimension is vectorizable, gemmM is not, and vice versa.
      // Vectorization width depending on length of K.
      if (dimKF == 3) {
        input1GemmKVectorizable = true;
      } else {
        input1GemmKVectorizable = false;
      }

      // For output tensor.
      // When K is the fastest changing dimension(3),
      // gemmK dimension is vectorizable, gemmN is not, and vice versa.
      // Vectorization width depending on length of K.
      if (dimKO == 3) {
        input2GemmKVectorizable = true;
      } else {
        input2GemmKVectorizable = false;
      }
    }
  });
}
}

std::unique_ptr<llvm::StringRef> mlir::translateModuleToMIOpenHeader(ModuleOp m) {
  llvm::raw_string_ostream output(resultStr);

  // Enumerate FuncOp instances inside the ModuleOp.
  for (auto f : m.getOps<FuncOp>()) {
    miopen::ConvOpType opType;
    llvm::SmallVector<std::string, 3> tensorDescs;
    llvm::SmallDenseMap<int64_t, std::string> gridwiseGemmArguments;

    // Obtain critical information from ModuleOp.
    ObtainModuleInfo(m, tensorDescs, opType);

    int srcLayoutAttrCtr = 0;

    // Start emitting.
    EmitHeaderPreamble(output, tensorDescs, opType);

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
            } else if (transform.getValue() == "Merge" || transform.getValue() == "Unfold") {
              // XXX treat Unfold just like Merge on non-XDLOPS path.
              ops << "Merge" << "<"
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

    bool input1GemmKVectorizable = false, input2GemmKVectorizable = false;
    obtainHeaderVectorizableArgs(input1GemmKVectorizable,
                                 input2GemmKVectorizable, f, opType);
    EmitHeaderEpilogue(output, gridwiseGemmArguments, input1GemmKVectorizable,
                       input2GemmKVectorizable, opType);
  }

  output.flush();
  return std::make_unique<llvm::StringRef>(resultStr);
}

std::unique_ptr<llvm::StringRef> mlir::translateModuleToMIOpenCpp(ModuleOp m) {
  llvm::raw_string_ostream output(resultStr);

  // Enumerate FuncOp instances inside the ModuleOp.
  for (auto f : m.getOps<FuncOp>()) {
    miopen::ConvOpType opType;
    llvm::SmallVector<std::string, 3> tensorDescs;

    // Obtain critical information from ModuleOp.
    ObtainModuleInfo(m, tensorDescs, opType);

    int srcLayoutAttrCtr = 0;

    // Start emitting.
    EmitCppPreamble(output, opType);

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

    EmitCppInterlude(output, opType);

    EmitCppEpilogue(output, tensorDescs, opType);
  }

  output.flush();
  return std::make_unique<llvm::StringRef>(resultStr);
}

std::unique_ptr<llvm::StringRef> mlir::translateModuleToMIOpenCFlags(ModuleOp m) {
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

      int64_t n = 0, k = 0, ho = 0, wo = 0, hi = 0, wi = 0;
      int64_t c = 0, y = 0, x = 0;

      size_t dimKF, dimCF, dimYF, dimXF;
      size_t dimNO, dimKO, dimHO, dimWO;
      size_t dimNI, dimCI;

      for (size_t i = 0; i < 4; ++i) {
        auto filterDim = filterLayoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();
        auto inputDim = inputLayoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();
        auto outputDim = outputLayoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();

        if (filterDim.str() == "k") {
          dimKF = i;
          k = filterDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_K=" << k;
        } else if (filterDim.str() == "c") {
          dimCF = i;
          c = filterDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_C=" << c;
        } else if (filterDim.str() == "y") {
          dimYF = i;
          y = filterDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_Y=" << y;
        } else if (filterDim.str() == "x") {
          dimXF = i;
          x = filterDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_X=" << x;
        }

        if (inputDim.str() == "ni") {
          dimNI = i;
          n = inputDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_N=" << n;
        } else if (inputDim.str() == "hi") {
          hi = inputDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_HI=" << hi;
        } else if (inputDim.str() == "wi") {
          wi = inputDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_WI=" << wi;
        } else if (inputDim.str() == "ci") {
          dimCI = i;
        }

        if (outputDim.str() == "ho") {
          dimHO = i;
          ho = outputDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_HO=" << ho;
        } else if (outputDim.str() == "wo") {
          dimWO = i;
          wo = outputDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_WO=" << wo;
        } else if (outputDim.str() == "no") {
          dimNO = i;
        } else if (outputDim.str() == "ko") {
          dimKO = i;
        }
      }

      auto strideAttr = op.getAttrOfType<ArrayAttr>("strides");
      int64_t strideH = strideAttr.getValue()[0].dyn_cast<IntegerAttr>().getInt();
      int64_t strideW = strideAttr.getValue()[1].dyn_cast<IntegerAttr>().getInt();
      output << " -DCK_PARAM_PROBLEM_CONV_STRIDE_H=" << strideH;
      output << " -DCK_PARAM_PROBLEM_CONV_STRIDE_W=" << strideW;

      auto dilationAttr = op.getAttrOfType<ArrayAttr>("dilations");
      int64_t dilationH = dilationAttr.getValue()[0].dyn_cast<IntegerAttr>().getInt();
      int64_t dilationW = dilationAttr.getValue()[1].dyn_cast<IntegerAttr>().getInt();
      output << " -DCK_PARAM_PROBLEM_CONV_DILATION_H=" << dilationH;
      output << " -DCK_PARAM_PROBLEM_CONV_DILATION_W=" << dilationW;

      auto paddingAttr = op.getAttrOfType<ArrayAttr>("padding");
      int64_t paddingHL = paddingAttr.getValue()[0].dyn_cast<ArrayAttr>().getValue()[0].dyn_cast<IntegerAttr>().getInt();
      int64_t paddingWL = paddingAttr.getValue()[0].dyn_cast<ArrayAttr>().getValue()[1].dyn_cast<IntegerAttr>().getInt();
      int64_t paddingHR = paddingAttr.getValue()[1].dyn_cast<ArrayAttr>().getValue()[0].dyn_cast<IntegerAttr>().getInt();
      int64_t paddingWR = paddingAttr.getValue()[1].dyn_cast<ArrayAttr>().getValue()[1].dyn_cast<IntegerAttr>().getInt();

      output << " -DCK_PARAM_PROBLEM_IN_LEFT_PAD_H=" << paddingHL;
      output << " -DCK_PARAM_PROBLEM_IN_LEFT_PAD_W=" << paddingWL;
      output << " -DCK_PARAM_PROBLEM_IN_RIGHT_PAD_H=" << paddingHR;
      output << " -DCK_PARAM_PROBLEM_IN_RIGHT_PAD_W=" << paddingWR;

      // TBD: be able to set data type.
      output << " -DMIOPEN_USE_FP32=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_BFP16=0";

      // TBD: be able to set convolution direction.
      output << " -DCK_PARAM_PROBLEM_CONV_DIRECTION_FORWARD=1";
      output << " -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_DATA=0";
      output << " -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_WEIGHT=0";

      // distinguish between:
      // - parameters truly need to be tuned.
      // - parameters deducible via transformations.
      // - parameters which have heuristic-based values.
      // - parameters which are related to code generation.

      TunableParameters params;
      params.init();

      // Determine vectorization dimensions and lengths.
      int64_t vectorizableLength = 0;

      // Filter tensor.
      // Find the fastest changing dimension.
      bool input1GemmKVectorizable = false;
      if (dimKF == 3) {
        // When K is the fastest changing dimension,
        // gemmM dimension is vectorizable.
        // vectorization width depending on length of K.
        vectorizableLength = k;

        // gemmK dimension non-vectorizable.
        input1GemmKVectorizable = false;
      } else {
        // gemmK dimension vectorizable,
        // depending on which among C, Y, X be the fastest changing dimension.
        if (dimKF == 0) {
          // dimKF is the lowest changing dimension, which means dimC/dimY/dimX
          vectorizableLength = c * y * x;
        } else {
          if (dimCF == 3) {
            vectorizableLength = c;
          } else if (dimXF == 3 && dimYF == 2) {
            vectorizableLength = y * x;
          }
        }

        input1GemmKVectorizable = true;
        // gemmM dimension non-vectorizable.
      }

      int perThreadOpsA = params["CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK"] * params["CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK"] / params["CK_PARAM_TUNABLE_BLOCK_SIZE"];
      int perThreadOpsAVectorLength = 1;
      if ((vectorizableLength > 0) && (vectorizableLength % 4 == 0)) {
        perThreadOpsAVectorLength = gcd(4, perThreadOpsA);
      } else if ((vectorizableLength > 0) && (vectorizableLength % 2 == 0)) {
        perThreadOpsAVectorLength = gcd(2, perThreadOpsA);
      }
      int perThreadOpsANonVectorizedLength = perThreadOpsA / perThreadOpsAVectorLength;
      params.setValue("CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM", perThreadOpsAVectorLength);
      if (input1GemmKVectorizable) {
        params.setValue("CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M", params["CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK"] / perThreadOpsANonVectorizedLength);
        params.setValue("CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K", params["CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK"] / perThreadOpsAVectorLength);
      } else {
        params.setValue("CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K", params["CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK"] / perThreadOpsANonVectorizedLength);
        params.setValue("CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M", params["CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK"] / perThreadOpsAVectorLength);
      }

      // Input tensor.
      bool input2GemmKVectorizable = false;
      vectorizableLength = 0;
      // Find the fastest changing dimension.
      if (dimNI == 3) {
        // When N is the fastest changing dimension,
        // gemmN dimension is vectorizable.
        // vectorization width depending on length of N.
        vectorizableLength = n;

        // gemmK dimension non-vectorizable.
        input2GemmKVectorizable = false;
      } else if (dimCI == 3) {
        // When C is the fastest changing dimension,
        // gemmK dimension vectorizable.
        // vectorization width depending on length of C.
        vectorizableLength = c;

        input2GemmKVectorizable = true;
        // gemmN dimension non-vectorizable.
      }

      int perThreadOpsB = params["CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK"] * params["CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK"] / params["CK_PARAM_TUNABLE_BLOCK_SIZE"];
      int perThreadOpsBVectorLength = 1;
      if ((vectorizableLength > 0) && (vectorizableLength % 4 == 0)) {
        perThreadOpsBVectorLength = gcd(4, perThreadOpsB);
      } else if ((vectorizableLength > 0) && (vectorizableLength % 2 == 0)) {
        perThreadOpsBVectorLength = gcd(2, perThreadOpsB);
      }
      int perThreadOpsBNonVectorizedLength = perThreadOpsB / perThreadOpsBVectorLength;
      params.setValue("CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM", perThreadOpsBVectorLength);
      if (input2GemmKVectorizable) {
        params.setValue("CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N", params["CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK"] / perThreadOpsBNonVectorizedLength);
        params.setValue("CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K", params["CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK"] / perThreadOpsBVectorLength);
      } else {
        params.setValue("CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K", params["CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK"] / perThreadOpsBNonVectorizedLength);
        params.setValue("CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N", params["CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK"] / perThreadOpsBVectorLength);
      }

      // Output tensor.
      if (dimKO == 3) {
        // gemmM vectorizable.
        // However, there is no parameters for vectorizing gemmM dimension for matrix C.
        // Do nothing here.

        // gemmN non-vectorizable.
      } else {
        // gemmN dimension vectorizable,
        // depending on which among, N, Ho, Wo be the fastest changing dimension.
        int vectorizableLength = 0;
        if (dimKO == 0) {
          vectorizableLength = n * ho * wo;
        } else {
          if (dimNO == 3) {
            vectorizableLength = n;
          } else if (dimWO == 3 && dimHO == 2) {
            vectorizableLength = ho * wo;
          }
        }
        if (vectorizableLength % 4 == 0) {
          params.setValue("CK_PARAM_TUNABLE_GEMM_C_THREAD_COPY_DST_DATA_PER_WRITE_GEMM_N1", 4);
        } else if (vectorizableLength % 2 == 0) {
          params.setValue("CK_PARAM_TUNABLE_GEMM_C_THREAD_COPY_DST_DATA_PER_WRITE_GEMM_N1", 2);
        }

        // gemmM non-vectorizable.
      }

      // Print out the tunable parameters.
      params.print(output);
      if (IsPopulateTunableParameters.getValue()) {
        // Populate YAML config file.
        params.dump();
      }

      // Emit parameters derived from tunable parameters.
      int64_t gemmMPerBlock = params["CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK"];
      int64_t gemmNPerBlock = params["CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK"];
      int64_t gemmM = k;
      int64_t gemmN = n * ho * wo;
      int64_t gridSize = (gemmM / gemmMPerBlock) * (gemmN / gemmNPerBlock);
      output << " -DCK_PARAM_DEPENDENT_GRID_SIZE=" << gridSize;

      // Emit code-gen related macros.
      output << " -DCK_THREADWISE_GEMM_USE_AMD_INLINE_ASM=1";
      output << " -std=c++14";
      output << " -D__HIP_PLATFORM_HCC__=1";
    });
  }
 
  output.flush();
  return std::make_unique<llvm::StringRef>(resultStr);
}
