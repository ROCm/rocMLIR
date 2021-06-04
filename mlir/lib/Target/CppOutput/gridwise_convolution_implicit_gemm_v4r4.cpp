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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
static constexpr StringLiteral kVarArgName[3] = {"p_wei_global", "p_in_global",
                                                 "p_out_global"};

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
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM;

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
    "GemmG",
    "GemmK", // Conv2DOpType and Conv2DBwdWeightOpType
    "GemmM", // Conv2DBwdDataOpType
};

void EmitCppPreamble(llvm::raw_ostream &output, miopen::ConvOpType opType) {
  output << kCppPreamblePart1;
// Between Preamble Part 1 and Part 2:
  if (opType == miopen::ConvOpType::Conv2DOpType) {
    output << R"(#include "mlir_gen_igemm_conv2d_cpp_v4r4_fwd.hpp")";
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    output << R"(#include "mlir_gen_igemm_conv2d_cpp_v4r1_bwd.hpp")";
  } else if (opType == miopen::ConvOpType::Conv2DBwdWeightOpType) {
    output << R"(#include "mlir_gen_igemm_conv2d_cpp_v4r4_wrw.hpp")";
  }

  output << kCppPreamblePart2;
// Between Preamble Part 2 and Par 3:
  if (opType == miopen::ConvOpType::Conv2DOpType) {
    output << R"(
    __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void mlir_gen_igemm_conv2d_cpp_v4r4_fwd)";
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    output << R"(
    __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void mlir_gen_igemm_conv2d_cpp_v4r1_bwd)";
  } else if (opType == miopen::ConvOpType::Conv2DBwdWeightOpType) {
    output << R"(
    __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void mlir_gen_igemm_conv2d_cpp_v4r4_wrw)";
  }

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
  } else if (opType == miopen::ConvOpType::Conv2DBwdWeightOpType) {
    output << llvm::format(kCppPreamblePart3Format.data(), argPInGlobal.c_str(),
                           argPOutGlobal.c_str(), argPWeiGlobal.c_str());
  }
}

void EmitCppInterlude(llvm::raw_ostream &output, miopen::ConvOpType opType) {
  std::string gemmNameABlockCopySrcDataPerRead;
  if (opType == miopen::ConvOpType::Conv2DOpType) {
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[0].str();
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[1].str();
  } else if (opType == miopen::ConvOpType::Conv2DBwdWeightOpType) {
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[0].str();
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
    constexpr auto gridwise_conv = MlirGenIgemmConv2dV4r4Fwd)";
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    output << R"(
    constexpr auto gridwise_conv = MlirGenIgemmConv2dV1r1Bwd)";
  } else if (opType == miopen::ConvOpType::Conv2DBwdWeightOpType) {
    output << R"(
    constexpr auto gridwise_conv = MlirGenIgemmConv2dV4r4Wrw)";
  }

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
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[0].str();
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[1].str();
  } else if (opType == miopen::ConvOpType::Conv2DBwdWeightOpType) {
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[0].str();
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
// GemmN = %s
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

static constexpr StringLiteral kHeaderPreamblePart2BwdWeight = R"(
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        Float* __restrict__ p_wei_global,
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
        constexpr bool not_need_atomic = (ConvStrideH >= ConvDilationH * (y - 1) + 1) and
                                         (ConvStrideW >= ConvDilationW * (x - 1) + 1);
        constexpr auto in_memory_op = 
            not_need_atomic ? InMemoryDataOperation::Set : InMemoryDataOperation::AtomicAdd;
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
                                                     %s,
                                                     %s,
)";

static constexpr StringLiteral kHeaderEpiloguePart3Format = R"(
                                                     GemmABlockCopySrcDataPerRead_%s,
                                                     GemmABlockCopyDstDataPerWrite_GemmM,
                                                     GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
                                                     GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
                                                     Sequence<0, 1>,
                                                     Sequence<0, 1>,
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
  std::string commentGemmG;
  std::string commentGemmM;
  std::string commentGemmN;
  std::string commentGemmK;
  std::string gemmNameABlockCopySrcDataPerRead;
  if (opType == miopen::ConvOpType::Conv2DOpType) {
    headerIncludeGuard = "MLIR_GEN_IGEMM_CONV2D_CPP_V4R4_FWD";
    commentGemmG = "G";
    commentGemmM = "K";
    commentGemmN = "N * H * W";
    commentGemmK = "C * Y * X";
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[0].str();
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    headerIncludeGuard = "MLIR_GEN_IGEMM_CONV2D_CPP_V1R1_BWD";
    commentGemmG = "G";
    commentGemmM = "C * Y * X";
    commentGemmN = "N * H * W";
    commentGemmK = "K";
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[1].str();
  } else if (opType == miopen::ConvOpType::Conv2DBwdWeightOpType) {
    headerIncludeGuard = "MLIR_GEN_IGEMM_CONV2D_CPP_V4R4_WRW";
    commentGemmG = "G";
    commentGemmM = "K";
    commentGemmN = "C * Y * X";
    commentGemmK = "N * H * W";
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[0].str();
  }
 output << llvm::format(
      kHeaderPreamblePart1Format.data(), headerIncludeGuard.c_str(),
      headerIncludeGuard.c_str(), commentGemmM.c_str(), commentGemmN.c_str(),
      commentGemmK.c_str(), gemmNameABlockCopySrcDataPerRead.c_str());

  if (opType == miopen::ConvOpType::Conv2DOpType) {
    output << R"(struct MlirGenIgemmConv2dV4r4Fwd)";
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    output << R"(struct MlirGenIgemmConv2dV1r1Bwd)";
  } else if (opType == miopen::ConvOpType::Conv2DBwdWeightOpType) {
    output << R"(struct MlirGenIgemmConv2dV4r4Wrw)";
  }

  if (opType == miopen::ConvOpType::Conv2DOpType) {
    output << kHeaderPreamblePart2Forward;
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    output << kHeaderPreamblePart2BwdData;
  } else if (opType == miopen::ConvOpType::Conv2DBwdWeightOpType) {
    output << kHeaderPreamblePart2BwdWeight;
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
  std::string gemmHeaderEpiloguePart2Sequence;
  if (opType == miopen::ConvOpType::Conv2DOpType) {
    inMemOp = "InMemoryDataOperation::Set";
    gemmHeaderEpiloguePart2Sequence = "Sequence<1, 0>";
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    inMemOp = "in_memory_op";
    gemmHeaderEpiloguePart2Sequence = "Sequence<0, 1>";
  } else if (opType == miopen::ConvOpType::Conv2DBwdWeightOpType) {
    inMemOp = "InMemoryDataOperation::Set";
    gemmHeaderEpiloguePart2Sequence = "Sequence<1, 0>";
  }
  output << llvm::format(kHeaderEpiloguePart2.data(), inMemOp.c_str(),
                         gemmHeaderEpiloguePart2Sequence.c_str(),
                         gemmHeaderEpiloguePart2Sequence.c_str());

  // Between Part2 and Part3 emit which dimension the vectorization takes place
  // for filter tensor. kcyx, kyxc, yxkc, ckyx: 0 yxck, cyxk: 1
  if (input1GemmKVectorizable) {
    output << "                                                     0,";
  } else {
    output << "                                                     1,";
  }

  std::string gemmNameABlockCopySrcDataPerRead;
  if (opType == miopen::ConvOpType::Conv2DOpType) {
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[0].str();
  } else if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[1].str();
  } else if (opType == miopen::ConvOpType::Conv2DBwdWeightOpType) {
    gemmNameABlockCopySrcDataPerRead = kGemmNameABlockCopySrcDataPerRead[0].str();
  }
  output << llvm::format(kHeaderEpiloguePart3Format.data(),
                         gemmNameABlockCopySrcDataPerRead.c_str());
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
  } else if (opType == miopen::ConvOpType::Conv2DBwdWeightOpType) {
    output << llvm::format(kHeaderEpiloguePart4.data(), argPOutGlobal.c_str(),
                           argPInGlobal.c_str(), argPWeiGlobal.c_str());
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

static void ObtainModuleInfo(ModuleOp &m,
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
      // Determine if this is the lowest layer.
      auto lowestLayerAttr = op->getAttrOfType<BoolAttr>("lowest_layer");
      if (lowestLayerAttr) {
        // get lower_layer_layout attribute.
        auto srcLayoutAttr = op->getAttrOfType<ArrayAttr>("lower_layer_layout");
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
    f.walk([&opType](miopen::GridwiseGemmOp op) {
      opType = ObtainConvDirection(op);
    });
  }
}

} // namespace

void mlir::translateModuleFromMIOpenToHeader(ModuleOp m, std::string &header) {
  llvm::raw_string_ostream output(header);

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
                ops << ">, Sequence<ConvDilationH, ConvStrideH, 0>>{}";
                convDilationCtr++;
              } else {
                ops << ">, Sequence<ConvDilationW, ConvStrideW, 0>>{}";
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
    f.walk([&input1GemmKVectorizable, &input2GemmKVectorizable,
            opType](miopen::GridwiseGemmOp op) {
      llvm::StringMap<std::pair<size_t, int64_t>> dimIndexVal;

      auto inputLayoutAttr = op->getAttrOfType<ArrayAttr>("input_layout");
      auto inputDimensionAttr = op->getAttrOfType<ArrayAttr>("input_dimension");
      populateDimVal(inputLayoutAttr, inputDimensionAttr, dimIndexVal);

      auto outputLayoutAttr = op->getAttrOfType<ArrayAttr>("output_layout");
      auto outputDimensionAttr =
          op->getAttrOfType<ArrayAttr>("output_dimension");
      populateDimVal(outputLayoutAttr, outputDimensionAttr, dimIndexVal);

      auto filterLayoutAttr = op->getAttrOfType<ArrayAttr>("filter_layout");
      auto filterDimensionAttr =
          op->getAttrOfType<ArrayAttr>("filter_dimension");
      populateDimVal(filterLayoutAttr, filterDimensionAttr, dimIndexVal);

      PopulateParamsBase::obtainGemmADimKVectorizable(opType, dimIndexVal,
                                                      input1GemmKVectorizable);
      PopulateParamsBase::obtainGemmBDimKVectorizable(opType, dimIndexVal,
                                                      input2GemmKVectorizable);
    });

    EmitHeaderEpilogue(output, gridwiseGemmArguments, input1GemmKVectorizable,
                       input2GemmKVectorizable, opType);
  }

  output.flush();
}

void mlir::translateModuleFromMIOpenToCpp(ModuleOp m, std::string &source) {
  llvm::raw_string_ostream output(source);

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

    EmitCppInterlude(output, opType);

    EmitCppEpilogue(output, tensorDescs, opType);
  }

  output.flush();
}

void mlir::translateModuleFromMIOpenToCFlags(ModuleOp m, std::string &cflags) {
  llvm::raw_string_ostream output(cflags);

  for (auto f : m.getOps<FuncOp>()) {
    output << f.getName() << "\n";

    f.walk([&output](miopen::GridwiseGemmOp op) {
      ConvolutionContext ctx = populateConvContext(op);
      InitParamsNonXDL validParams;
      DerivedParams gemmADerivedParam;
      DerivedParams gemmBDerivedParam;
      DerivedBlockGemmParams blockGemmDerivedParam;
      int64_t gemmCDstPerWrite;
      int64_t gridSize;
      PopulateParams populateParams;
      (void)populateParams.paramsFromCtx(
          ctx, 0, validParams, gemmADerivedParam, gemmBDerivedParam,
          blockGemmDerivedParam, gemmCDstPerWrite, gridSize);

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
      parameters["CK_PARAM_PROBLEM_IN_LEFT_PAD_H"] = ctx.paddingVal[0];
      parameters["CK_PARAM_PROBLEM_IN_LEFT_PAD_W"] = ctx.paddingVal[1];
      parameters["CK_PARAM_PROBLEM_IN_RIGHT_PAD_H"] = ctx.paddingVal[2];
      parameters["CK_PARAM_PROBLEM_IN_RIGHT_PAD_W"] = ctx.paddingVal[3];
      // Data type
      parameters["MIOPEN_USE_FP32"] = 1;
      parameters["MIOPEN_USE_FP16"] = 0;
      parameters["MIOPEN_USE_BFP16"] = 0;
      // This is only needed in forward, since its kernel has the ability
      // to run in more than one directions.
      miopen::ConvOpType opType = ObtainConvDirection(op);
      if (opType == miopen::ConvOpType::Conv2DOpType) {
        parameters["CK_PARAM_PROBLEM_CONV_DIRECTION_FORWARD"] = 1;
        parameters["CK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_DATA"] = 0;
        parameters["CK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_WEIGHT"] = 0;
      }

      // parameters truly tunable.
      parameters["CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK"] =
          validParams.gemmMPerBlock;
      parameters["CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK"] =
          validParams.gemmNPerBlock;
      parameters["CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK"] =
          validParams.gemmKPerBlock;
      parameters["CK_PARAM_TUNABLE_GEMM_M_PER_THREAD"] =
          validParams.gemmMPerThread;
      parameters["CK_PARAM_TUNABLE_GEMM_N_PER_THREAD"] =
          validParams.gemmNPerThread;

      // parameters derivable from tunable parameters.
      parameters["CK_PARAM_TUNABLE_BLOCK_SIZE"] = validParams.blockSize;
      parameters["CK_PARAM_DEPENDENT_GRID_SIZE"] = gridSize;

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

      parameters
          ["CK_PARAM_TUNABLE_GEMM_C_THREAD_COPY_DST_DATA_PER_WRITE_GEMM_N1"] =
              gemmCDstPerWrite;

      parameters["CK_PARAM_TUNABLE_GEMM_M_LEVEL0_CLUSTER"] =
          blockGemmDerivedParam.gemmMLevel0Cluster;
      parameters["CK_PARAM_TUNABLE_GEMM_N_LEVEL0_CLUSTER"] =
          blockGemmDerivedParam.gemmNLevel0Cluster;
      parameters["CK_PARAM_TUNABLE_GEMM_M_LEVEL1_CLUSTER"] =
          blockGemmDerivedParam.gemmMLevel1Cluster;
      parameters["CK_PARAM_TUNABLE_GEMM_N_LEVEL1_CLUSTER"] =
          blockGemmDerivedParam.gemmNLevel1Cluster;

      // Emit code-gen related macros.
      parameters["CK_THREADWISE_GEMM_USE_AMD_INLINE_ASM"] = 1;

      // Setting flag to 1 means using inline ASM to do atomic add
      // This is not supported in gfx906, disabling it now
      if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
        parameters["CK_USE_AMD_BUFFER_ATOMIC_ADD"] = 0;
      }

      parameters["__HIP_PLATFORM_HCC__"] = 1;

      auto printParams = [&parameters](llvm::raw_ostream &os) {
        for (auto kv : parameters) {
          os << " -D" << kv.first << "=" << kv.second;
        }
      };

      // Print out the tunable parameters.
      printParams(output);

      output << " -std=c++14";
      output << "\n";
    });
  }
 
  output.flush();
}
