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

#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
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
std::string resultStr;

class PopulateParamsXDL : public PopulateParamsBase {
private:
  struct InitParamsXDL : InitParams {
    InitParamsXDL(int64_t mPerBlock, int64_t nPerBlock, int64_t kPerBlock,
                  int64_t mPerWave, int64_t nPerWave)
        : InitParams{mPerBlock, nPerBlock, kPerBlock}, gemmMPerWave(mPerWave),
          gemmNPerWave(nPerWave) {}
    int64_t gemmMPerWave;
    int64_t gemmNPerWave;
  };

  llvm::SmallVector<InitParamsXDL, 4> initParameters = {
      // M/block N/block K/block M/wave N/wave
      {128, 128, 16, 64, 64},
      {8, 64, 8, 8, 64},
      {4, 64, 16, 4, 64},
      {16, 16, 4, 16, 16},
  };
  const int64_t waveSize = 64;

  int64_t obtainBlockSize(InitParamsXDL &params, int64_t waveSize) {
    return waveSize * params.gemmNPerBlock * params.gemmMPerBlock /
           (params.gemmMPerWave * params.gemmNPerWave);
  }

  LogicalResult calculateGemmABlockCopyPerformanceParameters(
      InitParamsXDL *param, ConvolutionContext &ctx, DerivedParams &derived) {
    int64_t blockSize = obtainBlockSize(*param, waveSize);
    return calculateInputDerivedParams(param, blockSize, ctx, true, derived);
  }

  LogicalResult calculateGemmBBlockCopyPerformanceParameters(
      InitParamsXDL *param, ConvolutionContext &ctx, DerivedParams &derived) {
    int64_t blockSize = obtainBlockSize(*param, waveSize);
    return calculateInputDerivedParams(param, blockSize, ctx, false, derived);
  }

  LogicalResult calculateLdsNumberOfByte(InitParamsXDL *param,
                                         ConvolutionContext &ctx,
                                         size_t &ldsSize) {
    DerivedParams gemmADerived;
    LogicalResult res =
        calculateGemmABlockCopyPerformanceParameters(param, ctx, gemmADerived);

    if (failed(res))
      return failure();

    DerivedParams gemmBDerived;
    res =
        calculateGemmBBlockCopyPerformanceParameters(param, ctx, gemmBDerived);

    if (failed(res))
      return failure();

    int64_t threadGemmDataPerRead_GemmM =
        param->gemmMPerBlock / gemmADerived.clusterLenGemmPos2;
    int64_t threadGemmDataPerRead_GemmN =
        param->gemmNPerBlock / gemmBDerived.clusterLenGemmPos2;

    const auto max_lds_align =
        lcm(gemmADerived.dstDataPerWrite, gemmBDerived.dstDataPerWrite,
            threadGemmDataPerRead_GemmM, threadGemmDataPerRead_GemmN);

    const auto a_block_space =
        param->gemmKPerBlock *
        integer_least_multiple(param->gemmMPerBlock, max_lds_align);
    const auto b_block_space =
        param->gemmKPerBlock *
        integer_least_multiple(param->gemmNPerBlock, max_lds_align);

    ldsSize = 2 * (a_block_space + b_block_space) * sizeof(float);

    return success();
  }

  LogicalResult isValidXDLOPSGemm(InitParamsXDL *param, int64_t blockSize) {
    // TBD: support fp16/bf16
    const auto gemmKPackedPerBlock = param->gemmKPerBlock;

    // unsupported xdlops-gemm
    if (param->gemmMPerWave == 16 && param->gemmNPerWave == 32)
      return failure();
    if (param->gemmMPerWave == 32 && param->gemmNPerWave == 16)
      return failure();
    if (param->gemmMPerWave == 8 && param->gemmNPerWave != 64)
      return failure();
    if (param->gemmMPerWave == 4 && param->gemmNPerWave != 64)
      return failure();
    if (param->gemmMPerWave == 32 && param->gemmNPerWave == 32 &&
        gemmKPackedPerBlock % 2 != 0)
      return failure();
    if (param->gemmMPerWave == 16 && param->gemmNPerWave == 16 &&
        gemmKPackedPerBlock % 4 != 0)
      return failure();

    // fail with blockSize >= 512
    /// \todo fix the issue with blockSize >= 512
    if(blockSize < 64 || blockSize > 256)
      return failure();

    if ((param->gemmMPerBlock % param->gemmMPerWave) != 0)
      return failure();

    if ((param->gemmNPerBlock % param->gemmNPerWave) != 0)
      return failure();

    return success();
  }

public:
  void paramsFromCtx(ConvolutionContext &ctx,
                     std::map<std::string, int> &params) {
    LogicalResult res(LogicalResult::Failure);
    InitParamsXDL validParams{0, 0, 0, 0, 0};
    int64_t blockSize;
    DerivedParams gemmADerivedParam;
    DerivedParams gemmBDerivedParam;

    GemmSize gemmSize;
    obtainGemmSize(ctx, gemmSize);

    for (auto &params : initParameters) {
      res = isValidGemm(&params, gemmSize);
      if (failed(res)) {
        continue;
      }

      blockSize = obtainBlockSize(params, waveSize);

      res = isValidXDLOPSGemm(&params, blockSize);
      if (failed(res)) {
        continue;
      }

      res = calculateGemmABlockCopyPerformanceParameters(&params, ctx,
                                                         gemmADerivedParam);
      if (failed(res)) {
        continue;
      }

      res = calculateGemmBBlockCopyPerformanceParameters(&params, ctx,
                                                         gemmBDerivedParam);

      if (failed(res)) {
        continue;
      }

      std::size_t ldsSize = 0;
      res = calculateLdsNumberOfByte(&params, ctx, ldsSize);
      if (failed(res)) {
        continue;
      }

      if (ldsSize > 64 * 1024) {
        continue;
      }

      validParams = params;
      break;
    }

    if (failed(res)) {
      // All initParameters have failed, shouldn't happen
      llvm_unreachable("FATAL ERROR! COULD NOT FIND VALID TUNING PARAMETERS!");
    }

    // parameters truly tunable.
    params["CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK"] = validParams.gemmMPerBlock;
    params["CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK"] = validParams.gemmNPerBlock;
    params["CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK"] = validParams.gemmKPerBlock;
    params["CK_PARAM_GEMM_M_PER_WAVE"] = validParams.gemmMPerWave;
    params["CK_PARAM_GEMM_N_PER_WAVE"] = validParams.gemmNPerWave;

    // parameters derivable from tunable parameters.
    params["CK_PARAM_TUNABLE_BLOCK_SIZE"] = blockSize;
    params["CK_PARAM_DEPENDENT_GRID_SIZE"] =
        obtainGridSize(gemmSize, &validParams);

    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K"] =
        gemmADerivedParam.clusterLenGemmPos1;
    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M"] =
        gemmADerivedParam.clusterLenGemmPos2;
    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM"] =
        gemmADerivedParam.srcDataPerRead;
    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M"] =
        gemmADerivedParam.dstDataPerWrite;

    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K"] =
        gemmBDerivedParam.clusterLenGemmPos1;
    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N"] =
        gemmBDerivedParam.clusterLenGemmPos2;
    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM"] =
        gemmBDerivedParam.srcDataPerRead;
    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N"] =
        gemmBDerivedParam.dstDataPerWrite;
  }
};

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
      bool hasUnfoldTransform = false;

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
    f.walk([&filterGemmKVectorizable, &inputGemmKVectorizable](miopen::GridwiseGemmOp op) {
      auto filterLayoutAttr = op.getAttrOfType<ArrayAttr>("filter_layout");
      auto inputLayoutAttr = op.getAttrOfType<ArrayAttr>("input_layout");

      size_t dimKF, dimCF, dimYF, dimXF;
      size_t dimNI, dimCI, dimHI, dimWI;

      for (size_t i = 0; i < 4; ++i) {
        auto filterDim = filterLayoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();

        if (filterDim.str() == "k") {
          dimKF = i;
        } else if (filterDim.str() == "c") {
          dimCF = i;
        } else if (filterDim.str() == "y") {
          dimYF = i;
        } else if (filterDim.str() == "x") {
          dimXF = i;
        }

        auto inputDim = inputLayoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();
        if (inputDim.str() == "ni") {
          dimNI = i;
        } else if (inputDim.str() == "ci") {
          dimCI = i;
        } else if (inputDim.str() == "hi") {
          dimHI = i;
        } else if (inputDim.str() == "wi") {
          dimWI = i;
        }
      }

      // Filter tensor.
      // Find the fastest changing dimension.
      if (dimKF == 3) {
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
      if (dimNI == 3) {
        // When N is the fastest changing dimension,
        // gemmN dimension is vectorizable.
        // vectorization width depending on length of N.

        // gemmK dimension non-vectorizable.
        inputGemmKVectorizable = false;
      } else if (dimCI == 3) {
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
  return std::make_unique<llvm::StringRef>(resultStr);
}

std::unique_ptr<llvm::StringRef> mlir::translateModuleToMIOpenCppXDLOPS(ModuleOp m) {
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
  llvm::raw_string_ostream output(resultStr);

  for (auto f : m.getOps<FuncOp>()) {
    f.walk([&output](miopen::GridwiseGemmOp op) {
      miopen::ConvOpType opType = ObtainConvDirection(op);

      llvm::StringMap<std::pair<size_t, int64_t>> dimIndexVal;
      // Filter
      auto filterLayoutAttr = op.getAttrOfType<ArrayAttr>("filter_layout");
      auto filterDimensionAttr =
          op.getAttrOfType<ArrayAttr>("filter_dimension");
      populateDimVal(filterLayoutAttr, filterDimensionAttr, dimIndexVal);
      output << " -DCK_PARAM_PROBLEM_K=" << dimIndexVal["k"].second;
      output << " -DCK_PARAM_PROBLEM_C=" << dimIndexVal["c"].second;
      output << " -DCK_PARAM_PROBLEM_Y=" << dimIndexVal["y"].second;
      output << " -DCK_PARAM_PROBLEM_X=" << dimIndexVal["x"].second;
      // Input
      auto inputLayoutAttr = op.getAttrOfType<ArrayAttr>("input_layout");
      auto inputDimensionAttr = op.getAttrOfType<ArrayAttr>("input_dimension");
      populateDimVal(inputLayoutAttr, inputDimensionAttr, dimIndexVal);
      output << " -DCK_PARAM_PROBLEM_N=" << dimIndexVal["ni"].second;
      output << " -DCK_PARAM_PROBLEM_HI=" << dimIndexVal["hi"].second;
      output << " -DCK_PARAM_PROBLEM_WI=" << dimIndexVal["wi"].second;
      // Output
      auto outputLayoutAttr = op.getAttrOfType<ArrayAttr>("output_layout");
      auto outputDimensionAttr = op.getAttrOfType<ArrayAttr>("output_dimension");
      populateDimVal(outputLayoutAttr, outputDimensionAttr, dimIndexVal);
      output << " -DCK_PARAM_PROBLEM_HO=" << dimIndexVal["ho"].second;
      output << " -DCK_PARAM_PROBLEM_WO=" << dimIndexVal["wo"].second;

      // Stride
      auto strideAttr = op.getAttrOfType<ArrayAttr>("strides");
      llvm::SmallVector<int64_t, 0> strideVal;
      populateSeqVal(strideAttr, strideVal);
      output << " -DCK_PARAM_PROBLEM_CONV_STRIDE_H=" << strideVal[0];
      output << " -DCK_PARAM_PROBLEM_CONV_STRIDE_W=" << strideVal[1];

      // Dilation
      auto dilationAttr = op.getAttrOfType<ArrayAttr>("dilations");
      llvm::SmallVector<int64_t, 0> dilationVal;
      populateSeqVal(dilationAttr, dilationVal);
      output << " -DCK_PARAM_PROBLEM_CONV_DILATION_H=" << dilationVal[0];
      output << " -DCK_PARAM_PROBLEM_CONV_DILATION_W=" << dilationVal[1];

      // Padding
      auto paddingAttr = op.getAttrOfType<ArrayAttr>("padding");
      llvm::SmallVector<int64_t, 0> paddingVal;
      populateSeqVal(paddingAttr, paddingVal);
      output << " -DCK_PARAM_PROBLEM_LEFT_PAD_H=" << paddingVal[0];
      output << " -DCK_PARAM_PROBLEM_LEFT_PAD_W=" << paddingVal[1];
      output << " -DCK_PARAM_PROBLEM_RIGHT_PAD_H=" << paddingVal[2];
      output << " -DCK_PARAM_PROBLEM_RIGHT_PAD_W=" << paddingVal[3];

      // TBD: be able to set data type.
      output << " -DMIOPEN_USE_FP32=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_BFP16=0";

      output << " -DCK_PARAM_PROBLEM_CONV_DIRECTION_FORWARD=1";
      output << " -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_DATA=0";
      output << " -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_WEIGHT=0";

      ConvolutionContext convContext = populateConvContext(op);
      std::map<std::string, int> parameters;
      PopulateParamsXDL populateParams;
      populateParams.paramsFromCtx(convContext, parameters);
      TunableParameters params(parameters);

      // Print out the tunable parameters.
      params.print(output);
      if (IsPopulateTunableParameters.getValue()) {
        // Populate YAML config file.
        params.dump("tunable.yaml");
      }

      // Emit code-gen related parameters.
      output << " -DCK_USE_AMD_XDLOPS=1";
      output << " -DCK_USE_AMD_XDLOPS_INLINE_ASM=1";
      output << " -std=c++14";
      output << " -D__HIP_PLATFORM_HCC__=1";
    });
  }

  output.flush();
  return std::make_unique<llvm::StringRef>(resultStr);
}
