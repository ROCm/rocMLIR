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
// result string to keep C++ source / header / flags emission.
std::string resultStr;

class TunableParameters : public TunableParametersBase {
public:
  TunableParameters() : TunableParametersBase("gridwise_convolution_implicit_gemm_v4r4_gen_xdlops.yaml") {}

  enum {
    gemmMPerBlock = 0,
    gemmNPerBlock = 1,
    gemmKPerBlock = 2,
    gemmMPerWave = 3,
    gemmNPerWave = 4
  };

  // TBD: review logic here as they may be tied to NCHW layout.
  std::tuple<int, int, int, int, bool>
  calculateGemmABlockCopyPerformanceParameters(llvm::SmallVector<int64_t, 5> &param) {
      int64_t clusterLengths_GemmK  = 0;
      int64_t clusterLengths_GemmM  = 0;
      int64_t srcDataPerRead_GemmK  = 4;
      int64_t dstDataPerWrite_GemmM = 4;
  
      const auto waveSize = 64;
      const auto blockSize =
          param[gemmNPerBlock] * param[gemmMPerBlock] / (param[gemmMPerWave] * param[gemmNPerWave]) * waveSize;
  
      // calculate threadwise copy size
      const auto a_data_per_thread_copy = (param[gemmKPerBlock] * param[gemmMPerBlock]) / blockSize;
  
      if(!(a_data_per_thread_copy > 0))
          return std::make_tuple(-1, -1, -1, -1, false);
  
      // calculate vector length on gemmk dimension
      srcDataPerRead_GemmK = gcd(srcDataPerRead_GemmK, param[gemmKPerBlock]);
  
      // GemmABlockCopySrcDataPerRead_GemmK also bounded by size of threadwise copy
      srcDataPerRead_GemmK = gcd(srcDataPerRead_GemmK, a_data_per_thread_copy);
  
      // decide threadwise copy lengths
      const auto a_data_per_thread_copy_gemmk = srcDataPerRead_GemmK;
      const auto a_data_per_thread_copy_gemmm =
          a_data_per_thread_copy / a_data_per_thread_copy_gemmk;
  
      // GemmABlockCopyDstDataPerWrite_GemmM also bounded by size of threadwise copy
      dstDataPerWrite_GemmM = gcd(dstDataPerWrite_GemmM, a_data_per_thread_copy_gemmm);
  
      // calculate blockwise copy thread cluster lengths
      clusterLengths_GemmK = param[gemmKPerBlock] / a_data_per_thread_copy_gemmk;
      clusterLengths_GemmM = param[gemmMPerBlock] / a_data_per_thread_copy_gemmm;
  
      if(!(clusterLengths_GemmK > 0 && clusterLengths_GemmM > 0))
          return std::make_tuple(-1, -1, -1, -1, false);
  
      return std::make_tuple(clusterLengths_GemmK,
                             clusterLengths_GemmM,
                             srcDataPerRead_GemmK,
                             dstDataPerWrite_GemmM,
                             true);
  }

  // TBD: review logic here as they may be tied to NCHW layout.
  std::tuple<int, int, int, int, bool>
  calculateGemmBBlockCopyPerformanceParameters(llvm::SmallVector<int64_t, 5> &param) {
      int64_t clusterLengths_GemmK  = 0;
      int64_t clusterLengths_GemmN  = 0;
      int64_t srcDataPerRead_GemmN  = 4;
      int64_t dstDataPerWrite_GemmN = 4;
  
      const auto waveSize = 64;
      const auto blockSize =
          param[gemmNPerBlock] * param[gemmMPerBlock] / (param[gemmMPerWave] * param[gemmNPerWave]) * waveSize;
  
      srcDataPerRead_GemmN = gcd(srcDataPerRead_GemmN, param[gemmNPerBlock]);
  
      // calculate vector length on gemmn dimension
      if(ctx.y == 1 && ctx.x == 1 && ctx.strideH == 1 && ctx.strideW == 1 && ctx.paddingHL == 0 &&
         ctx.paddingHR == 0 && ctx.paddingWL == 0 && ctx.paddingWR == 0)
      {
          // \todo there are more configs that can go through this if branch
          srcDataPerRead_GemmN = gcd(srcDataPerRead_GemmN, ctx.hi * ctx.wi);
      }
      else if(ctx.strideW == 1)
      {
          srcDataPerRead_GemmN =
              gcd(srcDataPerRead_GemmN, ctx.paddingWL, ctx.wi, ctx.paddingWR, ctx.dilationW);
      }
      else
      {
          srcDataPerRead_GemmN = 1;
      }
  
      // calculate threadwise copy size
      const auto b_data_per_thread_copy = (param[gemmKPerBlock] * param[gemmNPerBlock]) / blockSize;
  
      if(!(b_data_per_thread_copy > 0))
          return std::make_tuple(-1, -1, -1, -1, false);
  
      // GemmBBlockCopySrcDataPerRead_GemmN also bounded by size of threadwise copy
      srcDataPerRead_GemmN = gcd(srcDataPerRead_GemmN, b_data_per_thread_copy);

      const auto b_data_per_thread_copy_gemmn = srcDataPerRead_GemmN;
      const auto b_data_per_thread_copy_gemmk =
          b_data_per_thread_copy / b_data_per_thread_copy_gemmn;
  
      // GemmBBlockCopyDstDataPerWrite_GemmN also bounded by size of threadwise copy
      dstDataPerWrite_GemmN = gcd(dstDataPerWrite_GemmN, b_data_per_thread_copy_gemmn);
  
      // calculate blockwise copy thread cluster lengths
      clusterLengths_GemmK = param[gemmKPerBlock] / b_data_per_thread_copy_gemmk;
      clusterLengths_GemmN = param[gemmNPerBlock] / b_data_per_thread_copy_gemmn;
  
      if(!(clusterLengths_GemmK > 0 && clusterLengths_GemmN > 0))
          return std::make_tuple(-1, -1, -1, -1, false);
  
      return std::make_tuple(clusterLengths_GemmK,
                             clusterLengths_GemmN,
                             srcDataPerRead_GemmN,
                             dstDataPerWrite_GemmN,
                             true);
  }

  // TBD: review logic here as they may be tied to NCHW layout.
  std::tuple<std::size_t, bool> 
  calculateLdsNumberOfByte(llvm::SmallVector<int64_t, 5> &param) {
      std::size_t lds_size = 0;
  
      bool valid = false;
  
      int64_t gemmABlockCopyDescDataPerWriteGemmM = 0;
      int64_t gemmABlockCopyClusterLengths_GemmM  = 0;
      std::tie(std::ignore,
               gemmABlockCopyClusterLengths_GemmM,
               std::ignore,
               gemmABlockCopyDescDataPerWriteGemmM,
               valid) = calculateGemmABlockCopyPerformanceParameters(param);
  
      if(!valid)
          return std::make_tuple(0, false);
  
      int64_t gemmBBlockCopyDescDataPerWriteGemmN = 0;
      int64_t gemmBBlockCopyClusterLengths_GemmN  = 0;
      std::tie(std::ignore,
               gemmBBlockCopyClusterLengths_GemmN,
               std::ignore,
               gemmBBlockCopyDescDataPerWriteGemmN,
               valid) = calculateGemmBBlockCopyPerformanceParameters(param);
  
      if(!valid)
          return std::make_tuple(0, false);
  
      int64_t threadGemmDataPerRead_GemmM = param[gemmMPerBlock] / gemmABlockCopyClusterLengths_GemmM;
      int64_t threadGemmDataPerRead_GemmN = param[gemmNPerBlock] / gemmBBlockCopyClusterLengths_GemmN;
  
      const auto max_lds_align = lcm(gemmABlockCopyDescDataPerWriteGemmM,
                                     gemmBBlockCopyDescDataPerWriteGemmN,
                                     threadGemmDataPerRead_GemmM,
                                     threadGemmDataPerRead_GemmN);
  
      const auto a_block_space =
          param[gemmKPerBlock] * integer_least_multiple(param[gemmMPerBlock], max_lds_align);
      const auto b_block_space =
          param[gemmKPerBlock] * integer_least_multiple(param[gemmNPerBlock], max_lds_align);
  
      lds_size = 2 * (a_block_space + b_block_space) * sizeof(float);
  
      return std::make_tuple(lds_size, true);
  }

  bool isValidXDLOPSGemm(llvm::SmallVector<int64_t, 5> &param) {
    // TBD: support fp16/bf16
    const auto gemmKPackedPerBlock = param[gemmKPerBlock];

    // unsupported xdlops-gemm
    if(param[gemmMPerWave] == 16 && param[gemmNPerWave] == 32)
        return false;
    if(param[gemmMPerWave] == 32 && param[gemmNPerWave] == 16)
        return false;
    if(param[gemmMPerWave] == 8 && param[gemmNPerWave] != 64)
        return false;
    if(param[gemmMPerWave] == 4 && param[gemmNPerWave] != 64)
        return false;
    if(param[gemmMPerWave] == 32 && param[gemmNPerWave] == 32 && gemmKPackedPerBlock % 2 != 0)
        return false;
    if(param[gemmMPerWave] == 16 && param[gemmNPerWave] == 16 && gemmKPackedPerBlock % 4 != 0)
        return false;

    const auto waveSize  = 64;
    const auto blockSize = param[gemmNPerBlock] * param[gemmMPerBlock] / (param[gemmMPerWave] * param[gemmNPerWave]) * waveSize;

    // fail with blockSize >= 512
    /// \todo fix the issue with blockSize >= 512
    if(blockSize < 64 || blockSize > 256)
        return false;

    return (param[gemmMPerBlock] % param[gemmMPerWave]) == 0 && (param[gemmNPerBlock] % param[gemmNPerWave]) == 0;
  }

  // TBD review logic here for various layouts.
  bool isValidParameter(llvm::SmallVector<int64_t, 5> &param) {
    int64_t gemmM = ctx.k;
    int64_t gemmN = ctx.n * ctx.ho * ctx.wo;
    int64_t gemmK = ctx.c * ctx.y * ctx.x;

    llvm::errs() << "gemmM: " << gemmM << " gemmN: " << gemmN << " gemmK: " << gemmK << "\n";
    llvm::errs() << "MPerBlock: " << param[gemmMPerBlock] << "\n";
    llvm::errs() << "NPerBlock: " << param[gemmNPerBlock] << "\n";
    llvm::errs() << "KPerBlock: " << param[gemmKPerBlock] << "\n";
    llvm::errs() << "MPerWave: " << param[gemmMPerWave] << "\n";
    llvm::errs() << "NPerWave: " << param[gemmNPerWave] << "\n";

    if (!(gemmM % param[gemmMPerBlock] == 0 &&
          gemmN % param[gemmNPerBlock] == 0 &&
          gemmK % param[gemmKPerBlock] == 0)) {
      llvm::errs() << "NOT VALID\n";
      return false;
    }

    if (!isValidXDLOPSGemm(param)) {
      llvm::errs() << "NOT VALID\n";
      return false;
    }

    bool valid = false;

    // check blockwise copy of A matrix
    std::tie(std::ignore, std::ignore, std::ignore, std::ignore, valid) =
        calculateGemmABlockCopyPerformanceParameters(param);

    if(!valid) {
      llvm::errs() << "NOT VALID\n";
      return false;
    }

    // check blockwise copy of B matrix
    std::tie(std::ignore, std::ignore, std::ignore, std::ignore, valid) =
        calculateGemmBBlockCopyPerformanceParameters(param);

    if(!valid) {
      llvm::errs() << "NOT VALID\n";
      return false;
    }

    std::size_t lds_size = 0;
    std::tie(lds_size, valid) = calculateLdsNumberOfByte(param);

    if (!valid || (lds_size > 64 * 1024)) {
      llvm::errs() << "NOT VALID\n";
      return false;
    }

    llvm::errs() << "VALID WITH LDS SIZE: " << lds_size << "\n";
    return (valid && lds_size <= 64 * 1024);
  }

  void customInit() override {
    // Check the following initial tuning parameters and find the valid one.
    llvm::SmallVector<llvm::SmallVector<int64_t, 5>, 10> initParameters = {
      // 0: GEMM_M_PER_BLOCK
      // 1: GEMM_N_PER_BLOCK
      // 2: GEMM_K_PER_BLOCK
      // 3: GEMM_M_PER_WAVE
      // 4: GEMM_N_PER_WAVE
      {128, 128, 16, 64, 64},
      {  8,  64,  8,  8, 64},
      {  4,  64, 16,  4, 64},
      { 16,  16,  4, 16, 16},
    };

    bool foundValidParameters = false;
    llvm::SmallVector<int64_t, 5> validParameter;
    for (auto &param : initParameters) {
      if (isValidParameter(param)) {
        foundValidParameters = true;
        validParameter = param;
        break;
      }
    }

    if (!foundValidParameters) {
      llvm::errs() << "FATAL ERROR! COULD NOT FIND VALID TUNING PARAMETERS!";
    }

    // parameters truly tunable.
    params["CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK"] = validParameter[gemmMPerBlock];
    params["CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK"] = validParameter[gemmNPerBlock];
    params["CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK"] = validParameter[gemmKPerBlock];
    params["CK_PARAM_GEMM_M_PER_WAVE"] = validParameter[gemmMPerWave];
    params["CK_PARAM_GEMM_N_PER_WAVE"] = validParameter[gemmNPerWave];

    // parameters derivable from tunable parameters.
    const auto waveSize = 64;
    params["CK_PARAM_TUNABLE_BLOCK_SIZE"] = validParameter[gemmMPerBlock] * validParameter[gemmNPerBlock] / (validParameter[gemmMPerWave] * validParameter[gemmNPerWave]) * waveSize;

    int gemmABlockCopyClusterLengths_GemmK  = 0;
    int gemmABlockCopyClusterLengths_GemmM  = 0;
    int gemmABlockCopySrcDataPerRead_GemmK  = 0;
    int gemmABlockCopyDstDataPerWrite_GemmM = 0;
    int gemmBBlockCopyClusterLengths_GemmK  = 0;
    int gemmBBlockCopyClusterLengths_GemmN  = 0;
    int gemmBBlockCopySrcDataPerRead_GemmN  = 0;
    int gemmBBlockCopyDstDataPerWrite_GemmN = 0;

    std::tie(gemmABlockCopyClusterLengths_GemmK,
             gemmABlockCopyClusterLengths_GemmM,
             gemmABlockCopySrcDataPerRead_GemmK,
             gemmABlockCopyDstDataPerWrite_GemmM,
             std::ignore) = calculateGemmABlockCopyPerformanceParameters(validParameter);

    std::tie(gemmBBlockCopyClusterLengths_GemmK,
             gemmBBlockCopyClusterLengths_GemmN,
             gemmBBlockCopySrcDataPerRead_GemmN,
             gemmBBlockCopyDstDataPerWrite_GemmN,
             std::ignore) = calculateGemmBBlockCopyPerformanceParameters(validParameter);


    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K"] = gemmABlockCopyClusterLengths_GemmK;
    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M"] = gemmABlockCopyClusterLengths_GemmM;
    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM"] = gemmABlockCopySrcDataPerRead_GemmK;
    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M"] = gemmABlockCopyDstDataPerWrite_GemmM;

    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K"] = gemmBBlockCopyClusterLengths_GemmK;
    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N"] = gemmBBlockCopyClusterLengths_GemmN;
    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM"] = gemmBBlockCopySrcDataPerRead_GemmN;
    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N"] = gemmBBlockCopyDstDataPerWrite_GemmN;
  }
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
      // Emit flags immediately determined from convolution configs.
      auto inputLayoutAttr = op.getAttrOfType<ArrayAttr>("input_layout");
      auto inputDimensionAttr = op.getAttrOfType<ArrayAttr>("input_dimension");
      auto outputLayoutAttr = op.getAttrOfType<ArrayAttr>("output_layout");
      auto outputDimensionAttr = op.getAttrOfType<ArrayAttr>("output_dimension");
      auto filterLayoutAttr = op.getAttrOfType<ArrayAttr>("filter_layout");
      auto filterDimensionAttr = op.getAttrOfType<ArrayAttr>("filter_dimension");

      ConvolutionContext ctx;

      for (size_t i = 0; i < 4; ++i) {
        auto filterDim = filterLayoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();
        auto inputDim = inputLayoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();
        auto outputDim = outputLayoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();

        if (filterDim.str() == "k") {
          ctx.dimKF = i;
          ctx.k = filterDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_K=" << ctx.k;
        } else if (filterDim.str() == "c") {
          ctx.dimCF = i;
          ctx.c = filterDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_C=" << ctx.c;
        } else if (filterDim.str() == "y") {
          ctx.dimYF = i;
          ctx.y = filterDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_Y=" << ctx.y;
        } else if (filterDim.str() == "x") {
          ctx.dimXF = i;
          ctx.x = filterDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_X=" << ctx.x;
        }

        if (inputDim.str() == "ni") {
          ctx.dimNI = i;
          ctx.n = inputDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_N=" << ctx.n;
        } else if (inputDim.str() == "hi") {
          ctx.dimHI = i;
          ctx.hi = inputDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_HI=" << ctx.hi;
        } else if (inputDim.str() == "wi") {
          ctx.dimWI = i;
          ctx.wi = inputDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_WI=" << ctx.wi;
        } else if (inputDim.str() == "ci") {
          ctx.dimCI = i;
        }

        if (outputDim.str() == "ho") {
          ctx.dimHO = i;
          ctx.ho = outputDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_HO=" << ctx.ho;
        } else if (outputDim.str() == "wo") {
          ctx.dimWO = i;
          ctx.wo = outputDimensionAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
          output << " -DCK_PARAM_PROBLEM_WO=" << ctx.wo;
        } else if (outputDim.str() == "no") {
          ctx.dimNO = i;
        } else if (outputDim.str() == "ko") {
          ctx.dimKO = i;
        }
      }

      auto strideAttr = op.getAttrOfType<ArrayAttr>("strides");
      ctx.strideH = strideAttr.getValue()[0].dyn_cast<IntegerAttr>().getInt();
      ctx.strideW = strideAttr.getValue()[1].dyn_cast<IntegerAttr>().getInt();
      output << " -DCK_PARAM_PROBLEM_CONV_STRIDE_H=" << ctx.strideH;
      output << " -DCK_PARAM_PROBLEM_CONV_STRIDE_W=" << ctx.strideW;

      auto dilationAttr = op.getAttrOfType<ArrayAttr>("dilations");
      ctx.dilationH = dilationAttr.getValue()[0].dyn_cast<IntegerAttr>().getInt();
      ctx.dilationW = dilationAttr.getValue()[1].dyn_cast<IntegerAttr>().getInt();
      output << " -DCK_PARAM_PROBLEM_CONV_DILATION_H=" << ctx.dilationH;
      output << " -DCK_PARAM_PROBLEM_CONV_DILATION_W=" << ctx.dilationW;

      auto paddingAttr = op.getAttrOfType<ArrayAttr>("padding");
      ctx.paddingHL = paddingAttr.getValue()[0].dyn_cast<ArrayAttr>().getValue()[0].dyn_cast<IntegerAttr>().getInt();
      ctx.paddingWL = paddingAttr.getValue()[0].dyn_cast<ArrayAttr>().getValue()[1].dyn_cast<IntegerAttr>().getInt();
      ctx.paddingHR = paddingAttr.getValue()[1].dyn_cast<ArrayAttr>().getValue()[0].dyn_cast<IntegerAttr>().getInt();
      ctx.paddingWR = paddingAttr.getValue()[1].dyn_cast<ArrayAttr>().getValue()[1].dyn_cast<IntegerAttr>().getInt();

      output << " -DCK_PARAM_PROBLEM_LEFT_PAD_H=" << ctx.paddingHL;
      output << " -DCK_PARAM_PROBLEM_LEFT_PAD_W=" << ctx.paddingWL;
      output << " -DCK_PARAM_PROBLEM_RIGHT_PAD_H=" << ctx.paddingHR;
      output << " -DCK_PARAM_PROBLEM_RIGHT_PAD_W=" << ctx.paddingWR;

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
      params.initWithContext(ctx);

      // XXX disable for now.
      //// Determine vectorization dimensions and lengths.
      //int64_t vectorizableLength = 0;

      //// Filter tensor.
      //// Find the fastest changing dimension.
      //bool filterGemmKVectorizable = false;
      //if (ctx.dimKF == 3) {
      //  // When K is the fastest changing dimension,
      //  // gemmM dimension is vectorizable.
      //  // vectorization width depending on length of K.
      //  vectorizableLength = ctx.k;

      //  // gemmK dimension non-vectorizable.
      //  filterGemmKVectorizable = false;
      //} else {
      //  // gemmK dimension vectorizable,
      //  // depending on which among C, Y, X be the fastest changing dimension.
      //  if (ctx.dimKF == 0) {
      //    // dimKF is the lowest changing dimension, which means dimC/dimY/dimX
      //    vectorizableLength = ctx.c * ctx.y * ctx.x;
      //  } else {
      //    if (ctx.dimCF == 3) {
      //      vectorizableLength = ctx.c;
      //    } else if (ctx.dimXF == 3 && ctx.dimYF == 2) {
      //      vectorizableLength = ctx.y * ctx.x;
      //    }
      //  }

      //  filterGemmKVectorizable = true;
      //  // gemmM dimension non-vectorizable.
      //}

      // XXX disable vectorization logic on matrix A for now.
      //int perThreadOpsA = params["CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK"] * params["CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK"] / params["CK_PARAM_TUNABLE_BLOCK_SIZE"];
      //int perThreadOpsAVectorLength = 1;
      //if ((vectorizableLength > 0) && (vectorizableLength % 4 == 0)) {
      //  perThreadOpsAVectorLength = gcd(4, perThreadOpsA);
      //} else if ((vectorizableLength > 0) && (vectorizableLength % 2 == 0)) {
      //  perThreadOpsAVectorLength = gcd(2, perThreadOpsA);
      //}
      //int perThreadOpsANonVectorizedLength = perThreadOpsA / perThreadOpsAVectorLength;
      //params.setValue("CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM", perThreadOpsAVectorLength);
      //if (filterGemmKVectorizable) {
      //  params.setValue("CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M", params["CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK"] / perThreadOpsANonVectorizedLength);
      //  params.setValue("CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K", params["CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK"] / perThreadOpsAVectorLength);
      //} else {
      //  params.setValue("CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K", params["CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK"] / perThreadOpsANonVectorizedLength);
      //  params.setValue("CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M", params["CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK"] / perThreadOpsAVectorLength);
      //}

      // XXX disable for now.
      //// Input tensor.
      //bool inputGemmKVectorizable = false;
      //vectorizableLength = 0;
      //// Find the fastest changing dimension.
      //if (ctx.dimNI == 3) {
      //  // When N is the fastest changing dimension,
      //  // gemmN dimension is vectorizable.
      //  // vectorization width depending on length of N.
      //  vectorizableLength = ctx.n;

      //  // gemmK dimension non-vectorizable.
      //  inputGemmKVectorizable = false;
      //} else if (ctx.dimCI == 3) {
      //  // When C is the fastest changing dimension,
      //  // gemmK dimension vectorizable.
      //  // vectorization width depending on length of C.
      //  vectorizableLength = ctx.c;

      //  inputGemmKVectorizable = true;
      //  // gemmN dimension non-vectorizable.
      //}

      // XXX disable vectorization logic on matrix B for now.
      //int perThreadOpsB = params["CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK"] * params["CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK"] / params["CK_PARAM_TUNABLE_BLOCK_SIZE"];
      //int perThreadOpsBVectorLength = 1;
      //if ((vectorizableLength > 0) && (vectorizableLength % 4 == 0)) {
      //  perThreadOpsBVectorLength = gcd(4, perThreadOpsB);
      //} else if ((vectorizableLength > 0) && (vectorizableLength % 2 == 0)) {
      //  perThreadOpsBVectorLength = gcd(2, perThreadOpsB);
      //}
      //int perThreadOpsBNonVectorizedLength = perThreadOpsB / perThreadOpsBVectorLength;
      //params.setValue("CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM", perThreadOpsBVectorLength);
      //if (inputGemmKVectorizable) {
      //  params.setValue("CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N", params["CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK"] / perThreadOpsBNonVectorizedLength);
      //  params.setValue("CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K", params["CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK"] / perThreadOpsBVectorLength);
      //} else {
      //  params.setValue("CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K", params["CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK"] / perThreadOpsBNonVectorizedLength);
      //  params.setValue("CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N", params["CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK"] / perThreadOpsBVectorLength);
      //}

      // Output tensor.
      // Dont vectorize on matrix C for now.

      // Print out the tunable parameters.
      params.print(output);
      if (IsPopulateTunableParameters.getValue()) {
        // Populate YAML config file.
        params.dump();
      }

      // Emit parameters derived from tunable parameters.
      int64_t gemmMPerBlock = params["CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK"];
      int64_t gemmNPerBlock = params["CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK"];
      int64_t gemmM = ctx.k;
      int64_t gemmN = ctx.n * ctx.ho * ctx.wo;
      int64_t gridSize = (gemmM / gemmMPerBlock) * (gemmN / gemmNPerBlock);
      output << " -DCK_PARAM_DEPENDENT_GRID_SIZE=" << gridSize;

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
