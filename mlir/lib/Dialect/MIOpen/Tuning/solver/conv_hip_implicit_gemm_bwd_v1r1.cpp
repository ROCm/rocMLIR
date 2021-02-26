#include "mlir/Dialect/MIOpen/Tuning/GridwiseGemmParams.h"

#define DEBUG_TYPE "igemm-bwd-v1r1"

PerformanceImplicitGemmBwdDataV1R1::PerformanceImplicitGemmBwdDataV1R1(
    int64_t BlockSize_, int64_t GemmMPerBlock_, int64_t GemmNPerBlock_,
    int64_t GemmKPerBlock_, int64_t GemmMPerThread_, int64_t GemmNPerThread_)
    : BlockSize(BlockSize_), GemmMPerBlock(GemmMPerBlock_),
      GemmNPerBlock(GemmNPerBlock_), GemmKPerBlock(GemmKPerBlock_),
      GemmMPerThread(GemmMPerThread_), GemmNPerThread(GemmNPerThread_) {}

bool PerformanceImplicitGemmBwdDataV1R1::operator==(
    const PerformanceImplicitGemmBwdDataV1R1 &other) const {
  // clang-format off
    return BlockSize == other.BlockSize
        && GemmMPerBlock == other.GemmMPerBlock
        && GemmNPerBlock == other.GemmNPerBlock
        && GemmKPerBlock == other.GemmKPerBlock
        && GemmMPerThread == other.GemmMPerThread
        && GemmNPerThread == other.GemmNPerThread;
  // clang-format on
}

std::tuple<int64_t, LogicalResult>
PerformanceImplicitGemmBwdDataV1R1::CalculateGridSize(
    const ConvolutionContext &ctx) const {
  int64_t GridSize = 0;

  int64_t gemm_m = 0;
  int64_t gemm_n = 0;

  std::tie(gemm_m, gemm_n, std::ignore) =
      ConvHipImplicitGemmBwdDataV1R1::CalculateGemmSize(ctx);

  if (!(gemm_m % GemmMPerBlock == 0 && gemm_n % GemmNPerBlock == 0))
    return std::make_tuple(-1, failure());

  GridSize = (gemm_m / GemmMPerBlock) * (gemm_n / GemmNPerBlock);

  return std::make_tuple(GridSize, success());
}

std::tuple<int64_t, int64_t, int64_t, int64_t, LogicalResult>
PerformanceImplicitGemmBwdDataV1R1::CalculateBlockGemmPerformanceParameters(
    const ConvolutionContext &) const {
  int64_t GemmMLevel0Cluster = 0;
  int64_t GemmNLevel0Cluster = 0;
  int64_t GemmMLevel1Cluster = 0;
  int64_t GemmNLevel1Cluster = 0;

  if (BlockSize == 64) {
    GemmMLevel0Cluster = 4;
    GemmNLevel0Cluster = 4;
    GemmMLevel1Cluster = 2;
    GemmNLevel1Cluster = 2;
  } else if (BlockSize == 128) {
    GemmMLevel0Cluster = 4;
    GemmNLevel0Cluster = 4;
    GemmMLevel1Cluster = 4;
    GemmNLevel1Cluster = 2;
  } else if (BlockSize == 256) {
    GemmMLevel0Cluster = 4;
    GemmNLevel0Cluster = 4;
    GemmMLevel1Cluster = 4;
    GemmNLevel1Cluster = 4;
  } else {
    return std::make_tuple(-1, -1, -1, -1, failure());
  }

  if (!(GemmMPerBlock % GemmMPerThread == 0 &&
        GemmNPerBlock % GemmNPerThread == 0))
    return std::make_tuple(-1, -1, -1, -1, failure());

  const auto thread_gemm_per_block_m = GemmMPerBlock / GemmMPerThread;
  const auto thread_gemm_per_block_n = GemmNPerBlock / GemmNPerThread;

  const auto thread_gemm_per_cluster_m =
      GemmMLevel0Cluster * GemmMLevel1Cluster;
  const auto thread_gemm_per_cluster_n =
      GemmNLevel0Cluster * GemmNLevel1Cluster;

  if (!(thread_gemm_per_block_m % thread_gemm_per_cluster_m == 0) &&
      (thread_gemm_per_block_n % thread_gemm_per_cluster_n == 0))
    return std::make_tuple(-1, -1, -1, -1, failure());

  const auto cluster_per_block_m =
      thread_gemm_per_block_m / thread_gemm_per_cluster_m;
  const auto cluster_per_block_n =
      thread_gemm_per_block_n / thread_gemm_per_cluster_n;

  // inline asm only support cluster_per_block_m = 2 andcluster_per_block_n = 2
  if (!(cluster_per_block_m == 2 && cluster_per_block_n == 2))
    return std::make_tuple(-1, -1, -1, -1, failure());

  return std::make_tuple(GemmMLevel0Cluster, GemmNLevel0Cluster,
                         GemmMLevel1Cluster, GemmNLevel1Cluster, success());
}

std::tuple<int64_t, int64_t, int64_t, int64_t, LogicalResult>
PerformanceImplicitGemmBwdDataV1R1::
    CalculateGemmABlockCopyPerformanceParameters(
        const ConvolutionContext &ctx) const {
  int64_t ClusterLengths_GemmK = 0;
  int64_t ClusterLengths_GemmM = 0;
  int64_t SrcDataPerRead_GemmM = 4;
  int64_t DstDataPerWrite_GemmM = 4;
  int64_t DstDataPerWrite_GemmKPACK = 4;

  // calculate vector length on gemmk dimension
  SrcDataPerRead_GemmM =
      ImplicitGemmUtil::gcd(SrcDataPerRead_GemmM, GemmMPerBlock);

  // calculate threadwise copy size
  const auto a_data_per_thread_copy =
      (GemmKPerBlock * GemmMPerBlock) / BlockSize;

  if (!(a_data_per_thread_copy > 0))
    return std::make_tuple(-1, -1, -1, -1, failure());

  // GemmABlockCopySrcDataPerRead_GemmK also bounded by size of threadwise copy
  SrcDataPerRead_GemmM =
      ImplicitGemmUtil::gcd(SrcDataPerRead_GemmM, a_data_per_thread_copy);

  // decide threadwise copy lengths
  const auto a_data_per_thread_copy_gemmm = SrcDataPerRead_GemmM;
  const auto a_data_per_thread_copy_gemmk =
      a_data_per_thread_copy / a_data_per_thread_copy_gemmm;

  if (1) { // if (ctx.IsFp32())
    DstDataPerWrite_GemmM = ImplicitGemmUtil::gcd(DstDataPerWrite_GemmM,
                                                  a_data_per_thread_copy_gemmm);
  }

  // calculate blockwise copy thread cluster lengths
  ClusterLengths_GemmK = GemmKPerBlock / a_data_per_thread_copy_gemmk;
  ClusterLengths_GemmM = GemmMPerBlock / a_data_per_thread_copy_gemmm;

  if (!(ClusterLengths_GemmK > 0 && ClusterLengths_GemmM > 0))
    return std::make_tuple(-1, -1, -1, -1, failure());

  if (1) { // if(ctx.IsFp32())
    return std::make_tuple(ClusterLengths_GemmK, ClusterLengths_GemmM,
                           SrcDataPerRead_GemmM, DstDataPerWrite_GemmM,
                           success());
  } else {
    return std::make_tuple(ClusterLengths_GemmK, ClusterLengths_GemmM,
                           SrcDataPerRead_GemmM, DstDataPerWrite_GemmKPACK,
                           success());
  }
}

std::tuple<int64_t, int64_t, int64_t, int64_t, LogicalResult>
PerformanceImplicitGemmBwdDataV1R1::
    CalculateGemmBBlockCopyPerformanceParameters(
        const ConvolutionContext &ctx) const {
  int64_t ClusterLengths_GemmK = 0;
  int64_t ClusterLengths_GemmN = 0;
  int64_t SrcDataPerRead_GemmN = 4;
  int64_t DstDataPerWrite_GemmN = 4;
  int64_t DstDataPerWrite_GemmKPACK = 4; // GetEPackLength(ctx, false);

  SrcDataPerRead_GemmN =
      ImplicitGemmUtil::gcd(SrcDataPerRead_GemmN, GemmNPerBlock);

  // calculate vector length on gemmn dimension
  auto dimIndexVal = ctx.dimIndexVal;
  const auto ho = dimIndexVal["ho"].second;
  const auto wo = dimIndexVal["wo"].second;

  SrcDataPerRead_GemmN = ImplicitGemmUtil::gcd(SrcDataPerRead_GemmN, ho * wo);

  // calculate threadwise copy size
  const auto b_data_per_thread_copy =
      (GemmKPerBlock * GemmNPerBlock) / BlockSize;

  if (!(b_data_per_thread_copy > 0))
    return std::make_tuple(-1, -1, -1, -1, failure());

  // GemmBBlockCopySrcDataPerRead_GemmN also bounded by size of threadwise copy
  SrcDataPerRead_GemmN =
      ImplicitGemmUtil::gcd(SrcDataPerRead_GemmN, b_data_per_thread_copy);

  const auto b_data_per_thread_copy_gemmn = SrcDataPerRead_GemmN;
  const auto b_data_per_thread_copy_gemmk =
      b_data_per_thread_copy / b_data_per_thread_copy_gemmn;

  // GemmBBlockCopyDstDataPerWrite_GemmN also bounded by size of threadwise copy
  if (1) { // if(ctx.IsFp32())
    DstDataPerWrite_GemmN = ImplicitGemmUtil::gcd(DstDataPerWrite_GemmN,
                                                  b_data_per_thread_copy_gemmn);
  }

  // calculate blockwise copy thread cluster lengths
  ClusterLengths_GemmK = GemmKPerBlock / b_data_per_thread_copy_gemmk;
  ClusterLengths_GemmN = GemmNPerBlock / b_data_per_thread_copy_gemmn;

  if (!(ClusterLengths_GemmK > 0 && ClusterLengths_GemmN > 0))
    return std::make_tuple(-1, -1, -1, -1, failure());

  if (1) // if(ctx.IsFp32())
  {
    return std::make_tuple(ClusterLengths_GemmK, ClusterLengths_GemmN,
                           SrcDataPerRead_GemmN, DstDataPerWrite_GemmN,
                           success());
  } else {
    return std::make_tuple(ClusterLengths_GemmK, ClusterLengths_GemmN,
                           SrcDataPerRead_GemmN, DstDataPerWrite_GemmKPACK,
                           success());
  }
}

std::tuple<int64_t, LogicalResult> PerformanceImplicitGemmBwdDataV1R1::
    CalculateGemmCThreadCopyPerformanceParameters(
        const ConvolutionContext &ctx) const {
  int64_t DstDataPerWrite_GemmN1 = 4;

  // GemmCThreadCopyDstDataPerWrite_GemmN1 bounded by size of threadwise GEMM
  DstDataPerWrite_GemmN1 =
      ImplicitGemmUtil::gcd(DstDataPerWrite_GemmN1, GemmNPerThread);

  // GemmCThreadCopyDstDataPerWrite_GemmN1 limited by global memory layout of
  // input tensor calculate vector length on gemmn dimension
  auto dimIndexVal = ctx.dimIndexVal;
  const auto y = dimIndexVal["y"].second;
  const auto x = dimIndexVal["x"].second;
  const auto hi = dimIndexVal["hi"].second;
  const auto wi = dimIndexVal["wi"].second;
  const auto conv_stride_h = ctx.strideVal[0];
  const auto conv_stride_w = ctx.strideVal[1];
  const auto conv_dilation_w = ctx.dilationVal[1];
  const auto in_left_pad_h = ctx.paddingVal[0];
  const auto in_left_pad_w = ctx.paddingVal[1];
  const auto in_right_pad_h = ctx.paddingVal[2];
  const auto in_right_pad_w = ctx.paddingVal[3];

  if (y == 1 && x == 1 && conv_stride_h == 1 && conv_stride_w == 1 &&
      in_left_pad_h == 0 && in_left_pad_w == 0 && in_right_pad_h == 0 &&
      in_right_pad_w == 0) {
    // \todo there are more configs that can go through this if branch
    DstDataPerWrite_GemmN1 =
        ImplicitGemmUtil::gcd(DstDataPerWrite_GemmN1, hi * wi);
  } else if (conv_stride_w == 1) {
    DstDataPerWrite_GemmN1 =
        ImplicitGemmUtil::gcd(DstDataPerWrite_GemmN1, in_left_pad_w, wi,
                              in_right_pad_w, conv_dilation_w);
  } else {
    DstDataPerWrite_GemmN1 = 1;
  }

  return std::make_tuple(DstDataPerWrite_GemmN1, success());
}

std::tuple<std::size_t, LogicalResult>
PerformanceImplicitGemmBwdDataV1R1::CalculateLdsNumberOfByte(
    const ConvolutionContext &ctx) const {
  std::size_t lds_size = 0;

  LogicalResult valid = failure();

  int64_t GemmABlockCopyDescDataPerWriteGemm = 0;
  std::tie(std::ignore, std::ignore, std::ignore,
           GemmABlockCopyDescDataPerWriteGemm, valid) =
      CalculateGemmABlockCopyPerformanceParameters(ctx);

  if (failed(valid))
    return std::make_tuple(0, failure());

  int64_t GemmBBlockCopyDescDataPerWriteGemm = 0;
  std::tie(std::ignore, std::ignore, std::ignore,
           GemmBBlockCopyDescDataPerWriteGemm, valid) =
      CalculateGemmBBlockCopyPerformanceParameters(ctx);

  if (failed(valid))
    return std::make_tuple(0, failure());

  const auto ThreadGemmDataPerRead_GemmM = GemmMPerThread;
  const auto ThreadGemmDataPerRead_GemmN = GemmNPerThread;

  const auto max_lds_align = ImplicitGemmUtil::lcm(
      GemmABlockCopyDescDataPerWriteGemm, GemmBBlockCopyDescDataPerWriteGemm,
      ThreadGemmDataPerRead_GemmM, ThreadGemmDataPerRead_GemmN);

  const auto a_block_space =
      GemmKPerBlock *
      ImplicitGemmUtil::integer_least_multiple(GemmMPerBlock, max_lds_align);
  const auto b_block_space =
      GemmKPerBlock *
      ImplicitGemmUtil::integer_least_multiple(GemmNPerBlock, max_lds_align);

  lds_size = 2 * (a_block_space + b_block_space) * sizeof(float);

  return std::make_tuple(lds_size, success());
}

LogicalResult PerformanceImplicitGemmBwdDataV1R1::IsValidValue() const {
  // clang-format off
  if( ImplicitGemmUtil::IsTwoPower<64, 256>(BlockSize) &&
      ImplicitGemmUtil::IsTwoPower<32, 128>(GemmMPerBlock) &&
      ImplicitGemmUtil::IsTwoPower<32, 128>(GemmNPerBlock) &&
      ImplicitGemmUtil::IsTwoPower<4, 16>(GemmKPerBlock) &&
      ImplicitGemmUtil::IsTwoPower<2, 4>(GemmMPerThread) &&
      ImplicitGemmUtil::IsTwoPower<2, 4>(GemmNPerThread))
    return success();

  return failure();
  // clang-format on
}

LogicalResult PerformanceImplicitGemmBwdDataV1R1::IsValid(
    const ConvolutionContext &ctx) const {
  if (failed(IsValidValue()))
    return failure();

  LogicalResult valid = failure();

  // check blockwise GEMM size
  int64_t gemm_m = 0;
  int64_t gemm_n = 0;
  int64_t gemm_k = 0;

  std::tie(gemm_m, gemm_n, gemm_k) =
      ConvHipImplicitGemmBwdDataV1R1::CalculateGemmSize(ctx);

  if (!(gemm_m % GemmMPerBlock == 0 && gemm_n % GemmNPerBlock == 0 &&
        gemm_k % GemmKPerBlock == 0))
    return failure();

  if (!(GemmMPerBlock % GemmMPerThread == 0 &&
        GemmNPerBlock % GemmNPerThread == 0))
    return failure();

  // check thread cluster in blockwise GEMM
  std::tie(std::ignore, std::ignore, std::ignore, std::ignore, valid) =
      CalculateBlockGemmPerformanceParameters(ctx);

  if (failed(valid))
    return failure();

  // check blockwise copy of A matrix
  std::tie(std::ignore, std::ignore, std::ignore, std::ignore, valid) =
      CalculateGemmABlockCopyPerformanceParameters(ctx);

  if (failed(valid))
    return failure();

  // check blockwise copy of B matrix
  std::tie(std::ignore, std::ignore, std::ignore, std::ignore, valid) =
      CalculateGemmBBlockCopyPerformanceParameters(ctx);

  if (failed(valid))
    return failure();

  // check threadwise copy of C matrix
  std::tie(std::ignore, valid) =
      CalculateGemmCThreadCopyPerformanceParameters(ctx);

  if (failed(valid))
    return failure();

  // check LDS allocation
  std::size_t lds_size = 0;
  std::tie(lds_size, valid) = CalculateLdsNumberOfByte(ctx);

  if (succeeded(valid) &&
      lds_size <= ImplicitGemmUtil::get_lds_max_number_of_byte())
    return success();

  return failure();
}

LogicalResult PerformanceImplicitGemmBwdDataV1R1::EuristicInit(
    const ConvolutionContext &ctx) {
  PerformanceImplicitGemmBwdDataV1R1 config;

  using ConfigParam = llvm::SmallVector<int64_t, 6>;
  // blockSize M/block N/block K/block M/thread N/thread
  llvm::SmallVector<ConfigParam, 21> initParams = {
      // clang-format off
      {256, 128, 128, 8, 4, 4},
      {128, 128, 64, 8, 4, 4},
      {128, 64, 128, 4, 4, 4},
      {64, 64, 64, 16, 4, 4},
      {64, 32, 64, 16, 2, 4},
      {64, 64, 32, 16, 4, 2},
      {64, 32, 32, 4, 2, 2}};
  // clang-format on
  /* MIOpen logic */
  /*      {256, 128, 128, 16, 4, 4}, {256, 128, 128, 8, 4, 4},
        {256, 128, 128, 4, 4, 4},  {128, 128, 64, 16, 4, 4},
        {128, 128, 64, 8, 4, 4},   {128, 128, 64, 4, 4, 4},
        {128, 64, 128, 16, 4, 4},  {128, 64, 128, 8, 4, 4},
        {128, 64, 128, 4, 4, 4},   {64, 64, 64, 16, 4, 4},
        {64, 64, 64, 8, 4, 4},     {64, 64, 64, 4, 4, 4},
        {64, 64, 32, 16, 4, 2},    {64, 64, 32, 8, 4, 2},
        {64, 64, 32, 4, 4, 2},     {64, 32, 64, 16, 2, 4},
        {64, 32, 64, 8, 2, 4},     {64, 32, 64, 4, 2, 4},
        {64, 32, 32, 16, 2, 2},    {64, 32, 32, 8, 2, 2},
        {64, 32, 32, 4, 2, 2}};
  */
  for (auto &param : initParams) {
    config = {param[0], param[1], param[2], param[3], param[4], param[5]};
    if (succeeded(config.IsValid(ctx))) {
      LLVM_DEBUG(llvm::dbgs() << "Successfully picked tuning params from backup"
                              << " path.\n");
      *this = config;
      return success();
    }
  }
  LLVM_DEBUG(
      llvm::dbgs() << "FATAL ERROR! COULD NOT FIND VALID TUNING PARAMETERS!\n");
  return failure();
}

std::tuple<int64_t, int64_t, int64_t>
ConvHipImplicitGemmBwdDataV1R1::CalculateGemmSize(
    const ConvolutionContext &ctx) {
  auto dimIndexVal = ctx.dimIndexVal;
  const auto n = dimIndexVal["no"].second;
  const auto k = dimIndexVal["k"].second;
  const auto c = dimIndexVal["c"].second;
  const auto ho = dimIndexVal["ho"].second;
  const auto wo = dimIndexVal["wo"].second;
  const auto y = dimIndexVal["y"].second;
  const auto x = dimIndexVal["x"].second;

  const auto gemm_m = c * y * x;
  const auto gemm_n = n * ho * wo;
  const auto gemm_k = k; /// GetEPackLength(ctx, false);

  return std::make_tuple(gemm_m, gemm_n, gemm_k);
}

LogicalResult ConvHipImplicitGemmBwdDataV1R1::IsApplicable(
    const ConvolutionContext &ctx) const {
  if (!(ctx.opType == miopen::ConvOpType::Conv2DBwdDataOpType))
    return failure();

  if (ctx.isXdlOp)
    return failure();

  if (!(ctx.IsF32() || ctx.IsBF16()))
    return failure();

  int64_t gemm_m = 0;
  int64_t gemm_n = 0;
  int64_t gemm_k = 0;

  std::tie(gemm_m, gemm_n, gemm_k) = CalculateGemmSize(ctx);

  if (gemm_m % 32 == 0 && gemm_n % 32 == 0 && gemm_k % 4 == 0)
    return success();
  return failure();
}

PerformanceImplicitGemmBwdDataV1R1
ConvHipImplicitGemmBwdDataV1R1::GetPerformanceConfig(
    const ConvolutionContext &ctx) const {
  return GetPerformanceConfigBase<PerformanceImplicitGemmBwdDataV1R1>(ctx);
}

LogicalResult ConvHipImplicitGemmBwdDataV1R1::IsValidPerformanceConfig(
    const ConvolutionContext &ctx,
    const PerformanceImplicitGemmBwdDataV1R1 &config) const {
  return config.IsValid(ctx);
}

llvm::StringMap<int64_t> ConvHipImplicitGemmBwdDataV1R1::GetSolution(
    const ConvolutionContext &ctx,
    const PerformanceImplicitGemmBwdDataV1R1 &config) const {
  llvm::StringMap<int64_t> result;

  int64_t grid_size = 0;

  std::tie(grid_size, std::ignore) = config.CalculateGridSize(ctx);

  int64_t GemmMLevel0Cluster = 0;
  int64_t GemmNLevel0Cluster = 0;
  int64_t GemmMLevel1Cluster = 0;
  int64_t GemmNLevel1Cluster = 0;
  int64_t GemmABlockCopyClusterLengths_GemmK = 0;
  int64_t GemmABlockCopyClusterLengths_GemmM = 0;
  int64_t GemmABlockCopySrcDataPerRead_GemmM = 0;
  int64_t GemmABlockCopyDstDataPerWrite_GemmM = 0;
  int64_t GemmABlockCopyDstDataPerWrite_GemmKPACK = 0;
  int64_t GemmBBlockCopyClusterLengths_GemmK = 0;
  int64_t GemmBBlockCopyClusterLengths_GemmN = 0;
  int64_t GemmBBlockCopySrcDataPerRead_GemmN = 0;
  int64_t GemmBBlockCopyDstDataPerWrite_GemmN = 0;
  int64_t GemmBBlockCopyDstDataPerWrite_GemmKPACK = 0;
  int64_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 0;

  std::tie(GemmMLevel0Cluster, GemmNLevel0Cluster, GemmMLevel1Cluster,
           GemmNLevel1Cluster, std::ignore) =
      config.CalculateBlockGemmPerformanceParameters(ctx);

  if (1) { // if(ctx.IsFp32())
    std::tie(
        GemmABlockCopyClusterLengths_GemmK, GemmABlockCopyClusterLengths_GemmM,
        GemmABlockCopySrcDataPerRead_GemmM, GemmABlockCopyDstDataPerWrite_GemmM,
        std::ignore) = config.CalculateGemmABlockCopyPerformanceParameters(ctx);

    std::tie(
        GemmBBlockCopyClusterLengths_GemmK, GemmBBlockCopyClusterLengths_GemmN,
        GemmBBlockCopySrcDataPerRead_GemmN, GemmBBlockCopyDstDataPerWrite_GemmN,
        std::ignore) = config.CalculateGemmBBlockCopyPerformanceParameters(ctx);
  } else {
    std::tie(GemmABlockCopyClusterLengths_GemmK,
             GemmABlockCopyClusterLengths_GemmM,
             GemmABlockCopySrcDataPerRead_GemmM,
             GemmABlockCopyDstDataPerWrite_GemmKPACK, std::ignore) =
        config.CalculateGemmABlockCopyPerformanceParameters(ctx);

    std::tie(GemmBBlockCopyClusterLengths_GemmK,
             GemmBBlockCopyClusterLengths_GemmN,
             GemmBBlockCopySrcDataPerRead_GemmN,
             GemmBBlockCopyDstDataPerWrite_GemmKPACK, std::ignore) =
        config.CalculateGemmBBlockCopyPerformanceParameters(ctx);
  }

  std::tie(GemmCThreadCopyDstDataPerWrite_GemmN1, std::ignore) =
      config.CalculateGemmCThreadCopyPerformanceParameters(ctx);

  result["block_size"] = config.BlockSize;
  result["m_per_block"] = config.GemmMPerBlock;
  result["n_per_block"] = config.GemmNPerBlock;
  result["k_per_block"] = config.GemmKPerBlock;
  result["m_per_thread"] = config.GemmMPerThread;
  result["n_per_thread"] = config.GemmNPerThread;

  result["grid_size"] = grid_size;
  result["m_level0_cluster"] = GemmMLevel0Cluster;
  result["n_level0_cluster"] = GemmNLevel0Cluster;
  result["m_level1_cluster"] = GemmMLevel1Cluster;
  result["n_level1_cluster"] = GemmNLevel1Cluster;
  result["matrix_a_cluster_lengths_gemmk"] = GemmABlockCopyClusterLengths_GemmK;
  result["matrix_a_cluster_lengths_gemmM"] = GemmABlockCopyClusterLengths_GemmM;
  result["matrix_a_source_data_per_read"] = GemmABlockCopySrcDataPerRead_GemmM;
  result["matrix_a_dest_data_per_write_dim_m"] =
      GemmABlockCopyDstDataPerWrite_GemmM;
  result["matrix_b_cluster_lengths_gemmk"] = GemmBBlockCopyClusterLengths_GemmK;
  result["matrix_b_cluster_lengths_gemmN"] = GemmBBlockCopyClusterLengths_GemmN;
  result["matrix_b_source_data_per_read"] = GemmBBlockCopySrcDataPerRead_GemmN;
  result["matrix_b_dest_data_per_write_dim_n"] =
      GemmBBlockCopyDstDataPerWrite_GemmN;
  result["matrix_c_dest_data_per_write"] =
      GemmCThreadCopyDstDataPerWrite_GemmN1;

  result["matrix_a_source_vector_read_dim"] =
      CalculateGemmASrcVectorReadDim(ctx);
  result["matrix_b_source_vector_read_dim"] =
      CalculateGemmBSrcVectorReadDim(ctx);

  for (llvm::StringMap<int64_t>::iterator it = result.begin();
       it != result.end(); ++it)
    LLVM_DEBUG(llvm::dbgs() << it->first() << "=" << it->second << "\n");

  return result;
}
