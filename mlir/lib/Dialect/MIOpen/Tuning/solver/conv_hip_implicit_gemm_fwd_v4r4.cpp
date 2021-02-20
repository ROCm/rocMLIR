#include "mlir/Dialect/MIOpen/Tuning/GridwiseGemmParams.h"

#define DEBUG_TYPE "miopen-tuning-parameter"

PerformanceImplicitGemmV4R4Fwd::PerformanceImplicitGemmV4R4Fwd(
    int64_t BlockSize_, int64_t GemmMPerBlock_, int64_t GemmNPerBlock_,
    int64_t GemmKPerBlock_, int64_t GemmMPerThread_, int64_t GemmNPerThread_)
    : BlockSize(BlockSize_), GemmMPerBlock(GemmMPerBlock_),
      GemmNPerBlock(GemmNPerBlock_), GemmKPerBlock(GemmKPerBlock_),
      GemmMPerThread(GemmMPerThread_), GemmNPerThread(GemmNPerThread_) {}

bool PerformanceImplicitGemmV4R4Fwd::operator==(
    const PerformanceImplicitGemmV4R4Fwd &other) const {
  return BlockSize == other.BlockSize && GemmMPerBlock == other.GemmMPerBlock &&
         GemmNPerBlock == other.GemmNPerBlock &&
         GemmKPerBlock == other.GemmKPerBlock &&
         GemmMPerThread == other.GemmMPerThread &&
         GemmNPerThread == other.GemmNPerThread;
}

std::tuple<int64_t, LogicalResult>
PerformanceImplicitGemmV4R4Fwd::CalculateGridSize(
    const ConvolutionContext &ctx) const {
  int64_t GridSize = 0;
  int64_t gemmM = 0;
  int64_t gemmN = 0;

  std::tie(gemmM, gemmN, std::ignore) =
      ConvHipImplicitGemmV4R4Fwd::CalculateGemmSize(ctx);

  if (!(gemmM % GemmMPerBlock == 0 && gemmN % GemmNPerBlock == 0))
    return std::make_tuple(-1, failure());
  llvm::errs() << gemmM << " " << GemmMPerBlock << " " << gemmN << " "
               << GemmNPerBlock << "\n";
  GridSize = (gemmM / GemmMPerBlock) * (gemmN / GemmNPerBlock);
  return std::make_tuple(GridSize, success());
}

std::tuple<int64_t, int64_t, int64_t, int64_t, LogicalResult>
PerformanceImplicitGemmV4R4Fwd::CalculateBlockGemmPerformanceParameters(
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
PerformanceImplicitGemmV4R4Fwd::CalculateGemmABlockCopyPerformanceParameters(
    const ConvolutionContext &) const {
  int64_t ClusterLengths_GemmK = 0;
  int64_t ClusterLengths_GemmM = 0;
  int64_t SrcDataPerRead_GemmK = 4;
  int64_t DstDataPerWrite_GemmM = 4;

  // calculate vector length on gemmk dimension
  SrcDataPerRead_GemmK = gcd(SrcDataPerRead_GemmK, GemmKPerBlock);

  // calculate threadwise copy size
  const auto a_data_per_thread_copy =
      (GemmKPerBlock * GemmMPerBlock) / BlockSize;

  if (!(a_data_per_thread_copy > 0))
    return std::make_tuple(-1, -1, -1, -1, failure());

  // GemmABlockCopySrcDataPerRead_GemmK also bounded by size of threadwise copy
  SrcDataPerRead_GemmK = gcd(SrcDataPerRead_GemmK, a_data_per_thread_copy);

  // decide threadwise copy lengths
  const auto a_data_per_thread_copy_gemmk = SrcDataPerRead_GemmK;
  const auto a_data_per_thread_copy_gemmm =
      a_data_per_thread_copy / a_data_per_thread_copy_gemmk;

  // GemmABlockCopyDstDataPerWrite_GemmM also bounded by size of threadwise copy
  DstDataPerWrite_GemmM =
      gcd(DstDataPerWrite_GemmM, a_data_per_thread_copy_gemmm);

  // calculate blockwise copy thread cluster lengths
  ClusterLengths_GemmK = GemmKPerBlock / a_data_per_thread_copy_gemmk;
  ClusterLengths_GemmM = GemmMPerBlock / a_data_per_thread_copy_gemmm;

  if (!(ClusterLengths_GemmK > 0 && ClusterLengths_GemmM > 0))
    return std::make_tuple(-1, -1, -1, -1, failure());

  return std::make_tuple(ClusterLengths_GemmK, ClusterLengths_GemmM,
                         SrcDataPerRead_GemmK, DstDataPerWrite_GemmM,
                         success());
}

std::tuple<int64_t, int64_t, int64_t, int64_t, LogicalResult>
PerformanceImplicitGemmV4R4Fwd::CalculateGemmBBlockCopyPerformanceParameters(
    const ConvolutionContext &ctx) const {
  int64_t ClusterLengths_GemmK = 0;
  int64_t ClusterLengths_GemmN = 0;
  int64_t SrcDataPerRead_GemmN = 4;
  int64_t DstDataPerWrite_GemmN = 4;

  SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, GemmNPerBlock);

  // calculate vector length on gemmn dimension
  /* MIOpen logic */
  /*
    auto dimIndexVal = ctx.dimIndexVal;
    auto y = dimIndexVal["y"].second;
    auto x = dimIndexVal["x"].second;
    auto hi = dimIndexVal["hi"].second;
    auto wi = dimIndexVal["wi"].second;
    auto conv_stride_h = ctx.strideVal[0];
    auto conv_stride_w = ctx.strideVal[1];
    auto conv_dilation_w = ctx.dilationVal[1];
    auto in_left_pad_h = ctx.paddingVal[0];
    auto in_left_pad_w = ctx.paddingVal[1];
    auto in_right_pad_h = ctx.paddingVal[2];
    auto in_right_pad_w = ctx.paddingVal[3];

    if (y == 1 && x == 1 && conv_stride_h == 1 && conv_stride_w == 1 &&
        in_left_pad_h == 0 && in_left_pad_w == 0 && in_right_pad_h == 0 &&
        in_right_pad_w == 0) {
      // \todo there are more configs that can go through this if branch
      SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, hi * wi);
    } else if (conv_stride_w == 1) {
      SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, in_left_pad_w, wi,
                                 in_right_pad_w, conv_dilation_w);
    } else {
      SrcDataPerRead_GemmN = 1;
    }
  */
  // calculate threadwise copy size
  const auto b_data_per_thread_copy =
      (GemmKPerBlock * GemmNPerBlock) / BlockSize;

  if (!(b_data_per_thread_copy > 0))
    return std::make_tuple(-1, -1, -1, -1, failure());

  // GemmBBlockCopySrcDataPerRead_GemmN also bounded by size of threadwise copy
  SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, b_data_per_thread_copy);

  const auto b_data_per_thread_copy_gemmn = SrcDataPerRead_GemmN;
  const auto b_data_per_thread_copy_gemmk =
      b_data_per_thread_copy / b_data_per_thread_copy_gemmn;

  // GemmBBlockCopyDstDataPerWrite_GemmN also bounded by size of threadwise copy
  DstDataPerWrite_GemmN =
      gcd(DstDataPerWrite_GemmN, b_data_per_thread_copy_gemmn);

  // calculate blockwise copy thread cluster lengths
  ClusterLengths_GemmK = GemmKPerBlock / b_data_per_thread_copy_gemmk;
  ClusterLengths_GemmN = GemmNPerBlock / b_data_per_thread_copy_gemmn;

  if (!(ClusterLengths_GemmK > 0 && ClusterLengths_GemmN > 0))
    return std::make_tuple(-1, -1, -1, -1, failure());

  return std::make_tuple(ClusterLengths_GemmK, ClusterLengths_GemmN,
                         SrcDataPerRead_GemmN, DstDataPerWrite_GemmN,
                         success());
}

std::tuple<int64_t, LogicalResult>
PerformanceImplicitGemmV4R4Fwd::CalculateGemmCThreadCopyPerformanceParameters(
    const ConvolutionContext &ctx) const {
  int64_t DstDataPerWrite_GemmN1 = 4;

  // GemmCThreadCopyDstDataPerWrite_GemmN1 bounded by size of threadwise GEMM
  DstDataPerWrite_GemmN1 = gcd(DstDataPerWrite_GemmN1, GemmNPerThread);

  // GemmCThreadCopyDstDataPerWrite_GemmN1 limited by global memory layout of
  // output tensor
  auto dimIndexVal = ctx.dimIndexVal;
  auto ho = dimIndexVal["ho"].second;
  auto wo = dimIndexVal["wo"].second;
  DstDataPerWrite_GemmN1 = gcd(DstDataPerWrite_GemmN1, ho * wo);

  return std::make_tuple(DstDataPerWrite_GemmN1, success());
}

std::tuple<std::size_t, LogicalResult>
PerformanceImplicitGemmV4R4Fwd::CalculateLdsNumberOfByte(
    const ConvolutionContext &ctx) const {
  std::size_t lds_size = 0;

  LogicalResult res(LogicalResult::Failure);

  int64_t GemmABlockCopyDescDataPerWriteGemmM = 0;
  std::tie(std::ignore, std::ignore, std::ignore,
           GemmABlockCopyDescDataPerWriteGemmM, res) =
      CalculateGemmABlockCopyPerformanceParameters(ctx);

  if (failed(res))
    return std::make_tuple(0, failure());

  int64_t GemmBBlockCopyDescDataPerWriteGemmN = 0;
  std::tie(std::ignore, std::ignore, std::ignore,
           GemmBBlockCopyDescDataPerWriteGemmN, res) =
      CalculateGemmBBlockCopyPerformanceParameters(ctx);

  if (failed(res))
    return std::make_tuple(0, failure());

  const auto ThreadGemmDataPerRead_GemmM = GemmMPerThread;
  const auto ThreadGemmDataPerRead_GemmN = GemmNPerThread;

  const auto max_lds_align = lcm(
      GemmABlockCopyDescDataPerWriteGemmM, GemmBBlockCopyDescDataPerWriteGemmN,
      ThreadGemmDataPerRead_GemmM, ThreadGemmDataPerRead_GemmN);

  const auto a_block_space =
      GemmKPerBlock * integer_least_multiple(GemmMPerBlock, max_lds_align);
  const auto b_block_space =
      GemmKPerBlock * integer_least_multiple(GemmNPerBlock, max_lds_align);

  lds_size = 2 * (a_block_space + b_block_space) * sizeof(float);

  return std::make_tuple(lds_size, success());
}

LogicalResult PerformanceImplicitGemmV4R4Fwd::IsValidValue() const {
  // clang-format off
  if ( IsTwoPower<64, 256>(BlockSize) &&
       IsTwoPower<32, 128>(GemmMPerBlock) &&
       IsTwoPower<32, 128>(GemmNPerBlock) &&
       IsTwoPower<4, 16>(GemmKPerBlock) &&
       IsTwoPower<2, 4>(GemmMPerThread) &&
       IsTwoPower<2, 4>(GemmNPerThread) )
    return success();
  else
    return failure();
  // clang-format on
}

LogicalResult
PerformanceImplicitGemmV4R4Fwd::IsValid(const ConvolutionContext &ctx) const {
  LogicalResult res(LogicalResult::Failure);

  res = IsValidValue();

  if (failed(res))
    return failure();

  // check blockwise GEMM size
  int64_t gemm_m = 0;
  int64_t gemm_n = 0;
  int64_t gemm_k = 0;

  std::tie(gemm_m, gemm_n, gemm_k) =
      ConvHipImplicitGemmV4R4Fwd::CalculateGemmSize(ctx);

  if (!(gemm_m % GemmMPerBlock == 0 && gemm_n % GemmNPerBlock == 0 &&
        gemm_k % GemmKPerBlock == 0))
    return failure();

  if (!(GemmMPerBlock % GemmMPerThread == 0 &&
        GemmNPerBlock % GemmNPerThread == 0))
    return failure();

  // check thread cluster in blockwise GEMM
  std::tie(std::ignore, std::ignore, std::ignore, std::ignore, res) =
      CalculateBlockGemmPerformanceParameters(ctx);

  if (failed(res))
    return failure();

  // check blockwise copy of A matrix
  std::tie(std::ignore, std::ignore, std::ignore, std::ignore, res) =
      CalculateGemmABlockCopyPerformanceParameters(ctx);

  if (failed(res))
    return failure();

  // check blockwise copy of B matrix
  std::tie(std::ignore, std::ignore, std::ignore, std::ignore, res) =
      CalculateGemmBBlockCopyPerformanceParameters(ctx);

  if (failed(res))
    return failure();

  // check threadwise copy of C matrix
  std::tie(std::ignore, res) =
      CalculateGemmCThreadCopyPerformanceParameters(ctx);

  if (failed(res))
    return failure();

  // check LDS allocation
  std::size_t lds_size = 0;
  std::tie(lds_size, res) = CalculateLdsNumberOfByte(ctx);

  if (succeeded(res) && lds_size <= get_lds_max_number_of_byte())
    return success();

  return failure();
}

LogicalResult
PerformanceImplicitGemmV4R4Fwd::EuristicInit(const ConvolutionContext &ctx) {
  PerformanceImplicitGemmV4R4Fwd config;

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
  /*
        {256, 128, 128, 16, 4, 4}, {256, 128, 128, 8, 4, 4},
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

  llvm::errs() << "FATAL ERROR! COULD NOT FIND VALID TUNING PARAMETERS!\n";
  return failure();
}

std::tuple<int64_t, int64_t, int64_t>
ConvHipImplicitGemmV4R4Fwd::CalculateGemmSize(const ConvolutionContext &ctx) {
  auto dimIndexVal = ctx.dimIndexVal;
  const auto n = dimIndexVal["no"].second;
  const auto k = dimIndexVal["k"].second;
  const auto c = dimIndexVal["c"].second;
  const auto ho = dimIndexVal["ho"].second;
  const auto wo = dimIndexVal["wo"].second;
  const auto y = dimIndexVal["y"].second;
  const auto x = dimIndexVal["x"].second;
  llvm::errs() << "n=" << n << " ho=" << ho << " wo=" << wo << "\n";
  const auto gemm_m = k;
  const auto gemm_n = n * ho * wo;
  const auto gemm_k = c * y * x;

  return std::make_tuple(gemm_m, gemm_n, gemm_k);
}

LogicalResult
ConvHipImplicitGemmV4R4Fwd::IsApplicable(const ConvolutionContext &ctx) const {
  if (!(ctx.opType == miopen::ConvOpType::Conv2DOpType))
    return failure();

  if (ctx.isXdlOp)
    return failure();

  int64_t gemm_m = 0;
  int64_t gemm_n = 0;
  int64_t gemm_k = 0;

  std::tie(gemm_m, gemm_n, gemm_k) = CalculateGemmSize(ctx);
  if (gemm_m % 32 == 0 && gemm_n % 32 == 0 && gemm_k % 4 == 0)
    return success();
  return failure();
}

PerformanceImplicitGemmV4R4Fwd ConvHipImplicitGemmV4R4Fwd::GetPerformanceConfig(
    const ConvolutionContext &ctx) const {
  return GetPerformanceConfigBase<PerformanceImplicitGemmV4R4Fwd>(ctx);
}

LogicalResult ConvHipImplicitGemmV4R4Fwd::IsValidPerformanceConfig(
    const ConvolutionContext &ctx,
    const PerformanceImplicitGemmV4R4Fwd &config) const {
  return config.IsValid(ctx);
}

llvm::StringMap<int64_t> ConvHipImplicitGemmV4R4Fwd::GetSolution(
    const ConvolutionContext &ctx,
    const PerformanceImplicitGemmV4R4Fwd &config) const {
  llvm::StringMap<int64_t> result;
  int64_t grid_size = 0;

  std::tie(grid_size, std::ignore) = config.CalculateGridSize(ctx);

  int64_t GemmMLevel0Cluster = 0;
  int64_t GemmNLevel0Cluster = 0;
  int64_t GemmMLevel1Cluster = 0;
  int64_t GemmNLevel1Cluster = 0;
  int64_t GemmABlockCopyClusterLengths_GemmK = 0;
  int64_t GemmABlockCopyClusterLengths_GemmM = 0;
  int64_t GemmABlockCopySrcDataPerRead_GemmK = 0;
  int64_t GemmABlockCopyDstDataPerWrite_GemmM = 0;
  int64_t GemmBBlockCopyClusterLengths_GemmK = 0;
  int64_t GemmBBlockCopyClusterLengths_GemmN = 0;
  int64_t GemmBBlockCopySrcDataPerRead_GemmN = 0;
  int64_t GemmBBlockCopyDstDataPerWrite_GemmN = 0;
  int64_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 0;

  std::tie(GemmMLevel0Cluster, GemmNLevel0Cluster, GemmMLevel1Cluster,
           GemmNLevel1Cluster, std::ignore) =
      config.CalculateBlockGemmPerformanceParameters(ctx);

  std::tie(
      GemmABlockCopyClusterLengths_GemmK, GemmABlockCopyClusterLengths_GemmM,
      GemmABlockCopySrcDataPerRead_GemmK, GemmABlockCopyDstDataPerWrite_GemmM,
      std::ignore) = config.CalculateGemmABlockCopyPerformanceParameters(ctx);

  std::tie(
      GemmBBlockCopyClusterLengths_GemmK, GemmBBlockCopyClusterLengths_GemmN,
      GemmBBlockCopySrcDataPerRead_GemmN, GemmBBlockCopyDstDataPerWrite_GemmN,
      std::ignore) = config.CalculateGemmBBlockCopyPerformanceParameters(ctx);

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
  result["matrix_a_source_data_per_read"] = GemmABlockCopySrcDataPerRead_GemmK;
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
