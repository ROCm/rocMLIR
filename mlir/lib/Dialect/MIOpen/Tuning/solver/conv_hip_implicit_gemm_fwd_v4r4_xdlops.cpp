#include "mlir/Dialect/MIOpen/Tuning/GridwiseGemmParams.h"

#define DEBUG_TYPE "miopen-tuning-parameter"

PerformanceImplicitGemmForwardV4R4Xdlops::
    PerformanceImplicitGemmForwardV4R4Xdlops()
    : PerformanceImplicitGemmForwardV4R4Xdlops::
          PerformanceImplicitGemmForwardV4R4Xdlops(4, 4, 1, 4, 4, 1, false,
                                                   false, 1) {}

PerformanceImplicitGemmForwardV4R4Xdlops::
    PerformanceImplicitGemmForwardV4R4Xdlops(
        int64_t GemmMPerBlock_, int64_t GemmNPerBlock_, int64_t GemmKPerBlock_,
        int64_t GemmMPerWave_, int64_t GemmNPerWave_, int64_t GemmKPack_,
        bool GemmAThreadCopyMoreGemmK_, bool GemmBThreadCopyMoreGemmKPack_,
        int64_t GemmBThreadDataPerRead_GemmN_)
    : GemmMPerBlock(GemmMPerBlock_), GemmNPerBlock(GemmNPerBlock_),
      GemmKPerBlock(GemmKPerBlock_), GemmMPerWave(GemmMPerWave_),
      GemmNPerWave(GemmNPerWave_), GemmKPack(GemmKPack_),
      GemmAThreadCopyMoreGemmK(GemmAThreadCopyMoreGemmK_),
      GemmBThreadCopyMoreGemmKPack(GemmBThreadCopyMoreGemmKPack_),
      GemmBThreadDataPerRead_GemmN(GemmBThreadDataPerRead_GemmN_) {}

bool PerformanceImplicitGemmForwardV4R4Xdlops::operator==(
    const PerformanceImplicitGemmForwardV4R4Xdlops &other) const {
  // clang-format off
    return GemmMPerBlock == other.GemmMPerBlock
        && GemmNPerBlock == other.GemmNPerBlock
        && GemmKPerBlock == other.GemmKPerBlock
        && GemmMPerWave == other.GemmMPerWave
        && GemmNPerWave == other.GemmNPerWave
        && GemmKPack == other.GemmKPack
        && GemmAThreadCopyMoreGemmK  == other.GemmAThreadCopyMoreGemmK
        && GemmBThreadCopyMoreGemmKPack  == other.GemmBThreadCopyMoreGemmKPack
        && GemmBThreadDataPerRead_GemmN  == other.GemmBThreadDataPerRead_GemmN;
  // clang-format on
}

LogicalResult PerformanceImplicitGemmForwardV4R4Xdlops::EuristicInit(
    const ConvolutionContext &ctx) {
  PerformanceImplicitGemmForwardV4R4Xdlops tmp;

  // loop over certain ranges of tuning parameter
  auto get_euristic_config = [&](auto is_valid_func) {
    /* MIOpen logic */
    // tmp = {256, 256, 8, 128, 128, 4, false, true, 1};

    tmp = {256, 256, 8, 128, 128, 4, false, false, 1};
    bool all_visited = false;
    do {
      do {
        // list in reverse order of importance,
        // and favor large GEMM
        if (!PreviousTwoPower<1, 4>(tmp.GemmBThreadDataPerRead_GemmN))
          break;
        if (!PreviousTwoPower<1, 8>(tmp.GemmKPerBlock))
          break;
        if (!PreviousTwoPower<1, 4>(tmp.GemmKPack))
          break;
        if (!PreviousTwoPower<4, 128>(tmp.GemmNPerWave))
          break;
        if (!PreviousTwoPower<4, 128>(tmp.GemmMPerWave))
          break;
        if (!PreviousTwoPower<4, 256>(tmp.GemmNPerBlock))
          break;
        if (!PreviousTwoPower<4, 256>(tmp.GemmMPerBlock))
          break;

        all_visited = true;
      } while (false);

      if (succeeded(is_valid_func(tmp, ctx)))
        break;
    } while (!all_visited);
  };

  // first round: really valid and fast
  get_euristic_config([](auto config, auto conv_context) {
    return config.IsReallyValid(conv_context);
  });

  // final check
  LogicalResult valid = tmp.IsReallyValid(ctx);
  *this = tmp;

  return valid;
}

std::tuple<int64_t, LogicalResult>
PerformanceImplicitGemmForwardV4R4Xdlops::CalculateBlockSize() const {
  int64_t block_size = 0;

  const auto waveSize = 64;

  if (!(GemmMPerBlock % GemmMPerWave == 0 &&
        GemmNPerBlock % GemmNPerWave == 0)) {
    LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: "
                            << "V4R4Xdlop BlockSize\n");
    return std::make_tuple(-1, failure());
  }

  block_size = (GemmNPerBlock * GemmMPerBlock) / (GemmMPerWave * GemmNPerWave) *
               waveSize;

  return std::make_tuple(block_size, success());
}

std::tuple<int64_t, LogicalResult>
PerformanceImplicitGemmForwardV4R4Xdlops::CalculateGridSize(
    const ConvolutionContext &ctx) const {
  int64_t GridSize = 0;

  int64_t gemm_g = -1;
  int64_t gemm_m = -1;
  int64_t gemm_n = -1;

  std::tie(gemm_g, gemm_m, gemm_n, std::ignore) =
      ConvHipImplicitGemmForwardV4R4Xdlops::CalculateGemmSize(ctx);

  if (!(gemm_m % GemmMPerBlock == 0 && gemm_n % GemmNPerBlock == 0)) {
    LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: "
                            << "V4R4Xdlop GridSize\n");
    return std::make_tuple(-1, failure());
  }

  GridSize = gemm_g * (gemm_m / GemmMPerBlock) * (gemm_n / GemmNPerBlock);

  return std::make_tuple(GridSize, success());
}

std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, LogicalResult>
PerformanceImplicitGemmForwardV4R4Xdlops::
    CalculateGemmABlockCopyPerformanceParameters(
        const ConvolutionContext &ctx) const {
  // A tensor shape [GemmG, GemmK, GemmM, GemmKPack]

  int64_t ClusterLengths_GemmK = -1;
  int64_t ClusterLengths_GemmM = -1;
  int64_t ClusterLengths_GemmKPack = -1;
  int64_t SrcDataPerRead_GemmKPack = 4;
  int64_t DstDataPerWrite_GemmKPack = 4;

  LogicalResult valid = failure();

  int64_t block_size = -1;

  std::tie(block_size, valid) = CalculateBlockSize();

  if (failed(valid)) {
    LLVM_DEBUG(llvm::dbgs()
               << "invalid performance parameter: V4R4Xdlop GemmABlockCopy\n");
    return std::make_tuple(-1, -1, -1, -1, -1, failure());
  }

  // GemmKPack is src vector read dimension, bounded by GemmKPack
  SrcDataPerRead_GemmKPack = gcd(SrcDataPerRead_GemmKPack, GemmKPack);

  // calculate threadwise copy size
  auto data_per_thread_copy =
      std::max(static_cast<int64_t>(1),
               (GemmKPerBlock * GemmMPerBlock * GemmKPack) / block_size);

  // make sure a thread can do a full vector load, at the cost that some threads
  // may not do threadwise copy at all
  data_per_thread_copy = lcm(data_per_thread_copy, SrcDataPerRead_GemmKPack);

  const auto data_per_thread_copy_gemmkpack = SrcDataPerRead_GemmKPack;
  const auto tmp = data_per_thread_copy / data_per_thread_copy_gemmkpack;

  if (tmp == 0) {
    LLVM_DEBUG(llvm::dbgs()
               << "invalid performance parameter: V4R4Xdlop GemmABlockCopy\n");
    return std::make_tuple(-1, -1, -1, -1, -1, failure());
  }

  int64_t data_per_thread_copy_gemmk = -1;
  int64_t data_per_thread_copy_gemmm = -1;

  if (GemmAThreadCopyMoreGemmK) {
    data_per_thread_copy_gemmk = gcd(GemmKPerBlock, tmp);
    data_per_thread_copy_gemmm = tmp / data_per_thread_copy_gemmk;
  } else {
    data_per_thread_copy_gemmm = gcd(GemmMPerBlock, tmp);
    data_per_thread_copy_gemmk = tmp / data_per_thread_copy_gemmm;
  }

  // vector write int64_to LDS
  DstDataPerWrite_GemmKPack =
      gcd(DstDataPerWrite_GemmKPack, data_per_thread_copy_gemmkpack);

  if (!(GemmKPerBlock % data_per_thread_copy_gemmk == 0 &&
        GemmMPerBlock % data_per_thread_copy_gemmm == 0 &&
        GemmKPack % data_per_thread_copy_gemmkpack == 0)) {
    LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: "
                            << "V4R4Xdlop GemmABlockCopy\n");
    return std::make_tuple(-1, -1, -1, -1, -1, failure());
  }

  ClusterLengths_GemmK = GemmKPerBlock / data_per_thread_copy_gemmk;
  ClusterLengths_GemmM = GemmMPerBlock / data_per_thread_copy_gemmm;
  ClusterLengths_GemmKPack = GemmKPack / data_per_thread_copy_gemmkpack;

  // blockwise-copy support that block_size is larger than thread cluster size,
  // which means some threads may not do threadwise copy
  if (block_size <
      ClusterLengths_GemmK * ClusterLengths_GemmM * ClusterLengths_GemmKPack) {
    LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: "
                            << "V4R4Xdlop GemmABlockCopy\n");
    return std::make_tuple(-1, -1, -1, -1, -1, failure());
  }

  return std::make_tuple(ClusterLengths_GemmK, ClusterLengths_GemmM,
                         ClusterLengths_GemmKPack, SrcDataPerRead_GemmKPack,
                         DstDataPerWrite_GemmKPack, success());
}

std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, LogicalResult>
PerformanceImplicitGemmForwardV4R4Xdlops::
    CalculateGemmBBlockCopyPerformanceParameters(
        const ConvolutionContext &ctx) const {
  // B tensor shape [GemmG, GemmK, GemmN, GemmKPack]

  int64_t ClusterLengths_GemmK = -1;
  int64_t ClusterLengths_GemmN = -1;
  int64_t ClusterLengths_GemmKPack = -1;
  int64_t SrcDataPerRead_GemmN = 4;
  int64_t DstDataPerWrite_GemmKPack = 4;

  LogicalResult valid = failure();

  int64_t block_size = -1;

  std::tie(block_size, valid) = CalculateBlockSize();

  if (failed(valid))
    LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter\n");
  /* MIOpen logic */
  /*
    // GemmN is src vector read dimension
    // calculate vector length on gemmn dimension based on global tensor layout
    auto dimIndexVal = ctx.dimIndexVal;
    const auto y  = dimIndexVal["y"].second;
    const auto x  = dimIndexVal["x"].second;
    const auto ho = dimIndexVal["ho"].second;
    const auto wo = dimIndexVal["wo"].second;
    const auto conv_stride_h = ctx.strideVal[0];
    const auto conv_stride_w = ctx.strideVal[1];
    const auto conv_dilation_w = ctx.dilationVal[1];
    const auto in_left_pad_h  = ctx.paddingVal[0];
    const auto in_left_pad_w  = ctx.paddingVal[1];
    const auto in_right_pad_h = ctx.paddingVal[2];
    const auto in_right_pad_w = ctx.paddingVal[3];

      // GemmN is src vector read dimension, bounded by input tensor global
    memory layout
      // TODO this logic need to be more aggresive
      if(y == 1 && x == 1 && conv_stride_h == 1 && conv_stride_w == 1 &&
    in_left_pad_h == 0 && in_left_pad_w == 0 && in_right_pad_h == 0 &&
    in_right_pad_w == 0)
      {
          SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, ho * wo);
      }
      else if(conv_stride_w == 1 && in_left_pad_w == 0 && in_right_pad_w == 0)
      {
          SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, wo);
      }
      else if(conv_stride_w == 1)
      {
          SrcDataPerRead_GemmN =
              gcd(SrcDataPerRead_GemmN, wo, in_left_pad_w, in_right_pad_w,
    conv_dilation_w);
      }
      else
      {
          SrcDataPerRead_GemmN = 1;
      }
  */
  // SrcDataPerRead_GemmN also bounded by GemmNPerBlock
  SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, GemmNPerBlock);

  // calculate threadwise copy size
  auto data_per_thread_copy =
      std::max(static_cast<int64_t>(1),
               (GemmKPerBlock * GemmNPerBlock * GemmKPack) / block_size);

  // make sure a thread can do a full vector load, at the cost that some threads
  // may not do threadwise copy at all
  data_per_thread_copy = lcm(data_per_thread_copy, SrcDataPerRead_GemmN);

  const auto data_per_thread_copy_gemmn = SrcDataPerRead_GemmN;
  const auto tmp = data_per_thread_copy / data_per_thread_copy_gemmn;

  int64_t data_per_thread_copy_gemmkpack = -1;
  int64_t data_per_thread_copy_gemmk = -1;
  if (GemmBThreadCopyMoreGemmKPack) {
    data_per_thread_copy_gemmkpack = gcd(GemmKPack, tmp);
    data_per_thread_copy_gemmk = tmp / data_per_thread_copy_gemmkpack;
  } else {
    data_per_thread_copy_gemmk = gcd(GemmKPerBlock, tmp);
    data_per_thread_copy_gemmkpack = tmp / data_per_thread_copy_gemmk;
  }

  // vector write int64_to LDS
  DstDataPerWrite_GemmKPack =
      gcd(DstDataPerWrite_GemmKPack, data_per_thread_copy_gemmkpack);

  if (!(data_per_thread_copy_gemmkpack > 0 && data_per_thread_copy_gemmk > 0 &&
        data_per_thread_copy_gemmn > 0)) {
    LLVM_DEBUG(llvm::dbgs()
               << "invalid performance parameter: GemmBBlockCopy\n");
    return std::make_tuple(-1, -1, -1, -1, -1, failure());
  }
  if (!(GemmKPerBlock % data_per_thread_copy_gemmk == 0 &&
        GemmNPerBlock % data_per_thread_copy_gemmn == 0 &&
        GemmKPack % data_per_thread_copy_gemmkpack == 0)) {
    LLVM_DEBUG(llvm::dbgs()
               << "invalid performance parameter: GemmBBlockCopy\n");
    return std::make_tuple(-1, -1, -1, -1, -1, failure());
  }

  ClusterLengths_GemmK = GemmKPerBlock / data_per_thread_copy_gemmk;
  ClusterLengths_GemmN = GemmNPerBlock / data_per_thread_copy_gemmn;
  ClusterLengths_GemmKPack = GemmKPack / data_per_thread_copy_gemmkpack;

  // blockwise-copy support that block_size is larger than thread cluster size,
  // which means some threads may not do threadwise copy
  if (block_size <
      ClusterLengths_GemmK * ClusterLengths_GemmN * ClusterLengths_GemmKPack) {
    LLVM_DEBUG(llvm::dbgs()
               << "invalid performance parameter: GemmBBlockCopy\n");
    return std::make_tuple(-1, -1, -1, -1, -1, failure());
  }

  return std::make_tuple(ClusterLengths_GemmK, ClusterLengths_GemmN,
                         ClusterLengths_GemmKPack, SrcDataPerRead_GemmN,
                         DstDataPerWrite_GemmKPack, success());
}

std::tuple<std::size_t, LogicalResult>
PerformanceImplicitGemmForwardV4R4Xdlops::CalculateLdsNumberOfByte(
    const ConvolutionContext &ctx) const {
  const auto a_block_space = GemmKPerBlock * GemmMPerBlock * GemmKPack;
  const auto b_block_space = GemmKPerBlock * GemmNPerBlock * GemmKPack;

  std::size_t lds_size = (a_block_space + b_block_space) * sizeof(float);

  return std::make_tuple(lds_size, success());
}

// Used by IsReallyValid()
LogicalResult PerformanceImplicitGemmForwardV4R4Xdlops::IsValidValue() const {
  // clang-format off
  if ( IsTwoPower<4, 256>(GemmMPerBlock)
        && IsTwoPower<4, 256>(GemmNPerBlock)
        && IsTwoPower<1, 8>(GemmKPerBlock)
        && IsTwoPower<4, 128>(GemmMPerWave)
        && IsTwoPower<4, 128>(GemmNPerWave)
        && IsTwoPower<1, 8>(GemmKPack))
    return success();
  
  return failure();
  // clang-format on
}

// Used by EuristicInit() and GenericSearch
// Only return false if a performance config will violate requirements given by
// kernel algorithm
LogicalResult PerformanceImplicitGemmForwardV4R4Xdlops::IsReallyValid(
    const ConvolutionContext &ctx) const {
  if (failed(IsValidValue()))
    return failure();

  if (failed(IsValidBlockwiseGemmXdlops(ctx, GemmMPerBlock, GemmNPerBlock,
                                        GemmKPerBlock, GemmMPerWave,
                                        GemmNPerWave, GemmKPack)))
    return failure();

  LogicalResult valid = failure();

  // check blockwise GEMM size
  {
    int64_t gemm_m = -1;
    int64_t gemm_n = -1;
    int64_t gemm_k_total = -1;

    std::tie(std::ignore, gemm_m, gemm_n, gemm_k_total) =
        ConvHipImplicitGemmForwardV4R4Xdlops::CalculateGemmSize(ctx);

    if (gemm_k_total % GemmKPack != 0)
      return failure();

    const auto gemm_k = gemm_k_total / GemmKPack;

    if (!(gemm_m % GemmMPerBlock == 0 && gemm_n % GemmNPerBlock == 0 &&
          gemm_k % GemmKPerBlock == 0))
      return failure();
  }

  // check blockwise copy of A matrix
  {
    std::tie(std::ignore, std::ignore, std::ignore, std::ignore, std::ignore,
             valid) = CalculateGemmABlockCopyPerformanceParameters(ctx);

    if (failed(valid))
      return failure();
  }

  // check blockwise copy of B matrix
  {
    std::tie(std::ignore, std::ignore, std::ignore, std::ignore, std::ignore,
             valid) = CalculateGemmBBlockCopyPerformanceParameters(ctx);

    if (failed(valid))
      return failure();
  }

  // check LDS allocation
  std::size_t lds_size = 0;
  std::tie(lds_size, valid) = CalculateLdsNumberOfByte(ctx);

  if (succeeded(valid) && lds_size <= get_lds_max_number_of_byte())
    return success();

  return failure();
}

LogicalResult ConvHipImplicitGemmForwardV4R4Xdlops::IsValidPerformanceConfig(
    const ConvolutionContext &ctx,
    const PerformanceImplicitGemmForwardV4R4Xdlops &c) const {
  return c.IsReallyValid(ctx);
}

std::tuple<int64_t, int64_t, int64_t, int64_t>
ConvHipImplicitGemmForwardV4R4Xdlops::CalculateGemmSize(
    const ConvolutionContext &ctx) {
  auto dimIndexVal = ctx.dimIndexVal;
  int64_t g = 1;
  const auto n = dimIndexVal["no"].second;
  const auto k = dimIndexVal["k"].second;
  const auto c = dimIndexVal["c"].second;
  const auto ho = dimIndexVal["ho"].second;
  const auto wo = dimIndexVal["wo"].second;
  const auto y = dimIndexVal["y"].second;
  const auto x = dimIndexVal["x"].second;

  const auto k_per_group = k / g;
  const auto c_per_group = c / g;

  const auto gemm_g = g;
  const auto gemm_m = k_per_group;
  const auto gemm_n = n * ho * wo;
  const auto gemm_k_total = c_per_group * y * x;

  return std::make_tuple(gemm_g, gemm_m, gemm_n, gemm_k_total);
}

PerformanceImplicitGemmForwardV4R4Xdlops
ConvHipImplicitGemmForwardV4R4Xdlops::GetPerformanceConfig(
    const ConvolutionContext &ctx) const {
  return GetPerformanceConfigBase<PerformanceImplicitGemmForwardV4R4Xdlops>(
      ctx);
}

llvm::StringMap<int64_t> ConvHipImplicitGemmForwardV4R4Xdlops::GetSolution(
    const ConvolutionContext &ctx,
    const PerformanceImplicitGemmForwardV4R4Xdlops &config) const {
  llvm::StringMap<int64_t> result;

  assert(succeeded(config.IsReallyValid(ctx)));

  int64_t grid_size = 0;
  int64_t block_size = 0;

  std::tie(grid_size, std::ignore) = config.CalculateGridSize(ctx);
  std::tie(block_size, std::ignore) = config.CalculateBlockSize();

  int64_t GemmABlockCopyClusterLengths_GemmK = -1;
  int64_t GemmABlockCopyClusterLengths_GemmM = -1;
  int64_t GemmABlockCopyClusterLengths_GemmKPack = -1;
  int64_t GemmABlockCopySrcDataPerRead_GemmKPack = -1;
  int64_t GemmABlockCopyDstDataPerWrite_GemmKPack = -1;

  int64_t GemmBBlockCopyClusterLengths_GemmK = -1;
  int64_t GemmBBlockCopyClusterLengths_GemmN = -1;
  int64_t GemmBBlockCopyClusterLengths_GemmKPack = -1;
  int64_t GemmBBlockCopySrcDataPerRead_GemmN = -1;
  int64_t GemmBBlockCopyDstDataPerWrite_GemmKPack = -1;

  std::tie(GemmABlockCopyClusterLengths_GemmK,
           GemmABlockCopyClusterLengths_GemmM,
           GemmABlockCopyClusterLengths_GemmKPack,
           GemmABlockCopySrcDataPerRead_GemmKPack,
           GemmABlockCopyDstDataPerWrite_GemmKPack, std::ignore) =
      config.CalculateGemmABlockCopyPerformanceParameters(ctx);

  std::tie(GemmBBlockCopyClusterLengths_GemmK,
           GemmBBlockCopyClusterLengths_GemmN,
           GemmBBlockCopyClusterLengths_GemmKPack,
           GemmBBlockCopySrcDataPerRead_GemmN,
           GemmBBlockCopyDstDataPerWrite_GemmKPack, std::ignore) =
      config.CalculateGemmBBlockCopyPerformanceParameters(ctx);

  result["block_size"] = block_size;
  result["grid_size"] = grid_size;
  result["m_per_block"] = config.GemmMPerBlock;
  result["n_per_block"] = config.GemmNPerBlock;
  result["k_per_block"] = config.GemmKPerBlock;
  result["m_per_thread"] = config.GemmMPerWave;
  result["n_per_thread"] = config.GemmNPerWave;
  result["gemm_kpack"] = config.GemmKPack;

  // Derived parameters for gemmA.
  result["matrix_a_cluster_lengths_gemmk"] = GemmABlockCopyClusterLengths_GemmK;
  result["matrix_a_cluster_lengths_gemmM"] = GemmABlockCopyClusterLengths_GemmM;
  result["matrix_a_source_data_per_read"] =
      GemmABlockCopySrcDataPerRead_GemmKPack;
  result["matrix_a_dest_data_per_write_dim_m"] =
      GemmABlockCopyDstDataPerWrite_GemmKPack;
  result["matrix_a_source_vector_read_dim"] =
      CalculateGemmASrcVectorReadDim(ctx);

  // Derived parameters for gemmB.
  result["matrix_b_cluster_lengths_gemmk"] = GemmBBlockCopyClusterLengths_GemmK;
  result["matrix_b_cluster_lengths_gemmN"] = GemmBBlockCopyClusterLengths_GemmN;
  result["matrix_b_source_data_per_read"] = GemmBBlockCopySrcDataPerRead_GemmN;
  result["matrix_b_dest_data_per_write_dim_n"] =
      GemmBBlockCopyDstDataPerWrite_GemmKPack;
  result["matrix_b_source_vector_read_dim"] =
      CalculateGemmBSrcVectorReadDim(ctx);

  // result.invoker_factory = conv::MakeImplGemmDataInvokerFactory(ctx);
  // result.construction_params.push_back(construction_parameters);
  return result;
}

LogicalResult ConvHipImplicitGemmForwardV4R4Xdlops::IsApplicable(
    const ConvolutionContext &ctx) const {
  if (!(ctx.opType == miopen::ConvOpType::Conv2DOpType))
    return failure();

  if (!(ctx.isXdlOp))
    return failure();

  // gemm size
  {
    int64_t gemm_g = -1;
    int64_t gemm_m = -1;
    int64_t gemm_n = -1;
    int64_t gemm_k_total = -1;

    std::tie(gemm_g, gemm_m, gemm_n, gemm_k_total) = CalculateGemmSize(ctx);

    if (failed(IsValidGridGemmXdlops(gemm_m, gemm_n, gemm_k_total)))
      return failure();
  }

  // this particular EuristicInit is so comprehensive, that if it cannot predict
  // a valid performance config, the problem is probably not applicable
  PerformanceImplicitGemmForwardV4R4Xdlops config;
  config.EuristicInit(ctx);

  if (succeeded(config.IsReallyValid(ctx)))
    return success();

  return failure();
}
