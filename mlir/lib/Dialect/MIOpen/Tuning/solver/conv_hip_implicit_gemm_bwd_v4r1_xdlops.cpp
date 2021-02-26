#include "mlir/Dialect/MIOpen/Tuning/GridwiseGemmParams.h"

#define DEBUG_TYPE "igemm-bwd-v4r1-xdl"

PerformanceImplicitGemmBwdDataV4R1Xdlops::
    PerformanceImplicitGemmBwdDataV4R1Xdlops()
    : PerformanceImplicitGemmBwdDataV4R1Xdlops::
          PerformanceImplicitGemmBwdDataV4R1Xdlops(16, 4, 1, 1, 4, 16, false,
                                                   false) {}

PerformanceImplicitGemmBwdDataV4R1Xdlops::
    PerformanceImplicitGemmBwdDataV4R1Xdlops(
        int64_t GemmNPerBlock_, int64_t GemmMPerBlock_, int64_t GemmKPerBlock_,
        int64_t GemmKPACKSize_, int64_t GemmMPerWave_, int64_t GemmNPerWave_,
        bool GemmAThreadCopyMoreGemmK_, bool GemmBThreadCopyMoreGemmKPack_)
    : GemmNPerBlock(GemmNPerBlock_), GemmMPerBlock(GemmMPerBlock_),
      GemmKPerBlock(GemmKPerBlock_), GemmKPACKSize(GemmKPACKSize_),
      GemmMPerWave(GemmMPerWave_), GemmNPerWave(GemmNPerWave_),
      GemmAThreadCopyMoreGemmK(GemmAThreadCopyMoreGemmK_),
      GemmBThreadCopyMoreGemmKPack(GemmBThreadCopyMoreGemmKPack_) {}

bool PerformanceImplicitGemmBwdDataV4R1Xdlops::operator==(
    const PerformanceImplicitGemmBwdDataV4R1Xdlops &other) const {
  // clang-format off
    return GemmNPerBlock == other.GemmNPerBlock
        && GemmMPerBlock == other.GemmMPerBlock
        && GemmKPerBlock == other.GemmKPerBlock
        && GemmKPACKSize == other.GemmKPACKSize
        && GemmMPerWave == other.GemmMPerWave
        && GemmNPerWave == other.GemmNPerWave
        && GemmAThreadCopyMoreGemmK  == other.GemmAThreadCopyMoreGemmK
        && GemmBThreadCopyMoreGemmKPack  == other.GemmBThreadCopyMoreGemmKPack;
  // clang-format on
}

std::tuple<int64_t, LogicalResult>
PerformanceImplicitGemmBwdDataV4R1Xdlops::CalculateGridSize(
    const ConvolutionContext &ctx) const {
  int64_t GridSize = 0;

  int64_t gemm_g = 0;
  int64_t gemm_m = 0;
  int64_t gemm_n = 0;

  std::tie(gemm_g, gemm_m, gemm_n, std::ignore) =
      ConvHipImplicitGemmBwdDataV4R1Xdlops::CalculateGemmSize(ctx, 0);

  if (!(gemm_m % GemmMPerBlock == 0 && gemm_n % GemmNPerBlock == 0)) {
    LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl\n");
    return std::make_tuple(-1, failure());
  }

  GridSize = gemm_g * (gemm_m / GemmMPerBlock) * (gemm_n / GemmNPerBlock);

  return std::make_tuple(GridSize, success());
}

std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, LogicalResult>
PerformanceImplicitGemmBwdDataV4R1Xdlops::
    CalculateGemmABlockCopyPerformanceParameters(
        const ConvolutionContext &ctx) const {
  int64_t ClusterLengths_GemmK = 0;
  int64_t ClusterLengths_GemmM = 0;
  int64_t ClusterLengths_GemmKPack = 0;
  int64_t SrcDataPerRead_GemmM = 4;

  int64_t DstDataPerWrite_GemmKPack = 4;

  const auto waveSize = 64;

  const auto BlockSize =
      GemmNPerBlock * GemmMPerBlock / (GemmMPerWave * GemmNPerWave) * waveSize;

  // calculate vector length on gemmk dimension
  SrcDataPerRead_GemmM =
      ImplicitGemmUtil::gcd(SrcDataPerRead_GemmM, GemmMPerBlock);

  auto dimIndexVal = ctx.dimIndexVal;
  const auto y = dimIndexVal["y"].second;
  const auto x = dimIndexVal["x"].second;

  // \todo too conservative
  if (!(y == 1 && x == 1))
    SrcDataPerRead_GemmM = 1;

  // calculate threadwise copy size
  auto a_data_per_thread_copy =
      std::max(static_cast<int64_t>(1),
               (GemmKPerBlock * GemmMPerBlock * GemmKPACKSize) / BlockSize);

  a_data_per_thread_copy =
      ImplicitGemmUtil::lcm(a_data_per_thread_copy, SrcDataPerRead_GemmM);
  // decide threadwise copy lengths
  const auto a_data_per_thread_copy_gemmm = SrcDataPerRead_GemmM;
  if (!(a_data_per_thread_copy_gemmm > 0)) {
    LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
    return std::make_tuple(-1, -1, -1, -1, -1, failure());
  }
  const auto tmp = a_data_per_thread_copy / a_data_per_thread_copy_gemmm;

  int64_t data_per_thread_copy_gemmk = -1;
  int64_t data_per_thread_copy_gemmkpack = -1;

  if (GemmAThreadCopyMoreGemmK) {
    data_per_thread_copy_gemmk = ImplicitGemmUtil::gcd(GemmKPerBlock, tmp);
    if (!(data_per_thread_copy_gemmk > 0)) {
      LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
      return std::make_tuple(-1, -1, -1, -1, -1, failure());
    }
    data_per_thread_copy_gemmkpack = tmp / data_per_thread_copy_gemmk;
    if (!(data_per_thread_copy_gemmkpack > 0)) {
      LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
      return std::make_tuple(-1, -1, -1, -1, -1, failure());
    }
  } else {
    data_per_thread_copy_gemmkpack = ImplicitGemmUtil::gcd(GemmKPACKSize, tmp);
    if (!(data_per_thread_copy_gemmkpack > 0)) {
      LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
      return std::make_tuple(-1, -1, -1, -1, -1, failure());
    }
    data_per_thread_copy_gemmk = tmp / data_per_thread_copy_gemmkpack;
    if (!(data_per_thread_copy_gemmk > 0)) {
      LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
      return std::make_tuple(-1, -1, -1, -1, -1, failure());
    }
  }

  if (DstDataPerWrite_GemmKPack > data_per_thread_copy_gemmkpack)
    DstDataPerWrite_GemmKPack = data_per_thread_copy_gemmkpack;
  DstDataPerWrite_GemmKPack = ImplicitGemmUtil::gcd(
      DstDataPerWrite_GemmKPack, data_per_thread_copy_gemmkpack);

  if (!(GemmKPerBlock % data_per_thread_copy_gemmk == 0 &&
        GemmMPerBlock % a_data_per_thread_copy_gemmm == 0 &&
        GemmKPACKSize % data_per_thread_copy_gemmkpack == 0)) {
    LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
    return std::make_tuple(-1, -1, -1, -1, -1, failure());
  }

  ClusterLengths_GemmK = GemmKPerBlock / data_per_thread_copy_gemmk;
  ClusterLengths_GemmM = GemmMPerBlock / a_data_per_thread_copy_gemmm;
  ClusterLengths_GemmKPack = GemmKPACKSize / data_per_thread_copy_gemmkpack;
  // blockwise-copy support that block_size is larger than thread cluster size,
  // which means some threads may not do threadwise copy
  if (BlockSize <
      ClusterLengths_GemmK * ClusterLengths_GemmM * ClusterLengths_GemmKPack) {
    LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
    return std::make_tuple(-1, -1, -1, -1, -1, failure());
  }

  return std::make_tuple(ClusterLengths_GemmK, ClusterLengths_GemmM,
                         ClusterLengths_GemmKPack, SrcDataPerRead_GemmM,
                         DstDataPerWrite_GemmKPack, success());
}

std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, LogicalResult>
PerformanceImplicitGemmBwdDataV4R1Xdlops::
    CalculateGemmBBlockCopyPerformanceParameters(
        const ConvolutionContext &ctx) const {
  int64_t ClusterLengths_GemmK = 0;
  int64_t ClusterLengths_GemmN = 0;
  int64_t ClusterLengths_GemmKPack = 0;
  int64_t SrcDataPerRead_GemmN = 4;

  int64_t DstDataPerWrite_GemmKPack = 4;

  const auto waveSize = 64;
  const auto BlockSize =
      GemmNPerBlock * GemmMPerBlock / (GemmMPerWave * GemmNPerWave) * waveSize;

  SrcDataPerRead_GemmN =
      ImplicitGemmUtil::gcd(SrcDataPerRead_GemmN, GemmNPerBlock);

  // calculate vector length on gemmn dimension
  auto dimIndexVal = ctx.dimIndexVal;
  const auto y = dimIndexVal["y"].second;
  const auto x = dimIndexVal["x"].second;

  // \todo too conversative
  if (y == 1 && x == 1) {
    const auto ho = dimIndexVal["ho"].second;
    const auto wo = dimIndexVal["wo"].second;
    SrcDataPerRead_GemmN = ImplicitGemmUtil::gcd(SrcDataPerRead_GemmN, ho * wo);
  } else {
    SrcDataPerRead_GemmN = 1;
  }

  // calculate threadwise copy size
  int64_t b_data_per_thread_copy =
      std::max(static_cast<int64_t>(1),
               (GemmKPerBlock * GemmMPerBlock * GemmKPACKSize) / BlockSize);

  if (!(b_data_per_thread_copy > 0)) {
    LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
    return std::make_tuple(-1, -1, -1, -1, -1, failure());
  }

  b_data_per_thread_copy =
      ImplicitGemmUtil::lcm(SrcDataPerRead_GemmN, b_data_per_thread_copy);
  if (BlockSize > GemmNPerBlock && GemmKPACKSize > BlockSize / GemmNPerBlock) {
    LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
    return std::make_tuple(-1, -1, -1, -1, -1, failure());
  }

  const auto data_per_thread_copy_gemmn = SrcDataPerRead_GemmN;
  if (!(data_per_thread_copy_gemmn > 0)) {
    LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
    return std::make_tuple(-1, -1, -1, -1, -1, failure());
  }

  const auto tmp = b_data_per_thread_copy / data_per_thread_copy_gemmn;
  if (!(tmp > 0)) {
    LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
    return std::make_tuple(-1, -1, -1, -1, -1, failure());
  }
  int64_t data_per_thread_copy_gemmkpack = -1;
  int64_t data_per_thread_copy_gemmk = -1;

  if (GemmBThreadCopyMoreGemmKPack) {
    data_per_thread_copy_gemmkpack = ImplicitGemmUtil::gcd(GemmKPACKSize, tmp);
    if (!(data_per_thread_copy_gemmkpack > 0)) {
      LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
      return std::make_tuple(-1, -1, -1, -1, -1, failure());
    }

    data_per_thread_copy_gemmk = tmp / data_per_thread_copy_gemmkpack;
    if (!(data_per_thread_copy_gemmk > 0)) {
      LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
      return std::make_tuple(-1, -1, -1, -1, -1, failure());
    }
  } else {
    data_per_thread_copy_gemmk = ImplicitGemmUtil::gcd(GemmKPerBlock, tmp);
    if (!(data_per_thread_copy_gemmk > 0)) {
      LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
      return std::make_tuple(-1, -1, -1, -1, -1, failure());
    }
    data_per_thread_copy_gemmkpack = tmp / data_per_thread_copy_gemmk;
    if (!(data_per_thread_copy_gemmkpack > 0)) {
      LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
      return std::make_tuple(-1, -1, -1, -1, -1, failure());
    }
  }

  // vector write int64_to LDS
  if (DstDataPerWrite_GemmKPack > data_per_thread_copy_gemmkpack)
    DstDataPerWrite_GemmKPack = data_per_thread_copy_gemmkpack;

  DstDataPerWrite_GemmKPack = ImplicitGemmUtil::gcd(
      DstDataPerWrite_GemmKPack, data_per_thread_copy_gemmkpack);

  if (!(GemmKPerBlock % data_per_thread_copy_gemmk == 0 &&
        GemmNPerBlock % data_per_thread_copy_gemmn == 0 &&
        GemmKPACKSize % data_per_thread_copy_gemmkpack == 0)) {
    LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
    return std::make_tuple(-1, -1, -1, -1, -1, failure());
  }

  ClusterLengths_GemmK = GemmKPerBlock / data_per_thread_copy_gemmk;
  ClusterLengths_GemmN = GemmNPerBlock / data_per_thread_copy_gemmn;
  ClusterLengths_GemmKPack = GemmKPACKSize / data_per_thread_copy_gemmkpack;

  if (BlockSize <
      ClusterLengths_GemmK * ClusterLengths_GemmN * ClusterLengths_GemmKPack) {
    LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
    return std::make_tuple(-1, -1, -1, -1, -1, failure());
  }

  return std::make_tuple(ClusterLengths_GemmK, ClusterLengths_GemmN,
                         ClusterLengths_GemmKPack, SrcDataPerRead_GemmN,
                         DstDataPerWrite_GemmKPack, success());
}

std::tuple<std::size_t, LogicalResult>
PerformanceImplicitGemmBwdDataV4R1Xdlops::CalculateLdsNumberOfByte(
    const ConvolutionContext &ctx) const {
  std::size_t lds_size = 0;

  LogicalResult valid = failure();

  int64_t GemmABlockCopyClusterLengths_GemmM = 0;
  int64_t GemmABlockCopyDescDataPerWriteGemmKPACK = 0;
  int64_t GemmKPack = GemmKPACKSize;

  std::tie(std::ignore, GemmABlockCopyClusterLengths_GemmM, std::ignore,
           std::ignore, GemmABlockCopyDescDataPerWriteGemmKPACK, valid) =
      CalculateGemmABlockCopyPerformanceParameters(ctx);

  if (failed(valid)) {
    LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
    return std::make_tuple(0, failure());
  }

  int64_t GemmBBlockCopyClusterLengths_GemmN = 0;
  int64_t GemmBBlockCopyDescDataPerWriteGemmKPACK = 0;
  std::tie(std::ignore, GemmBBlockCopyClusterLengths_GemmN, std::ignore,
           std::ignore, GemmBBlockCopyDescDataPerWriteGemmKPACK, valid) =
      CalculateGemmBBlockCopyPerformanceParameters(ctx);

  if (failed(valid)) {
    LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
    return std::make_tuple(0, failure());
  }

  if (GemmABlockCopyClusterLengths_GemmM == 0 ||
      GemmBBlockCopyClusterLengths_GemmN == 0) {
    LLVM_DEBUG(llvm::dbgs() << "invalid performance parameter: BwdV4R1Xdl");
    return std::make_tuple(0, failure());
  }

  const auto ThreadGemmDataPerRead_GemmM =
      GemmMPerBlock / GemmABlockCopyClusterLengths_GemmM;
  const auto ThreadGemmDataPerRead_GemmN =
      GemmNPerBlock / GemmBBlockCopyClusterLengths_GemmN;

  const auto max_lds_align = ImplicitGemmUtil::lcm(
      GemmABlockCopyDescDataPerWriteGemmKPACK,
      GemmBBlockCopyDescDataPerWriteGemmKPACK, ThreadGemmDataPerRead_GemmM,
      ThreadGemmDataPerRead_GemmN);

  const auto a_block_space =
      GemmKPerBlock *
      ImplicitGemmUtil::integer_least_multiple(GemmMPerBlock, max_lds_align);
  const auto b_block_space =
      GemmKPerBlock *
      ImplicitGemmUtil::integer_least_multiple(GemmNPerBlock, max_lds_align);
  lds_size = (a_block_space + b_block_space) * 4 * GemmKPack;

  return std::make_tuple(lds_size, success());
}

LogicalResult PerformanceImplicitGemmBwdDataV4R1Xdlops::IsReallyValid(
    const ConvolutionContext &ctx) const {
  if (failed(IsValidValue()))
    return failure();

  int64_t GemmM = 0, GemmN = 0, GemmK = 0, gemm_k_total = 0;

  // check blockwise GEMM size
  for (int64_t gemm_id = 0;
       gemm_id <
       ConvHipImplicitGemmBwdDataV4R1Xdlops::CalculateNumberOfGemm(ctx);
       ++gemm_id) {

    std::tie(std::ignore, GemmM, GemmN, gemm_k_total) =
        ConvHipImplicitGemmBwdDataV4R1Xdlops::CalculateGemmSize(ctx, gemm_id);

    if (gemm_k_total % GemmKPACKSize != 0)
      return failure();

    GemmK = gemm_k_total / GemmKPACKSize;

    if (!(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 &&
          GemmK % GemmKPerBlock == 0))
      return failure(); // wrong! cannot divice N evenly among thread
  }
  // heuristic to reduce search space
  {
    // use largest XdlopsGemm
    if (GemmMPerBlock % GemmMPerWave != 0)
      return failure();
    if (GemmNPerBlock % GemmNPerWave != 0)
      return failure();
  }

  if (!(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 &&
        GemmK % GemmKPerBlock == 0))
    return failure(); // wrong! cannot divice N evenly among thread

  if (failed(ImplicitGemmUtil::IsValidBlockwiseGemmXdlops(
          ctx, GemmMPerBlock, GemmNPerBlock, GemmKPerBlock, GemmMPerWave,
          GemmNPerWave, GemmKPACKSize)))
    return failure();

  LogicalResult valid = failure();

  // check blockwise copy of A matrix
  std::tie(std::ignore, std::ignore, std::ignore, std::ignore, std::ignore,
           valid) = CalculateGemmABlockCopyPerformanceParameters(ctx);

  if (failed(valid))
    return failure();

  // check blockwise copy of B matrix
  std::tie(std::ignore, std::ignore, std::ignore, std::ignore, std::ignore,
           valid) = CalculateGemmBBlockCopyPerformanceParameters(ctx);

  if (failed(valid))
    return failure();

  std::size_t lds_size = 0;
  std::tie(lds_size, valid) = CalculateLdsNumberOfByte(ctx);

  if (succeeded(valid) && lds_size <= 64 * 1024)
    return success();

  return failure();
}

LogicalResult PerformanceImplicitGemmBwdDataV4R1Xdlops::IsValidValue() const {
  // clang-format off
  if(ImplicitGemmUtil::IsTwoPower<16,256>(GemmNPerBlock) &&
     ImplicitGemmUtil::IsTwoPower<4,256>(GemmMPerBlock) &&
     ImplicitGemmUtil::IsTwoPower<1,8>(GemmKPerBlock) &&
     ImplicitGemmUtil::IsTwoPower<1,8>(GemmKPACKSize) &&
     ImplicitGemmUtil::IsTwoPower<4,128>(GemmMPerWave) &&
     ImplicitGemmUtil::IsTwoPower<16,128>(GemmNPerWave))
    // clang-format on
    return success();

  return failure();
}

LogicalResult PerformanceImplicitGemmBwdDataV4R1Xdlops::EuristicInit(
    const ConvolutionContext &ctx) {
  PerformanceImplicitGemmBwdDataV4R1Xdlops tmp;

  auto get_euristic_config = [&](auto is_valid_func) {
    /* MIOpen logic */
    //    tmp              = {256, 256, 8, 4, 128, 128, true, true};

    tmp = {256, 256, 8, 4, 128, 128, false, false};
    bool all_visited = false;
    do {
      do {
        // list in reverse order of importance,
        // and favor large GEMM
        if (!ImplicitGemmUtil::PreviousTwoPower<1, 8>(tmp.GemmKPerBlock))
          break;
        if (!ImplicitGemmUtil::PreviousTwoPower<1, 4>(tmp.GemmKPACKSize))
          break;
        if (!ImplicitGemmUtil::PreviousTwoPower<16, 128>(tmp.GemmNPerWave))
          break;
        if (!ImplicitGemmUtil::PreviousTwoPower<4, 128>(tmp.GemmMPerWave))
          break;
        if (!ImplicitGemmUtil::PreviousTwoPower<16, 256>(tmp.GemmNPerBlock))
          break;
        if (!ImplicitGemmUtil::PreviousTwoPower<4, 256>(tmp.GemmMPerBlock))
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

int64_t ConvHipImplicitGemmBwdDataV4R1Xdlops::CalculateNumberOfGemm(
    const ConvolutionContext &ctx) {

  const auto conv_stride_h = ctx.strideVal[0];
  const auto conv_stride_w = ctx.strideVal[1];

  const auto conv_dilation_h = ctx.dilationVal[0];
  const auto conv_dilation_w = ctx.dilationVal[1];

  const auto gcd_stride_dilation_h =
      ImplicitGemmUtil::gcd(conv_stride_h, conv_dilation_h);
  const auto gcd_stride_dilation_w =
      ImplicitGemmUtil::gcd(conv_stride_w, conv_dilation_w);

  const auto ytilda = conv_stride_h / gcd_stride_dilation_h;
  const auto xtilda = conv_stride_w / gcd_stride_dilation_w;

  return ytilda * xtilda;
}

std::tuple<int64_t, int64_t, int64_t, int64_t>
ConvHipImplicitGemmBwdDataV4R1Xdlops::CalculateGemmSize(
    const ConvolutionContext &ctx, int64_t gemm_id) {
  auto dimIndexVal = ctx.dimIndexVal;
  const int64_t g = 1;
  const auto n = dimIndexVal["no"].second;
  const auto k = dimIndexVal["k"].second;
  const auto c = dimIndexVal["c"].second;
  const auto hi = dimIndexVal["hi"].second;
  const auto wi = dimIndexVal["wi"].second;
  const auto y = dimIndexVal["y"].second;
  const auto x = dimIndexVal["x"].second;
  const auto ho = dimIndexVal["ho"].second;
  const auto wo = dimIndexVal["wo"].second;
  const auto conv_stride_h = ctx.strideVal[0];
  const auto conv_stride_w = ctx.strideVal[1];
  const auto conv_dilation_h = ctx.dilationVal[0];
  const auto conv_dilation_w = ctx.dilationVal[1];
  const auto in_left_pad_h = ctx.paddingVal[0];
  const auto in_left_pad_w = ctx.paddingVal[1];

  const auto gcd_stride_dilation_h =
      ImplicitGemmUtil::gcd(conv_stride_h, conv_dilation_h);
  const auto gcd_stride_dilation_w =
      ImplicitGemmUtil::gcd(conv_stride_w, conv_dilation_w);

  const auto ytilda = conv_stride_h / gcd_stride_dilation_h;
  const auto xtilda = conv_stride_w / gcd_stride_dilation_w;

  const auto ydot = ImplicitGemmUtil::integer_divide_ceil(y, ytilda);
  const auto xdot = ImplicitGemmUtil::integer_divide_ceil(x, xtilda);

  const auto htilda = ho + ImplicitGemmUtil::integer_divide_ceil(
                               conv_dilation_h * (y - 1), conv_stride_h);
  const auto wtilda = wo + ImplicitGemmUtil::integer_divide_ceil(
                               conv_dilation_w * (x - 1), conv_stride_w);

  // int64_termediate result could be negative, use int64_t instead of size_t
  const auto htilda_left =
      std::max(static_cast<int64_t>(0),
               in_left_pad_h - conv_dilation_h * (ytilda - 1)) /
      conv_stride_h;
  const auto wtilda_left =
      std::max(static_cast<int64_t>(0),
               in_left_pad_w - conv_dilation_w * (xtilda - 1)) /
      conv_stride_w;

  const auto htilda_right =
      std::min(htilda, ImplicitGemmUtil::integer_divide_ceil(
                           in_left_pad_h + hi - 1, conv_stride_h) +
                           1);
  const auto wtilda_right =
      std::min(wtilda, ImplicitGemmUtil::integer_divide_ceil(
                           in_left_pad_w + wi - 1, conv_stride_w) +
                           1);

  const auto htilda_slice = htilda_right - htilda_left;
  const auto wtilda_slice = wtilda_right - wtilda_left;

  // gemm_k size is different for each GEMM
  const auto i_ytilda = gemm_id / xtilda;
  const auto i_xtilda = gemm_id % xtilda;

  const auto ydot_slice = (i_ytilda + 1) * ydot <= y ? ydot : y % ydot;
  const auto xdot_slice = (i_xtilda + 1) * xdot <= x ? xdot : x % xdot;

  const auto gemm_m = c / g;
  const auto gemm_n = n * htilda_slice * wtilda_slice;
  const auto gemm_k = (k / g) * ydot_slice * xdot_slice;

  return std::make_tuple(g, gemm_m, gemm_n, gemm_k);
}

LogicalResult ConvHipImplicitGemmBwdDataV4R1Xdlops::IsApplicable(
    const ConvolutionContext &ctx) const {
  if (!(ctx.opType == miopen::ConvOpType::Conv2DBwdDataOpType))
    return failure();

  if (!(ctx.isXdlOp))
    return failure();

  if (!(ctx.IsF32() || ctx.IsF16() || ctx.IsBF16()))
    return failure();

  int64_t gemm_g = 0;
  int64_t gemm_m = 0;
  int64_t gemm_n = 0;
  int64_t gemm_k_total = 0;

  for (int64_t gemm_id = 0; gemm_id < CalculateNumberOfGemm(ctx); ++gemm_id) {
    std::tie(gemm_g, gemm_m, gemm_n, gemm_k_total) =
        CalculateGemmSize(ctx, gemm_id);
    if (failed(ImplicitGemmUtil::IsValidGridGemmXdlops(gemm_m, gemm_n,
                                                       gemm_k_total)))
      return failure();
  }
  return success();
}

PerformanceImplicitGemmBwdDataV4R1Xdlops
ConvHipImplicitGemmBwdDataV4R1Xdlops::GetPerformanceConfig(
    const ConvolutionContext &ctx) const {
  return GetPerformanceConfigBase<PerformanceImplicitGemmBwdDataV4R1Xdlops>(
      ctx);
}

LogicalResult ConvHipImplicitGemmBwdDataV4R1Xdlops::IsValidPerformanceConfig(
    const ConvolutionContext &ctx,
    const PerformanceImplicitGemmBwdDataV4R1Xdlops &c) const {
  return c.IsReallyValid(ctx);
}

llvm::StringMap<int64_t> ConvHipImplicitGemmBwdDataV4R1Xdlops::GetSolution(
    const ConvolutionContext &ctx,
    const PerformanceImplicitGemmBwdDataV4R1Xdlops &config) const {
  llvm::StringMap<int64_t> result;

  assert(succeeded(config.IsReallyValid(ctx)));

  // a series of kernels
  for (int64_t gemm_id = 0; gemm_id < CalculateNumberOfGemm(ctx); ++gemm_id) {
    int64_t gemm_g = 0;
    int64_t gemm_m = 0;
    int64_t gemm_n = 0;
    int64_t gemm_k = 0;

    const auto waveSize = 64;
    std::tie(gemm_g, gemm_m, gemm_n, gemm_k) = CalculateGemmSize(ctx, gemm_id);

    // don't compile or launch an empty gridwise GEMM
    if (gemm_k > 0) {
      int64_t grid_size = 0;

      const std::size_t GemmMPerBlock = config.GemmMPerBlock;
      const std::size_t GemmNPerBlock = config.GemmNPerBlock;
      const std::size_t GemmKPerBlock = config.GemmKPerBlock;
      const std::size_t GemmMPerWave = config.GemmMPerWave;
      const std::size_t GemmNPerWave = config.GemmNPerWave;

      const std::size_t block_size = GemmNPerBlock * GemmMPerBlock /
                                     (GemmMPerWave * GemmNPerWave) * waveSize;

      std::tie(grid_size, std::ignore) = config.CalculateGridSize(ctx);

      int64_t GemmABlockCopySrcDataPerRead_GemmM = 1;
      int64_t GemmBBlockCopySrcDataPerRead_GemmN = 1;
      int64_t GemmABlockCopyClusterLengths_GemmK = 0;
      int64_t GemmABlockCopyClusterLengths_GemmM = 0;
      int64_t GemmBBlockCopyClusterLengths_GemmK = 0;
      int64_t GemmBBlockCopyClusterLengths_GemmN = 0;

      int64_t GemmABlockCopyClusterLengths_GemmKPack = 1;
      int64_t GemmABlockCopyDstDataPerWrite_GemmKPack = 1;

      std::tie(GemmABlockCopyClusterLengths_GemmK,
               GemmABlockCopyClusterLengths_GemmM,
               GemmABlockCopyClusterLengths_GemmKPack,
               GemmABlockCopySrcDataPerRead_GemmM,
               GemmABlockCopyDstDataPerWrite_GemmKPack, std::ignore) =
          config.CalculateGemmABlockCopyPerformanceParameters(ctx);

      int64_t GemmBBlockCopyClusterLengths_GemmKPack = 1;
      int64_t GemmBBlockCopyDstDataPerWrite_GemmKPack = 1;

      std::tie(GemmBBlockCopyClusterLengths_GemmK,
               GemmBBlockCopyClusterLengths_GemmN,
               GemmBBlockCopyClusterLengths_GemmKPack,
               GemmBBlockCopySrcDataPerRead_GemmN,
               GemmBBlockCopyDstDataPerWrite_GemmKPack, std::ignore) =
          config.CalculateGemmBBlockCopyPerformanceParameters(ctx);

      result["block_size"] = block_size;
      result["grid_size"] = grid_size;
      result["m_per_block"] = GemmMPerBlock;
      result["n_per_block"] = GemmNPerBlock;
      result["k_per_block"] = GemmKPerBlock;
      result["m_per_thread"] = GemmMPerWave;
      result["n_per_thread"] = GemmNPerWave;
      result["gemm_kpack"] = config.GemmKPACKSize;

      // Derived parameters for gemmA.
      result["matrix_a_cluster_lengths_gemmk"] =
          GemmABlockCopyClusterLengths_GemmK;
      result["matrix_a_cluster_lengths_gemmM"] =
          GemmABlockCopyClusterLengths_GemmM;
      result["matrix_a_source_data_per_read"] =
          GemmABlockCopySrcDataPerRead_GemmM;
      result["matrix_a_dest_data_per_write_dim_m"] =
          GemmABlockCopyDstDataPerWrite_GemmKPack;
      result["matrix_a_source_vector_read_dim"] =
          CalculateGemmASrcVectorReadDim(ctx);

      // Derived parameters for gemmB.
      result["matrix_b_cluster_lengths_gemmk"] =
          GemmBBlockCopyClusterLengths_GemmK;
      result["matrix_b_cluster_lengths_gemmN"] =
          GemmBBlockCopyClusterLengths_GemmN;
      result["matrix_b_source_data_per_read"] =
          GemmBBlockCopySrcDataPerRead_GemmN;
      result["matrix_b_dest_data_per_write_dim_n"] =
          GemmBBlockCopyDstDataPerWrite_GemmKPack;
      result["matrix_b_source_vector_read_dim"] =
          CalculateGemmBSrcVectorReadDim(ctx);

      break;
    }
  }

  return result;
}
