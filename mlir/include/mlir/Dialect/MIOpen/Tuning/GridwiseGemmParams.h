//===- GridwiseGemmParams.h - MLIR tuning parameter generation --------*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MLIR tuning parameter generation
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIOPEN_GRIDWISE_GEMM_PARAMS_H
#define MLIR_DIALECT_MIOPEN_GRIDWISE_GEMM_PARAMS_H

#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Tuning/ConvContext.h"
#include "mlir/Dialect/MIOpen/Tuning/Serializable.h"
#include "mlir/Dialect/MIOpen/Tuning/SqliteDb.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"

#include <string>
#include <unordered_map>

using namespace mlir;

extern llvm::cl::opt<std::string> TunableParametersYAMLFile;
extern llvm::cl::opt<bool> IsPopulateTunableParameters;

LLVM_YAML_IS_STRING_MAP(int)

// greatest common divisor, aka highest common factor
template <typename T> T gcd(T x, T y) {
  if (x == y || x == 0) {
    return y;
  } else if (y == 0) {
    return x;
  } else if (x > y) {
    return gcd(x - y, y);
  } else {
    return gcd(x, y - x);
  }
}

template <typename T, typename... Ys> T gcd(T x, Ys... ys) {
  return gcd(x, gcd(ys...));
}

// least common multiple
template <typename T> T lcm(T x, T y) {
  if (x == 0 || y == 0) {
    return 0;
  } else {
    return (x * y) / gcd(x, y);
  }
}

template <typename T, typename... Ys> T lcm(T x, Ys... ys) {
  return lcm(x, lcm(ys...));
}

template <typename T> T integer_divide_ceil(T x, T y) {
  return (x + y - 1) / y;
}

template <typename T> T integer_least_multiple(T x, T y) {
  return y * integer_divide_ceil(x, y);
}

template <int64_t L, int64_t H> inline static bool IsTwoPower(const int64_t v) {
  static_assert(L <= H, "L <= H");
  if (((v - 1) & v) != 0)
    return false;
  return L <= v && v <= H;
}

template <int64_t L, int64_t H>
inline static bool PreviousTwoPower(int64_t &v) {
  static_assert((((L - 1) & L) == 0), "L is not power of 2");
  static_assert((((H - 1) & H) == 0), "H is not power of 2");
  assert((IsTwoPower<L, H>(v)));
  if (v == L) {
    v = H;
    return true;
  }
  v /= 2;
  return false;
}

constexpr std::size_t get_lds_max_number_of_byte() { return 65536; }

inline static uint32_t GetEPackLength(const ConvolutionContext &ctx,
                                      bool isXdlopsInvoked) {
  // Based on data type, Es are packed
  int64_t EPACK = 1;
  /*
      if(ctx.IsFp16()) // for fp16, either 2 or 4 Es could be packed
      {
          if(IsXdlopsSupport(ctx) && isXdlopsInvoked) // in xdlops, 4 fp16s are
     packed EPACK = 4; else // for fp16, either 2 or 4 Es could be packed in
     non-xdlops scenarios.
              // EPACK = (C * Y * X % 32) == 0 ? 4 : 2;
              EPACK = 2;
      }
      else if(ctx.IsBfp16()) // for bfp16, only 2 Es could be packed
      {
          EPACK = 2;
      }
  */
  return EPACK;
}

static inline LogicalResult IsValidBlockwiseGemmXdlops(
    const ConvolutionContext &ctx, const int64_t GemmMPerBlock,
    const int64_t GemmNPerBlock, const int64_t GemmKPerBlock,
    const int64_t GemmMPerWave, const int64_t GemmNPerWave,
    const int64_t GemmKPack) {
  // check M, N and K
  std::vector<std::tuple<int64_t, int64_t, int64_t>> validWaveGemmSize = {
      // std::make_tuple(128, 128, 1),
      std::make_tuple(128, 64, 1),
      // std::make_tuple(128, 32, 1),
      // std::make_tuple(128, 16, 1),
      std::make_tuple(64, 128, 1), std::make_tuple(64, 64, 1),
      std::make_tuple(64, 32, 1), std::make_tuple(64, 16, 1),
      // std::make_tuple(32, 128, 1),
      std::make_tuple(32, 64, 1), std::make_tuple(32, 32, 2),
      // std::make_tuple(16, 128, 1),
      std::make_tuple(16, 64, 1), std::make_tuple(16, 16, 4),
      // std::make_tuple(8, 128, 1),
      std::make_tuple(8, 64, 1),
      // std::make_tuple(4, 128, 1),
      std::make_tuple(4, 64, 1)};

  if (!std::any_of(validWaveGemmSize.cbegin(), validWaveGemmSize.cend(),
                   [GemmMPerWave, GemmNPerWave,
                    GemmKPerBlock](const auto it) noexcept -> bool {
                     int64_t validMPerWave, validNPerWave, validKPerWave;
                     std::tie(validMPerWave, validNPerWave, validKPerWave) = it;
                     return (GemmMPerWave == validMPerWave) &&
                            (GemmNPerWave == validNPerWave) &&
                            (GemmKPerBlock % validKPerWave == 0);
                   }))
    return failure();

  const auto WaveSize = 64;
  const auto BlockSize = (GemmNPerBlock * GemmMPerBlock) /
                         (GemmMPerWave * GemmNPerWave) * WaveSize;

  if (BlockSize < 64 || BlockSize > 256)
    return failure();

  if ((GemmMPerBlock % GemmMPerWave) == 0 &&
      (GemmNPerBlock % GemmNPerWave) == 0)
    return success();

  return failure();
}

static inline LogicalResult IsValidGridGemmXdlops(const std::size_t GemmM,
                                                  const std::size_t GemmN,
                                                  const std::size_t GemmK) {
  // unsupported xdlops-gemm
  if (GemmM % 16 != 0 && GemmN % 64 != 0)
    return failure();

  const auto WaveSize = 64;

  if ((GemmM * GemmN) % 256 == 0 && (GemmK * GemmM) % WaveSize == 0 &&
      (GemmK * GemmN) % WaveSize == 0 && GemmN % 16 == 0 && GemmM % 4 == 0 &&
      GemmK % 4 == 0)
    return success();

  return failure();
}

template <typename PerformanceImplicitGemm_t>
inline static auto GetPerformanceConfigBase(const ConvolutionContext &ctx) {
  PerformanceImplicitGemm_t pp;
  pp.EuristicInit(ctx);
  return pp;
}

class PopulateParamsBase {
public:
  static void obtainGemmADimKVectorizable(
      mlir::miopen::ConvOpType opType,
      llvm::StringMap<std::pair<size_t, int64_t>> &dimIndexVal,
      bool &input1GemmKVectorizable) {
    // Vectorizable flag is opposite between forwad and bwd_data
    if (opType == mlir::miopen::ConvOpType::Conv2DOpType) {
      // When K is not the fastest changing dimension,
      // gemmK dimension is vectorizable, gemmM is not, and vice versa.
      // Vectorization width depending on which among C, Y, X be the fastest
      // changing dimension.
      if (dimIndexVal["k"].first == 3) {
        input1GemmKVectorizable = false;
      } else {
        input1GemmKVectorizable = true;
      }
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdDataOpType) {
      // When K is the fastest changing dimension(3),
      // gemmK dimension is vectorizable, gemmM is not, and vice versa.
      // Vectorization width depending on length of K.
      if (dimIndexVal["k"].first == 3) {
        input1GemmKVectorizable = true;
      } else {
        input1GemmKVectorizable = false;
      }
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdWeightOpType) {
      // When K is the fastest changing dimension,
      // gemmM dimension is vectorizable, gemmK is not, and vice versa.
      // Vectorization width depending on which among N, and HoWo be the fastest
      // changing dimension.
      if (dimIndexVal["k"].first == 3) {
        input1GemmKVectorizable = false;
      } else {
        input1GemmKVectorizable = true;
      }
    }
  }

  static void obtainGemmBDimKVectorizable(
      mlir::miopen::ConvOpType opType,
      llvm::StringMap<std::pair<size_t, int64_t>> &dimIndexVal,
      bool &input2GemmKVectorizable) {
    // Vectorizable flag is opposite between forwad and bwd_data
    if (opType == mlir::miopen::ConvOpType::Conv2DOpType) {
      // For input tensor.
      // When C is the fastest changing dimension,
      // gemmK dimension is vectorizable, gemmN is not, and vice versa.
      // Vectorization width depending on length of C.
      if (dimIndexVal["ci"].first == 3) {
        input2GemmKVectorizable = true;
      } else {
        input2GemmKVectorizable = false;
      }
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdDataOpType) {
      // For output tensor.
      // When K is the fastest changing dimension(3),
      // gemmK dimension is vectorizable, gemmN is not, and vice versa.
      // Vectorization width depending on length of K.
      if (dimIndexVal["ko"].first == 3) {
        input2GemmKVectorizable = true;
      } else {
        input2GemmKVectorizable = false;
      }
    } else if (opType == mlir::miopen::ConvOpType::Conv2DBwdWeightOpType) {
      // For input tensor
      // When C is the fastest changing dimension,
      // gemmN dimension is vectorizable, gemmK is not, and vice versa.
      // Vectorization width depending on length of C.
      if (dimIndexVal["ci"].first == 3) {
        input2GemmKVectorizable = false;
      } else {
        input2GemmKVectorizable = true;
      }
    }
  }
};

struct PerformanceImplicitGemmV4R4Fwd
    : Serializable<PerformanceImplicitGemmV4R4Fwd> {
  int64_t BlockSize;

  int64_t GemmMPerBlock;
  int64_t GemmNPerBlock;
  int64_t GemmKPerBlock;

  int64_t GemmMPerThread;
  int64_t GemmNPerThread;

  PerformanceImplicitGemmV4R4Fwd(int64_t, int64_t, int64_t, int64_t, int64_t,
                                 int64_t);

  PerformanceImplicitGemmV4R4Fwd()
      : PerformanceImplicitGemmV4R4Fwd(-1, -1, -1, -1, -1, -1) {}

  bool operator==(const PerformanceImplicitGemmV4R4Fwd &other) const;

  template <class Self, class F> static void visit(Self &&self, F f) {
    f(self.BlockSize, "BlockSize");
    f(self.GemmMPerBlock, "GemmMPerBlock");
    f(self.GemmNPerBlock, "GemmNPerBlock");
    f(self.GemmKPerBlock, "GemmKPerBlock");
    f(self.GemmMPerThread, "GemmMPerThread");
    f(self.GemmNPerThread, "GemmNPerThread");
  }

  std::tuple<int64_t, LogicalResult>
  CalculateGridSize(const ConvolutionContext &ctx) const;

  std::tuple<int64_t, int64_t, int64_t, int64_t, LogicalResult>
  CalculateBlockGemmPerformanceParameters(const ConvolutionContext &ctx) const;

  std::tuple<int64_t, int64_t, int64_t, int64_t, LogicalResult>
  CalculateGemmABlockCopyPerformanceParameters(
      const ConvolutionContext &ctx) const;

  std::tuple<int64_t, int64_t, int64_t, int64_t, LogicalResult>
  CalculateGemmBBlockCopyPerformanceParameters(
      const ConvolutionContext &ctx) const;

  std::tuple<int64_t, LogicalResult>
  CalculateGemmCThreadCopyPerformanceParameters(
      const ConvolutionContext &ctx) const;

  std::tuple<std::size_t, LogicalResult>
  CalculateLdsNumberOfByte(const ConvolutionContext &ctx) const;

  int64_t CalculateGemmASrcVectorReadDim(const ConvolutionContext &ctx) const;
  int64_t CalculateGemmBSrcVectorReadDim(const ConvolutionContext &ctx) const;

  LogicalResult IsValidValue() const;
  LogicalResult IsValid(const ConvolutionContext &ctx) const;
  LogicalResult EuristicInit(const ConvolutionContext &ctx);
};

struct PerformanceImplicitGemmV4R4WrW
    : Serializable<PerformanceImplicitGemmV4R4WrW> {
  int64_t BlockSize;

  int64_t GemmMPerBlock;
  int64_t GemmNPerBlock;
  int64_t GemmKPerBlock;

  int64_t GemmMPerThread;
  int64_t GemmNPerThread;

  int64_t srcVectorReadDim;

  PerformanceImplicitGemmV4R4WrW(int64_t, int64_t, int64_t, int64_t, int64_t,
                                 int64_t);

  PerformanceImplicitGemmV4R4WrW()
      : PerformanceImplicitGemmV4R4WrW(-1, -1, -1, -1, -1, -1) {}

  bool operator==(const PerformanceImplicitGemmV4R4WrW &other) const;

  template <class Self, class F> static void visit(Self &&self, F f) {
    f(self.BlockSize, "BlockSize");
    f(self.GemmMPerBlock, "GemmMPerBlock");
    f(self.GemmNPerBlock, "GemmNPerBlock");
    f(self.GemmKPerBlock, "GemmKPerBlock");
    f(self.GemmMPerThread, "GemmMPerThread");
    f(self.GemmNPerThread, "GemmNPerThread");
  }

  std::tuple<int64_t, LogicalResult>
  CalculateGridSize(const ConvolutionContext &ctx) const;

  std::tuple<int64_t, int64_t, int64_t, int64_t, LogicalResult>
  CalculateBlockGemmPerformanceParameters(const ConvolutionContext &ctx) const;

  std::tuple<int64_t, int64_t, int64_t, int64_t, LogicalResult>
  CalculateGemmABlockCopyPerformanceParameters(
      const ConvolutionContext &ctx) const;

  std::tuple<int64_t, int64_t, int64_t, int64_t, LogicalResult>
  CalculateGemmBBlockCopyPerformanceParameters(
      const ConvolutionContext &ctx) const;

  std::tuple<int64_t, LogicalResult>
  CalculateGemmCThreadCopyPerformanceParameters(
      const ConvolutionContext &ctx) const;

  std::tuple<std::size_t, LogicalResult>
  CalculateLdsNumberOfByte(const ConvolutionContext &ctx) const;

  int64_t CalculateGemmASrcVectorReadDim(const ConvolutionContext &ctx) const;
  int64_t CalculateGemmBSrcVectorReadDim(const ConvolutionContext &ctx) const;

  LogicalResult IsValidValue() const;
  LogicalResult IsValid(const ConvolutionContext &ctx) const;
  LogicalResult EuristicInit(const ConvolutionContext &ctx);
};

struct PerformanceImplicitGemmBwdDataV1R1
    : Serializable<PerformanceImplicitGemmBwdDataV1R1> {
  int64_t BlockSize;

  int64_t GemmMPerBlock;
  int64_t GemmNPerBlock;
  int64_t GemmKPerBlock;

  int64_t GemmMPerThread;
  int64_t GemmNPerThread;

  int64_t srcVectorReadDim;
  PerformanceImplicitGemmBwdDataV1R1(int64_t, int64_t, int64_t, int64_t,
                                     int64_t, int64_t);

  PerformanceImplicitGemmBwdDataV1R1()
      : PerformanceImplicitGemmBwdDataV1R1(-1, -1, -1, -1, -1, -1) {}

  bool operator==(const PerformanceImplicitGemmBwdDataV1R1 &other) const;

  template <class Self, class F> static void visit(Self &&self, F f) {
    f(self.BlockSize, "BlockSize");
    f(self.GemmMPerBlock, "GemmMPerBlock");
    f(self.GemmNPerBlock, "GemmNPerBlock");
    f(self.GemmKPerBlock, "GemmKPerBlock");
    f(self.GemmMPerThread, "GemmMPerThread");
    f(self.GemmNPerThread, "GemmNPerThread");
  }

  std::tuple<int64_t, LogicalResult>
  CalculateGridSize(const ConvolutionContext &ctx) const;

  std::tuple<int64_t, int64_t, int64_t, int64_t, LogicalResult>
  CalculateBlockGemmPerformanceParameters(const ConvolutionContext &ctx) const;

  std::tuple<int64_t, int64_t, int64_t, int64_t, LogicalResult>
  CalculateGemmABlockCopyPerformanceParameters(
      const ConvolutionContext &ctx) const;

  std::tuple<int64_t, int64_t, int64_t, int64_t, LogicalResult>
  CalculateGemmBBlockCopyPerformanceParameters(
      const ConvolutionContext &ctx) const;

  std::tuple<int64_t, LogicalResult>
  CalculateGemmCThreadCopyPerformanceParameters(
      const ConvolutionContext &ctx) const;

  std::tuple<std::size_t, mlir::LogicalResult>
  CalculateLdsNumberOfByte(const ConvolutionContext &ctx) const;

  int64_t CalculateGemmASrcVectorReadDim(const ConvolutionContext &ctx) const;
  int64_t CalculateGemmBSrcVectorReadDim(const ConvolutionContext &ctx) const;

  LogicalResult IsValidValue() const;
  LogicalResult IsValid(const ConvolutionContext &ctx) const;
  LogicalResult EuristicInit(const ConvolutionContext &ctx);
};

struct PerformanceImplicitGemmForwardV4R4Xdlops
    : Serializable<PerformanceImplicitGemmForwardV4R4Xdlops> {
  int64_t GemmMPerBlock;
  int64_t GemmNPerBlock;
  int64_t GemmKPerBlock;
  int64_t GemmMPerWave;
  int64_t GemmNPerWave;
  int64_t GemmKPack;
  bool GemmAThreadCopyMoreGemmK;
  bool GemmBThreadCopyMoreGemmKPack;
  int64_t GemmBThreadDataPerRead_GemmN;
  int64_t srcVectorReadDim;

  PerformanceImplicitGemmForwardV4R4Xdlops(int64_t, int64_t, int64_t, int64_t,
                                           int64_t, int64_t, bool, bool,
                                           int64_t);
  PerformanceImplicitGemmForwardV4R4Xdlops();

  template <class Self, class F> static void visit(Self &&self, F f) {
    f(self.GemmMPerBlock, "GemmMPerBlock");
    f(self.GemmNPerBlock, "GemmNPerBlock");
    f(self.GemmKPerBlock, "GemmKPerBlock");
    f(self.GemmMPerWave, "GemmMPerWave");
    f(self.GemmNPerWave, "GemmNPerWave");
    f(self.GemmKPack, "GemmKPack");
    f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
    f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
    f(self.GemmBThreadDataPerRead_GemmN, "GemmBThreadDataPerRead_GemmN");
  }

  bool operator==(const PerformanceImplicitGemmForwardV4R4Xdlops &other) const;
  std::string ToString() const;

  LogicalResult EuristicInit(const ConvolutionContext &ctx);
  LogicalResult IsValidValue() const;
  LogicalResult IsReallyValid(const ConvolutionContext &ctx) const;

  std::tuple<int64_t, LogicalResult> CalculateBlockSize() const;
  std::tuple<int64_t, LogicalResult>
  CalculateGridSize(const ConvolutionContext &ctx) const;
  std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, LogicalResult>
  CalculateGemmABlockCopyPerformanceParameters(
      const ConvolutionContext &ctx) const;
  std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, LogicalResult>
  CalculateGemmBBlockCopyPerformanceParameters(
      const ConvolutionContext &ctx) const;
  std::tuple<std::size_t, LogicalResult>
  CalculateLdsNumberOfByte(const ConvolutionContext &ctx) const;

  int64_t CalculateGemmASrcVectorReadDim(const ConvolutionContext &ctx) const;
  int64_t CalculateGemmBSrcVectorReadDim(const ConvolutionContext &ctx) const;
};

struct PerformanceImplicitGemmBwdDataV4R1Xdlops
    : Serializable<PerformanceImplicitGemmBwdDataV4R1Xdlops> {
  int64_t GemmNPerBlock; // 2^n[8..16]
  int64_t GemmMPerBlock; // 2^n[32..128]
  int64_t GemmKPerBlock; // 2^n[4..16]

  int64_t GemmKPACKSize; // 2^[1..4]

  int64_t GemmMPerWave;
  int64_t GemmNPerWave;
  int64_t srcVectorReadDim;

  // GemmAThreadCopyMoreGemmK is currently a fix value, is untunable
  bool GemmAThreadCopyMoreGemmK;
  bool GemmBThreadCopyMoreGemmKPack;

  PerformanceImplicitGemmBwdDataV4R1Xdlops(int64_t, int64_t, int64_t, int64_t,
                                           int64_t, int64_t, bool, bool);

  PerformanceImplicitGemmBwdDataV4R1Xdlops();

  bool operator==(const PerformanceImplicitGemmBwdDataV4R1Xdlops &other) const;

  template <class Self, class F> static void visit(Self &&self, F f) {
    f(self.GemmNPerBlock, "GemmNPerBlock");
    f(self.GemmMPerBlock, "GemmMPerBlock");
    f(self.GemmKPerBlock, "GemmKPerBlock");
    f(self.GemmKPACKSize, "GemmKPACKSize");
    f(self.GemmMPerWave, "GemmMPerWave");
    f(self.GemmNPerWave, "GemmNPerWave");
    f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
    f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
  }

  std::tuple<int64_t, LogicalResult>
  CalculateGridSize(const ConvolutionContext &ctx) const;
  std::tuple<std::size_t, LogicalResult>
  CalculateLdsNumberOfByte(const ConvolutionContext &ctx) const;
  std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, LogicalResult>
  CalculateGemmABlockCopyPerformanceParameters(
      const ConvolutionContext &ctx) const;
  std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, LogicalResult>
  CalculateGemmBBlockCopyPerformanceParameters(
      const ConvolutionContext &ctx) const;
  LogicalResult IsValidValue() const;

  int64_t CalculateGemmASrcVectorReadDim(const ConvolutionContext &ctx) const;
  int64_t CalculateGemmBSrcVectorReadDim(const ConvolutionContext &ctx) const;
  // bool IsValid(const ConvolutionContext& ctx) const;
  LogicalResult IsReallyValid(const ConvolutionContext &ctx) const;
  // bool IsFastToBeUsedForTuning(const ConvolutionContext& ctx) const;
  LogicalResult EuristicInit(const ConvolutionContext &ctx);
};

struct PerformanceImplicitGemmWrwV4R4Xdlops
    : Serializable<PerformanceImplicitGemmWrwV4R4Xdlops> {
  int64_t GemmMPerBlock;
  int64_t GemmNPerBlock;
  int64_t GemmKPerBlock;
  int64_t GemmMPerWave;
  int64_t GemmNPerWave;
  int64_t GemmKPack;
  bool GemmAThreadCopyMoreGemmK;
  bool GemmBThreadCopyMoreGemmK;
  int64_t srcVectorReadDim;

  PerformanceImplicitGemmWrwV4R4Xdlops(int64_t, int64_t, int64_t, int64_t,
                                       int64_t, int64_t, bool, bool);
  PerformanceImplicitGemmWrwV4R4Xdlops();

  template <class Self, class F> static void visit(Self &&self, F f) {
    f(self.GemmMPerBlock, "GemmMPerBlock");
    f(self.GemmNPerBlock, "GemmNPerBlock");
    f(self.GemmKPerBlock, "GemmKPerBlock");
    f(self.GemmMPerWave, "GemmMPerWave");
    f(self.GemmNPerWave, "GemmNPerWave");
    f(self.GemmKPack, "GemmKPack");
    f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
    f(self.GemmBThreadCopyMoreGemmK, "GemmBThreadCopyMoreGemmK");
  }

  bool operator==(const PerformanceImplicitGemmWrwV4R4Xdlops &other) const;
  std::string ToString() const;

  LogicalResult EuristicInit(const ConvolutionContext &ctx);
  LogicalResult IsValidValue() const;
  // LogicalResult IsValid(const ConvolutionContext& ctx) const;
  LogicalResult IsReallyValid(const ConvolutionContext &ctx) const;
  // LogicalResult IsFastToBeUsedForTuning(const ConvolutionContext& ctx) const;

  std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, LogicalResult>
  CalculateGemmSizeAndGemmKBlock(const ConvolutionContext &ctx) const;
  std::tuple<int64_t, LogicalResult> CalculateBlockSize() const;
  std::tuple<int64_t, LogicalResult>
  CalculateGridSize(const ConvolutionContext &ctx) const;
  std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, LogicalResult>
  CalculateGemmABlockCopyPerformanceParameters(
      const ConvolutionContext &ctx) const;
  std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, LogicalResult>
  CalculateGemmBBlockCopyPerformanceParameters(
      const ConvolutionContext &ctx) const;
  std::tuple<std::size_t, LogicalResult>
  CalculateLdsNumberOfByte(const ConvolutionContext &ctx) const;

  int64_t CalculateGemmASrcVectorReadDim(const ConvolutionContext &ctx) const;
  int64_t CalculateGemmBSrcVectorReadDim(const ConvolutionContext &ctx) const;
};

struct SolverBase {
  virtual LogicalResult IsApplicable(const ConvolutionContext &ctx) const = 0;
};

struct ConvHipImplicitGemmV4R4Fwd : SolverBase {
  static std::tuple<int64_t, int64_t, int64_t>
  CalculateGemmSize(const ConvolutionContext &ctx);

  LogicalResult IsApplicable(const ConvolutionContext &ctx) const;

  PerformanceImplicitGemmV4R4Fwd
  GetPerformanceConfig(const ConvolutionContext &ctx) const;

  LogicalResult
  IsValidPerformanceConfig(const ConvolutionContext &ctx,
                           const PerformanceImplicitGemmV4R4Fwd &config) const;

  llvm::StringMap<int64_t>
  GetSolution(const ConvolutionContext &ctx,
              const PerformanceImplicitGemmV4R4Fwd &config) const;

  std::string getId() { return "ConvHipImplicitGemmV4R4Fwd"; }
};

struct ConvHipImplicitGemmV4R4WrW : SolverBase {
  static std::tuple<int64_t, int64_t, int64_t>

  CalculateGemmSize(const ConvolutionContext &ctx);

  LogicalResult IsApplicable(const ConvolutionContext &ctx) const;

  PerformanceImplicitGemmV4R4WrW
  GetPerformanceConfig(const ConvolutionContext &ctx) const;

  LogicalResult
  IsValidPerformanceConfig(const ConvolutionContext &ctx,
                           const PerformanceImplicitGemmV4R4WrW &config) const;

  llvm::StringMap<int64_t>
  GetSolution(const ConvolutionContext &ctx,
              const PerformanceImplicitGemmV4R4WrW &config) const;

  std::string getId() { return "ConvHipImplicitGemmV4R4WrW"; }
};

struct ConvHipImplicitGemmBwdDataV1R1 : SolverBase {
  static std::tuple<int64_t, int64_t, int64_t>
  CalculateGemmSize(const ConvolutionContext &ctx);

  LogicalResult IsApplicable(const ConvolutionContext &ctx) const;

  PerformanceImplicitGemmBwdDataV1R1
  GetPerformanceConfig(const ConvolutionContext &ctx) const;

  LogicalResult IsValidPerformanceConfig(
      const ConvolutionContext &ctx,
      const PerformanceImplicitGemmBwdDataV1R1 &config) const;

  llvm::StringMap<int64_t>
  GetSolution(const ConvolutionContext &ctx,
              const PerformanceImplicitGemmBwdDataV1R1 &config) const;

  std::string getId() { return "ConvHipImplicitGemmBwdDataV1R1"; }
};

struct ConvHipImplicitGemmForwardV4R4Xdlops : SolverBase {
  static std::tuple<int64_t, int64_t, int64_t, int64_t>
  CalculateGemmSize(const ConvolutionContext &ctx);

  PerformanceImplicitGemmForwardV4R4Xdlops
  GetPerformanceConfig(const ConvolutionContext &ctx) const;

  LogicalResult IsValidPerformanceConfig(
      const ConvolutionContext &ctx,
      const PerformanceImplicitGemmForwardV4R4Xdlops &c) const;

  LogicalResult IsApplicable(const ConvolutionContext &ctx) const;

  llvm::StringMap<int64_t>
  GetSolution(const ConvolutionContext &ctx,
              const PerformanceImplicitGemmForwardV4R4Xdlops &config) const;

  std::string getId() { return "ConvHipImplicitGemmForwardV4R4Xdlops"; }
};

struct ConvHipImplicitGemmBwdDataV4R1Xdlops : SolverBase {
  static int64_t CalculateNumberOfGemm(const ConvolutionContext &ctx);

  static std::tuple<int64_t, int64_t, int64_t, int64_t>
  CalculateGemmSize(const ConvolutionContext &ctx, int64_t gemm_id);

  PerformanceImplicitGemmBwdDataV4R1Xdlops
  GetPerformanceConfig(const ConvolutionContext &ctx) const;

  LogicalResult IsValidPerformanceConfig(
      const ConvolutionContext &ctx,
      const PerformanceImplicitGemmBwdDataV4R1Xdlops &c) const;

  LogicalResult IsApplicable(const ConvolutionContext &ctx) const;

  llvm::StringMap<int64_t>
  GetSolution(const ConvolutionContext &ctx,
              const PerformanceImplicitGemmBwdDataV4R1Xdlops &config) const;

  std::string getId() { return "ConvHipImplicitGemmBwdDataV4R1Xdlops"; }
};

struct ConvHipImplicitGemmWrwV4R4Xdlops : SolverBase {
  PerformanceImplicitGemmWrwV4R4Xdlops
  GetPerformanceConfig(const ConvolutionContext &ctx) const;

  LogicalResult
  IsValidPerformanceConfig(const ConvolutionContext &ctx,
                           const PerformanceImplicitGemmWrwV4R4Xdlops &c) const;

  LogicalResult IsApplicable(const ConvolutionContext &ctx) const;

  llvm::StringMap<int64_t>
  GetSolution(const ConvolutionContext &ctx,
              const PerformanceImplicitGemmWrwV4R4Xdlops &config) const;
  std::string getId() { return "ConvHipImplicitGemmWrwV4R4Xdlops"; }
};

template <class... Solvers> struct SolverContainer {
  std::tuple<llvm::StringMap<int64_t>, LogicalResult>
  SearchForConfigParameters(const ConvolutionContext &ctx);
};

std::tuple<llvm::StringMap<int64_t>, LogicalResult>
GetConfigParameters(const ConvolutionContext &ctx);

#endif // MLIR_DIALECT_lMIOPEN_GRIDWISE_GEMM_PARAMS_H
