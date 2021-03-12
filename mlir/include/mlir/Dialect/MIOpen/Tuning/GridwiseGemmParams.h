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
#include "mlir/Dialect/MIOpen/Tuning/ImplicitGemm_util.h"
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

template <typename PerformanceImplicitGemm_t>
inline static auto GetPerformanceConfigBase(const ConvolutionContext &ctx) {
  PerformanceImplicitGemm_t pp;
  pp.HeuristicInit(ctx);
  return pp;
}

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

  LogicalResult IsValidValue() const;
  LogicalResult IsValid(const ConvolutionContext &ctx) const;
  LogicalResult HeuristicInit(const ConvolutionContext &ctx);
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

  LogicalResult IsValidValue() const;
  LogicalResult IsValid(const ConvolutionContext &ctx) const;
  LogicalResult HeuristicInit(const ConvolutionContext &ctx);
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

  LogicalResult IsValidValue() const;
  LogicalResult IsValid(const ConvolutionContext &ctx) const;
  LogicalResult HeuristicInit(const ConvolutionContext &ctx);
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

  LogicalResult HeuristicInit(const ConvolutionContext &ctx);
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

  LogicalResult IsReallyValid(const ConvolutionContext &ctx) const;
  LogicalResult HeuristicInit(const ConvolutionContext &ctx);
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

  LogicalResult HeuristicInit(const ConvolutionContext &ctx);
  LogicalResult IsValidValue() const;
  LogicalResult IsReallyValid(const ConvolutionContext &ctx) const;

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
};

// Base class for problem solvers.
struct SolverBase {
  virtual LogicalResult IsApplicable(const ConvolutionContext &ctx) const = 0;

  int64_t CalculateGemmASrcVectorReadDim(const ConvolutionContext &ctx) const {
    auto dimIndexVal = ctx.dimIndexVal;
    bool Vectorizable = false;
    ImplicitGemmUtil::obtainGemmADimKVectorizable(ctx.opType, dimIndexVal,
                                                  Vectorizable);
    if (Vectorizable)
      return 1;
    else
      return 0;
  }

  int64_t CalculateGemmBSrcVectorReadDim(const ConvolutionContext &ctx) const {
    auto dimIndexVal = ctx.dimIndexVal;
    bool Vectorizable;
    ImplicitGemmUtil::obtainGemmBDimKVectorizable(ctx.opType, dimIndexVal,
                                                  Vectorizable);
    if (Vectorizable)
      return 1;
    else
      return 0;
  }
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
