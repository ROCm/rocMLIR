//===-- WinogradSolver.cpp - Winograd kernel selection ---------------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//
//
// Ports MIOpen's Winograd solver applicability and kernel selection logic.
// Each isApplicable* method mirrors the constraints of the corresponding
// MIOpen solver class.  The WTI (work-time index) computation uses a
// simplified cost model that captures tile granularity and arithmetic
// reduction, yielding a score where higher = predicted faster.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Winograd/WinogradSolver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <string>

using namespace mlir::rock::winograd;

// ===--------------------------------------------------------------------=== //
// Anonymous-namespace helpers
// ===--------------------------------------------------------------------=== //

namespace {

/// Map a chip name to the generation digit used in kernel entry-point names.
std::string getArchGen(llvm::StringRef chip) {
  if (chip.starts_with("gfx12"))
    return "12";
  if (chip.starts_with("gfx11"))
    return "11";
  if (chip.starts_with("gfx10"))
    return "10";
  if (chip.starts_with("gfx9"))
    return "9";
  return "unknown";
}

std::string getDtypeStr(const WinogradConvProblem &p) {
  if (p.isFp16)
    return "fp16";
  if (p.isBf16)
    return "bf16";
  return "fp32";
}

std::string getFilterStr(const WinogradConvProblem &p) {
  if (p.direction == WinogradDirection::BackwardWeight)
    return "f3x2";
  return "f2x3";
}

std::string getStrideStr(const WinogradConvProblem &p) {
  if (p.strideH == 2 || p.strideW == 2)
    return "stride2";
  return "stride1";
}

bool fitsInBits(int64_t val, int bits) {
  return val >= 0 && val < (static_cast<int64_t>(1) << bits);
}

/// Return true when a * b < 2^bits (overflow-safe).
bool productFitsInBits(int64_t a, int64_t b, int bits) {
  if (a <= 0 || b <= 0)
    return true;
  int64_t limit = static_cast<int64_t>(1) << bits;
  return a < limit / b;
}

/// Return true when a * b * c < 2^bits (overflow-safe).
bool product3FitsInBits(int64_t a, int64_t b, int64_t c, int bits) {
  if (a <= 0 || b <= 0 || c <= 0)
    return true;
  int64_t limit = static_cast<int64_t>(1) << bits;
  // Check a*b overflow-safely using division
  if (a >= limit / b)
    return false;
  // Now a*b < limit, so the multiply is safe
  int64_t ab = a * b;
  if (c > 0 && ab >= limit / c)
    return false;
  return true;
}

/// Return true when a * b * c * d < 2^bits (overflow-safe).
bool product4FitsInBits(int64_t a, int64_t b, int64_t c, int64_t d, int bits) {
  if (a <= 0 || b <= 0 || c <= 0 || d <= 0)
    return true;
  int64_t limit = static_cast<int64_t>(1) << bits;
  // Check each intermediate product using division before multiplying
  if (a >= limit / b)
    return false;
  int64_t ab = a * b;
  if (c > 0 && ab >= limit / c)
    return false;
  int64_t abc = ab * c;
  if (d > 0 && abc >= limit / d)
    return false;
  return true;
}

/// Common feasibility predicate shared by V21 / V30 / V40.
bool isCommonSpConvApplicable(const WinogradConvProblem &p) {
  if (!p.isNCHW)
    return false; // Winograd assembly kernels only support NCHW layout
  if (p.direction != WinogradDirection::Forward &&
      p.direction != WinogradDirection::BackwardData)
    return false;
  if (!p.isFp32 && !p.isFp16)
    return false;
  if (p.R != 3 || p.S != 3)
    return false;
  if (p.strideH < 1 || p.strideH > 2 || p.strideW < 1 || p.strideW > 2)
    return false;
  if (p.dilationH != 1 || p.dilationW != 1)
    return false;
  if (p.groupCount != 1)
    return false;
  return true;
}

/// Enumerate candidate n_groups values for the standard Sp3 families.
llvm::SmallVector<int64_t> getCandidateNGroups(int64_t numCU) {
  llvm::SmallVector<int64_t> cands;
  for (int64_t ng = 1; ng <= numCU * 4 && ng <= 512; ng *= 2)
    cands.push_back(ng);
  if (numCU > 0) {
    cands.push_back(numCU);
    cands.push_back(numCU * 2);
  }
  llvm::sort(cands);
  cands.erase(std::unique(cands.begin(), cands.end()), cands.end());
  return cands;
}

std::string channelModeToStr(WinogradChannelMode mode) {
  switch (mode) {
  case WinogradChannelMode::C16:
    return "c16";
  case WinogradChannelMode::C32:
    return "c32";
  default:
    return "default";
  }
}

WinogradChannelMode strToChannelMode(llvm::StringRef s) {
  if (s == "c16" || s == "C16")
    return WinogradChannelMode::C16;
  if (s == "c32" || s == "C32")
    return WinogradChannelMode::C32;
  return WinogradChannelMode::Default;
}

std::string familyToStr(WinogradFamily family) {
  switch (family) {
  case WinogradFamily::V21:
    return "V21";
  case WinogradFamily::V30:
    return "V30";
  case WinogradFamily::V40:
    return "V40";
  case WinogradFamily::Fury_V2:
    return "FuryV2";
  case WinogradFamily::Fury_V4:
    return "FuryV4";
  case WinogradFamily::Rage_V4_6:
    return "RageV4_6";
  case WinogradFamily::Rage_V4_9:
    return "RageV4_9";
  }
  llvm_unreachable("unhandled WinogradFamily");
}

std::optional<WinogradFamily> strToFamily(llvm::StringRef s) {
  return llvm::StringSwitch<std::optional<WinogradFamily>>(s)
      .Case("V21", WinogradFamily::V21)
      .Case("V30", WinogradFamily::V30)
      .Case("V40", WinogradFamily::V40)
      .Case("FuryV2", WinogradFamily::Fury_V2)
      .Case("FuryV4", WinogradFamily::Fury_V4)
      .Case("RageV4_6", WinogradFamily::Rage_V4_6)
      .Case("RageV4_9", WinogradFamily::Rage_V4_9)
      .Default(std::nullopt);
}

/// Common feasibility predicate for Fury / Rage families.
bool isFuryRageBaseApplicable(const WinogradConvProblem &p) {
  if (!p.isNCHW)
    return false; // Winograd assembly kernels only support NCHW layout
  if (p.direction != WinogradDirection::Forward &&
      p.direction != WinogradDirection::BackwardData)
    return false;
  if (p.strideH != 1 || p.strideW != 1)
    return false;
  if (p.dilationH != 1 || p.dilationW != 1)
    return false;
  if (p.R > 3 || p.S > 3)
    return false;
  if (p.groupCount != 1)
    return false;
  return true;
}

/// Determine the viable Fury channel modes for a problem.
llvm::SmallVector<WinogradChannelMode>
getFuryChannelModes(const WinogradConvProblem &p) {
  llvm::SmallVector<WinogradChannelMode> modes;
  if (p.C % 32 == 0 && p.K % 32 == 0)
    modes.push_back(WinogradChannelMode::C32);
  if (p.C % 16 == 0 && p.K % 16 == 0)
    modes.push_back(WinogradChannelMode::C16);
  return modes;
}

} // anonymous namespace

// ===--------------------------------------------------------------------=== //
// Public helpers
// ===--------------------------------------------------------------------=== //

bool mlir::rock::winograd::archStartsWith(llvm::StringRef arch,
                                          llvm::StringRef prefix) {
  std::string chip = extractChipName(arch);
  return llvm::StringRef(chip).starts_with(prefix);
}

std::string mlir::rock::winograd::extractChipName(llvm::StringRef arch) {
  // Handle "amdgcn-amd-amdhsa:gfx942" -> "gfx942"
  auto colonPos = arch.rfind(':');
  if (colonPos != llvm::StringRef::npos)
    return arch.substr(colonPos + 1).str();
  // Already a bare chip name like "gfx942"
  return arch.str();
}

// ===--------------------------------------------------------------------=== //
// Shader constraint helpers
// ===--------------------------------------------------------------------=== //

bool WinogradSolver::isShaderConstraintsMetV21(
    const WinogradConvProblem &problem) {
  const auto &p = problem;
  if (!fitsInBits(p.N, 16) || !fitsInBits(p.C, 16) || !fitsInBits(p.H, 16) ||
      !fitsInBits(p.W, 16) || !fitsInBits(p.K, 16) || !fitsInBits(p.outH, 16) ||
      !fitsInBits(p.outW, 16))
    return false;
  if (!product3FitsInBits(p.C, p.R, p.S, 22))
    return false;
  if (!product3FitsInBits(p.K, p.R, p.S, 28))
    return false;
  if (p.isFp16 && (p.C % 2 != 0))
    return false;
  return true;
}

bool WinogradSolver::isShaderConstraintsMetV30(
    const WinogradConvProblem &problem) {
  const auto &p = problem;
  if (!fitsInBits(p.N, 16) || !fitsInBits(p.C, 16) || !fitsInBits(p.H, 16) ||
      !fitsInBits(p.W, 16) || !fitsInBits(p.K, 16) || !fitsInBits(p.outH, 16) ||
      !fitsInBits(p.outW, 16))
    return false;
  if (!productFitsInBits(p.H, p.W, 29))
    return false;
  if (!productFitsInBits(p.C + 1, p.H * p.W, 30))
    return false;
  if (!product3FitsInBits(p.C + 1, p.R, p.S, 22))
    return false;
  if (!product3FitsInBits(p.K + 1, p.outH, p.outW, 30))
    return false;
  return true;
}

// ===--------------------------------------------------------------------=== //
// Per-family applicability (ported from MIOpen solver classes)
// ===--------------------------------------------------------------------=== //

bool WinogradSolver::isApplicableV21(const WinogradConvProblem &problem) {
  std::string chip = extractChipName(problem.arch);
  if (chip != "gfx900" && chip != "gfx906")
    return false;
  if (problem.isXnackEnabled)
    return false;
  if (!isCommonSpConvApplicable(problem))
    return false;
  return isShaderConstraintsMetV21(problem);
}

bool WinogradSolver::isApplicableV30(const WinogradConvProblem &problem) {
  std::string chip = extractChipName(problem.arch);
  llvm::StringRef chipRef(chip);
  if (!chipRef.starts_with("gfx9") && !chipRef.starts_with("gfx10") &&
      !chipRef.starts_with("gfx11"))
    return false;
  if (problem.isXnackEnabled)
    return false;
  if (!isCommonSpConvApplicable(problem))
    return false;
  return isShaderConstraintsMetV30(problem);
}

bool WinogradSolver::isApplicableV40(const WinogradConvProblem &problem) {
  std::string chip = extractChipName(problem.arch);
  if (!llvm::StringRef(chip).starts_with("gfx12"))
    return false;
  if (problem.isXnackEnabled)
    return false;
  if (!isCommonSpConvApplicable(problem))
    return false;
  return isShaderConstraintsMetV30(problem);
}

bool WinogradSolver::isApplicableFuryV2(const WinogradConvProblem &problem) {
  std::string chip = extractChipName(problem.arch);
  if (!llvm::StringRef(chip).starts_with("gfx11"))
    return false;
  if (!problem.isFp16)
    return false;
  if (!isFuryRageBaseApplicable(problem))
    return false;

  auto modes = getFuryChannelModes(problem);
  if (modes.empty())
    return false;

  if (!product4FitsInBits(problem.N, problem.C, problem.H, problem.W, 31))
    return false;
  if (!product4FitsInBits(problem.N, problem.K, problem.outH, problem.outW, 31))
    return false;
  return true;
}

bool WinogradSolver::isApplicableFuryV4(const WinogradConvProblem &problem) {
  std::string chip = extractChipName(problem.arch);
  if (!llvm::StringRef(chip).starts_with("gfx12"))
    return false;
  if (!problem.isFp16)
    return false;
  if (!isFuryRageBaseApplicable(problem))
    return false;

  auto modes = getFuryChannelModes(problem);
  if (modes.empty())
    return false;
  return true;
}

bool WinogradSolver::isApplicableRageV4_6(const WinogradConvProblem &problem) {
  std::string chip = extractChipName(problem.arch);
  llvm::StringRef chipRef(chip);
  if (chip != "gfx942" && !chipRef.starts_with("gfx12"))
    return false;
  if (!problem.isFp16)
    return false;
  if (!isFuryRageBaseApplicable(problem))
    return false;
  return true;
}

bool WinogradSolver::isApplicableRageV4_9(const WinogradConvProblem &problem) {
  std::string chip = extractChipName(problem.arch);
  llvm::StringRef chipRef(chip);
  if (chip != "gfx942" && !chipRef.starts_with("gfx12"))
    return false;

  // gfx942 supports fp16/fp32/bf16; gfx12 supports fp16 only
  if (chip == "gfx942") {
    if (!problem.isFp16 && !problem.isFp32 && !problem.isBf16)
      return false;
  } else {
    if (!problem.isFp16)
      return false;
  }

  if (problem.direction != WinogradDirection::Forward &&
      problem.direction != WinogradDirection::BackwardData)
    return false;
  if (!problem.isNCHW)
    return false;
  if (problem.strideH != 1 || problem.strideW != 1)
    return false;
  if (problem.dilationH != 1 || problem.dilationW != 1)
    return false;
  if (problem.R > 3 || problem.S > 3)
    return false;
  if (problem.groupCount != 1)
    return false;

  // Dimension limits
  if (!fitsInBits(problem.N, 16) || !fitsInBits(problem.C, 16) ||
      !fitsInBits(problem.H, 16) || !fitsInBits(problem.W, 16) ||
      !fitsInBits(problem.K, 16) || !fitsInBits(problem.outH, 16) ||
      !fitsInBits(problem.outW, 16))
    return false;

  // Batch tensor sizes must fit in 31 bits
  if (!product4FitsInBits(problem.N, problem.C, problem.H, problem.W, 31))
    return false;
  if (!product4FitsInBits(problem.N, problem.K, problem.outH, problem.outW, 31))
    return false;
  return true;
}

// ===--------------------------------------------------------------------=== //
// WTI computation
// ===--------------------------------------------------------------------=== //

// F(2,3) Winograd reduces 2x2 * 3x3 = 36 MACs to 4x4 = 16 multiplications
// per output tile (2.25x arithmetic reduction).  The WTI score captures this
// benefit minus the granularity losses from tiling and channel alignment.

float WinogradSolver::computeWtiV30(const WinogradConvProblem &problem,
                                    int64_t nGroups) {
  int64_t tileH = (problem.outH + 1) / 2;
  int64_t tileW = (problem.outW + 1) / 2;
  int64_t totalTiles = problem.N * tileH * tileW;

  if (totalTiles == 0 || nGroups <= 0)
    return -1.0f;

  double directFlops =
      2.0 * problem.N * problem.K * problem.outH * problem.outW * problem.C *
      problem.R * problem.S;

  // 1.3 accounts for input/output transform overhead
  constexpr double kTransformOverhead = 1.3;
  double winogradFlops =
      2.0 * totalTiles * problem.K * problem.C * 16.0 * kTransformOverhead;

  if (winogradFlops <= 0.0)
    return -1.0f;

  // Spatial granularity: how evenly tiles distribute across workgroups
  int64_t tilesPerWg = (totalTiles + nGroups - 1) / nGroups;
  double spatialGran =
      static_cast<double>(totalTiles) / (static_cast<double>(tilesPerWg) * nGroups);

  // Channel granularity: how well K maps to wavefront-aligned tiles
  int64_t kAlign = problem.isFp16 ? 32 : 16;
  int64_t alignedK = ((problem.K + kAlign - 1) / kAlign) * kAlign;
  double channelGran = static_cast<double>(problem.K) / static_cast<double>(alignedK);

  return static_cast<float>((directFlops / winogradFlops) * spatialGran *
                            channelGran);
}

float WinogradSolver::computeWtiFury(const WinogradConvProblem &problem,
                                     int64_t nGroups,
                                     WinogradChannelMode mode) {
  int64_t tileH = (problem.outH + 1) / 2;
  int64_t tileW = (problem.outW + 1) / 2;
  int64_t totalTiles = problem.N * tileH * tileW;

  if (totalTiles == 0 || nGroups <= 0)
    return -1.0f;

  double directFlops =
      2.0 * problem.N * problem.K * problem.outH * problem.outW * problem.C *
      problem.R * problem.S;

  // Fury kernels have tighter transform pipelines than V30
  constexpr double kTransformOverhead = 1.1;
  double winogradFlops =
      2.0 * totalTiles * problem.K * problem.C * 16.0 * kTransformOverhead;

  if (winogradFlops <= 0.0)
    return -1.0f;

  int64_t tilesPerWg = (totalTiles + nGroups - 1) / nGroups;
  double spatialGran =
      static_cast<double>(totalTiles) / (static_cast<double>(tilesPerWg) * nGroups);

  // C32 mode packs more channels per iteration -> better utilization
  int64_t cAlign = (mode == WinogradChannelMode::C32) ? 32 : 16;
  int64_t alignedC = ((problem.C + cAlign - 1) / cAlign) * cAlign;
  double channelGran = static_cast<double>(problem.C) / static_cast<double>(alignedC);

  return static_cast<float>((directFlops / winogradFlops) * spatialGran *
                            channelGran);
}

float WinogradSolver::computeWtiRage(const WinogradConvProblem &problem,
                                     int64_t nGroups) {
  int64_t tileH = (problem.outH + 1) / 2;
  int64_t tileW = (problem.outW + 1) / 2;
  int64_t totalTiles = problem.N * tileH * tileW;

  if (totalTiles == 0 || nGroups <= 0)
    return -1.0f;

  double directFlops =
      2.0 * problem.N * problem.K * problem.outH * problem.outW * problem.C *
      problem.R * problem.S;

  // Rage kernels have the best-optimised transform pipeline
  constexpr double kTransformOverhead = 1.05;
  double winogradFlops =
      2.0 * totalTiles * problem.K * problem.C * 16.0 * kTransformOverhead;

  if (winogradFlops <= 0.0)
    return -1.0f;

  int64_t tilesPerWg = (totalTiles + nGroups - 1) / nGroups;
  double spatialGran =
      static_cast<double>(totalTiles) / (static_cast<double>(tilesPerWg) * nGroups);

  int64_t kAlign = 16;
  if (problem.isFp16)
    kAlign = 32;
  int64_t alignedK = ((problem.K + kAlign - 1) / kAlign) * kAlign;
  double channelGran = static_cast<double>(problem.K) / static_cast<double>(alignedK);

  return static_cast<float>((directFlops / winogradFlops) * spatialGran *
                            channelGran);
}

// ===--------------------------------------------------------------------=== //
// buildSelection – construct kernel file / entry-point names
// ===--------------------------------------------------------------------=== //

WinogradKernelSelection
WinogradSolver::buildSelection(WinogradFamily family,
                               const WinogradConvProblem &problem,
                               int64_t nGroups, WinogradChannelMode mode) {
  WinogradKernelSelection sel;
  sel.family = family;
  sel.nGroups = nGroups;
  sel.channelMode = mode;

  std::string chip = extractChipName(problem.arch);
  std::string gen = getArchGen(chip);
  std::string dtype = getDtypeStr(problem);

  // Defaults
  sel.blockSize = 512;
  sel.gridSize = nGroups;
  sel.compOptions = "-mcpu=" + chip;

  auto buildSp3Name = [&](llvm::StringRef verTag, llvm::StringRef verNums) {
    std::string filter = getFilterStr(problem);
    std::string stride = getStrideStr(problem);
    sel.kernelFile = "Conv_Winograd_" + verTag.str() + "_" + verNums.str() +
                     "_" + dtype + "_" + filter + "_" + stride + ".s";
    sel.kernelName = "miopenSp3AsmConv_" + verTag.str() + "_" + verNums.str() +
                     "_gfx" + gen + "_" + dtype + "_" + filter + "_" + stride;
  };

  switch (family) {
  case WinogradFamily::V21:
    buildSp3Name("v21", "1_3");
    sel.abiVersion = 1;
    break;

  case WinogradFamily::V30:
    buildSp3Name("v30", "3_1");
    sel.abiVersion = 1;
    break;

  case WinogradFamily::V40:
    buildSp3Name("v40", "6_0");
    sel.abiVersion = 1;
    break;

  case WinogradFamily::Fury_V2: {
    sel.blockSize = 384;
    std::string cmode = channelModeToStr(mode);
    // Fury v2 uses fp16_fp16acc in filenames
    std::string furyDtype = "fp16_fp16acc";
    sel.kernelFile =
        "Conv_Winograd_Fury_v2_4_1_" + furyDtype + "_f2x3_" + cmode + "_stride1.s";
    // Fury kernel names include vgpr count based on arch
    sel.kernelName = "miopenSp3AsmConvFury_v2_4_1_gfx" + gen + "_1536vgprs" +
                     "_" + furyDtype + "_f2x3_" + cmode + "_stride1";
    sel.abiVersion = 2;
    break;
  }

  case WinogradFamily::Fury_V4: {
    sel.blockSize = 384;
    std::string cmode = channelModeToStr(mode);
    // Fury v4 uses fp16_fp32acc in filenames
    std::string furyDtype = "fp16_fp32acc";
    sel.kernelFile =
        "Conv_Winograd_Fury_v4_6_0_" + furyDtype + "_f2x3_" + cmode + "_stride1.s";
    std::string vgprTag = "1536vgprs";
    sel.kernelName = "miopenSp3AsmConvFury_v4_6_0_gfx" + gen + "_" + vgprTag +
                     "_" + furyDtype + "_f2x3_" + cmode + "_stride1";
    sel.abiVersion = 2;
    break;
  }

  case WinogradFamily::Rage_V4_6: {
    // Rage v4_6 is fp16 only, uses fp16_fp32acc in filename
    sel.blockSize = llvm::StringRef(chip).starts_with("gfx12") ? 384 : 768;
    std::string rageDtype = "fp16_fp32acc";
    sel.kernelFile =
        "Conv_Winograd_Rage_v4_6_1_" + rageDtype + "_f2x3_stride1.s";
    std::string archTag = (chip == "gfx942") ? "gfx9" : "gfx12";
    sel.kernelName = "miopenSp3AsmConvRage_v4_6_1_" + archTag + "_" +
                     rageDtype + "_f2x3_stride1";
    sel.abiVersion = 2;
    break;
  }

  case WinogradFamily::Rage_V4_9: {
    sel.blockSize = llvm::StringRef(chip).starts_with("gfx12") ? 384 : 768;
    // Rage v4_9 supports multiple dtypes with fp32acc suffix
    std::string rageDtype;
    if (problem.isFp16)
      rageDtype = "fp16_fp32acc";
    else if (problem.isBf16)
      rageDtype = "bf16_fp32acc";
    else
      rageDtype = "fp32_fp32acc";
    sel.kernelFile =
        "Conv_Winograd_Rage_v4_9_0_" + rageDtype + "_f2x3_stride1.s";
    std::string archTag = (chip == "gfx942") ? "gfx9" : "gfx12";
    sel.kernelName = "miopenSp3AsmConvRage_v4_9_0_" + archTag + "_" +
                     rageDtype + "_f2x3_stride1";
    sel.abiVersion = 2;
    break;
  }
  }

  // Compute WTI for the selection
  switch (family) {
  case WinogradFamily::V21:
  case WinogradFamily::V30:
  case WinogradFamily::V40:
    sel.wti = computeWtiV30(problem, nGroups);
    break;
  case WinogradFamily::Fury_V2:
  case WinogradFamily::Fury_V4:
    sel.wti = computeWtiFury(problem, nGroups, mode);
    break;
  case WinogradFamily::Rage_V4_6:
  case WinogradFamily::Rage_V4_9:
    sel.wti = computeWtiRage(problem, nGroups);
    break;
  }

  return sel;
}

// ===--------------------------------------------------------------------=== //
// Public API
// ===--------------------------------------------------------------------=== //

bool WinogradSolver::isApplicable(const WinogradConvProblem &problem) {
  return isApplicableV21(problem) || isApplicableV30(problem) ||
         isApplicableV40(problem) || isApplicableFuryV2(problem) ||
         isApplicableFuryV4(problem) || isApplicableRageV4_6(problem) ||
         isApplicableRageV4_9(problem);
}

llvm::SmallVector<WinogradKernelSelection>
WinogradSolver::findApplicable(const WinogradConvProblem &problem) {
  llvm::SmallVector<WinogradKernelSelection> results;
  auto nGroupCands = getCandidateNGroups(problem.numCU);

  // Helper: try all n_groups for a Sp3-family (V21/V30/V40)
  auto trySp3 = [&](WinogradFamily family, bool applicable) {
    if (!applicable)
      return;
    for (int64_t ng : nGroupCands) {
      auto sel =
          buildSelection(family, problem, ng, WinogradChannelMode::Default);
      if (sel.wti > 0.0f)
        results.push_back(std::move(sel));
    }
  };

  trySp3(WinogradFamily::V21, isApplicableV21(problem));
  trySp3(WinogradFamily::V30, isApplicableV30(problem));
  trySp3(WinogradFamily::V40, isApplicableV40(problem));

  // Helper: try all n_groups x channel_modes for a Fury family
  auto tryFury = [&](WinogradFamily family, bool applicable) {
    if (!applicable)
      return;
    auto modes = getFuryChannelModes(problem);
    int64_t minNg = std::max<int64_t>(1, (problem.K + 15) / 16);
    for (int64_t ng : nGroupCands) {
      if (ng < minNg)
        continue;
      for (auto cmode : modes) {
        auto sel = buildSelection(family, problem, ng, cmode);
        if (sel.wti > 0.0f)
          results.push_back(std::move(sel));
      }
    }
  };

  tryFury(WinogradFamily::Fury_V2, isApplicableFuryV2(problem));
  tryFury(WinogradFamily::Fury_V4, isApplicableFuryV4(problem));

  // Helper: try all n_groups for a Rage family
  auto tryRage = [&](WinogradFamily family, bool applicable) {
    if (!applicable)
      return;
    for (int64_t ng : nGroupCands) {
      auto sel =
          buildSelection(family, problem, ng, WinogradChannelMode::Default);
      if (sel.wti > 0.0f)
        results.push_back(std::move(sel));
    }
  };

  tryRage(WinogradFamily::Rage_V4_6, isApplicableRageV4_6(problem));
  tryRage(WinogradFamily::Rage_V4_9, isApplicableRageV4_9(problem));

  // Sort by WTI descending (best first)
  llvm::sort(results,
             [](const WinogradKernelSelection &a,
                const WinogradKernelSelection &b) { return a.wti > b.wti; });

  return results;
}

std::optional<WinogradKernelSelection>
WinogradSolver::selectBest(const WinogradConvProblem &problem) {
  auto all = findApplicable(problem);
  if (all.empty())
    return std::nullopt;
  return all.front();
}

// ===--------------------------------------------------------------------=== //
// Perf-config serialization
// ===--------------------------------------------------------------------=== //

std::string
WinogradSolver::toPerfConfigStr(const WinogradKernelSelection &selection) {
  // Extract the data path portion from the kernel file name.
  // The file name format is Conv_Winograd_<version>_<dataPath>_stride1.s
  // For Fury/Rage: <dataPath> includes accumulation type like fp16_fp32acc_f2x3
  // For V21/V30/V40: <dataPath> is like fp32_f2x3
  std::string dataPath;
  llvm::StringRef file(selection.kernelFile);

  // Remove prefix and .s suffix to extract the variable part
  if (file.ends_with(".s"))
    file = file.drop_back(2);

  // For different families, extract what comes after the version prefix
  switch (selection.family) {
  case WinogradFamily::Fury_V2:
    if (auto pos = file.find("v2_4_1_"); pos != llvm::StringRef::npos)
      dataPath = file.substr(pos + 7).str(); // after "v2_4_1_"
    break;
  case WinogradFamily::Fury_V4:
    if (auto pos = file.find("v4_6_0_"); pos != llvm::StringRef::npos)
      dataPath = file.substr(pos + 7).str();
    break;
  case WinogradFamily::Rage_V4_6:
    if (auto pos = file.find("v4_6_1_"); pos != llvm::StringRef::npos)
      dataPath = file.substr(pos + 7).str();
    break;
  case WinogradFamily::Rage_V4_9:
    if (auto pos = file.find("v4_9_0_"); pos != llvm::StringRef::npos)
      dataPath = file.substr(pos + 7).str();
    break;
  default:
    // V21/V30/V40: extract after version tag
    if (auto pos = file.find("1_3_"); pos != llvm::StringRef::npos)
      dataPath = file.substr(pos + 4).str();
    else if (auto pos2 = file.find("3_1_"); pos2 != llvm::StringRef::npos)
      dataPath = file.substr(pos2 + 4).str();
    else if (auto pos3 = file.find("6_0_"); pos3 != llvm::StringRef::npos)
      dataPath = file.substr(pos3 + 4).str();
    break;
  }

  if (dataPath.empty())
    dataPath = "unknown";

  std::string result;
  llvm::raw_string_ostream os(result);
  os << "winograd:v1," << familyToStr(selection.family) << ","
     << selection.nGroups << "," << channelModeToStr(selection.channelMode)
     << "," << dataPath;
  return result;
}

std::optional<WinogradKernelSelection>
WinogradSolver::resolveFromPerfConfig(const WinogradConvProblem &problem,
                                      llvm::StringRef perfConfig) {
  // Expected format: "winograd:v1,<family>,<nGroups>,<channelMode>,<dataPath>"
  if (!perfConfig.starts_with("winograd:v1,"))
    return std::nullopt;

  llvm::StringRef rest =
      perfConfig.drop_front(llvm::StringRef("winograd:v1,").size());

  // Split on commas
  llvm::SmallVector<llvm::StringRef, 4> parts;
  rest.split(parts, ',');
  if (parts.size() < 4)
    return std::nullopt;

  auto family = strToFamily(parts[0]);
  if (!family)
    return std::nullopt;

  int64_t nGroups = 0;
  if (parts[1].getAsInteger(10, nGroups) || nGroups <= 0)
    return std::nullopt;

  WinogradChannelMode cmode = strToChannelMode(parts[2]);

  return buildSelection(*family, problem, nGroups, cmode);
}
