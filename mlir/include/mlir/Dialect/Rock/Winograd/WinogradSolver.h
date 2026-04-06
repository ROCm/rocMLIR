//===-- WinogradSolver.h - Winograd kernel selection -------------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//
//
// Selects the best Winograd assembly kernel for a given convolution problem.
// Ports MIOpen's Winograd solver applicability and selection logic to rocMLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_WINOGRAD_WINOGRAD_SOLVER_H
#define MLIR_DIALECT_ROCK_WINOGRAD_WINOGRAD_SOLVER_H

#include "mlir/Dialect/Rock/Winograd/WinogradConvProblem.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace rock {
namespace winograd {

enum class WinogradFamily {
  V21,       // Conv_Winograd_v21_1_3 - gfx900/gfx906
  V30,       // Conv_Winograd_v30_3_1 - gfx9/gfx10/gfx11
  V40,       // Conv_Winograd_v40_6_0 - gfx12
  Fury_V2,   // Conv_Winograd_Fury_v2_4_1 - gfx11, fp16 only
  Fury_V4,   // Conv_Winograd_Fury_v4_6_0 - gfx12, fp16 only
  Rage_V4_6, // Conv_Winograd_Rage_v4_6_1 - gfx942/gfx12, fp16 only
  Rage_V4_9, // Conv_Winograd_Rage_v4_9_0 - gfx942/gfx12, fp16/fp32/bf16
};

enum class WinogradChannelMode {
  C16,
  C32,
  Default, // for non-Fury families
};

/// Result of kernel selection.
struct WinogradKernelSelection {
  WinogradFamily family = WinogradFamily::V30;
  std::string kernelFile;
  std::string kernelName;
  std::string compOptions;
  int64_t blockSize = 0;
  int64_t gridSize = 0;
  int64_t nGroups = 0;
  WinogradChannelMode channelMode = WinogradChannelMode::Default;
  float wti = 0.0f;
  int abiVersion = 0;
};

class WinogradSolver {
public:
  /// Check if any Winograd kernel is applicable for this problem.
  static bool isApplicable(const WinogradConvProblem &problem);

  /// Return all applicable kernel variants, sorted by predicted performance
  /// (best first).
  static llvm::SmallVector<WinogradKernelSelection>
  findApplicable(const WinogradConvProblem &problem);

  /// Select the single best kernel for this problem.
  static std::optional<WinogradKernelSelection>
  selectBest(const WinogradConvProblem &problem);

  /// Parse a winograd perf_config string and resolve to a kernel selection.
  /// Format: "winograd:v1,<family>,<nGroups>,<channelMode>,<dataPath>"
  static std::optional<WinogradKernelSelection>
  resolveFromPerfConfig(const WinogradConvProblem &problem,
                        llvm::StringRef perfConfig);

  /// Serialize a kernel selection to a perf_config string.
  static std::string toPerfConfigStr(const WinogradKernelSelection &selection);

private:
  // Per-family applicability checks (ported from MIOpen).
  static bool isApplicableV21(const WinogradConvProblem &problem);
  static bool isApplicableV30(const WinogradConvProblem &problem);
  static bool isApplicableV40(const WinogradConvProblem &problem);
  static bool isApplicableFuryV2(const WinogradConvProblem &problem);
  static bool isApplicableFuryV4(const WinogradConvProblem &problem);
  static bool isApplicableRageV4_6(const WinogradConvProblem &problem);
  static bool isApplicableRageV4_9(const WinogradConvProblem &problem);

  // Per-family WTI computation (ported from MIOpen).
  static float computeWtiV30(const WinogradConvProblem &problem,
                             int64_t nGroups);
  static float computeWtiFury(const WinogradConvProblem &problem,
                              int64_t nGroups, WinogradChannelMode mode);
  static float computeWtiRage(const WinogradConvProblem &problem,
                              int64_t nGroups);

  // Build kernel file/name for a given family + problem.
  static WinogradKernelSelection
  buildSelection(WinogradFamily family, const WinogradConvProblem &problem,
                 int64_t nGroups, WinogradChannelMode mode);

  // Shader constraint checks (ported from MIOpen).
  static bool isShaderConstraintsMetV21(const WinogradConvProblem &problem);
  static bool isShaderConstraintsMetV30(const WinogradConvProblem &problem);
};

/// Check if arch string starts with a prefix.
bool archStartsWith(llvm::StringRef arch, llvm::StringRef prefix);

/// Extract gfx chip name from full arch string.
/// e.g. "amdgcn-amd-amdhsa:gfx942" -> "gfx942"
std::string extractChipName(llvm::StringRef arch);

} // namespace winograd
} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_WINOGRAD_WINOGRAD_SOLVER_H
