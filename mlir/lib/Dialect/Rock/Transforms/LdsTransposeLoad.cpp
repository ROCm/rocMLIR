//===- LdsTransposeLoad.cpp - MLIR helper for rock.lds_transpose_load
//
// Copyright 2025 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file defines helper functions for MLIR code generation related to
// rock.lds_transpose_load operations. It provides utilities for computing
// panel offsets, generating indices, and emitting calls to the LDS
// transpose load operation in a MFMA-friendly layout.
//
// It is intended to simplify the IR generation logic and ensure
// consistent handling of f16/bf16 panel loads from LDS memory.
//
//===----------------------------------------------------------------------===//

#include "LdsTransposeLoad.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "rock-hw-transpose-support"

using namespace mlir;
using namespace mlir::rock;

namespace mlir::rock::hwtranspose {
namespace {

bool archSupported(StringRef arch) { return arch.contains("gfx950"); }

// Describes available hardware layouts and their MFMA geometry.
struct LayoutConfig {
  LayoutKind kind;
  int64_t mnDim;
  int64_t kDim;
  StringRef name;
};

static constexpr LayoutConfig kLayoutConfigs[] = {
    {LayoutKind::L16x32, 16, 32, "16x32"},
    {LayoutKind::L32x16, 32, 16, "32x16"},
    {LayoutKind::L16x16, 16, 16, "16x16"},
    {LayoutKind::L32x8, 32, 8, "32x8"}};
} // namespace

// Validates that the block dimensions evenly divide into panels based on the
// MFMA instruction shape. Returns `true` if valid, otherwise `false`.
bool validatePaneling(const MfmaInstrShape &shape, OperandKind operandA,
                      OperandKind operandB, int64_t mPerBlock,
                      int64_t nPerBlock, int64_t kPerBlock) {

  if (kPerBlock % shape.kMfma != 0) {
    return false;
  }

  if (operandA == OperandKind::A && operandB == OperandKind::B) {
    if (mPerBlock % shape.mnMfma != 0)
      return false;
    if (nPerBlock % shape.mnMfma != 0)
      return false;
    return true;
  }
  return true;
}

LayoutKind selectLayout(int64_t mnDim, int64_t kDim) {
  for (const auto &config : kLayoutConfigs) {
    if (config.mnDim == mnDim && config.kDim == kDim) {
      return config.kind;
    }
  }
  return LayoutKind::None;
}

static DecisionLdsTransposeContext LdsTransposeDecison;

DecisionLdsTransposeContext &getDecisionLdsTransposeContext() {
  return LdsTransposeDecison;
}

// Analyzes GEMM tiling and MFMA instruction parameters to determine
// if the hardware LDS transpose optimization can be applied.
// Returns a `Decision` struct indicating applicability and layout details.
Decision makeDecision(StringRef arch, Type elemTypeA, Type elemTypeB,
                      bool DirectToLds, const MfmaInstrShape &shape,
                      OperandKind operandA, OperandKind operandB,
                      int64_t mPerBlock, int64_t nPerBlock, int64_t kPerBlock,
                      int64_t mPerWave, int64_t nPerWave,
                      bool doubleBuffering) {
  Decision dec;
  dec.operandA = operandA;
  dec.operandB = operandB;
  dec.mPerBlock = mPerBlock;
  dec.nPerBlock = nPerBlock;
  dec.kPerBlock = kPerBlock;
  dec.mPerWave = mPerWave;
  dec.nPerWave = nPerWave;
  dec.doubleBuffering = doubleBuffering;

  // Basic applicability checks
  if (!archSupported(arch) || !DirectToLds) {
    return dec;
  }

  if (elemTypeA != elemTypeB) {
    return dec;
  }
  if (!(elemTypeA.isF16() || elemTypeA.isBF16()) ||
      !(elemTypeB.isF16() || elemTypeB.isBF16())) {
    return dec;
  }

  // Check MFMA instruction shape and select a layout
  bool geomOk = ((shape.mnMfma == 16 || shape.mnMfma == 32) &&
                 (shape.kMfma == 8 || shape.kMfma == 16 || shape.kMfma == 32));
  if (!geomOk) {
    return dec;
  }

  dec.layout = selectLayout(shape.mnMfma, shape.kMfma);
  if (dec.layout == LayoutKind::None) {
    return dec;
  }

  if (!validatePaneling(shape, dec.operandA, dec.operandB, mPerBlock, nPerBlock,
                        kPerBlock)) {
    return dec;
  }

  // If all checks pass, the decision is usable
  dec.usable = true;
  return dec;
}

StringRef layoutName(LayoutKind kind) {
  for (const auto &config : kLayoutConfigs)
    if (config.kind == kind)
      return config.name;
  return "none";
}

// Attaches attributes to a `ThreadwiseReadIntoOp` to encode the chosen
// LDS transpose configuration for later lowering.
void attachAttributes(Operation *readIntoOp, const Decision &dec,
                      PatternRewriter &rewriter, bool isA) {
  if (!dec.usable)
    return;
  readIntoOp->setAttr("rock.hw_lds_transpose_enabled", rewriter.getUnitAttr());
  readIntoOp->setAttr("rock.hw_lds_transpose_layout",
                      rewriter.getStringAttr(layoutName(dec.layout)));

  if (isA) {
    readIntoOp->setAttr("rock.hw_lds_transpose_operand",
                        rewriter.getStringAttr("A"));
  } else {
    readIntoOp->setAttr("rock.hw_lds_transpose_operand",
                        rewriter.getStringAttr("B"));
  }

  readIntoOp->setAttr("rock.hw_lds_transpose_mperblock",
                      rewriter.getI64IntegerAttr(dec.mPerBlock));
  readIntoOp->setAttr("rock.hw_lds_transpose_mperwave",
                      rewriter.getI64IntegerAttr(dec.mPerWave));
  readIntoOp->setAttr("rock.hw_lds_transpose_nperblock",
                      rewriter.getI64IntegerAttr(dec.nPerBlock));
  readIntoOp->setAttr("rock.hw_lds_transpose_nperwave",
                      rewriter.getI64IntegerAttr(dec.nPerWave));
  readIntoOp->setAttr("rock.hw_lds_transpose_kperblock",
                      rewriter.getI64IntegerAttr(dec.kPerBlock));

  readIntoOp->setAttr("rock.hw_lds_transpose_double_buffering",
                      rewriter.getBoolAttr(dec.doubleBuffering));

  LLVM_DEBUG(llvm::dbgs() << "[lds_transpose] attachAttributes: enabled layout="
                          << layoutName(dec.layout) << " doubleBuffering="
                          << dec.doubleBuffering << "\n");
}

static LayoutKind layoutFromString(StringRef s) {
  for (const auto &config : kLayoutConfigs) {
    if (config.name == s) {
      return config.kind;
    }
  }
  return LayoutKind::None;
}

// Derived lowering-time configuration extracted from operation attributes.
// Used to drive emission of LDS transpose load instructions.
LoweringInfo deriveLoweringInfo(ThreadwiseReadIntoOp op, PatternRewriter &b) {
  LoweringInfo info;
  auto layoutAttr =
      op->getAttrOfType<StringAttr>("rock.hw_lds_transpose_layout");
  if (!layoutAttr)
    return info;

  info.layout = layoutFromString(layoutAttr.getValue());
  if (info.layout == LayoutKind::None)
    return info;

  // Destination buffer type
  auto dest = op.getDest();
  auto destType = cast<MemRefType>(dest.getType());
  Type elemType = destType.getElementType();
  info.elemType = elemType;

  // Read mPerBlock, nPerBlock, kPerBlock, mPerWave, nPerWave
  if (auto mPerBlockAttr =
          op->getAttrOfType<IntegerAttr>("rock.hw_lds_transpose_mperblock"))
    info.mPerBlock = mPerBlockAttr.getInt();
  if (auto nPerBlockAttr =
          op->getAttrOfType<IntegerAttr>("rock.hw_lds_transpose_nperblock"))
    info.nPerBlock = nPerBlockAttr.getInt();
  if (auto kPerBlockAttr =
          op->getAttrOfType<IntegerAttr>("rock.hw_lds_transpose_kperblock"))
    info.kPerBlock = kPerBlockAttr.getInt();
  if (auto mPerWaveAttr =
          op->getAttrOfType<IntegerAttr>("rock.hw_lds_transpose_mperwave"))
    info.mPerWave = mPerWaveAttr.getInt();
  if (auto nPerWaveAttr =
          op->getAttrOfType<IntegerAttr>("rock.hw_lds_transpose_nperwave"))
    info.nPerWave = nPerWaveAttr.getInt();

  // Read doubleBuffering flag
  if (auto doubleBufferingAttr =
          op->getAttrOfType<BoolAttr>("rock.hw_lds_transpose_double_buffering"))
    info.doubleBuffering = doubleBufferingAttr.getValue();

  // Operand-specific identification (A or B)
  if (auto operandAttr =
          op->getAttrOfType<StringAttr>("rock.hw_lds_transpose_operand")) {
    StringRef val = operandAttr.getValue();
    if (val == "A") {
      info.operand = OperandKind::A;
    } else if (val == "B") {
      info.operand = OperandKind::B;
    }
  }

  info.usable = true;
  return info;
}

// Helper to get layout dimensions consistently
static std::pair<int64_t, int64_t> getLayoutDims(LayoutKind kind) {
  for (const auto &config : kLayoutConfigs) {
    if (config.kind == kind)
      return {config.mnDim, config.kDim};
  }
  return {0, 0};
}

//===----------------------------------------------------------------------===//
// getBasePanelOffsets - Compute per-panel LDS offsets for a given lane ID
//
// Given a wavefront lane ID and a specific MFMA layout (L16x32, L16x16, etc.),
// this function computes the base byte offsets into LDS memory where each
// lane should read its operands from.
//
// These offsets are derived from AMD’s LDS tiling and MFMA operand layout
// conventions (e.g., 16x16, 16x32 panels). The goal is to map each lane’s
// register to the correct element position in LDS.
//===----------------------------------------------------------------------===//
static SmallVector<Value> getBasePanelOffsets(LayoutKind layout, Value lane,
                                              PatternRewriter &b,
                                              Location loc) {
  Value c16 = arith::ConstantIndexOp::create(b, loc, 16);
  Value c4 = arith::ConstantIndexOp::create(b, loc, 4);
  Value c2 = arith::ConstantIndexOp::create(b, loc, 2);

  Value blockId = arith::DivUIOp::create(b, loc, lane, c16);
  Value laneInBlock = arith::RemUIOp::create(b, loc, lane, c16);

  // Base offset calculations
  Value mOffsetBase = arith::MulIOp::create(
      b, loc, arith::RemUIOp::create(b, loc, laneInBlock, c4), c4);
  Value kOffsetBase = arith::DivUIOp::create(b, loc, laneInBlock, c4);

  SmallVector<Value> panelOffsets;
  switch (layout) {
  case LayoutKind::L16x32: {
    panelOffsets = {kOffsetBase, mOffsetBase};
    break;
  }
  case LayoutKind::L16x16: {
    // kbase = kOffsetBase + (blockId * 4)
    Value kBase = arith::AddIOp::create(
        b, loc, arith::MulIOp::create(b, loc, blockId, c4), kOffsetBase);
    panelOffsets = {kBase, mOffsetBase};
    break;
  }
  case LayoutKind::L32x16: {
    // mbase = mOffsetBase + (blockId % 2) * 16
    Value c16_2 = arith::ConstantIndexOp::create(b, loc, 16);
    Value mBase = arith::AddIOp::create(
        b, loc,
        arith::MulIOp::create(
            b, loc, arith::RemUIOp::create(b, loc, blockId, c2), c16_2),
        mOffsetBase);
    panelOffsets = {kOffsetBase, mBase};
    break;
  }
  case LayoutKind::L32x8: {
    // k_base_local = kOffsetBase + (blockId / 2) * 4
    Value kBase = arith::AddIOp::create(
        b, loc,
        arith::MulIOp::create(b, loc,
                              arith::DivUIOp::create(b, loc, blockId, c2), c4),
        kOffsetBase);

    // m_offset_base = mOffsetBase + (blockId % 2) * 16
    Value c16_2 = arith::ConstantIndexOp::create(b, loc, 16);
    Value mBase = arith::AddIOp::create(
        b, loc,
        arith::MulIOp::create(
            b, loc, arith::RemUIOp::create(b, loc, blockId, c2), c16_2),
        mOffsetBase);
    panelOffsets = {kBase, mBase};
    break;
  }
  default:
    llvm_unreachable("Unsupported layout in getBasePanelOffsets");
  }
  return panelOffsets;
}

LogicalResult emitThreadwiseHWTranspose(ThreadwiseReadIntoOp op,
                                        const LoweringInfo &info,
                                        PatternRewriter &b, int64_t blockSize,
                                        int64_t waveSize) {
  if (!info.usable)
    return failure();

  Location loc = op.getLoc();
  auto dest = op.getDest();
  auto destType = cast<MemRefType>(dest.getType());
  Type elemType = info.elemType;
  Value sourceView = op.getSource();
  auto [rawSrc, _, __] = untransform(b, sourceView);

  Value tid = b.createOrFold<rock::WorkitemIdOp>(loc, b.getIndexType());

  // Compute lane ID within the wavefront (0–63).
  Value waveSizeVal = arith::ConstantIndexOp::create(b, loc, waveSize);
  Value lane = arith::RemUIOp::create(b, loc, tid, waveSizeVal);
  Value waveId = arith::DivUIOp::create(b, loc, tid, waveSizeVal);
  int64_t physicalWaves = blockSize / waveSize;

  int64_t mPerWave = info.mPerWave;
  int64_t nPerWave = info.nPerWave;
  int64_t mPerBlock = info.mPerBlock;
  int64_t nPerBlock = info.nPerBlock;

  // Calculate how many wave-sized tiles fit in the block dimensions
  // These determine the wave grid, not accounting for outer loop repeats
  int64_t waveTilesInM = (mPerBlock + mPerWave - 1) / mPerWave;
  int64_t waveTilesInN = (nPerBlock + nPerWave - 1) / nPerWave;

  // Determine wave grid layout based on physical waves and wave tiles
  // This distributes waves spatially across M and N dimensions
  int64_t wavesInM = 1;
  int64_t wavesInN = 1;

  if (physicalWaves >= 4) {
    // Try to make a grid matching the wave tile layout
    for (int64_t m = 1; m <= physicalWaves; ++m) {
      if (physicalWaves % m == 0) {
        int64_t n = physicalWaves / m;
        // Prefer layouts where physical waves evenly divide wave tiles
        if (m <= waveTilesInM && n <= waveTilesInN) {
          wavesInM = m;
          wavesInN = n;
          break;
        }
      }
    }
    // Fallback: distribute waves based on which dimension has more tiles
    if (wavesInM == 1 && wavesInN == 1) {
      if (waveTilesInM <= waveTilesInN) {
        wavesInM = (physicalWaves >= waveTilesInM) ? waveTilesInM : 1;
        wavesInN = physicalWaves / wavesInM;
      } else {
        wavesInN = (physicalWaves >= waveTilesInN) ? waveTilesInN : 1;
        wavesInM = physicalWaves / wavesInN;
      }
    }
  }

  // Decompose wave_id into 2D grid position (wave_m, wave_n)
  Value wavesInNVal = arith::ConstantIndexOp::create(b, loc, wavesInN);
  Value waveM = arith::DivUIOp::create(b, loc, waveId, wavesInNVal);
  Value waveN = arith::RemUIOp::create(b, loc, waveId, wavesInNVal);

  // Use mPerBlock as stride for operand A, nPerBlock for operand B
  int64_t ldsStride =
      (info.operand == OperandKind::A) ? info.mPerBlock : info.nPerBlock;

  // Compute base LDS panel offsets according to the layout and lane mapping.
  SmallVector<Value> panelOffsets =
      getBasePanelOffsets(info.layout, lane, b, loc);

  // Determine if this is a double-rate instruction
  // Double-rate ONLY for L32x16 (32x32x16 MFMA) and L16x32 (16x16x32 MFMA)
  // L16x16 (16x16x16 MFMA) and L32x8 (32x32x8 MFMA) are SINGLE-RATE
  // instruction.
  auto [nonKDim, instrK] = getLayoutDims(info.layout);
  bool isDoubleRate =
      (info.layout == LayoutKind::L32x16 || info.layout == LayoutKind::L16x32);

  // Each ds_read_tr16_b64 call ALWAYS returns vector<4xf16>
  // For double-rate, we make 2 calls and store all 8 elements separately
  VectorType panelVecType = VectorType::get({4}, elemType);

  // panelVectors will contain:
  // - Single-rate: 1 vector<4xf16> per K tile
  // - Double-rate: 2 vector<4xf16> per K tile (low + high)
  SmallVector<Value> panelVectors;

  // Get base offsets from getBasePanelOffsets
  // For panelOffsets[0] = k_base_local, panelOffsets[1] = m_offset_base
  Value k_base_local = panelOffsets[0];
  Value m_offset_base = panelOffsets[1];

  // M/N stride: MNMfma (e.g., 32)
  // K stride per tile: KMfma (e.g., 8)
  int64_t mnStride = nonKDim;
  int64_t kTileStride = instrK;
  Value mnStrideVal = arith::ConstantIndexOp::create(b, loc, mnStride);
  Value kTileStrideVal = arith::ConstantIndexOp::create(b, loc, kTileStride);
  Value ldsStrideVal = arith::ConstantIndexOp::create(b, loc, ldsStride);

  // The extra indices tell us WHICH M/N tile we're loading in this iteration.
  // Check if there's an extra index for M/N tile selection
  ValueRange extraIndices = op.getExtraIndices();
  Value mnTileIndex = nullptr;

  // Extra indices format: [tid, m_tile_idx] for A or [tid, n_tile_idx] for B
  if (extraIndices.size() >= 2) {
    mnTileIndex = extraIndices[1]; // Second index is the M/N tile iterator
  }

  // Determine how many M/N tiles each wave covers vs total tiles needed
  // Wave grid covers: wavesInM × mPerWave or wavesInN × nPerWave
  // Outer loop should only iterate over tiles NOT covered by wave grid
  int64_t waveCoverageM = wavesInM * mPerWave;
  int64_t waveCoverageN = wavesInN * nPerWave;

  // Calculate how many MFMA tiles (mnStride-sized) the wave grid covers
  int64_t mfmaTilesCoveredByWavesM = waveCoverageM / mnStride;
  int64_t mfmaTilesCoveredByWavesN = waveCoverageN / mnStride;

  // If we have an M/N tile index from the outer loop, use it
  // Otherwise, generate all M/N tiles (fallback for single-tile case)
  // EXCEPTION: For double buffering, always generate ALL tiles at once
  int64_t startMnIdx = 0;
  int64_t endMnIdx = 1;
  bool useDynamicMnIndex = false;

  // Compute panels on-demand from layout dimensions
  int64_t mPanels = info.mPerBlock / nonKDim;
  int64_t nPanels = info.nPerBlock / nonKDim;
  int64_t kPanels = info.kPerBlock / instrK;

  if (info.doubleBuffering) {
    // Double buffering: load ALL M/N panels at once into the larger buffer
    // This bypasses the outer loop and generates all loads
    if (info.operand == OperandKind::A) {
      endMnIdx = mPanels;
    } else if (info.operand == OperandKind::B) {
      endMnIdx = nPanels;
    }
    useDynamicMnIndex = false;
  } else if (mnTileIndex) {
    // Outer loop handles M/N iteration, we load only ONE M/N tile per call
    useDynamicMnIndex = true;
    endMnIdx = 1;
  } else {
    // No outer loop - check how many tiles we need to generate statically
    // Only generate tiles that wave grid does NOT cover
    if (info.operand == OperandKind::A) {
      int64_t totalMfmaTilesM = mPanels;
      endMnIdx =
          std::max<int64_t>(1, totalMfmaTilesM - mfmaTilesCoveredByWavesM);
    } else if (info.operand == OperandKind::B) {
      int64_t totalMfmaTilesN = nPanels;
      endMnIdx =
          std::max<int64_t>(1, totalMfmaTilesN - mfmaTilesCoveredByWavesN);
    }
  }

  // For double-rate layouts ONLY (L32x16, L16x32), compute k_offset_base
  // L32x16 (32x32x16): k_offset_base = (block_id / 2) * 8
  // L16x32 (16x16x32): k_offset_base = block_id * 8
  Value blockId = nullptr;
  Value kOffsetBase = nullptr;

  if (isDoubleRate) {
    Value c16 = arith::ConstantIndexOp::create(b, loc, 16);
    Value c2 = arith::ConstantIndexOp::create(b, loc, 2);
    Value c8 = arith::ConstantIndexOp::create(b, loc, 8);
    blockId = arith::DivUIOp::create(b, loc, lane, c16);

    if (info.layout == LayoutKind::L32x16) {
      // k_offset_base = (block_id / 2) * 8
      kOffsetBase = arith::MulIOp::create(
          b, loc, arith::DivUIOp::create(b, loc, blockId, c2), c8);
    } else if (info.layout == LayoutKind::L16x32) {
      // k_offset_base = block_id * 8
      kOffsetBase = arith::MulIOp::create(b, loc, blockId, c8);
    }
  }

  // Generate loads: If outer loop exists, load one M/N tile with all K tiles
  //                 Otherwise, load all M/N tiles with all K tiles
  for (int64_t mnIdxLocal = startMnIdx; mnIdxLocal < endMnIdx; ++mnIdxLocal) {
    for (int64_t kIdx = 0; kIdx < kPanels; ++kIdx) {
      // Calculate m_base for this M/N tile
      Value m_base = m_offset_base;

      // Add mfma offset to distribute work across waves spatially
      // For operand A: m_base += wave_m * mmfma
      // For operand B: n_base (m_base) += wave_n * nmfma
      if (info.operand == OperandKind::A) {
        Value waveOffsetM = arith::MulIOp::create(b, loc, waveM, mnStrideVal);
        m_base = arith::AddIOp::create(b, loc, m_base, waveOffsetM);
      } else if (info.operand == OperandKind::B) {
        Value waveOffsetN = arith::MulIOp::create(b, loc, waveN, mnStrideVal);
        m_base = arith::AddIOp::create(b, loc, m_base, waveOffsetN);
      }

      // Add iteration offset from outer loop (if exists) or static tile index
      if (useDynamicMnIndex) {
        // Dynamic index from outer loop: m_base += mnTileIndex * mnStride
        Value mnOffsetAdd =
            arith::MulIOp::create(b, loc, mnTileIndex, mnStrideVal);
        m_base = arith::AddIOp::create(b, loc, m_base, mnOffsetAdd);
      } else if (mnIdxLocal > 0) {
        // Static index: m_base += mnIdxLocal * mnStride
        Value mnIdxLocalVal =
            arith::ConstantIndexOp::create(b, loc, mnIdxLocal);
        Value mnOffsetAdd =
            arith::MulIOp::create(b, loc, mnStrideVal, mnIdxLocalVal);
        m_base = arith::AddIOp::create(b, loc, m_base, mnOffsetAdd);
      }

      if (!isDoubleRate) {
        // SINGLE-RATE (L32x8, L16x16): One load per K tile
        // k_base = k_base_local + kIdx * kTileStride
        Value k_base = k_base_local;
        if (kIdx > 0) {
          Value kIdxVal = arith::ConstantIndexOp::create(b, loc, kIdx);
          Value kOffsetAdd =
              arith::MulIOp::create(b, loc, kTileStrideVal, kIdxVal);
          k_base = arith::AddIOp::create(b, loc, k_base, kOffsetAdd);
        }

        // final_offset = k_base * ldsStride + m_base
        Value final_offset = arith::AddIOp::create(
            b, loc, m_base,
            arith::MulIOp::create(b, loc, k_base, ldsStrideVal));

        // Perform LDS transpose load (ds_read_tr16_b64) -> returns
        // vector<4xf16>
        auto l = rock::LDSTransposeLoadOp::create(b, loc, panelVecType, rawSrc,
                                                  ValueRange{final_offset});
        panelVectors.push_back(l.getFragment());

      } else {
        // DOUBLE-RATE (L32x16, L16x32): TWO loads per K tile
        // Each load returns vector<4xf16>, total 8 elements per K tile
        // k_offset_low = k_offset_base + k_tile * KMfma
        // k_offset_high = k_offset_base + 4 + k_tile * KMfma

        Value kIdxVal = arith::ConstantIndexOp::create(b, loc, kIdx);
        Value kTileOffset =
            arith::MulIOp::create(b, loc, kTileStrideVal, kIdxVal);
        Value k_offset_low =
            arith::AddIOp::create(b, loc, kOffsetBase, kTileOffset);
        Value c4 = arith::ConstantIndexOp::create(b, loc, 4);
        Value k_offset_high = arith::AddIOp::create(b, loc, k_offset_low, c4);

        Value k_base_low =
            arith::AddIOp::create(b, loc, k_base_local, k_offset_low);
        Value k_base_high =
            arith::AddIOp::create(b, loc, k_base_local, k_offset_high);

        // offset_low = k_base_low * ldsStride + m_base
        Value offset_low = arith::AddIOp::create(
            b, loc, m_base,
            arith::MulIOp::create(b, loc, k_base_low, ldsStrideVal));

        // offset_high = k_base_high * ldsStride + m_base
        Value offset_high = arith::AddIOp::create(
            b, loc, m_base,
            arith::MulIOp::create(b, loc, k_base_high, ldsStrideVal));

        // Load low half: returns vector<4xf16>
        auto load_low = rock::LDSTransposeLoadOp::create(
            b, loc, panelVecType, rawSrc, ValueRange{offset_low});
        panelVectors.push_back(load_low.getFragment());

        // Load high half: returns vector<4xf16>
        auto load_high = rock::LDSTransposeLoadOp::create(
            b, loc, panelVecType, rawSrc, ValueRange{offset_high});
        panelVectors.push_back(load_high.getFragment());
      }
    }
  }

  // Calculate expected number of loads
  // - For double buffering: we generate ALL M/N panels → endMnIdx panels ×
  // kPanels × (1 or 2 for rate)
  // - Single-rate: 1 load per K tile → actualMnTiles × kPanels loads
  // - Double-rate: 2 loads per K tile → actualMnTiles × kPanels × 2 loads
  int64_t actualMnTiles = endMnIdx - startMnIdx;
  int64_t loadsPerKTile = isDoubleRate ? 2 : 1;
  int64_t expectedLoads = actualMnTiles * kPanels * loadsPerKTile;

  // Each load ALWAYS produces 4 elements (ds_read_tr16_b64 → vector<4xf16>)
  int64_t sliceElems = expectedLoads * 4;

  // Verify we generated the expected number of loads
  if (panelVectors.size() != (size_t)expectedLoads) {
    return op.emitOpError("Mismatch in number of generated loads: expected ")
           << expectedLoads << ", got " << panelVectors.size();
  }

  // Scalar buffer path rank-1.
  int64_t destCap = destType.getShape()[0];
  int64_t targetElems = std::min<int64_t>(sliceElems, destCap);
  int64_t produced = 0;

  // Write each extracted element from the loaded panel vectors into `dest`.
  // The destination is rank-1, meaning scalar sequential layout.
  for (Value pv : panelVectors) {
    for (int lane = 0; lane < 4 && produced < targetElems; ++lane) {
      Value ciLane = arith::ConstantIndexOp::create(b, loc, lane);
      Value elem = vector::ExtractOp::create(b, loc, pv, ciLane);
      Value idx = arith::ConstantIndexOp::create(b, loc, produced++);
      InBoundsStoreOp::create(b, loc, elem, dest, ValueRange{idx});
    }
    if (produced >= targetElems)
      break; // Stop once we have written all target elements
  }

  b.replaceOp(op, ValueRange{});
  return success();
}

} // namespace mlir::rock::hwtranspose
