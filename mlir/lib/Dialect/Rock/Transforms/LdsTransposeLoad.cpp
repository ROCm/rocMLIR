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
bool validatePaneling(const MfmaInstrShape &shape, OperandKind operand,
                      int64_t mPerBlock, int64_t nPerBlock, int64_t kPerBlock) {

  if (kPerBlock % shape.kMfma != 0) {
    return false;
  }

  int64_t dPerBlock = operand == OperandKind::A ? mPerBlock : nPerBlock;
  if (dPerBlock % shape.mnMfma != 0)
    return false;
  return true;
}

// Analyzes GEMM tiling and MFMA instruction parameters to determine
// if the hardware LDS transpose optimization can be applied.
// Returns a `Decision` struct indicating applicability and layout details.
Decision makeDecision(StringRef arch, Type elemTypeA, Type elemTypeB,
                      bool DirectToLds, const MfmaInstrShape &shape,
                      OperandKind operand, int64_t mPerBlock, int64_t nPerBlock,
                      int64_t kPerBlock, int64_t mPerWave, int64_t nPerWave,
                      bool doubleBuffering) {
  Decision dec;
  dec.operand = operand;
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

  if (elemTypeA != elemTypeB || !(elemTypeA.isF16() || elemTypeA.isBF16()))
    return dec;

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

  if (!validatePaneling(shape, operand, mPerBlock, nPerBlock, kPerBlock)) {
    return dec;
  }

  // If all checks pass, the decision is usable
  dec.usable = true;
  return dec;
}

LayoutKind selectLayout(int64_t mnDim, int64_t kDim) {
  for (const auto &config : kLayoutConfigs) {
    if (config.mnDim == mnDim && config.kDim == kDim) {
      return config.kind;
    }
  }
  return LayoutKind::None;
}

StringRef layoutName(LayoutKind kind) {
  for (const auto &config : kLayoutConfigs) {
    if (config.kind == kind)
      return config.name;
  }
  return "none";
}

static LayoutKind layoutFromString(StringRef s) {
  for (const auto &config : kLayoutConfigs) {
    if (config.name == s) {
      return config.kind;
    }
  }
  return LayoutKind::None;
}

// Helper to get layout dimensions consistently
static std::pair<int64_t, int64_t> getLayoutDims(LayoutKind kind) {
  for (const auto &config : kLayoutConfigs) {
    if (config.kind == kind)
      return {config.mnDim, config.kDim};
  }
  return {0, 0};
}

// Attaches attributes to a `ThreadwiseReadIntoOp` to encode the chosen
// LDS transpose configuration for later lowering.
DictionaryAttr buildTransposeAttr(const Decision &dec, bool isOperandA,
                                  PatternRewriter &rewriter) {
  if (!dec.usable)
    return nullptr;

  NamedAttrList attrs;
  attrs.append("rock.lds_transpose_enabled", rewriter.getUnitAttr());
  attrs.append("rock.mfma_layout",
               rewriter.getStringAttr(layoutName(dec.layout)));
  attrs.append("rock.operand", rewriter.getStringAttr(isOperandA ? "A" : "B"));
  attrs.append("rock.mperblock", rewriter.getI64IntegerAttr(dec.mPerBlock));
  attrs.append("rock.nperblock", rewriter.getI64IntegerAttr(dec.nPerBlock));
  attrs.append("rock.kperblock", rewriter.getI64IntegerAttr(dec.kPerBlock));
  attrs.append("rock.mperwave", rewriter.getI64IntegerAttr(dec.mPerWave));
  attrs.append("rock.nperwave", rewriter.getI64IntegerAttr(dec.nPerWave));
  attrs.append("rock.double_buffering",
               rewriter.getBoolAttr(dec.doubleBuffering));
  return rewriter.getDictionaryAttr(attrs);
}

// Derived lowering-time configuration extracted from operation attributes.
// Used to drive emission of LDS transpose load instructions.
LoweringInfo deriveLoweringInfo(ThreadwiseReadIntoOp op, PatternRewriter &b) {
  LoweringInfo info;
  auto layoutAttr = op->getAttrOfType<StringAttr>("rock.mfma_layout");
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
  if (auto mPerBlockAttr = op->getAttrOfType<IntegerAttr>("rock.mperblock"))
    info.mPerBlock = mPerBlockAttr.getInt();
  if (auto nPerBlockAttr = op->getAttrOfType<IntegerAttr>("rock.nperblock"))
    info.nPerBlock = nPerBlockAttr.getInt();
  if (auto kPerBlockAttr = op->getAttrOfType<IntegerAttr>("rock.kperblock"))
    info.kPerBlock = kPerBlockAttr.getInt();
  if (auto mPerWaveAttr = op->getAttrOfType<IntegerAttr>("rock.mperwave"))
    info.mPerWave = mPerWaveAttr.getInt();
  if (auto nPerWaveAttr = op->getAttrOfType<IntegerAttr>("rock.nperwave"))
    info.nPerWave = nPerWaveAttr.getInt();

  // Read doubleBuffering flag
  if (auto doubleBufferingAttr =
          op->getAttrOfType<BoolAttr>("rock.double_buffering"))
    info.doubleBuffering = doubleBufferingAttr.getValue();

  // Operand-specific identification (A or B)
  if (auto operandAttr = op->getAttrOfType<StringAttr>("rock.operand")) {
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

//===----------------------------------------------------------------------===//
// MNTileBounds - Result structure for M/N tile iteration bounds
//
// Encapsulates the iteration bounds for M/N tile generation loops.
//===----------------------------------------------------------------------===//
struct MNTileBounds {
  int64_t startIdx;     // Start index
  int64_t endIdx;       // End index
  bool useDynamicIndex; // Whether to use runtime tile index from outer loop
};

//===----------------------------------------------------------------------===//
// computeMNTileIterationBounds - Compute M/N tile iteration bounds
//
// Determines how many M/N tiles to generate in the main loop based on two
// scenarios:
//
// SCENARIO 1: Double Buffering
//   - Load ALL M/N panels at once into a larger buffer
//   - endIdx = total number of panels (mPanels or nPanels)
//   - useDynamicIndex = false (static generation)
//
// SCENARIO 2: Single Buffering with Outer Loop (default)
//   - Outer loop in BlockwiseGemmToThreadwise always exists
//   - Load ONE tile per call
//   - endIdx = 1
//   - useDynamicIndex = true (use mnTileIndex from outer loop)
//
// Parameters:
//   doubleBuffering - Whether double buffering is enabled
//   operand        - Operand kind (A or B)
//   mPanels        - Number of M panels (mPerBlock / MNMfma)
//   nPanels        - Number of N panels (nPerBlock / MNMfma)
//
// Returns:
//   MNTileBounds with startIdx, endIdx, and useDynamicIndex
//===----------------------------------------------------------------------===//
static MNTileBounds computeMNTileIterationBounds(bool doubleBuffering,
                                                 OperandKind operand,
                                                 int64_t mPanels,
                                                 int64_t nPanels) {
  MNTileBounds bounds;
  bounds.startIdx = 0;

  if (doubleBuffering) {
    bounds.endIdx = (operand == OperandKind::A) ? mPanels : nPanels;
    bounds.useDynamicIndex = false;

    LLVM_DEBUG(llvm::dbgs() << "[lds_transpose] Double buffering: "
                            << "loading ALL " << bounds.endIdx << " panels\n");
  } else {
    bounds.endIdx = 1;
    bounds.useDynamicIndex = true;

    LLVM_DEBUG(llvm::dbgs() << "[lds_transpose] Single buffering: "
                            << "loading 1 panel per outer loop iteration\n");
  }

  return bounds;
}

//===----------------------------------------------------------------------===//
// getDoubleRateKOffsetBase - Compute K offset base for double-rate layouts
//
// For double-rate layouts (L32x16, L16x32), each K tile is split into two
// loads (low and high halves). The K offset base is computed from the thread's
// lane ID to determine which K "block" the thread belongs to.
//
// Formula:
//   - L32x16 (32x32x16): k_offset_base = ((lane / 16) / 2) * 8
//   - L16x32 (16x16x32): k_offset_base = (lane / 16) * 8
//
// The lane / 16 gives the "block_id" (which 16-lane group the thread is in).
// For L32x16, we further divide by 2 and multiply by 8 (half of KMfma=16).
// For L16x32, we directly multiply by 8 (half of KMfma=16).
//
// This offset is used as the starting point for computing low/high half offsets
// in double-rate loads.
//
// Parameters:
//   isDoubleRate - Whether the layout is double-rate (L32x16, L16x32)
//   layout       - Specific layout kind (L32x16 or L16x32)
//   lane         - Thread's lane ID (workitem ID within the block)
//
// Returns:
//   K offset base value for double-rate layouts, or nullptr for single-rate
//===----------------------------------------------------------------------===//
static Value getDoubleRateKOffsetBase(PatternRewriter &b, Location loc,
                                      bool isDoubleRate, LayoutKind layout,
                                      Value lane) {
  if (!isDoubleRate)
    return nullptr;

  Value c16 = arith::ConstantIndexOp::create(b, loc, 16);
  Value c2 = arith::ConstantIndexOp::create(b, loc, 2);
  Value c8 = arith::ConstantIndexOp::create(b, loc, 8);
  Value blockId = arith::DivUIOp::create(b, loc, lane, c16);

  Value kOffsetBase;
  if (layout == LayoutKind::L32x16) {
    kOffsetBase = arith::MulIOp::create(
        b, loc, arith::DivUIOp::create(b, loc, blockId, c2), c8);
  } else if (layout == LayoutKind::L16x32) {
    kOffsetBase = arith::MulIOp::create(b, loc, blockId, c8);
  } else {
    return nullptr; // Should not happen for double-rate
  }

  return kOffsetBase;
}

//===----------------------------------------------------------------------===//
// computePanelFinalOffset - Compute final K offset for a specific K tile
//
// This function centralizes the K offset computation logic for both single-rate
// and double-rate layouts. It handles the tile-based offset calculation and
// optional low/high half splitting for double-rate layouts.
//
// Formula:
//   Single-rate: k_final = k_base_local + (kTileIdx * kTileStride)
//   Double-rate: k_final = k_base_local + kOffsetBase + (kTileIdx *
//   kTileStride) + halfOffset
//     where halfOffset = 0 for low half, 4 for high half
//
// Parameters:
//   isDoubleRate  - Whether this is a double-rate layout (L32x16, L16x32)
//   kBaseLocal    - Local K base offset from computeLDSBaseOffsets()
//   kOffsetBase   - Double-rate K offset base (from getDoubleRateKOffsetBase)
//   kTileIdx      - Current K tile index (0, 1, 2, ...)
//   kTileStride   - K stride per tile (instrK, e.g., 8 or 16)
//   isHighHalf    - For double-rate: true = high half (+4), false = low half
//
// Returns:
//   Final K offset value to use for emitPanelLoad()
//===----------------------------------------------------------------------===//
static Value computePanelFinalOffset(PatternRewriter &b, Location loc,
                                     bool isDoubleRate, Value kBaseLocal,
                                     Value kOffsetBase, int64_t kTileIdx,
                                     Value kTileStride,
                                     bool isHighHalf = false) {
  Value kBase = kBaseLocal;

  if (isDoubleRate) {
    // Double-rate: k_offset = kOffsetBase + kTileIdx * kTileStride [+ 4 for
    // high]
    Value kTileOffset;
    if (kTileIdx > 0) {
      Value kIdxVal = arith::ConstantIndexOp::create(b, loc, kTileIdx);
      kTileOffset = arith::MulIOp::create(b, loc, kTileStride, kIdxVal);
    } else {
      kTileOffset = arith::ConstantIndexOp::create(b, loc, 0);
    }

    Value k_offset = arith::AddIOp::create(b, loc, kOffsetBase, kTileOffset);

    // For high half, add 4
    if (isHighHalf) {
      Value c4 = arith::ConstantIndexOp::create(b, loc, 4);
      k_offset = arith::AddIOp::create(b, loc, k_offset, c4);
    }

    // k_base = k_base_local + k_offset
    kBase = arith::AddIOp::create(b, loc, kBaseLocal, k_offset);

  } else {
    // Single-rate: k_base = k_base_local + kTileIdx * kTileStride
    if (kTileIdx > 0) {
      Value kIdxVal = arith::ConstantIndexOp::create(b, loc, kTileIdx);
      Value kOffsetAdd = arith::MulIOp::create(b, loc, kTileStride, kIdxVal);
      kBase = arith::AddIOp::create(b, loc, kBase, kOffsetAdd);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "[lds_transpose] Computed panel K offset for tile "
                          << kTileIdx << (isHighHalf ? " (high)" : " (low)")
                          << "\n");

  return kBase;
}

//===----------------------------------------------------------------------===//
// emitPanelLoad - Emit an LDS transpose load operation
//
// Computes the final LDS offset and emits a hardware LDS transpose load
// instruction (ds_read_tr16_b64). This instruction always returns vector<4xf16>
// regardless of the layout.
//
// The final offset is computed as: final_offset = k_base * ldsStride + m_base
// where ldsStride depends on the operand (mPerBlock for A, nPerBlock for B).
//
// This function is called once per K tile for single-rate layouts (L32x8,
// L16x16) and twice per K tile for double-rate layouts (L32x16, L16x32).
//
// Parameters:
//   rawSrc       - Source LDS buffer (raw, untransformed memref)
//   kBase        - K dimension base offset for this panel
//   mBase        - M/N dimension base offset for this panel
//   ldsStride    - Stride between K rows in LDS (mPerBlock or nPerBlock)
//   panelVecType - Result type (always vector<4xf16> or vector<4xbf16>)
//
// Returns:
//   The loaded panel vector (vector<4xf16/bf16>)
//===----------------------------------------------------------------------===//
static Value emitPanelLoad(PatternRewriter &b, Location loc, Value rawSrc,
                           Value kBase, Value mBase, Value ldsStride,
                           VectorType panelVecType) {
  // Compute final offset: k_base * ldsStride + m_base
  Value kOffset = arith::MulIOp::create(b, loc, kBase, ldsStride);
  Value finalOffset = arith::AddIOp::create(b, loc, mBase, kOffset);

  // Emit hardware LDS transpose load: ds_read_tr16_b64
  auto loadOp = rock::LDSTransposeLoadOp::create(b, loc, panelVecType, rawSrc,
                                                 ValueRange{finalOffset});

  return loadOp.getFragment();
}

//===----------------------------------------------------------------------===//
// writePanelVectorsToDestination - Write loaded panel vectors to destination
//
// Extracts individual f16/bf16 elements from loaded panel vectors and writes
// them sequentially to the destination buffer. Each panel vector contains 4
// elements (ds_read_tr16_b64 always returns vector<4xf16>).
//
// Parameters:
//   panelVectors - Array of loaded panel vectors (each is vector<4xf16>)
//   dest         - Destination memref (rank-1, scalar layout)
//   targetElems  - Maximum number of elements to write
//
// Returns:
//   success() if all target elements were written
//   failure() if destination capacity was insufficient
//===----------------------------------------------------------------------===//
static LogicalResult
writePanelVectorsToDestination(PatternRewriter &b, Location loc,
                               ArrayRef<Value> panelVectors, Value dest,
                               int64_t targetElems) {

  int64_t produced = 0;

  LLVM_DEBUG(llvm::dbgs() << "[lds_transpose] Writing " << panelVectors.size()
                          << " panel vectors (" << (panelVectors.size() * 4)
                          << " elements) to destination, target=" << targetElems
                          << "\n");

  // Each panel vector contains 4 elements (ds_read_tr16_b64 → vector<4xf16>)
  constexpr int64_t elementsPerVector = 4;

  for (Value panelVec : panelVectors) {
    for (int64_t laneIdx = 0; laneIdx < elementsPerVector; ++laneIdx) {
      if (produced >= targetElems) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[lds_transpose] Reached target element count: "
                   << produced << "\n");
        return success();
      }

      // Extract element at lane index
      Value laneIdxVal = arith::ConstantIndexOp::create(b, loc, laneIdx);
      Value elem = vector::ExtractOp::create(b, loc, panelVec, laneIdxVal);

      // Store to destination at sequential index
      Value storeIdx = arith::ConstantIndexOp::create(b, loc, produced);
      InBoundsStoreOp::create(b, loc, elem, dest, ValueRange{storeIdx});

      ++produced;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "[lds_transpose] Wrote " << produced
                          << " elements to destination\n");

  // Check if we wrote all target elements
  if (produced < targetElems) {
    return failure(); // Insufficient panel vectors to fill destination
  }

  return success();
}

//===----------------------------------------------------------------------===//
// getBasePanelOffsets - Compute per-panel LDS offsets for a given lane ID
//
// Given a wavefront lane ID and a specific MFMA layout (L16x32, L16x16, etc.),
// this function computes the base byte offsets into LDS memory where each
// lane should read its operands from.
//
// These offsets are derived from AMD's LDS tiling and MFMA operand layout
// conventions (e.g., 16x16, 16x32 panels). The goal is to map each lane's
// register to the correct element position in LDS.
//
// Note: This is an internal helper function. Use computeLDSBaseOffsets()
// instead for better readability.
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
    Value mBase = arith::AddIOp::create(
        b, loc,
        arith::MulIOp::create(b, loc,
                              arith::RemUIOp::create(b, loc, blockId, c2), c16),
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
    Value mBase = arith::AddIOp::create(
        b, loc,
        arith::MulIOp::create(b, loc,
                              arith::RemUIOp::create(b, loc, blockId, c2), c16),
        mOffsetBase);
    panelOffsets = {kBase, mBase};
    break;
  }
  default:
    llvm_unreachable("Unsupported layout in getBasePanelOffsets");
  }
  return panelOffsets;
}

//===----------------------------------------------------------------------===//
// computeLDSBaseOffsets - Compute LDS base offsets for K and M/N dimensions
//
// This is the main entry point for computing LDS base offsets. It calls the
// internal getBasePanelOffsets() function and returns the results in a
// structured format.
//
// The returned offsets represent the starting position in LDS where a specific
// thread should begin reading data. These are the "local" offsets within a
// panel, before adding wave-level and tile-level offsets.
//
// Usage:
//   auto [k_base_local, m_offset_base] = computeLDSBaseOffsets(...);
//
// Parameters:
//   layout - MFMA layout kind (L16x32, L16x16, L32x16, L32x8)
//   lane   - Thread's lane ID within the workgroup
//
// Returns:
//   std::pair<Value, Value>:
//     - first:  K dimension base offset (k_base_local)
//     - second: M/N dimension base offset (m_offset_base)
//===----------------------------------------------------------------------===//
static std::pair<Value, Value> computeLDSBaseOffsets(LayoutKind layout,
                                                     Value lane,
                                                     PatternRewriter &b,
                                                     Location loc) {
  SmallVector<Value> offsets = getBasePanelOffsets(layout, lane, b, loc);

  LLVM_DEBUG(llvm::dbgs() << "[lds_transpose] Computed LDS base offsets: "
                          << "k_base_local=offsets[0], "
                          << "m_offset_base=offsets[1]\n");

  return {offsets[0], offsets[1]};
}

//===----------------------------------------------------------------------===//
// WaveGridLayout
//===----------------------------------------------------------------------===//
// Structure to hold the computed wave grid layout and wave ID decomposition.
//
// Fields:
//   wavesInM - Number of waves distributed along M dimension
//   wavesInN - Number of waves distributed along N dimension
//   waveM    - This thread's wave position in M dimension (runtime value)
//   waveN    - This thread's wave position in N dimension (runtime value)
//===----------------------------------------------------------------------===//
struct WaveGridLayout {
  int64_t wavesInM;
  int64_t wavesInN;
  Value waveM;
  Value waveN;
};

//===----------------------------------------------------------------------===//
// StrideConfig - Result structure for stride computation
//
// Encapsulates the computed stride values for wave offset and tile offset.
//
// Fields:
//   waveOffsetStride - Stride for spatial wave separation (always nonKDim)
//   tileOffsetStride - Stride for outer loop tile iteration
//===----------------------------------------------------------------------===//
struct StrideConfig {
  int64_t waveOffsetStride; // Stride for spatial wave separation
  int64_t tileOffsetStride; // Stride for outer loop tile iteration
};

//===----------------------------------------------------------------------===//
// computeWaveGridLayout()
//===----------------------------------------------------------------------===//
//
// Computes how physical waves are spatially distributed across the M and N
// dimensions, and decomposes the wave ID into a 2D grid position.
//
// This version uses a deterministic layout selection based solely on the number
// of physical waves (1, 2, 3, or 4). The goal is to match the wave grid to the
// number of available wave tiles (waveTilesInM, waveTilesInN) while choosing a
// stable and predictable layout.
//
// Key principles:
//  - physicalWaves ∈ {1, 2, 3, 4} (corresponding to 64–256 threads)
//  - Prefer balanced or natural layouts when possible:
//        1 wave  → 1×1
//        2 waves → prefer 1×2
//        3 waves → prefer 1×3
//        4 waves → prefer 2×2
//  - If a preferred layout does not fit the available tiles, fallback logic
//    selects the best possible layout while maintaining determinism.
//  - The result defines which spatial tile each wave is responsible for,
//    which is essential when performing LDS transpose loads.
//
// Usage:
//   WaveGridLayout grid = computeWaveGridLayout(...);
//   // grid.wavesInM, grid.wavesInN are compile-time constants
//   // grid.waveM,    grid.waveN    are runtime indices
//
// Parameters:
//   waveId        - Runtime wave ID inside the workgroup.
//   physicalWaves - Total number of waves (compile-time).
//
// Returns:
//   WaveGridLayout containing:
//     - wavesInM, wavesInN: Grid dimensions.
//     - waveM, waveN:      This wave's assigned 2D grid coordinates.
//===----------------------------------------------------------------------===//
static WaveGridLayout computeWaveGridLayout(Value waveId, int64_t physicalWaves,
                                            int64_t mPerWave, int64_t nPerWave,
                                            int64_t mPerBlock,
                                            int64_t nPerBlock,
                                            PatternRewriter &b, Location loc) {
  // Calculate how many wave-sized tiles fit in the block dimensions
  // These determine the wave grid, not accounting for outer loop repeats
  int64_t waveTilesInM = mPerBlock / mPerWave;
  int64_t waveTilesInN = nPerBlock / nPerWave;

  // Determine wave grid layout based on physical waves and wave tiles
  // This distributes waves spatially across M and N dimensions
  // Note: physicalWaves can only be 1, 2, 3, or 4 (for 64, 128, 192, 256
  // threads)
  int64_t wavesInM = 1;
  int64_t wavesInN = 1;

  switch (physicalWaves) {
  case 1:
    // Single wave: always 1×1
    wavesInM = 1;
    wavesInN = 1;
    break;

  case 2:
    // Two waves: prefer 1×2, fallback to 2×1 if needed
    if (waveTilesInN >= 2) {
      wavesInM = 1;
      wavesInN = 2;
    } else if (waveTilesInM >= 2) {
      wavesInM = 2;
      wavesInN = 1;
    } else {
      // Rare: both tiles < 2, use 1×2 (outer loop handles overflow)
      wavesInM = 1;
      wavesInN = 2;
    }
    break;

  case 3:
    // Three waves: prefer 1×3, fallback to 3×1 or dimension-based
    if (waveTilesInN >= 3) {
      wavesInM = 1;
      wavesInN = 3;
    } else if (waveTilesInM >= 3) {
      wavesInM = 3;
      wavesInN = 1;
    } else {
      // Fallback: choose dimension with more tiles (outer loop handles rest)
      wavesInM = (waveTilesInN >= waveTilesInM) ? 1 : 3;
      wavesInN = (waveTilesInN >= waveTilesInM) ? 3 : 1;
    }
    break;

  case 4:
    // Four waves: prefer 2×2 (balanced), then 1×4, 4×1, or fallback
    if (waveTilesInM >= 2 && waveTilesInN >= 2) {
      wavesInM = 2;
      wavesInN = 2;
    } else if (waveTilesInN >= 4) {
      wavesInM = 1;
      wavesInN = 4;
    } else if (waveTilesInM >= 4) {
      wavesInM = 4;
      wavesInN = 1;
    } else {
      // Fallback: prefer 2×2 if at least one dimension >= 2
      if (waveTilesInN >= 2 || waveTilesInM >= 2) {
        wavesInM = 2;
        wavesInN = 2;
      } else {
        // Edge case: very small tiles, default to 1×4 (outer loop iterates)
        wavesInM = 1;
        wavesInN = 4;
      }
    }
    break;
  default:
    llvm_unreachable("Invalid physicalWaves: blockSize / waveSize");
  }

  LLVM_DEBUG(llvm::dbgs() << "[lds_transpose] Wave grid layout: " << wavesInM
                          << "×" << wavesInN << " (M×N), covering "
                          << waveTilesInM << " M-tiles and " << waveTilesInN
                          << " N-tiles\n");

  // Decompose wave_id into 2D grid position (wave_m, wave_n)
  Value wavesInNVal = arith::ConstantIndexOp::create(b, loc, wavesInN);
  Value waveM = arith::DivUIOp::create(b, loc, waveId, wavesInNVal);
  Value waveN = arith::RemUIOp::create(b, loc, waveId, wavesInNVal);

  return {wavesInM, wavesInN, waveM, waveN};
}

//===----------------------------------------------------------------------===//
// computeStrideConfiguration()
//===----------------------------------------------------------------------===//
// Computes stride values for LDS transpose load offset calculations.
//
// Two types of strides are computed:
// 1. waveOffsetStride - Spatial separation between different waves in the grid
// 2. tileOffsetStride - Step size for outer loop iterations through tiles
//
// KEY INSIGHT:
// - waveOffsetStride is ALWAYS nonKDim (one MFMA tile)
// - tileOffsetStride depends on wave grid layout:
//   * Single wave: nonKDim (sequential tiles)
//   * Multiple waves: wavesInDim × nonKDim (interleaved tiles)
//
// EXAMPLE (2×2 grid, nonKDim=16):
//   Wave 0 iterates: tiles [0, 2, 4, 6] → step = 2×16 = 32
//   Wave 1 iterates: tiles [1, 3, 5, 7] → step = 2×16 = 32
//
// Parameters:
//   operand   - Whether this is operand A (M dimension) or B (N dimension)
//   waveGrid  - Wave grid layout containing wavesInM and wavesInN
//   nonKDim   - Size of MFMA tile in M/N dimension (e.g., 16 or 32)
//
// Returns:
//   StrideConfig containing waveOffsetStride and tileOffsetStride
//===----------------------------------------------------------------------===//
static StrideConfig computeStrideConfiguration(OperandKind operand,
                                               const WaveGridLayout &waveGrid,
                                               int64_t nonKDim) {
  StrideConfig config;

  // Wave offset stride: ALWAYS nonKDim for spatial wave separation
  // This ensures each wave accesses a different spatial region (one MFMA tile
  // apart)
  config.waveOffsetStride = nonKDim;

  // Tile offset stride: step size for outer loop (accounts for wave
  // interleaving)
  if (operand == OperandKind::A) {
    // Operand A (M dimension)
    if (waveGrid.wavesInM >= 2) {
      // Multiple waves in M dimension → tiles are interleaved
      // Each wave skips wavesInM tiles to reach its next tile
      config.tileOffsetStride = waveGrid.wavesInM * nonKDim;
    } else {
      // Single wave in M dimension → tiles are sequential
      config.tileOffsetStride = nonKDim;
    }
  } else {
    // Operand B (N dimension)
    if (waveGrid.wavesInN >= 2 && waveGrid.wavesInN == waveGrid.wavesInM) {
      // BALANCED GRID (2×2, 3×3, 4×4) → tiles are interleaved in N dimension
      // Special case: balanced grids require interleaved tile access
      config.tileOffsetStride = waveGrid.wavesInN * nonKDim;
    } else {
      // UNBALANCED GRID (1×N, N×1) or single wave → tiles are sequential
      config.tileOffsetStride = nonKDim;
    }
  }

  LLVM_DEBUG(
      llvm::dbgs() << "[lds_transpose] Computed stride configuration for "
                   << (operand == OperandKind::A ? "A" : "B") << ":\n"
                   << "  waveOffsetStride = " << config.waveOffsetStride << "\n"
                   << "  tileOffsetStride = " << config.tileOffsetStride
                   << "\n");

  return config;
}

//===----------------------------------------------------------------------===//
// computeWaveOffset()
//===----------------------------------------------------------------------===//
// Computes the spatial offset for wave-level work distribution.
//
// Each wave in the wave grid handles a different spatial region of the M or N
// dimension. This function calculates the base offset for the current wave's
// region.
//
// For operand A (M dimension): offset = waveM * waveOffsetStride
// For operand B (N dimension): offset = waveN * waveOffsetStride
//
// Parameters:
//   operand   - Whether this is operand A or B
//   waveM     - Wave's M position in the wave grid
//   waveN     - Wave's N position in the wave grid
//   waveOffsetStride  - Stride for wave separation (depends on wave grid
//   layout)
//
// Returns:
//   Value representing the wave offset (runtime value)
//===----------------------------------------------------------------------===//
static Value computeWaveOffset(OperandKind operand, Value waveM, Value waveN,
                               Value waveOffsetStride, PatternRewriter &b,
                               Location loc) {
  Value wavePos = (operand == OperandKind::A) ? waveM : waveN;
  Value waveOffset = arith::MulIOp::create(b, loc, wavePos, waveOffsetStride);

  LLVM_DEBUG(llvm::dbgs() << "[lds_transpose] Wave offset for operand "
                          << (operand == OperandKind::A ? "A" : "B")
                          << ": wave_pos * waveOffsetStride\n");

  return waveOffset;
}

//===----------------------------------------------------------------------===//
// computeTileIterationOffset()
//===----------------------------------------------------------------------===//
// Computes the offset for the current M/N tile iteration.
//
// This offset accounts for which M/N tile we're currently loading in the
// outer loop. There are two modes:
//
// 1. Dynamic mode (useDynamicIndex=true):
//    - The tile index comes from an outer loop iterator
//    - Used when single buffering with outer loop
//    - offset = mnTileIndex * tileOffsetStride
//
// 2. Static mode (useDynamicIndex=false):
//    - The tile index is a compile-time constant
//    - Used when double buffering (loading all tiles at once)
//    - offset = mnIdxLocal * tileOffsetStride
//
// Parameters:
//   useDynamicIndex   - Whether to use runtime tile index (from outer loop)
//   mnTileIndex       - Runtime tile index from outer loop (nullptr if static)
//   mnIdxLocal        - Compile-time tile index
//   tileOffsetStride  - Stride for tile iteration (depends on wave grid layout)
//   b, loc            - MLIR builder and location
//
// Returns:
//   Value representing the tile iteration offset
//===----------------------------------------------------------------------===//
static Value computeTileIterationOffset(bool useDynamicIndex, Value mnTileIndex,
                                        int64_t mnIdxLocal,
                                        Value tileOffsetStride,
                                        PatternRewriter &b, Location loc) {
  if (useDynamicIndex) {
    // Dynamic mode: offset = mnTileIndex * tileOffsetStride
    Value tileOffset =
        arith::MulIOp::create(b, loc, mnTileIndex, tileOffsetStride);

    LLVM_DEBUG(llvm::dbgs() << "[lds_transpose] Dynamic tile offset: "
                            << "mnTileIndex * tileOffsetStride\n");

    return tileOffset;
  } else if (mnIdxLocal > 0) {
    // Static mode: offset = mnIdxLocal * tileOffsetStride (only if mnIdxLocal >
    // 0)
    Value mnIdxLocalVal = arith::ConstantIndexOp::create(b, loc, mnIdxLocal);
    Value tileOffset =
        arith::MulIOp::create(b, loc, mnIdxLocalVal, tileOffsetStride);

    LLVM_DEBUG(llvm::dbgs() << "[lds_transpose] Static tile offset: "
                            << mnIdxLocal << " * tileOffsetStride\n");

    return tileOffset;
  } else {
    // No offset needed (mnIdxLocal == 0)
    return arith::ConstantIndexOp::create(b, loc, 0);
  }
}

//===----------------------------------------------------------------------===//
// computeFinalMNOffset()
//===----------------------------------------------------------------------===//
// Computes the final M/N base offset by combining:
// 1. Base offset (from lane-level addressing)
// 2. Wave offset (for spatial wave distribution)
// 3. Tile iteration offset (for M/N tile iteration)
//
// Final formula:
//   m_base = m_offset_base + waveOffset + tileIterationOffset
//
// This is the primary offset used for addressing LDS in the M/N dimension.
//
// Parameters:
//   baseOffset        - Base offset from lane-level addressing
//   operand           - Whether this is operand A or B
//   waveM, waveN      - Wave position in the wave grid
//   mnTileIndex       - Runtime tile index (nullptr if static mode)
//   mnIdxLocal        - Compile-time tile index
//   useDynamicIndex   - Whether to use runtime tile index
//   waveOffsetStride  - Stride for wave offset (depends on wave grid layout)
//   tileOffsetStride  - Stride for tile iteration offset (depends on wave grid)
//   b, loc            - MLIR builder and location
//
// Returns:
//   Value representing the final M/N base offset
//===----------------------------------------------------------------------===//
static Value computeFinalMNOffset(Value baseOffset, OperandKind operand,
                                  Value waveM, Value waveN, Value mnTileIndex,
                                  int64_t mnIdxLocal, bool useDynamicIndex,
                                  Value waveOffsetStride,
                                  Value tileOffsetStride, PatternRewriter &b,
                                  Location loc) {
  // Start with base offset
  Value finalOffset = baseOffset;

  // Add wave offset for spatial distribution
  Value waveOffset =
      computeWaveOffset(operand, waveM, waveN, waveOffsetStride, b, loc);
  finalOffset = arith::AddIOp::create(b, loc, finalOffset, waveOffset);

  // Add tile iteration offset
  Value tileOffset = computeTileIterationOffset(
      useDynamicIndex, mnTileIndex, mnIdxLocal, tileOffsetStride, b, loc);
  finalOffset = arith::AddIOp::create(b, loc, finalOffset, tileOffset);

  LLVM_DEBUG(llvm::dbgs() << "[lds_transpose] Final M/N offset: "
                          << "base + wave + tile\n");

  return finalOffset;
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
  auto [rawSrc, /*transformStack*/ _, /*needs64BitIndices*/ __] =
      untransform(b, sourceView);

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

  // Compute wave grid layout and decompose wave ID into 2D position
  WaveGridLayout waveGrid = computeWaveGridLayout(
      waveId, physicalWaves, mPerWave, nPerWave, mPerBlock, nPerBlock, b, loc);
  Value waveM = waveGrid.waveM;
  Value waveN = waveGrid.waveN;

  // Use mPerBlock as stride for operand A, nPerBlock for operand B
  int64_t ldsStride =
      (info.operand == OperandKind::A) ? info.mPerBlock : info.nPerBlock;

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

  // Get base offsets using computeLDSBaseOffsets helper
  auto [k_base_local, m_offset_base] =
      computeLDSBaseOffsets(info.layout, lane, b, loc);

  // K stride per tile: KMfma (e.g., 8)
  int64_t kTileStride = instrK;
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

  // Compute panels on-demand from layout dimensions
  int64_t mPanels = info.mPerBlock / nonKDim;
  int64_t nPanels = info.nPerBlock / nonKDim;
  int64_t kPanels = info.kPerBlock / instrK;

  // Compute M/N tile iteration bounds
  MNTileBounds tileBounds = computeMNTileIterationBounds(
      info.doubleBuffering, info.operand, mPanels, nPanels);
  int64_t startMnIdx = tileBounds.startIdx;
  int64_t endMnIdx = tileBounds.endIdx;
  bool useDynamicMnIndex = tileBounds.useDynamicIndex;

  // For double-rate layouts ONLY (L32x16, L16x32), compute k_offset_base
  Value kOffsetBase =
      getDoubleRateKOffsetBase(b, loc, isDoubleRate, info.layout, lane);

  // Compute stride configuration for offset calculations
  StrideConfig strideConfig =
      computeStrideConfiguration(info.operand, waveGrid, nonKDim);

  Value waveOffsetStrideVal =
      arith::ConstantIndexOp::create(b, loc, strideConfig.waveOffsetStride);
  Value tileOffsetStrideVal =
      arith::ConstantIndexOp::create(b, loc, strideConfig.tileOffsetStride);

  // Generate loads: If outer loop exists, load one M/N tile with all K tiles
  //                 Otherwise, load all M/N tiles with all K tiles
  for (int64_t mnIdxLocal = startMnIdx; mnIdxLocal < endMnIdx; ++mnIdxLocal) {
    for (int64_t kIdx = 0; kIdx < kPanels; ++kIdx) {
      // Compute final M/N base offset combining:
      // - Base offset (lane-level addressing)
      // - Wave offset (spatial separation between waves)
      // - Tile offset (outer loop iteration through MFMA tiles)
      Value m_base = computeFinalMNOffset(
          m_offset_base, info.operand, waveM, waveN, mnTileIndex, mnIdxLocal,
          useDynamicMnIndex, waveOffsetStrideVal, tileOffsetStrideVal, b, loc);

      if (!isDoubleRate) {
        // SINGLE-RATE (L32x8, L16x16): One load per K tile
        Value k_base =
            computePanelFinalOffset(b, loc, isDoubleRate, k_base_local,
                                    kOffsetBase, kIdx, kTileStrideVal);

        // Emit LDS transpose load for this K tile (single-rate: one per K tile)
        Value panelVec = emitPanelLoad(b, loc, rawSrc, k_base, m_base,
                                       ldsStrideVal, panelVecType);
        panelVectors.push_back(panelVec);

      } else {
        // DOUBLE-RATE (L32x16, L16x32): TWO loads per K tile
        // Each load returns vector<4xf16>, total 8 elements per K tile
        // Compute K offsets for low and high halves
        Value k_base_low = computePanelFinalOffset(
            b, loc, isDoubleRate, k_base_local, kOffsetBase, kIdx,
            kTileStrideVal, /*isHighHalf=*/false);
        Value k_base_high = computePanelFinalOffset(
            b, loc, isDoubleRate, k_base_local, kOffsetBase, kIdx,
            kTileStrideVal, /*isHighHalf=*/true);

        // Emit low half load
        Value panelVecLow = emitPanelLoad(b, loc, rawSrc, k_base_low, m_base,
                                          ldsStrideVal, panelVecType);
        panelVectors.push_back(panelVecLow);

        // Emit high half load
        Value panelVecHigh = emitPanelLoad(b, loc, rawSrc, k_base_high, m_base,
                                           ldsStrideVal, panelVecType);
        panelVectors.push_back(panelVecHigh);
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

  // Write loaded panel vectors to destination buffer
  // Destination is rank-1 with scalar sequential layout
  int64_t destCap = destType.getShape()[0];
  int64_t targetElems = std::min<int64_t>(sliceElems, destCap);

  if (failed(writePanelVectorsToDestination(b, loc, panelVectors, dest,
                                            targetElems))) {
    return op.emitOpError(
        "Failed to write panel vectors: insufficient panel count for "
        "destination capacity");
  }

  b.eraseOp(op);
  return success();
}

} // namespace mlir::rock::hwtranspose
