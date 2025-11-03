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

// Calculates the number of M/N/K panels per block based on the MFMA instruction
// shape. Returns `true` if the dimensions divide evenly, otherwise `false`.
bool calculatePanels(const MfmaInstrShape &shape, OperandKind operandA,
                     OperandKind operandB, int64_t &mPerBlock,
                     int64_t &nPerBlock, int64_t kPerBlock, int64_t &mPanels,
                     int64_t &nPanels, int64_t &kPanels) {

  if (kPerBlock % shape.kMfma != 0) {
    return false;
  }
  kPanels = kPerBlock / shape.kMfma;

  if (operandA == OperandKind::A && operandB == OperandKind::B) {
    if (mPerBlock % shape.mnMfma != 0)
      return false;
    mPanels = mPerBlock / shape.mnMfma;
    if (nPerBlock % shape.mnMfma != 0)
      return false;
    nPanels = nPerBlock / shape.mnMfma;
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
                      int64_t mPerBlock, int64_t nPerBlock, int64_t kPerBlock) {
  Decision dec;
  dec.operandA = operandA;
  dec.operandB = operandB;
  dec.mPerBlock = mPerBlock;
  dec.nPerBlock = nPerBlock;

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

  // Calculate and validate paneling
  if (!calculatePanels(shape, dec.operandA, dec.operandB, dec.mPerBlock,
                       dec.nPerBlock, kPerBlock, dec.mPanels, dec.nPanels,
                       dec.kPanels)) {
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
    if (dec.mPerBlock)
      readIntoOp->setAttr("rock.hw_lds_transpose_mperblock",
                          rewriter.getI64IntegerAttr(dec.mPerBlock));
    if (dec.mPanels > 1)
      readIntoOp->setAttr("rock.hw_lds_transpose_mpanels",
                          rewriter.getI64IntegerAttr(dec.mPanels));
  } else {
    readIntoOp->setAttr("rock.hw_lds_transpose_operand",
                        rewriter.getStringAttr("B"));
    if (dec.nPerBlock)
      readIntoOp->setAttr("rock.hw_lds_transpose_nperblock",
                          rewriter.getI64IntegerAttr(dec.nPerBlock));
    if (dec.nPanels > 1)
      readIntoOp->setAttr("rock.hw_lds_transpose_npanels",
                          rewriter.getI64IntegerAttr(dec.nPanels));
  }
  if (dec.kPanels > 1)
    readIntoOp->setAttr("rock.hw_lds_transpose_kpanels",
                        rewriter.getI64IntegerAttr(dec.kPanels));

  LLVM_DEBUG(llvm::dbgs() << "[lds_transpose] attachAttributes: enabled layout="
                          << layoutName(dec.layout) << "\n");
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

  // Operand kind
  if (auto operandAttr =
          op->getAttrOfType<StringAttr>("rock.hw_lds_transpose_operand")) {
    StringRef val = operandAttr.getValue();
    if (val == "A") {
      info.operand = OperandKind::A;
      if (auto mPerBlockAttr =
              op->getAttrOfType<IntegerAttr>("rock.hw_lds_transpose_mperblock"))
        info.mPerBlock = mPerBlockAttr.getInt();
      if (auto mPanelsAttr =
              op->getAttrOfType<IntegerAttr>("rock.hw_lds_transpose_mpanels"))
        info.mPanels = mPanelsAttr.getInt();
      if (auto kPanelsAttr =
              op->getAttrOfType<IntegerAttr>("rock.hw_lds_transpose_kpanels"))
        info.kPanels = kPanelsAttr.getInt();

    } else if (val == "B") {
      info.operand = OperandKind::B;
      if (auto nPerBlockAttr =
              op->getAttrOfType<IntegerAttr>("rock.hw_lds_transpose_nperblock"))
        info.nPerBlock = nPerBlockAttr.getInt();
      if (auto nPanelsAttr =
              op->getAttrOfType<IntegerAttr>("rock.hw_lds_transpose_npanels"))
        info.nPanels = nPanelsAttr.getInt();
      if (auto kPanelsAttr =
              op->getAttrOfType<IntegerAttr>("rock.hw_lds_transpose_kpanels"))
        info.kPanels = kPanelsAttr.getInt();
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
  auto cst = [&](int64_t v) {
    return b.create<arith::ConstantIndexOp>(loc, v);
  };

  auto add = [&](Value a, Value m) {
    return b.create<arith::AddIOp>(loc, a, m);
  };
  auto mul = [&](Value a, Value m) {
    return b.create<arith::MulIOp>(loc, a, m);
  };
  auto div = [&](Value a, Value m) {
    return b.create<arith::DivUIOp>(loc, a, m);
  };
  auto rem = [&](Value a, Value m) {
    return b.create<arith::RemUIOp>(loc, a, m);
  };
  SmallVector<Value> panelOffsets;
  Value c16 = cst(16), c4 = cst(4), c2 = cst(2);
  Value blockId = div(lane, c16);
  Value laneInBlock = rem(lane, c16);
  // Base offset calculations
  Value mOffsetBase = mul(rem(laneInBlock, c4), c4);
  Value kOffsetBase = div(laneInBlock, c4);

  switch (layout) {
  case LayoutKind::L16x32: {
    panelOffsets = {kOffsetBase, mOffsetBase};
    break;
  }
  case LayoutKind::L16x16: {
    // kbase = kOffsetBase + (blockId * 4)
    Value kBase = add(mul(blockId, c4), kOffsetBase);
    panelOffsets = {kBase, mOffsetBase};
    break;
  }
  case LayoutKind::L32x16: {
    // mbase = mOffsetBase + (blockId % 2) * 16
    Value mBase = add(mul(rem(blockId, c2), cst(16)), mOffsetBase);
    panelOffsets = {kOffsetBase, mBase};
    break;
  }
  case LayoutKind::L32x8: {
    // k_base_local = kOffsetBase + (blockId / 2) * 4
    Value kBase = add(mul(div(blockId, c2), c4), kOffsetBase);

    // m_offset_base = mOffsetBase + (blockId % 2) * 16
    Value mBase = add(mul(rem(blockId, c2), cst(16)), mOffsetBase);
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
                                        PatternRewriter &b) {
  if (!info.usable)
    return failure();

  Location loc = op.getLoc();
  auto dest = op.getDest();
  auto destType = cast<MemRefType>(dest.getType());
  Type elemType = info.elemType;
  Value sourceView = op.getSource();
  auto [rawSrc, _, __] = untransform(b, sourceView);

  Value tid = b.createOrFold<rock::WorkitemIdOp>(loc, b.getIndexType());
  auto cst = [&](int64_t v) {
    return b.create<arith::ConstantIndexOp>(loc, v);
  };
  // Compute lane ID within the wavefront (0–63).
  Value lane = b.create<arith::RemUIOp>(loc, tid, cst(64));

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

  Value mnStrideVal = cst(mnStride);
  Value kTileStrideVal = cst(kTileStride);
  Value ldsStrideVal = cst(ldsStride);

  // The extra indices tell us WHICH M/N tile we're loading in this iteration.
  // Check if there's an extra index for M/N tile selection
  ValueRange extraIndices = op.getExtraIndices();
  Value mnTileIndex = nullptr;

  // Extra indices format: [tid, m_tile_idx] for A or [tid, n_tile_idx] for B
  if (extraIndices.size() >= 2) {
    mnTileIndex = extraIndices[1]; // Second index is the M/N tile iterator
  }

  // If we have an M/N tile index from the outer loop, use it
  // Otherwise, generate all M/N tiles (fallback for single-tile case)
  int64_t startMnIdx = 0;
  int64_t endMnIdx = 1;
  bool useDynamicMnIndex = false;

  if (mnTileIndex) {
    // Outer loop handles M/N iteration, we load only ONE M/N tile per call
    useDynamicMnIndex = true;
    endMnIdx = 1;
  } else {
    // No outer loop, generate all M/N tiles statically
    if (info.operand == OperandKind::A) {
      endMnIdx = info.mPanels;
    } else if (info.operand == OperandKind::B) {
      endMnIdx = info.nPanels;
    }
  }

  int64_t kPanels = info.kPanels;

  // For double-rate layouts ONLY (L32x16, L16x32), compute k_offset_base
  // L32x16 (32x32x16): k_offset_base = (block_id / 2) * 8
  // L16x32 (16x16x32): k_offset_base = block_id * 8
  Value blockId = nullptr;
  Value kOffsetBase = nullptr;

  if (isDoubleRate) {
    Value c16 = cst(16), c2 = cst(2), c8 = cst(8);
    blockId = b.create<arith::DivUIOp>(loc, lane, c16);

    if (info.layout == LayoutKind::L32x16) {
      // k_offset_base = (block_id / 2) * 8
      kOffsetBase = b.create<arith::MulIOp>(
          loc, b.create<arith::DivUIOp>(loc, blockId, c2), c8);
    } else if (info.layout == LayoutKind::L16x32) {
      // k_offset_base = block_id * 8
      kOffsetBase = b.create<arith::MulIOp>(loc, blockId, c8);
    }
  }

  // Generate loads: If outer loop exists, load one M/N tile with all K tiles
  //                 Otherwise, load all M/N tiles with all K tiles
  for (int64_t mnIdxLocal = startMnIdx; mnIdxLocal < endMnIdx; ++mnIdxLocal) {
    for (int64_t kIdx = 0; kIdx < kPanels; ++kIdx) {
      // Calculate m_base for this M/N tile
      Value m_base = m_offset_base;
      if (useDynamicMnIndex) {
        // Use dynamic index from outer loop: m_base += mnTileIndex * mnStride
        Value mnOffsetAdd =
            b.create<arith::MulIOp>(loc, mnTileIndex, mnStrideVal);
        m_base = b.create<arith::AddIOp>(loc, m_base, mnOffsetAdd);
      } else if (mnIdxLocal > 0) {
        // Use static index: m_base += mnIdxLocal * mnStride
        Value mnOffsetAdd =
            b.create<arith::MulIOp>(loc, mnStrideVal, cst(mnIdxLocal));
        m_base = b.create<arith::AddIOp>(loc, m_base, mnOffsetAdd);
      }

      if (!isDoubleRate) {
        // SINGLE-RATE (L32x8, L16x16): One load per K tile
        // k_base = k_base_local + kIdx * kTileStride
        Value k_base = k_base_local;
        if (kIdx > 0) {
          Value kOffsetAdd =
              b.create<arith::MulIOp>(loc, kTileStrideVal, cst(kIdx));
          k_base = b.create<arith::AddIOp>(loc, k_base, kOffsetAdd);
        }

        // final_offset = k_base * ldsStride + m_base
        Value final_offset = b.create<arith::AddIOp>(
            loc, m_base, b.create<arith::MulIOp>(loc, k_base, ldsStrideVal));

        // Perform LDS transpose load (ds_read_tr16_b64) -> returns
        // vector<4xf16>
        auto l = b.create<rock::LDSTransposeLoadOp>(loc, panelVecType, rawSrc,
                                                    ValueRange{final_offset});
        panelVectors.push_back(l.getFragment());

      } else {
        // DOUBLE-RATE (L32x16, L16x32): TWO loads per K tile
        // Each load returns vector<4xf16>, total 8 elements per K tile
        // k_offset_low = k_offset_base + k_tile * KMfma
        // k_offset_high = k_offset_base + 4 + k_tile * KMfma

        Value kTileOffset =
            b.create<arith::MulIOp>(loc, kTileStrideVal, cst(kIdx));
        Value k_offset_low =
            b.create<arith::AddIOp>(loc, kOffsetBase, kTileOffset);
        Value k_offset_high =
            b.create<arith::AddIOp>(loc, k_offset_low, cst(4));

        Value k_base_low =
            b.create<arith::AddIOp>(loc, k_base_local, k_offset_low);
        Value k_base_high =
            b.create<arith::AddIOp>(loc, k_base_local, k_offset_high);

        // offset_low = k_base_low * ldsStride + m_base
        Value offset_low = b.create<arith::AddIOp>(
            loc, m_base,
            b.create<arith::MulIOp>(loc, k_base_low, ldsStrideVal));

        // offset_high = k_base_high * ldsStride + m_base
        Value offset_high = b.create<arith::AddIOp>(
            loc, m_base,
            b.create<arith::MulIOp>(loc, k_base_high, ldsStrideVal));

        // Load low half: returns vector<4xf16>
        auto load_low = b.create<rock::LDSTransposeLoadOp>(
            loc, panelVecType, rawSrc, ValueRange{offset_low});
        panelVectors.push_back(load_low.getFragment());

        // Load high half: returns vector<4xf16>
        auto load_high = b.create<rock::LDSTransposeLoadOp>(
            loc, panelVecType, rawSrc, ValueRange{offset_high});
        panelVectors.push_back(load_high.getFragment());
      }
    }
  }

  // Calculate expected number of loads
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
      Value ciLane = cst(lane);
      Value elem = b.create<vector::ExtractOp>(loc, pv, ciLane);
      Value idx = cst(produced++);
      b.create<InBoundsStoreOp>(loc, elem, dest, ValueRange{idx});
    }
    if (produced >= targetElems)
      break; // Stop once we have written all target elements
  }

  b.replaceOp(op, ValueRange{});
  return success();
}

} // namespace mlir::rock::hwtranspose
