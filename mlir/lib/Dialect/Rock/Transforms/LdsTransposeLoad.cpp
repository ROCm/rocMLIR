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
bool calculatePanels(const MfmaInstrShape &shape, OperandKind operand,
                     int64_t mPerBlock, int64_t nPerBlock, int64_t kPerBlock,
                     int64_t &mPanels, int64_t &nPanels, int64_t &kPanels) {

  /*if (mPerBlock > 32 || nPerBlock > 32) {
    return false;
  }*/

  if (kPerBlock % shape.kMfma != 0) {
    return false;
  }
  kPanels = kPerBlock / shape.kMfma;

  if (operand == OperandKind::A) {
    if (mPerBlock % shape.mnMfma != 0)
      return false;
    mPanels = mPerBlock / shape.mnMfma;
    return true;
  }

  if (nPerBlock % shape.mnMfma != 0)
    return false;
  nPanels = nPerBlock / shape.mnMfma;
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

// Analyzes GEMM tiling and MFMA instruction parameters to determine
// if the hardware LDS transpose optimization can be applied.
// Returns a `Decision` struct indicating applicability and layout details.
Decision makeDecision(StringRef arch, Type elemType, bool ldsLayoutAIsMxK,
                      bool ldsLayoutBIsNxK, const MfmaInstrShape &shape,
                      OperandKind operand, int64_t mPerBlock, int64_t nPerBlock,
                      int64_t kPerBlock) {
  Decision dec;
  dec.operand = operand;

  // Basic applicability checks
  if (!archSupported(arch) || !(elemType.isF16() || elemType.isBF16()) ||
      ldsLayoutAIsMxK || ldsLayoutBIsNxK) {
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
  if (!calculatePanels(shape, operand, mPerBlock, nPerBlock, kPerBlock,
                       dec.mPanels, dec.nPanels, dec.kPanels)) {
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
                      PatternRewriter &rewriter) {
  if (!dec.usable)
    return;
  readIntoOp->setAttr("rock.hw_lds_transpose_enabled", rewriter.getUnitAttr());
  readIntoOp->setAttr("rock.hw_lds_transpose_layout",
                      rewriter.getStringAttr(layoutName(dec.layout)));
  readIntoOp->setAttr(
      "rock.hw_lds_transpose_operand",
      rewriter.getStringAttr(dec.operand == OperandKind::A ? "A" : "B"));

  if (dec.mPanels > 1)
    readIntoOp->setAttr("rock.hw_lds_transpose_mpanels",
                        rewriter.getI64IntegerAttr(dec.mPanels));
  if (dec.nPanels > 1)
    readIntoOp->setAttr("rock.hw_lds_transpose_npanels",
                        rewriter.getI64IntegerAttr(dec.nPanels));
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

  // Read paneling info directly from attributes
  if (auto mPanelsAttr =
          op->getAttrOfType<IntegerAttr>("rock.hw_lds_transpose_mpanels"))
    info.mPanels = mPanelsAttr.getInt();
  if (auto nPanelsAttr =
          op->getAttrOfType<IntegerAttr>("rock.hw_lds_transpose_npanels"))
    info.nPanels = nPanelsAttr.getInt();
  if (auto kPanelsAttr =
          op->getAttrOfType<IntegerAttr>("rock.hw_lds_transpose_kpanels"))
    info.kPanels = kPanelsAttr.getInt();

  // Destination buffer type
  auto dest = op.getDest();
  auto destType = cast<MemRefType>(dest.getType());
  Type elemType = destType.getElementType();
  info.elemType = elemType;

  // Operand kind
  if (auto operandAttr =
          op->getAttrOfType<StringAttr>("rock.hw_lds_transpose_operand")) {
    if (operandAttr.getValue() == "B")
      info.operand = OperandKind::B;
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

  switch (layout) {
  case LayoutKind::L16x32: {
    Value c16 = cst(16), c32 = cst(32), c4 = cst(4), cPanel = cst(16);
    Value laneDiv16 = div(lane, c16);
    Value laneMod16 = rem(lane, c16);
    Value offset = add(mul(laneDiv16, c32), laneMod16);
    Value off0 = mul(c4, offset);
    Value off1 = mul(c4, add(offset, cPanel));
    panelOffsets = {off0, off1};
    break;
  }
  case LayoutKind::L16x16: {
    Value c16 = cst(16), c4 = cst(4);
    Value laneDiv16 = div(lane, c16);
    Value laneMod16 = rem(lane, c16);
    Value offset = add(mul(laneDiv16, c16), laneMod16);
    Value off0 = mul(c4, offset);
    panelOffsets = {off0};
    break;
  }
  case LayoutKind::L32x16: {
    Value c32 = cst(32), c64 = cst(64), c16 = cst(16), c4 = cst(4), c8 = cst(8),
          c32el = cst(32);
    Value tidMod32 = rem(lane, c32);

    Value term0 = mul(div(lane, c32), c64);
    Value term1 = mul(div(tidMod32, c16), c4);
    Value tidMod32Mod16 = rem(tidMod32, c16);
    Value term2 = mul(div(tidMod32Mod16, c4), c8);
    Value term3 = rem(tidMod32Mod16, c4);

    Value offset = add(add(term0, term1), add(term2, term3));
    Value off0 = mul(c4, offset);
    Value off1 = add(off0, c32el);
    panelOffsets = {off0, off1};
    break;
  }
  case LayoutKind::L32x8: {
    Value c32 = cst(32), c16 = cst(16), c4 = cst(4), c8 = cst(8);
    Value tidMod32 = rem(lane, c32);

    Value term0 = mul(div(lane, c32), c32);
    Value term1 = mul(div(tidMod32, c16), c4);
    Value tidMod32Mod16 = rem(tidMod32, c16);
    Value term2 = mul(div(tidMod32Mod16, c4), c8);
    Value term3 = rem(tidMod32Mod16, c4);

    Value offset = add(term0, add(term1, add(term2, term3)));
    Value off0 = mul(c4, offset);
    panelOffsets = {off0};
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
  // Compute base LDS panel offsets according to the layout and lane mapping.
  SmallVector<Value> panelOffsets =
      getBasePanelOffsets(info.layout, lane, b, loc);

  // Each LDS transpose load returns a vector<4 x elemType>.
  VectorType panelVecType = VectorType::get({4}, elemType);
  SmallVector<Value> panelVectors;

  // Multi-K fused case:
  // Expand the offsets across multiple K panels.
  // Each additional K-panel is offset by (nonKDim * instrK) elements.
  if (info.kPanels > 1) {
    auto [nonKDim, instrK] = getLayoutDims(info.layout);
    int64_t tileStrideElems = nonKDim * instrK; // full slice size
    Value strideC = cst(tileStrideElems);
    SmallVector<Value> baseOffsets = panelOffsets;
    SmallVector<Value> expanded;
    for (int64_t kp = 0; kp < info.kPanels; ++kp) {
      Value offsetKP = b.create<arith::MulIOp>(loc, strideC, cst(kp));
      for (Value off : panelOffsets) {
        expanded.push_back(b.create<arith::AddIOp>(loc, off, offsetKP));
      }
    }
    panelOffsets = std::move(expanded);
  }
  // Main LDS load: perform transpose loads for all panels.
  for (Value off : panelOffsets) {
    auto l = b.create<rock::LDSTransposeLoadOp>(loc, panelVecType, rawSrc,
                                                ValueRange{off});
    panelVectors.push_back(l.getFragment());
  }

  int64_t sliceElems = (int64_t)panelOffsets.size() * 4;

  // Scalar buffer path rank-1.
  int64_t destCap = destType.getShape()[0];
  int64_t targetElems = std::min<int64_t>(sliceElems, destCap);
  int64_t produced = 0;

  // Write each extracted element from the loaded panel vectors into `dest`.
  // The destination is rank-1, meaning scalar sequential layout.
  for (Value pv : panelVectors) {
    for (int lane = 0; lane < 4 && produced < targetElems; ++lane) {
      Value ciLane = cst(lane);
      Value elem =
          b.create<vector::ExtractElementOp>(loc, elemType, pv, ciLane);
      Value idx = cst(produced++);
      b.create<InBoundsStoreOp>(loc, elem, dest, ValueRange{idx});
    }
    if (produced >= targetElems)
      break;
  }

  b.replaceOp(op, ValueRange{});
  return success();
}

} // namespace mlir::rock::hwtranspose
