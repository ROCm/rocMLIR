//===- GridwiseGemmToBlockwise - MLIR Rock ops lowering passes -----===//
//
// Copyright 2026 The MLIR Authors.
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
// ============================================================
//
// This pass converts rock.gridwise_gemm_accel and rock.gridwise_attention_accel
// into block- and threadwise ops
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "GridLayoutEmitter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>
#include <optional>
#include <tuple>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKTRANSFORMSTOPTRPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-transforms-to-ptr"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {
struct RockTransformsToPtrPass
    : public rock::impl::RockTransformsToPtrPassBase<
          RockTransformsToPtrPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

namespace {

// Helper function to create GPU allocations for pointer and mask tensors
static std::pair<Value, Value> createPointerAndMaskTensors(
    PatternRewriter &b, Location loc, ArrayRef<int64_t> shape) {
  auto privateMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
      gpu::GPUDialect::getPrivateAddressSpace());

  // Create pointer tensor allocation (i32 type)
  auto pointerTensorType = MemRefType::get(shape, b.getI32Type(), AffineMap{},
                                           privateMemoryAddressSpace);
  Value pointerTensor = GpuAllocOp::create(b, loc, pointerTensorType);

  // Create mask tensor allocation (i1 type, same shape)
  auto maskTensorType = MemRefType::get(shape, b.getI1Type(), AffineMap{},
                                        privateMemoryAddressSpace);
  Value maskTensor = GpuAllocOp::create(b, loc, maskTensorType);

  return {pointerTensor, maskTensor};
}

//===----------------------------------------------------------------------===//
// BlockwiseLoadTileOp lowering.
//===----------------------------------------------------------------------===//
struct BlockwiseLoadTileRewritePattern
    : public OpRewritePattern<BlockwiseLoadTileOp> {
  using OpRewritePattern<BlockwiseLoadTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlockwiseLoadTileOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    Value source = op.getSource();
    Value dest = op.getDestRegisters();
    auto sourceIndices = op.getSourceIndices();

    // Get the shape from the destination registers
    auto destType = cast<MemRefType>(dest.getType());
    auto shape = destType.getShape();

    // Create pointer and mask tensors
    auto [pointerTensor, maskTensor] = createPointerAndMaskTensors(b, loc, shape);

    // Create rock.transforms_to_ptr operation
    TransformsToPtrOp::create(b, loc, source, pointerTensor, maskTensor, sourceIndices);

    // Create rock.blockwise_load_tile_ptr operation
    BlockwiseLoadTilePtrOp::create(b, loc, pointerTensor, maskTensor, dest);

    b.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseStoreTileOp lowering.
//===----------------------------------------------------------------------===//
struct BlockwiseStoreTileRewritePattern
    : public OpRewritePattern<BlockwiseStoreTileOp> {
  using OpRewritePattern<BlockwiseStoreTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlockwiseStoreTileOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    Value source = op.getSource();
    Value dest = op.getDest();
    auto extraIndices = op.getExtraIndices();
    auto storeMethod = op.getStoreMethod();

    // Get the shape from the source registers
    auto sourceType = cast<MemRefType>(source.getType());
    auto shape = sourceType.getShape();

    // Create pointer and mask tensors
    auto [pointerTensor, maskTensor] = createPointerAndMaskTensors(b, loc, shape);

    // Create rock.transforms_to_ptr operation
    TransformsToPtrOp::create(b, loc, dest, pointerTensor, maskTensor, extraIndices);

    // Create rock.blockwise_store_tile_ptr operation
    BlockwiseStoreTilePtrOp::create(b, loc, pointerTensor, maskTensor, source, storeMethod);

    b.eraseOp(op);
    return success();
  }
};

} // end anonymous namespace

void RockTransformsToPtrPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<BlockwiseLoadTileOp, BlockwiseStoreTileOp>();
  target.addLegalOp<BlockwiseLoadTilePtrOp, BlockwiseStoreTilePtrOp,
                    TransformsToPtrOp, GpuAllocOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<BlockwiseLoadTileRewritePattern, BlockwiseStoreTileRewritePattern>(ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
