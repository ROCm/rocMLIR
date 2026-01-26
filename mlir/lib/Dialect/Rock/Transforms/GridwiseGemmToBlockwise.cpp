//===- GridwiseGemmToBlockwise - MLIR Rock ops lowering passes -----===//
//
// Copyright 2020 The MLIR Authors.
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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#define GEN_PASS_DEF_ROCKGRIDWISEGEMMTOBLOCKWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-gridwise-to-blockwise"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {
struct RockGridwiseGemmToBlockwisePass
    : public rock::impl::RockGridwiseGemmToBlockwisePassBase<
          RockGridwiseGemmToBlockwisePass> {
  void runOnOperation() override;
};

} // end anonymous namespace

static Value blockwiseGemmAccel(PatternRewriter &rewriter, Location loc,
                                Value bufferA, Value bufferB, Value matrixC,
                                Value bufferScaleA, Value bufferScaleB) {
  auto cType = cast<RankedTensorType>(matrixC.getType());
  auto gemmOp = BlockwiseGemmOp::create(rewriter, loc, cType, bufferA, bufferB,
                                        matrixC, bufferScaleA, bufferScaleB);
  return gemmOp.getResult();
}

static scf::ForOp createMainLoop(PatternRewriter &rewriter, Location loc,
                                 Value end, ValueRange iterArgs) {
  Value one = rewriter.createOrFold<arith::ConstantIntOp>(
      loc, rewriter.getI32Type(), 1);
  Value start = rewriter.createOrFold<arith::ConstantIntOp>(
      loc, rewriter.getI32Type(), 0);
  scf::ForOp loopOp =
      scf::ForOp::create(rewriter, loc, start, end, one, iterArgs);
  return loopOp;
}

// This function will process a tile of gemm input into LDS (or register)
// buffer in a way it could be fed to blockwise_gemm_accel op
static Value loadAndStoreGemmInputTile(PatternRewriter &rewriter, Location loc,
                                       Value in, Value kIter, StringRef dName,
                                       rock::layout::GridCoordinates gridCoords,
                                       int64_t kPerBlock, int64_t dPerBlock,
                                       SmallVector<int64_t, 3> &bidGridLengths) {
  FailureOr<RegsAsMatrixSubTiles> maybeBufferViews = getLoadRegsAsTileViews(
      rewriter, loc, in, dName, bidGridLengths, kPerBlock, dPerBlock);
  assert(succeeded(maybeBufferViews));
  Value wrappedSource = transform(rewriter, in, maybeBufferViews->gridSubTile);

  // Determine the result type from the source shape (last two dimensions are the tile)
  auto sourceType = cast<RankedTensorType>(wrappedSource.getType());
  auto sourceShape = sourceType.getShape();
  auto resultType = RankedTensorType::get(
      sourceShape.take_back(2), sourceType.getElementType());

  // Load from global memory to LDS or register buffer.
  auto loadOp = BlockwiseLoadTileOp::create(
      rewriter, loc, resultType, wrappedSource,
      ValueRange{kIter, gridCoords.g_block, gridCoords.m_block,
                 gridCoords.n_block});
  return loadOp.getResult();
}

// This function creates a zero-initialized accumulator tensor
static Value createZeroAccBuffer(PatternRewriter &rewriter, Location loc,
                                 int64_t mPerBlock, int64_t nPerBlock,
                                 Type accType) {
  auto tensorType = RankedTensorType::get({mPerBlock, nPerBlock}, accType);
  auto zeroAttr = rewriter.getZeroAttr(tensorType);
  return arith::ConstantOp::create(rewriter, loc, tensorType, zeroAttr);
}

//===----------------------------------------------------------------------===//
// GridwiseGemm lowering.
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// GridwiseGemmAccel lowering.
//===----------------------------------------------------------------------===//
struct GridwiseGemmAccelRewritePattern
    : public OpRewritePattern<GridwiseGemmOp> {
  using OpRewritePattern<GridwiseGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GridwiseGemmOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    // Obtain data types of inputs.
    auto elementTypeA = op.getA().getType().getElementType();
    auto maybeElementTypeALoad = getInputFusionElementType(op.getA());
    auto elementTypeALoad = failed(maybeElementTypeALoad)
                                ? elementTypeA
                                : maybeElementTypeALoad.value();

    auto elementTypeB = op.getB().getType().getElementType();
    auto maybeElementTypeBLoad = getInputFusionElementType(op.getB());
    auto elementTypeBLoad = failed(maybeElementTypeBLoad)
                                ? elementTypeB
                                : maybeElementTypeBLoad.value();
    auto destType = op.getResult().getType().getElementType();
    auto scaleA = op.getScaleA();
    auto scaleB = op.getScaleB();
    bool hasScaleA = scaleA != nullptr;
    bool hasScaleB = scaleB != nullptr;
    bool isScaledGemm = hasScaleA && hasScaleB;
    auto elementTypeScaleA =
        isScaledGemm ? scaleA.getType().getElementType() : nullptr;
    auto elementTypeScaleB =
        isScaledGemm ? scaleB.getType().getElementType() : nullptr;

    // Get 'features' from the op
    auto features = rock::getFeatures(op);
    auto featuresAttr = op.getFeaturesAttr();

    // Prepare some useful constants.
    Value matA = op.getA();
    Value matB = op.getB();

    // Obtain critical matrix dimensions.
    ArrayRef<int64_t> aShape, bShape;
    aShape = op.getA().getType().getShape();
    bShape = op.getB().getType().getShape();
    // Obtain critical matrix dimensions.
    int64_t G = aShape[0];
    int64_t M = aShape[1];
    int64_t K = aShape[2];
    int64_t N = bShape[2];

    // Obtain critical tuning parameters.
    StringRef arch = rock::getArchValue(op);
    uint32_t blockSize = rock::getBlockSize(op).value().getInt();
    uint32_t gridSize = rock::getGridSize(op).value().getInt();
    GemmParamsAttr tuningParams = op.getParams();
    int64_t kpack = tuningParams.getKpack();
    int64_t kPerBlock = tuningParams.getKPerBlock();
    int64_t mPerBlock = tuningParams.getMPerBlock();
    int64_t nPerBlock = tuningParams.getNPerBlock();
    int64_t mBlocks = M / mPerBlock;
    int64_t nBlocks = N / nPerBlock;

    LLVM_DEBUG(llvm::dbgs() << "gridSize: " << gridSize << "\n"
                            << "blockSize: " << blockSize << "\n"
                            << "elementTypeALoad: " << elementTypeALoad << "\n"
                            << "elementTypeBLoad: " << elementTypeBLoad << "\n"
                            << "\n"
                            << "kPerBlock: " << kPerBlock << "\n"
                            << "mPerBlock: " << mPerBlock << "\n"
                            << "nPerBlock: " << nPerBlock << "\n");
    SmallVector<int64_t, 3> bidGridLengths = {G, mBlocks, nBlocks};

    // Get current workgroup ID.
    auto bid = WorkgroupIdOp::create(b, loc, b.getI32Type());

    // Compute grid coordinates
    int64_t gridGroupSize = tuningParams.getGridGroupSize();
    auto gridCoords = layout::makeGroupedGridLayout(
        b, loc, bid,
        {G, mBlocks, nBlocks, rock::getNumCUValue(op),
         rock::getNumChipletsValue(op), elementTypeA, destType, gridGroupSize},
        arch);

    // Obtain Accelerator-related attributes.
    int64_t numWaves = tuningParams.getNumWaves();
    int64_t numCTAs = tuningParams.getNumCTAs();

    LLVM_DEBUG(llvm::dbgs() << "M: " << M << "\n"
                            << "N: " << N << "\n"
                            << "K: " << K << "\n"
                            << "G: " << G << "\n"
                            << "mPerBlock: " << mPerBlock << "\n"
                            << "nPerBlock: " << nPerBlock << "\n"
                            << "kPerBlock: " << kPerBlock << "\n"
                            << "kpack: " << kpack << "\n"
                            << "mBlocks = M / mPerBlock: " << mBlocks << "\n"
                            << "nBlocks = N / nPerBlock: " << nBlocks << "\n"
                            << "numWaves: " << numWaves << "\n"
                            << "numCTAs: " << numCTAs << "\n");

    // TODO(roctriton): f32 if float, i32 if int
    Type accType = b.getF32Type();
    Value initAcc = createZeroAccBuffer(b, loc, mPerBlock, nPerBlock, accType);

    // Emit loop with iter_args for the accumulator
    int64_t kIterations = K / kPerBlock;
    Value nIterations =
        ConstantIntOp::create(b, loc, b.getI32Type(), kIterations);

    scf::ForOp loopOp = createMainLoop(b, loc, nIterations, ValueRange{initAcc});
    Value loopResult;
    {
      PatternRewriter::InsertionGuard guard(b);
      b.setInsertionPointToStart(loopOp.getBody());
      Value iv = loopOp.getInductionVar();
      Value accArg = loopOp.getRegionIterArg(0);

      // Load from global memory to registers
      Value loadedB = loadAndStoreGemmInputTile(
          b, loc, matB, /*kiter=*/iv, "n", gridCoords, kPerBlock,
          nPerBlock, bidGridLengths);
      Value loadedA = loadAndStoreGemmInputTile(
          b, loc, matA, /*kiter=*/iv, "m", gridCoords, kPerBlock,
          mPerBlock, bidGridLengths);

      Value loadedScaleA, loadedScaleB;
      if (isScaledGemm) {
        loadedScaleB = loadAndStoreGemmInputTile(
            b, loc, scaleB, /*kiter=*/iv, "n", gridCoords,
            kPerBlock, nPerBlock, bidGridLengths);
        loadedScaleA = loadAndStoreGemmInputTile(
            b, loc, scaleA, /*kiter=*/iv, "m", gridCoords,
            kPerBlock, mPerBlock, bidGridLengths);
      }

      // Emit blockwise GEMM. This will load data from LDS (or registers) and
      // compute the MMA at the same time
      Value newAcc = blockwiseGemmAccel(b, loc, loadedA, loadedB, accArg,
                                        /*bufferScaleA=*/loadedScaleA,
                                        /*bufferScaleB=*/loadedScaleB);

      // Yield the new accumulator
      scf::YieldOp::create(b, loc, ValueRange{newAcc});
      loopResult = loopOp.getResult(0);
    }

    auto untileResult =
        rock::UntileOp::create(b, loc, op.getResult().getType(), loopResult);
    b.replaceOp(op, untileResult);
    return success();
  }
};

} // end anonymous namespace

void RockGridwiseGemmToBlockwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<rock::GridwiseGemmOp, GridwiseAttentionOp>();
  target.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                         memref::MemRefDialect, affine::AffineDialect,
                         vector::VectorDialect, linalg::LinalgDialect,
                         scf::SCFDialect, math::MathDialect,
                         tensor::TensorDialect>();
  target.addLegalOp<gpu::PrintfOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<GridwiseGemmAccelRewritePattern>(ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
