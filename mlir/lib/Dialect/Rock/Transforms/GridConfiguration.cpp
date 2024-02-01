//===- GridConfiguration.cpp - Rock GEMM implementation ------------===//
//
// Copyright 2022 Advanced Micro Devices.
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
// This pass computes grid size for rock.gemm and rock.attention
// operations
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include <memory>
#include <type_traits>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKGRIDCONFIGURATIONPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-gemm-to-gridwise"

using namespace mlir;
using namespace mlir::rock;

namespace {
class RockGridConfigurationPass
    : public rock::impl::RockGridConfigurationPassBase<
          RockGridConfigurationPass> {
public:
  using rock::impl::RockGridConfigurationPassBase<
      RockGridConfigurationPass>::RockGridConfigurationPassBase;
  void runOnOperation() override;

  void compute(RockGemmWrapperInterface op);
  void compute(AttentionOp op);
};
} // namespace

int64_t obtainGridSize(const GemmSize &gemmSize, const InitParams &param) {
  return (gemmSize.m / param.gemmMPerBlock) *
         (gemmSize.n / param.gemmNPerBlock) * gemmSize.g;
}

LogicalResult getKBlocks(const int64_t batchSize, const GemmSize &gemmSize,
                         const InitParamsAccel &params, int64_t &gemmKBlocks,
                         uint32_t numCu) {
  return calculateKBlockNum(batchSize, gemmSize, params.gemmMPerBlock,
                            params.gemmNPerBlock, params.gemmKPerBlock,
                            params.gemmKPack, numCu, gemmKBlocks);
}

template <typename InitParams>
GemmSize getPaddedGemmSize(const InitParams &params, GemmSize gemmSize) {
  int64_t gemmKPack{1};
  if constexpr (std::is_same_v<InitParams, InitParamsAccel>) {
    gemmKPack = params.gemmKPack;
  }

  auto gemmExtraPad =
      calculatePadding(params.gemmKPerBlock, params.gemmMPerBlock,
                       params.gemmNPerBlock, gemmSize, gemmKPack);

  if (gemmExtraPad.has_value()) {
    gemmSize.m += gemmExtraPad->m;
    gemmSize.k += gemmExtraPad->k;
    gemmSize.n += gemmExtraPad->n;
  }
  return gemmSize;
}

void RockGridConfigurationPass::compute(RockGemmWrapperInterface op) {
  OpBuilder b(op.getContext());
  GemmFeatures features = op.getGemmFeatures();
  PopulateParamsInfo info = PopulateParamsInfo::fromOp(op);

  auto origGemmSize = op.getGemmSize();

  if (isAccel(features)) {
    auto populateParamsAccelPtr = PopulateParamsAccel::select(features);

    InitParamsAccel initParams;
    if (auto xdlopsAttr = op.getGemmParams()->cast<XdlopsGemmParamsAttr>()) {
      initParams = InitParamsAccel(xdlopsAttr);
    } else {
      auto wmmaAttr = op.getGemmParams()->cast<WmmaGemmParamsAttr>();
      initParams = InitParamsAccel(wmmaAttr);
    }

    auto paddedGemmSize = getPaddedGemmSize(initParams, origGemmSize);
    bool requiredPadding = !(paddedGemmSize == origGemmSize);

    auto gridSize = obtainGridSize(paddedGemmSize, initParams);

    int64_t gemmKBlocks = 1;
    auto maybeWrwOp = (info.kernelType == KernelType::Conv2DBwdWeight);
    if (maybeWrwOp &&
        isWrWAtomicKernel(info.gemmFeatures, info.gemmAType, requiredPadding)) {
      auto res = getKBlocks(info.batchSize, paddedGemmSize, initParams,
                            gemmKBlocks, info.numCu);

      if (failed(res)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Invalid tuning parameters for computing KBlocks.\n");
        signalPassFailure();
        return;
      }
      gridSize *= gemmKBlocks;
    }

    op.setGridSizeAttr(b.getI32IntegerAttr(gridSize));
    getOperation()->setAttr("grid_size", b.getI32IntegerAttr(gridSize));

    // Set kblocks attribute only for backward weight convolutions.
    if (auto bwdOp = dyn_cast<Conv2DBwdWeightOp>(op.getOperation()))
      bwdOp->setAttr(bwdOp.getKBlocksAttrName(), b.getIndexAttr(gemmKBlocks));

  } else {
    auto attr = op.getGemmParams()->cast<GeneralGemmParamsAttr>();
    InitParamsNonAccel initParams(attr);

    auto paddedGemmSize = getPaddedGemmSize(initParams, origGemmSize);
    auto gridSize = obtainGridSize(paddedGemmSize, initParams);

    op.setGridSizeAttr(b.getI32IntegerAttr(gridSize));
    getOperation()->setAttr("grid_size", b.getI32IntegerAttr(gridSize));
  }
}

void RockGridConfigurationPass::compute(AttentionOp op) {
  OpBuilder builder(op.getContext());

  RockAccelTuningParamAttrInterface accelParams0 =
      op.getParams0Attr().cast<RockAccelTuningParamAttrInterface>();

  RockAccelTuningParamAttrInterface accelParams1 =
      op.getParams1Attr().cast<RockAccelTuningParamAttrInterface>();

  Value queries = op.getQueries();
  Value keys = op.getKeys();
  Value values = op.getValues();

  // Calculate (padded) grid size
  SmallVector<int64_t, 3> queriesShape =
      llvm::to_vector<3>(queries.getType().cast<MemRefType>().getShape());
  // Note: the gridwise ops take K x M and K x N, so Q must be transposed if
  // it's in the natural M x K form
  if (!op.getQTransposed()) {
    std::iter_swap(queriesShape.rbegin(), queriesShape.rbegin() + 1);
  }
  SmallVector<int64_t, 3> keysShape =
      llvm::to_vector<3>(keys.getType().cast<MemRefType>().getShape());
  if (op.getKTransposed()) {
    std::iter_swap(keysShape.rbegin(), keysShape.rbegin() + 1);
  }
  SmallVector<int64_t, 3> valuesShape =
      llvm::to_vector<3>(values.getType().cast<MemRefType>().getShape());
  if (op.getVTransposed()) {
    std::iter_swap(valuesShape.rbegin(), valuesShape.rbegin() + 1);
  }
  GemmSize gemm0Size(/*g=*/queriesShape[0], /*m=*/keysShape[2],
                     /*k=*/queriesShape[1],
                     /*n=*/queriesShape[2]);
  GemmSize gemm0ExtraPad =
      requiredPadding(accelParams0, gemm0Size).value_or(GemmSize{0, 0, 0, 0});
  GemmSize gemm1Size(/*g=*/queriesShape[0], /*m=*/valuesShape[2],
                     /*k=*/valuesShape[1],
                     /*n=*/keysShape[2]);
  GemmSize gemm1ExtraPad =
      requiredPadding(accelParams1, gemm1Size).value_or(GemmSize{0, 0, 0, 0});

  int64_t gridSize =
      ((gemm0Size.n + gemm0ExtraPad.n) / accelParams0.getNPerBlock()) *
      ((gemm1Size.m + gemm1ExtraPad.m) / accelParams1.getMPerBlock()) *
      gemm0Size.g;

  IntegerAttr gridSizeAttr = builder.getI32IntegerAttr(gridSize);
  func::FuncOp funcOp = getOperation();
  funcOp->setAttr("grid_size", gridSizeAttr);
}

void RockGridConfigurationPass::runOnOperation() {
  func::FuncOp func = getOperation();
  func.walk([&](RockGemmWrapperInterface op) { compute(op); });
  func.walk([&](AttentionOp op) { compute(op); });
}
