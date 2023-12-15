//===- ViewToRock.cpp - Lowering Tensor to Rock Dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes Tensor view operations to the Rock
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKCHECKRESIDENCYPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-check-residency"

using namespace mlir;

namespace {

struct RockCheckResidencyPass
    : public rock::impl::RockCheckResidencyPassBase<RockCheckResidencyPass> {
public:
  LogicalResult checkFunc(gpu::GPUModuleOp mod, LLVM::LLVMFuncOp func) {
    bool hasGlobalSync = func->hasAttr("rock.has_global_sync");

    auto removeBinary = [&]() -> LogicalResult {
      if (hasGlobalSync) {
        // from mod and remove func
        assert(0);
      }
      return failure(hasGlobalSync);
    };

#define GET_INT_VALUE(var, op, attr_name)                                      \
  int64_t var = 0;                                                             \
  if (auto attr = op->getAttrOfType<IntegerAttr>(attr_name))                   \
    var = attr.getInt();                                                       \
  else                                                                         \
    return removeBinary()

    GET_INT_VALUE(ldsRequired, func, "rock.shared_buffer_size");
    GET_INT_VALUE(gridSize, func, "grid_size");
    GET_INT_VALUE(blockSize, func, "block_size");

    auto funcName = func.getName();
    LLVM_DEBUG(llvm::dbgs()
               << "\nfunc:                 " << funcName << "\n"
               << "  Grid Size:          " << gridSize << "\n"
               << "  Block Size:         " << blockSize << "\n"
               << "  Shared Buffer Size: " << ldsRequired << "\n"
               << "  has Global Sync:    " << hasGlobalSync << "\n");

#define GET_INT_VALUE2(var, dict, name)                                        \
  int64_t var = 0;                                                             \
  if (auto var##Attr = dict.get(name))                                         \
    var = var##Attr.cast<IntegerAttr>().getInt();                              \
  else                                                                         \
    return removeBinary()

    auto metaData = mod->getAttrOfType<DictionaryAttr>("rocdl.metadata");
    if (!metaData)
      return removeBinary();

    GET_INT_VALUE2(nextVGPR, metaData, "amdhsa_next_free_vgpr");
    GET_INT_VALUE2(nextSGPR, metaData, "amdhsa_next_free_sgpr");

    LLVM_DEBUG(llvm::dbgs() << "  VGPR Size:          " << nextVGPR << "\n"
                            << "  SGPR Size:          " << nextSGPR << "\n");

    auto archAttr = mod->getParentOfType<ModuleOp>()->getAttrOfType<StringAttr>(
        "mhal.arch");
    if (!archAttr)
      return removeBinary();

    auto arch = rock::lookupArchInfo(archAttr);

    int64_t numThreadsPerSIMD = arch.waveSize;
    int64_t numWavesPerBlock = ((blockSize - 1) / numThreadsPerSIMD) + 1;

    // Compute total per CU based on Register Allocation
    int64_t wavesPerSIMDByVGPRs = arch.totalVGPRPerEU / nextVGPR;
    int64_t wavesPerSIMDBySGPRs = arch.totalSGPRPerEU / nextSGPR;
    int64_t wavesPerSIMDByGPRs = std::min(
        std::min(wavesPerSIMDByVGPRs, wavesPerSIMDBySGPRs), arch.maxWavesPerEU);

    int64_t wavesPerCUByGPRs = wavesPerSIMDByGPRs * arch.numEUPerCU;
    int64_t blocksPerCUByGPRs = wavesPerCUByGPRs / numWavesPerBlock;

    // Compute total blocks per CU based on Shared Memory Allocation
    int64_t blocksPerCUByLDS = arch.totalSharedMemPerCU / ldsRequired;

    // Total blocks per CU
    int64_t blocksPerCU = std::min(blocksPerCUByGPRs, blocksPerCUByLDS);

    LLVM_DEBUG(llvm::dbgs() << "  ==================================\n"
                            << "  Blocks Per CU:      " << blocksPerCU << "\n");

    // All blocks resident
    int64_t numCUsRequired = (gridSize / blocksPerCU);
    bool allBlocksResident = arch.minNumCU >= numCUsRequired;

    // Add attribute to kernel func
    auto intTy = IntegerType::get(func->getContext(), 32);
    func->setAttr("rock.blocks_per_cu", IntegerAttr::get(intTy, blocksPerCU));

    if (hasGlobalSync && !allBlocksResident)
      return removeBinary();
    return success();
  }

  void runOnOperation() override {
    auto mod = getOperation();

    mod->walk([&](LLVM::LLVMFuncOp f) {
      if (failed(checkFunc(mod, f))) {
        // Disabled failure: may be valid to simply remove the binary
        // signalPassFailure();
      }
    });
  }
};
} // namespace
