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
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
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
  LogicalResult checkFunc(gpu::BinaryOp bop, gpu::KernelAttr &kernel) {
    bool hasGlobalSync = kernel.getAttr("rock.has_global_sync") != nullptr;

    auto removeBinary = [&]() -> LogicalResult {
      if (hasGlobalSync) {
        // from bop and remove func
        assert(0);
      }
      return failure(hasGlobalSync);
    };

#define GET_INT_VALUE(var, kernelMD, attr_name)                                \
  int64_t var = 0;                                                             \
  if (auto attr = kernelMD.getAttr<IntegerAttr>(attr_name))                    \
    var = attr.getInt();                                                       \
  else                                                                         \
    return removeBinary()

    GET_INT_VALUE(ldsRequired, kernel, "rock.shared_buffer_size");
    GET_INT_VALUE(gridSize, kernel, "grid_size");
    GET_INT_VALUE(blockSize, kernel, "block_size");

    auto funcName = kernel.getName();
    LLVM_DEBUG(llvm::dbgs()
               << "\nfunc:                 " << funcName << "\n"
               << "  Grid Size:          " << gridSize << "\n"
               << "  Block Size:         " << blockSize << "\n"
               << "  Shared Buffer Size: " << ldsRequired << "\n"
               << "  has Global Sync:    " << hasGlobalSync << "\n");

    if (!kernel.getMetadata())
      return removeBinary();

    GET_INT_VALUE(nextVGPR, kernel, "vgpr_count");
    GET_INT_VALUE(nextSGPR, kernel, "sgpr_count");

    LLVM_DEBUG(llvm::dbgs() << "  VGPR Size:          " << nextVGPR << "\n"
                            << "  SGPR Size:          " << nextSGPR << "\n");

    auto archAttr = bop->getParentOfType<ModuleOp>()->getAttrOfType<StringAttr>(
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
    if (blocksPerCU != 0) {
      int64_t numCUsRequired = (gridSize / blocksPerCU);
      bool allBlocksResident = arch.minNumCU >= numCUsRequired;

      // Add attribute to kernel func
      auto intTy = IntegerType::get(&getContext(), 32);

      kernel = kernel.appendMetadata(
          {NamedAttribute(StringAttr::get(&getContext(), "rock.blocks_per_cu"),
                          IntegerAttr::get(intTy, blocksPerCU))});

      if (hasGlobalSync && !allBlocksResident)
        return removeBinary();
    } else
      return removeBinary();
    return success();
  }

  void runOnOperation() override {
    Operation *topOp = getOperation();
    MLIRContext *context = &getContext();
    // Search for nested binaries:
    for (Region &region : topOp->getRegions()) {
      for (Block &block : region) {
        for (Operation &op : block) {
          auto binOp = dyn_cast<gpu::BinaryOp>(&op);
          if (!binOp)
            continue;
          SmallVector<Attribute, 8> objects;
          bool changedObj = false;
          // Walk all objects in the binary.
          for (Attribute objRaw : binOp.getObjects()) {
            auto obj = cast<gpu::ObjectAttr>(objRaw);
            gpu::KernelTableAttr metadata = obj.getKernels();
            // Continue if the object has invalid metadata.
            if (!metadata)
              continue;
            bool changedMD = false;
            NamedAttrList updatedMD;
            // Walk each of the kernels in the object.
            for (auto [name, attr] : metadata) {
              gpu::KernelAttr kernel = attr;
              if (failed(checkFunc(binOp, kernel))) {
                // Disabled failure: may be valid to simply remove the binary
                // signalPassFailure();
              }
              if (kernel != attr)
                changedMD = true;
              // Append the kernel metadata in case it was updated.
              updatedMD.append(name, kernel);
            }
            if (!changedMD) {
              objects.push_back(obj);
              continue;
            }
            // Update the object attribute in case any of the objects was
            // updated with info from checkFunc.
            objects.push_back(gpu::ObjectAttr::get(
                context, obj.getTarget(), obj.getFormat(), obj.getObject(),
                obj.getProperties(),
                gpu::KernelTableAttr::get(updatedMD.getDictionary(context))));
            changedObj = true;
          }
          // If any of the objects was updated, update the binary.
          if (changedObj)
            binOp.setObjectsAttr(ArrayAttr::get(context, objects));
        }
      }
    }
  }
};
} // namespace
