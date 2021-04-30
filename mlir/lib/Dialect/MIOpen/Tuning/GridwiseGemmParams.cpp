#include "mlir/Dialect/MIOpen/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/MIOpen/Tuning/SqliteDb.h"

#define DEBUG_TYPE "miopen-tuning-parameter"

LogicalResult PopulateParams::populateDerived(
    ConvolutionContext &ctx, InitParamsNonXDL &params, GemmSize &gemmSize,
    DerivedParams &gemmADerivedParam, DerivedParams &gemmBDerivedParam,
    DerivedBlockGemmParams &blockGemmDerivedParam, int64_t &gemmCDstPerWrite,
    int64_t &gridSize) {

  LogicalResult res = failure();
  res = isValidGemm(&params, gemmSize);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Gemm sizes, M: " << gemmSize.gemmM
               << " N: " << gemmSize.gemmN << " K: " << gemmSize.gemmK << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Gemm size and gemm/block "
                            << "size does not divide exactly.\n");
    return failure();
  }

  res = calculateGemmABlockCopyPerformanceParameters(&params, ctx,
                                                     gemmADerivedParam);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent gemmA tuning parameter "
                            << " size.\n");
    return failure();
  }

  res = calculateGemmBBlockCopyPerformanceParameters(&params, ctx,
                                                     gemmBDerivedParam);

  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent gemmB tuning parameter "
                            << " size.\n");
    return failure();
  }

  res = CalculateBlockGemmPerformanceParameters(params, ctx,
                                                blockGemmDerivedParam);

  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent blockGemm tuning parameter "
                            << " size.\n");
    return failure();
  }

  gridSize = obtainGridSize(gemmSize, &params);
  gemmCDstPerWrite = calculateGemmCDestDataPerWrite(params, ctx);
  return success();
}

LogicalResult PopulateParams::paramsFromCtx(
    ConvolutionContext &ctx, int64_t blockSizeOverride,
    InitParamsNonXDL &validParams, DerivedParams &gemmADerivedParam,
    DerivedParams &gemmBDerivedParam,
    DerivedBlockGemmParams &blockGemmDerivedParam, int64_t &gemmCDstPerWrite,
    int64_t &gridSize) {

  GemmSize gemmSize;
  obtainGemmSize(ctx, gemmSize);

#if __MLIR_ENABLE_SQLITE__
  std::string solverId;
  if (ctx.opType == miopen::ConvOpType::Conv2DOpType) {
    solverId = "ConvHipImplicitGemmV4R4Fwd";
  } else if (ctx.opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    solverId = "ConvHipImplicitGemmBwdDataV1R1";
  } else {
    solverId = "ConvHipImplicitGemmV4R4WrW";
  }

  SQLitePerfDb perfDb = getDb(ctx.arch, ctx.num_cu);
  bool loadRes = perfDb.load(ctx, solverId, validParams);
  if (loadRes) {
    LLVM_DEBUG(llvm::dbgs()
               << "DB load succeed,"
               << " block size: " << validParams.blockSize
               << " M/block: " << validParams.gemmMPerBlock
               << " N/block: " << validParams.gemmNPerBlock
               << " K/block: " << validParams.gemmKPerBlock
               << " M/thread: " << validParams.gemmMPerThread
               << " N/thread: " << validParams.gemmNPerThread << "\n");
    return populateDerived(ctx, validParams, gemmSize, gemmADerivedParam,
                           gemmBDerivedParam, blockGemmDerivedParam,
                           gemmCDstPerWrite, gridSize);
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "DB load failed, falling back to backup path.\n");
  }
#endif // MLIR_ENABLE_SQLITE

  // Backup path: Use the set of default tuning parameters
  LogicalResult res = failure();

  for (auto &params : initParameters) {
    // We have an override on the blockSize, only loop through the
    // initParameters with the same blockSize
    if ((blockSizeOverride != 0) && (blockSizeOverride != params.blockSize)) {
      continue;
    }

    res = populateDerived(ctx, params, gemmSize, gemmADerivedParam,
                          gemmBDerivedParam, blockGemmDerivedParam,
                          gemmCDstPerWrite, gridSize);
    if (failed(res)) {
      continue;
    }

    validParams = params;
    break;
  }

  if (failed(res)) {
    // All initParameters have failed, shouldn't happen
    LLVM_DEBUG(llvm::dbgs() << "FATAL ERROR! COULD NOT FIND VALID TUNING"
                            << " PARAMETERS!\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Successfully picked tuning params from backup"
                            << " path.\n");
  }

  return res;
}

LogicalResult PopulateParamsXDL::populateDerived(
    ConvolutionContext &ctx, InitParamsXDL &params, GemmSize &gemmSize,
    DerivedParams &gemmADerivedParam, DerivedParams &gemmBDerivedParam,
    int64_t &blockSize, int64_t &gridSize) {
  LogicalResult res = isValidGemm(&params, gemmSize);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Gemm sizes, M: " << gemmSize.gemmM
               << " N: " << gemmSize.gemmN << " K: " << gemmSize.gemmK << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Gemm size and gemm/block "
                            << "size does not divide exactly.\n");
    return failure();
  }

  blockSize = obtainBlockSize(params, waveSize);

  res = isValidblockwisegemmxdlops(params, blockSize);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Invalid XDLOps gemm.\n");
    return failure();
  }

  res = calculateGemmABlockCopyPerformanceParameters(&params, ctx,
                                                     gemmADerivedParam);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent gemmA tuning parameter "
                            << " size.\n");
    return failure();
  }

  res = calculateGemmBBlockCopyPerformanceParameters(&params, ctx,
                                                     gemmBDerivedParam);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent gemmB tuning parameter "
                            << " size.\n");
    return failure();
  }

  std::size_t ldsSize = 0;
  res = calculateLdsNumberOfByte(params, ctx, gemmADerivedParam,
                                 gemmBDerivedParam, ldsSize);

  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "LDS size too large.\n");
    return failure();
  }

  // parameters derivable from tunable parameters.
  gridSize = obtainGridSize(gemmSize, &params);
  return success();
}

LogicalResult PopulateParamsXDL::paramsFromCtx(
    ConvolutionContext &ctx, int64_t blockSizeOverride,
    InitParamsXDL &validParams, DerivedParams &gemmADerivedParam,
    DerivedParams &gemmBDerivedParam, int64_t &blockSize, int64_t &gridSize) {

  GemmSize gemmSize;
  obtainGemmSize(ctx, gemmSize);

#if __MLIR_ENABLE_SQLITE__
  std::string solverId;
  if (ctx.opType == miopen::ConvOpType::Conv2DOpType) {
    solverId = "ConvHipImplicitGemmForwardV4R4Xdlops";
  } else if (ctx.opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
    solverId = "ConvHipImplicitGemmBwdDataV4R1Xdlops";
  } else {
    solverId = "ConvHipImplicitGemmWrwV4R4Xdlops";
  }

  SQLitePerfDb perfDb = getDb(ctx.arch, ctx.num_cu);
  bool loadRes = perfDb.load(ctx, solverId, validParams);
  if (loadRes) {
    LLVM_DEBUG(llvm::dbgs()
               << "DB load succeed,"
               << " M/block: " << validParams.gemmMPerBlock
               << " N/block: " << validParams.gemmNPerBlock
               << " K/block: " << validParams.gemmKPerBlock
               << " M/Wave: " << validParams.gemmMPerWave
               << " N/Wave: " << validParams.gemmNPerWave
               << " KPack: " << validParams.gemmKPack
               << " ACopyGemmK: " << validParams.gemmAThreadCopyMoreGemmK
               << " BCopyKPack: " << validParams.gemmBThreadCopyMoreGemmKPack
               << "\n");
    return populateDerived(ctx, validParams, gemmSize, gemmADerivedParam,
                           gemmBDerivedParam, blockSize, gridSize);
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "DB load failed, falling back to backup path.\n");
  }
#endif // MLIR_ENABLE_SQLITE

  LogicalResult res = failure();
  for (auto &params : initParameters) {
    blockSize = obtainBlockSize(params, waveSize);
    // We have an override on the blockSize, only loop through the
    // initParameters with the same blockSize
    if ((blockSizeOverride != 0) && (blockSizeOverride != blockSize)) {
      continue;
    }

    res = populateDerived(ctx, params, gemmSize, gemmADerivedParam,
                          gemmBDerivedParam, blockSize, gridSize);
    if (failed(res)) {
      continue;
    }

    validParams = params;
    break;
  }

  if (failed(res)) {
    // All initParameters have failed, shouldn't happen
    LLVM_DEBUG(llvm::dbgs() << "FATAL ERROR! COULD NOT FIND VALID TUNING"
                            << " PARAMETERS!\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Successfully picked tuning params from backup"
                            << " path.\n");
  }

  return res;
}
