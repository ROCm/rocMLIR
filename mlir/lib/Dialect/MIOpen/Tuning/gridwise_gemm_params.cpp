#include "mlir/Dialect/MIOpen/gridwise_gemm_params.h"
#include "mlir/Dialect/MIOpen/sqlite_db.h"

#define DEBUG_TYPE "miopen-tuning-parameter"

LogicalResult PopulateParams::populateDerived(
    ConvolutionContext &ctx, InitParamsNonXDL &params, GemmSize &gemmSize,
    DerivedParams &gemmADerivedParam, DerivedParams &gemmBDerivedParam,
    DerivedBlockGemmParams &blockGemmDerivedParam, int64_t &gemmCDstPerWrite,
    int64_t &gridSize) {

  LogicalResult res(LogicalResult::Failure);
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
    ConvolutionContext &ctx, InitParamsNonXDL &validParams, GemmSize &gemmSize,
    DerivedParams &gemmADerivedParam, DerivedParams &gemmBDerivedParam,
    DerivedBlockGemmParams &blockGemmDerivedParam, int64_t &gemmCDstPerWrite,
    int64_t &gridSize) {

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
               << "DB load succeed, block size: " << validParams.blockSize
               << " M/block: " << validParams.gemmMPerBlock
               << " N/block: " << validParams.gemmNPerBlock
               << " K/block: " << validParams.gemmKPerBlock
               << " M/thread: " << validParams.gemmMPerThread
               << " N/thread: " << validParams.gemmNPerThread << "\n");
    return populateDerived(ctx, validParams, gemmSize, gemmADerivedParam,
                           gemmBDerivedParam, blockGemmDerivedParam,
                           gemmCDstPerWrite, gridSize);
  }
#endif // MLIR_ENABLE_SQLITE

  // Backup path: Use the set of default tuning parameters
  LogicalResult res(LogicalResult::Failure);

  for (auto &params : initParameters) {
    // We have an override on the blockSize, only loop through the
    // initParameters with the same blockSize
    if ((validParams.blockSize != 0) &&
        (validParams.blockSize != params.blockSize)) {
      return failure();
    }

    res = populateDerived(ctx, params, gemmSize, gemmADerivedParam,
                          gemmBDerivedParam, blockGemmDerivedParam,
                          gemmCDstPerWrite, gridSize);

    validParams = params;
    break;
  }

  if (failed(res)) {
    // All initParameters have failed, shouldn't happen
    llvm::errs() << "FATAL ERROR! COULD NOT FIND VALID TUNING PARAMETERS!\n";
  }

  return res;
}
