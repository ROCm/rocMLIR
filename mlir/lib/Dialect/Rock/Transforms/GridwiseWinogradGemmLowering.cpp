//===- GridwiseWinogradGemmLowering.cpp - Lower gridwise_winograd_gemm ----===//
//
// Copyright 2025 The MLIR Authors.
// Licensed under the Apache License, Version 2.0.
// =============================================================================

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/WinogradConsts.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKGRIDWISEWINOGRADGEMMLOWERINGPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

namespace {

struct GridwiseWinogradGemmLoweringPattern
    : public OpRewritePattern<GridwiseWinogradGemmOp> {
  using OpRewritePattern<GridwiseWinogradGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GridwiseWinogradGemmOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    Value filter = op.getFilter(); // flat 1D
    Value input = op.getInput();   // flat 1D
    Value output = op.getOutput(); // flat 1D

    auto filterType = cast<MemRefType>(filter.getType());
    Type elemType = filterType.getElementType();
    bool needsPromotion = elemType.isF16() || elemType.isBF16();
    Type computeType = needsPromotion ? b.getF32Type() : elemType;

    int64_t G = op.getGroups();
    int64_t C = op.getChannels();
    int64_t K = op.getNumFilters();
    int64_t N = op.getBatchSize();
    int64_t inH = op.getInputH();
    int64_t inW = op.getInputW();
    int64_t outH = op.getOutputH();
    int64_t outW = op.getOutputW();

    auto wp = winograd::getParams(op.getFmr());
    int64_t m = wp.m;
    int64_t r = wp.r;
    int64_t alpha = wp.alpha;
    int64_t alphaSq = wp.alphaSq;

    int64_t tileH = (outH + m - 1) / m;
    int64_t tileW = (outW + m - 1) / m;

    auto padding = extractFromIntegerArrayAttr<int64_t>(op.getPadding());
    int64_t padH_l = padding[0], padW_l = padding[2];

    constexpr int64_t kBatch = 2;
    int64_t kGroups = (K + kBatch - 1) / kBatch;
    int64_t totalTiles = N * G * kGroups * tileH * tileW;
    int64_t blockSize = op.getBlockSize();

    auto idxConst = [&](int64_t v) -> Value {
      return arith::ConstantIndexOp::create(b, loc, v);
    };
    auto fpConst = [&](double v) -> Value {
      return arith::ConstantOp::create(b, loc, FloatAttr::get(computeType, v));
    };
    auto promote = [&](OpBuilder &builder, Value v) -> Value {
      if (needsPromotion)
        return arith::ExtFOp::create(builder, loc, computeType, v);
      return v;
    };
    auto demote = [&](OpBuilder &builder, Value v) -> Value {
      if (needsPromotion)
        return arith::TruncFOp::create(builder, loc, elemType, v);
      return v;
    };

    Value zeroFp = fpConst(0.0);

    Value bid = rock::WorkgroupIdOp::create(b, loc, b.getIndexType());
    Value tid = rock::WorkitemIdOp::create(b, loc, b.getIndexType());
    Value globalTid = arith::AddIOp::create(
        b, loc, arith::MulIOp::create(b, loc, bid, idxConst(blockSize)), tid);

    Value inBounds = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::ult, globalTid, idxConst(totalTiles));

    scf::IfOp ifOp = scf::IfOp::create(b, loc, inBounds, false);
    {
      OpBuilder tb = ifOp.getThenBodyBuilder();

      // Decompose globalTid into (n, g, kGroup, ty, tx)
      Value rem = globalTid;
      Value tx = arith::RemUIOp::create(tb, loc, rem, idxConst(tileW));
      rem = arith::DivUIOp::create(tb, loc, rem, idxConst(tileW));
      Value ty = arith::RemUIOp::create(tb, loc, rem, idxConst(tileH));
      rem = arith::DivUIOp::create(tb, loc, rem, idxConst(tileH));
      Value kGroup = arith::RemUIOp::create(tb, loc, rem, idxConst(kGroups));
      rem = arith::DivUIOp::create(tb, loc, rem, idxConst(kGroups));
      Value g_idx = arith::RemUIOp::create(tb, loc, rem, idxConst(G));
      Value n_idx = arith::DivUIOp::create(tb, loc, rem, idxConst(G));
      Value k_base = arith::MulIOp::create(tb, loc, kGroup, idxConst(kBatch));

      Value tileOriginH = arith::SubIOp::create(
          tb, loc, arith::MulIOp::create(tb, loc, ty, idxConst(m)), idxConst(padH_l));
      Value tileOriginW = arith::SubIOp::create(
          tb, loc, arith::MulIOp::create(tb, loc, tx, idxConst(m)), idxConst(padW_l));

      // Channel loop with KBATCH*16 accumulators as iter_args.
      // Input load+transform is shared across all K in the batch.
      // Filter load+transform and MAC are per-K.
      SmallVector<Value> initAccs(kBatch * alphaSq, zeroFp);
      scf::ForOp cLoop = scf::ForOp::create(tb, loc, idxConst(0), idxConst(C),
                                             idxConst(1), initAccs);
      {
        OpBuilder lb(cLoop.getBody(), cLoop.getBody()->begin());
        Value c_idx = cLoop.getInductionVar();

        // Load 4x4 input tile (shared across all K in batch)
        Value ngc_base = arith::MulIOp::create(lb, loc,
            arith::AddIOp::create(lb, loc,
                arith::MulIOp::create(lb, loc, n_idx, idxConst(G)), g_idx),
            idxConst(C));
        ngc_base = arith::AddIOp::create(lb, loc, ngc_base, c_idx);
        ngc_base = arith::MulIOp::create(lb, loc, ngc_base, idxConst(inH));

        Value zeroIdx = idxConst(0);
        Value d[16];
        for (int ih = 0; ih < alpha; ih++) {
          for (int iw = 0; iw < alpha; iw++) {
            Value hPos = arith::AddIOp::create(lb, loc, tileOriginH, idxConst(ih));
            Value wPos = arith::AddIOp::create(lb, loc, tileOriginW, idxConst(iw));
            Value hOk = arith::AndIOp::create(lb, loc,
                arith::CmpIOp::create(lb, loc, arith::CmpIPredicate::sge, hPos, idxConst(0)),
                arith::CmpIOp::create(lb, loc, arith::CmpIPredicate::slt, hPos, idxConst(inH)));
            Value wOk = arith::AndIOp::create(lb, loc,
                arith::CmpIOp::create(lb, loc, arith::CmpIPredicate::sge, wPos, idxConst(0)),
                arith::CmpIOp::create(lb, loc, arith::CmpIPredicate::slt, wPos, idxConst(inW)));
            Value ok = arith::AndIOp::create(lb, loc, hOk, wOk);
            Value safeH = arith::SelectOp::create(lb, loc, hOk, hPos, zeroIdx);
            Value safeW = arith::SelectOp::create(lb, loc, wOk, wPos, zeroIdx);
            Value flatIdx = arith::AddIOp::create(lb, loc,
                arith::MulIOp::create(lb, loc,
                    arith::AddIOp::create(lb, loc, ngc_base, safeH),
                    idxConst(inW)),
                safeW);
            Value loaded = promote(lb, memref::LoadOp::create(lb, loc, input, flatIdx));
            d[ih * alpha + iw] = arith::SelectOp::create(lb, loc, ok, loaded, zeroFp);
          }
        }

        // Input transform: V = BT * d * B (shared across K batch)
        Value v[16];
        Value d0md8  = arith::SubFOp::create(lb, loc, d[0], d[8]);
        Value d10md2 = arith::SubFOp::create(lb, loc, d[10], d[2]);
        Value d1md9  = arith::SubFOp::create(lb, loc, d[1], d[9]);
        Value d5md9  = arith::SubFOp::create(lb, loc, d[5], d[9]);
        Value d10pd6 = arith::AddFOp::create(lb, loc, d[10], d[6]);
        Value d10md6 = arith::SubFOp::create(lb, loc, d[10], d[6]);
        Value d6md14 = arith::SubFOp::create(lb, loc, d[6], d[14]);
        Value d5md13 = arith::SubFOp::create(lb, loc, d[5], d[13]);
        v[0]  = arith::AddFOp::create(lb, loc, d0md8, d10md2);
        v[1]  = arith::AddFOp::create(lb, loc, d1md9, arith::SubFOp::create(lb, loc, d[2], d[10]));
        v[2]  = arith::SubFOp::create(lb, loc, arith::SubFOp::create(lb, loc, d[10], d[2]), d1md9);
        v[3]  = arith::AddFOp::create(lb, loc, d1md9, arith::SubFOp::create(lb, loc, d[11], d[3]));
        v[4]  = arith::SubFOp::create(lb, loc, arith::SubFOp::create(lb, loc, d[8], d[4]), d10pd6);
        v[5]  = arith::AddFOp::create(lb, loc, arith::AddFOp::create(lb, loc, d[5], d[9]), d10pd6);
        v[6]  = arith::SubFOp::create(lb, loc, d10pd6, arith::AddFOp::create(lb, loc, d[5], d[9]));
        v[7]  = arith::SubFOp::create(lb, loc, arith::SubFOp::create(lb, loc, d[9], d[11]), d5md9);
        v[8]  = arith::AddFOp::create(lb, loc, arith::SubFOp::create(lb, loc, d[8], d[4]), d10md6);
        v[9]  = arith::SubFOp::create(lb, loc, arith::SubFOp::create(lb, loc, d[9], d[5]), d10md6);
        v[10] = arith::AddFOp::create(lb, loc, d5md9, d10md6);
        v[11] = arith::SubFOp::create(lb, loc, arith::SubFOp::create(lb, loc, d[7], d[11]), d5md9);
        v[12] = arith::SubFOp::create(lb, loc, d6md14, arith::SubFOp::create(lb, loc, d[4], d[12]));
        v[13] = arith::SubFOp::create(lb, loc, d5md13, arith::SubFOp::create(lb, loc, d[14], d[6]));
        v[14] = arith::AddFOp::create(lb, loc, d6md14, d5md13);
        v[15] = arith::AddFOp::create(lb, loc, arith::SubFOp::create(lb, loc, d[15], d[7]), d5md13);

        Value half = fpConst(0.5);
        Value quarter = fpConst(0.25);

        // For each K in the batch: filter transform + MAC
        SmallVector<Value> newAccs;
        for (int kb = 0; kb < kBatch; kb++) {
          Value k_idx = arith::AddIOp::create(lb, loc, k_base, idxConst(kb));

          // Filter transform for this K
          Value gkc_base = arith::MulIOp::create(lb, loc,
              arith::AddIOp::create(lb, loc,
                  arith::MulIOp::create(lb, loc, g_idx, idxConst(K)), k_idx),
              idxConst(C));
          gkc_base = arith::AddIOp::create(lb, loc, gkc_base, c_idx);
          gkc_base = arith::MulIOp::create(lb, loc, gkc_base, idxConst(r));

          Value f[9];
          for (int fh = 0; fh < r; fh++)
            for (int fw = 0; fw < r; fw++) {
              Value fIdx = arith::AddIOp::create(lb, loc,
                  arith::MulIOp::create(lb, loc,
                      arith::AddIOp::create(lb, loc, gkc_base, idxConst(fh)),
                      idxConst(r)),
                  idxConst(fw));
              f[fh * r + fw] = promote(lb, memref::LoadOp::create(lb, loc, filter, fIdx));
            }

          Value rs0 = arith::AddFOp::create(lb, loc, arith::AddFOp::create(lb, loc, f[0], f[1]), f[2]);
          Value rd0 = arith::AddFOp::create(lb, loc, arith::SubFOp::create(lb, loc, f[0], f[1]), f[2]);
          Value rs1 = arith::AddFOp::create(lb, loc, arith::AddFOp::create(lb, loc, f[3], f[4]), f[5]);
          Value rd1 = arith::AddFOp::create(lb, loc, arith::SubFOp::create(lb, loc, f[3], f[4]), f[5]);
          Value rs2 = arith::AddFOp::create(lb, loc, arith::AddFOp::create(lb, loc, f[6], f[7]), f[8]);
          Value rd2 = arith::AddFOp::create(lb, loc, arith::SubFOp::create(lb, loc, f[6], f[7]), f[8]);

          Value u[16];
          u[0]=f[0]; u[3]=f[2]; u[12]=f[6]; u[15]=f[8];
          u[1]=arith::MulFOp::create(lb,loc,half,rs0);
          u[2]=arith::MulFOp::create(lb,loc,half,rd0);
          u[13]=arith::MulFOp::create(lb,loc,half,rs2);
          u[14]=arith::MulFOp::create(lb,loc,half,rd2);
          u[4]=arith::MulFOp::create(lb,loc,half,arith::AddFOp::create(lb,loc,arith::AddFOp::create(lb,loc,f[0],f[3]),f[6]));
          u[8]=arith::MulFOp::create(lb,loc,half,arith::AddFOp::create(lb,loc,arith::SubFOp::create(lb,loc,f[0],f[3]),f[6]));
          u[7]=arith::MulFOp::create(lb,loc,half,arith::AddFOp::create(lb,loc,arith::AddFOp::create(lb,loc,f[2],f[5]),f[8]));
          u[11]=arith::MulFOp::create(lb,loc,half,arith::AddFOp::create(lb,loc,arith::SubFOp::create(lb,loc,f[2],f[5]),f[8]));
          Value sa=arith::AddFOp::create(lb,loc,arith::AddFOp::create(lb,loc,rs0,rs1),rs2);
          Value sd=arith::AddFOp::create(lb,loc,arith::SubFOp::create(lb,loc,rs0,rs1),rs2);
          Value sb=arith::AddFOp::create(lb,loc,arith::AddFOp::create(lb,loc,rd0,rd1),rd2);
          Value sc=arith::AddFOp::create(lb,loc,arith::SubFOp::create(lb,loc,rd0,rd1),rd2);
          u[5]=arith::MulFOp::create(lb,loc,quarter,sa);
          u[9]=arith::MulFOp::create(lb,loc,quarter,sd);
          u[6]=arith::MulFOp::create(lb,loc,quarter,sb);
          u[10]=arith::MulFOp::create(lb,loc,quarter,sc);

          // MAC: acc[kb*16+i] += U[i] * V[i]
          for (int i = 0; i < alphaSq; i++) {
            Value av = cLoop.getRegionIterArg(kb * alphaSq + i);
            Value prod = arith::MulFOp::create(lb, loc, u[i], v[i]);
            newAccs.push_back(arith::AddFOp::create(lb, loc, av, prod));
          }
        }
        scf::YieldOp::create(lb, loc, newAccs);
      } // end channel loop

      // For each K in batch: output transform + write
      for (int kb = 0; kb < kBatch; kb++) {
        Value k_idx = arith::AddIOp::create(tb, loc, k_base, idxConst(kb));
        Value kValid = arith::CmpIOp::create(tb, loc, arith::CmpIPredicate::slt,
                                              k_idx, idxConst(K));
        scf::IfOp kIf = scf::IfOp::create(tb, loc, kValid, false);
        {
          OpBuilder kb_b = kIf.getThenBodyBuilder();

          Value acc[16];
          for (int i = 0; i < alphaSq; i++)
            acc[i] = cLoop.getResult(kb * alphaSq + i);

          // Output transform: Y = AT * acc * A
          Value t[8];
          for (int j = 0; j < alpha; j++) {
            t[0*alpha+j] = arith::AddFOp::create(kb_b, loc,
                arith::AddFOp::create(kb_b, loc, acc[0*alpha+j], acc[1*alpha+j]), acc[2*alpha+j]);
            t[1*alpha+j] = arith::SubFOp::create(kb_b, loc,
                arith::SubFOp::create(kb_b, loc, acc[1*alpha+j], acc[2*alpha+j]), acc[3*alpha+j]);
          }
          Value yOut[4];
          for (int i = 0; i < m; i++) {
            yOut[i*m+0] = arith::AddFOp::create(kb_b, loc,
                arith::AddFOp::create(kb_b, loc, t[i*alpha+0], t[i*alpha+1]), t[i*alpha+2]);
            yOut[i*m+1] = arith::SubFOp::create(kb_b, loc,
                arith::SubFOp::create(kb_b, loc, t[i*alpha+1], t[i*alpha+2]), t[i*alpha+3]);
          }

          // Write output
          for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
              Value oh = arith::AddIOp::create(kb_b, loc,
                  arith::MulIOp::create(kb_b, loc, ty, idxConst(m)), idxConst(i));
              Value ow = arith::AddIOp::create(kb_b, loc,
                  arith::MulIOp::create(kb_b, loc, tx, idxConst(m)), idxConst(j));
              Value ohOk = arith::CmpIOp::create(kb_b, loc, arith::CmpIPredicate::slt, oh, idxConst(outH));
              Value owOk = arith::CmpIOp::create(kb_b, loc, arith::CmpIPredicate::slt, ow, idxConst(outW));
              Value outOk = arith::AndIOp::create(kb_b, loc, ohOk, owOk);

              scf::IfOp stIf = scf::IfOp::create(kb_b, loc, outOk, false);
              {
                OpBuilder sb = stIf.getThenBodyBuilder();
                Value yval = demote(sb, yOut[i * m + j]);
                Value outBase = arith::MulIOp::create(sb, loc,
                    arith::AddIOp::create(sb, loc,
                        arith::MulIOp::create(sb, loc, n_idx, idxConst(G)), g_idx),
                    idxConst(K));
                outBase = arith::AddIOp::create(sb, loc, outBase, k_idx);
                outBase = arith::MulIOp::create(sb, loc, outBase, idxConst(outH));
                Value flatOut = arith::AddIOp::create(sb, loc,
                    arith::MulIOp::create(sb, loc,
                        arith::AddIOp::create(sb, loc, outBase, oh),
                        idxConst(outW)),
                    ow);
                memref::StoreOp::create(sb, loc, yval, output, flatOut);
              }
            }
          }
        }
      }
    }

    b.eraseOp(op);
    return success();
  }
};

struct RockGridwiseWinogradGemmLoweringPass
    : public rock::impl::RockGridwiseWinogradGemmLoweringPassBase<
          RockGridwiseWinogradGemmLoweringPass> {
  using RockGridwiseWinogradGemmLoweringPassBase::RockGridwiseWinogradGemmLoweringPassBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<GridwiseWinogradGemmLoweringPattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // end anonymous namespace
