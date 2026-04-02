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

    int64_t totalTiles = N * G * K * tileH * tileW;
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

      // Decompose globalTid into (n, g, k, ty, tx)
      Value rem = globalTid;
      Value tx = arith::RemUIOp::create(tb, loc, rem, idxConst(tileW));
      rem = arith::DivUIOp::create(tb, loc, rem, idxConst(tileW));
      Value ty = arith::RemUIOp::create(tb, loc, rem, idxConst(tileH));
      rem = arith::DivUIOp::create(tb, loc, rem, idxConst(tileH));
      Value k_idx = arith::RemUIOp::create(tb, loc, rem, idxConst(K));
      rem = arith::DivUIOp::create(tb, loc, rem, idxConst(K));
      Value g_idx = arith::RemUIOp::create(tb, loc, rem, idxConst(G));
      Value n_idx = arith::DivUIOp::create(tb, loc, rem, idxConst(G));

      Value tileOriginH = arith::SubIOp::create(
          tb, loc, arith::MulIOp::create(tb, loc, ty, idxConst(m)), idxConst(padH_l));
      Value tileOriginW = arith::SubIOp::create(
          tb, loc, arith::MulIOp::create(tb, loc, tx, idxConst(m)), idxConst(padW_l));

      // Allocate private accumulators in computeType (f32 for f16 inputs)
      auto privAS = tb.getAttr<gpu::AddressSpaceAttr>(gpu::GPUDialect::getPrivateAddressSpace());
      auto accMemType = MemRefType::get({alphaSq}, computeType, AffineMap{}, privAS);
      Value accBuf = rock::GpuAllocOp::create(tb, loc, accMemType);
      for (int i = 0; i < alphaSq; i++)
        memref::StoreOp::create(tb, loc, zeroFp, accBuf, idxConst(i));

      // Channel loop
      scf::ForOp cLoop = scf::ForOp::create(tb, loc, idxConst(0), idxConst(C), idxConst(1));
      {
        OpBuilder lb(cLoop.getBody(), cLoop.getBody()->begin());
        Value c_idx = cLoop.getInductionVar();

        // Load 4x4 input tile as SSA values, promoting to computeType.
        // Interior tiles (fully in-bounds) use branchless direct loads.
        // Border tiles use per-element bounds checking with scf.if.
        Value ngc_base = arith::MulIOp::create(lb, loc,
            arith::AddIOp::create(lb, loc,
                arith::MulIOp::create(lb, loc, n_idx, idxConst(G)), g_idx),
            idxConst(C));
        ngc_base = arith::AddIOp::create(lb, loc, ngc_base, c_idx);
        ngc_base = arith::MulIOp::create(lb, loc, ngc_base, idxConst(inH));

        // Check if this is an interior tile (all 16 elements in-bounds)
        Value hOriginOk = arith::AndIOp::create(lb, loc,
            arith::CmpIOp::create(lb, loc, arith::CmpIPredicate::sge,
                                  tileOriginH, idxConst(0)),
            arith::CmpIOp::create(lb, loc, arith::CmpIPredicate::sle,
                                  tileOriginH, idxConst(inH - alpha)));
        Value wOriginOk = arith::AndIOp::create(lb, loc,
            arith::CmpIOp::create(lb, loc, arith::CmpIPredicate::sge,
                                  tileOriginW, idxConst(0)),
            arith::CmpIOp::create(lb, loc, arith::CmpIPredicate::sle,
                                  tileOriginW, idxConst(inW - alpha)));
        Value isInterior = arith::AndIOp::create(lb, loc, hOriginOk, wOriginOk);

        SmallVector<Type> d16Types(alphaSq, computeType);
        auto tileIf = scf::IfOp::create(lb, loc, d16Types, isInterior, true);

        // Interior path: branchless direct loads
        {
          OpBuilder ib = tileIf.getThenBodyBuilder();
          SmallVector<Value> dVals;
          for (int ih = 0; ih < alpha; ih++) {
            for (int iw = 0; iw < alpha; iw++) {
              Value hPos = arith::AddIOp::create(ib, loc, tileOriginH,
                                                 idxConst(ih));
              Value wPos = arith::AddIOp::create(ib, loc, tileOriginW,
                                                 idxConst(iw));
              Value flatIdx = arith::AddIOp::create(ib, loc,
                  arith::MulIOp::create(ib, loc,
                      arith::AddIOp::create(ib, loc, ngc_base, hPos),
                      idxConst(inW)),
                  wPos);
              dVals.push_back(promote(ib,
                  memref::LoadOp::create(ib, loc, input, flatIdx)));
            }
          }
          scf::YieldOp::create(ib, loc, dVals);
        }

        // Border path: per-element bounds checking
        {
          OpBuilder eb = tileIf.getElseBodyBuilder();
          SmallVector<Value> dVals;
          for (int ih = 0; ih < alpha; ih++) {
            for (int iw = 0; iw < alpha; iw++) {
              Value hPos = arith::AddIOp::create(eb, loc, tileOriginH,
                                                 idxConst(ih));
              Value wPos = arith::AddIOp::create(eb, loc, tileOriginW,
                                                 idxConst(iw));
              Value hOk = arith::AndIOp::create(eb, loc,
                  arith::CmpIOp::create(eb, loc, arith::CmpIPredicate::sge,
                                        hPos, idxConst(0)),
                  arith::CmpIOp::create(eb, loc, arith::CmpIPredicate::slt,
                                        hPos, idxConst(inH)));
              Value wOk = arith::AndIOp::create(eb, loc,
                  arith::CmpIOp::create(eb, loc, arith::CmpIPredicate::sge,
                                        wPos, idxConst(0)),
                  arith::CmpIOp::create(eb, loc, arith::CmpIPredicate::slt,
                                        wPos, idxConst(inW)));
              Value ok = arith::AndIOp::create(eb, loc, hOk, wOk);

              auto ldIf = scf::IfOp::create(eb, loc, TypeRange{computeType},
                                             ok, true);
              {
                OpBuilder thenB = ldIf.getThenBodyBuilder();
                Value flatIdx = arith::AddIOp::create(thenB, loc,
                    arith::MulIOp::create(thenB, loc,
                        arith::AddIOp::create(thenB, loc, ngc_base, hPos),
                        idxConst(inW)),
                    wPos);
                Value val = promote(thenB,
                    memref::LoadOp::create(thenB, loc, input, flatIdx));
                scf::YieldOp::create(thenB, loc, ValueRange{val});
              }
              {
                OpBuilder elseB2 = ldIf.getElseBodyBuilder();
                scf::YieldOp::create(elseB2, loc, ValueRange{zeroFp});
              }
              dVals.push_back(ldIf.getResult(0));
            }
          }
          scf::YieldOp::create(eb, loc, dVals);
        }

        Value d[16];
        for (int i = 0; i < alphaSq; i++)
          d[i] = tileIf.getResult(i);

        // Input transform: V = BT * d * B as direct SSA for F(2,3).
        // BT and B contain only {0, 1, -1}, so all ops are add/sub.
        Value v[16];
        // Row 0: BT row 0 = [1, 0, -1, 0] applied to columns of d
        // Then B applied to result columns
        // V[0][0] = d0 - d8 + (d10 - d2)
        // etc. -- closed-form from BT * d * B
        Value d0md8  = arith::SubFOp::create(lb, loc, d[0], d[8]);
        Value d10md2 = arith::SubFOp::create(lb, loc, d[10], d[2]);
        Value d1md9  = arith::SubFOp::create(lb, loc, d[1], d[9]);
        Value d5md9  = arith::SubFOp::create(lb, loc, d[5], d[9]);
        Value d10pd6 = arith::AddFOp::create(lb, loc, d[10], d[6]);
        Value d10md6 = arith::SubFOp::create(lb, loc, d[10], d[6]);
        Value d6md14 = arith::SubFOp::create(lb, loc, d[6], d[14]);
        Value d5md13 = arith::SubFOp::create(lb, loc, d[5], d[13]);

        // V[0][j]: BT row [1,0,-1,0] -> uses d rows 0 and 2
        v[0]  = arith::AddFOp::create(lb, loc, d0md8, d10md2);
        v[1]  = arith::AddFOp::create(lb, loc, d1md9,
            arith::SubFOp::create(lb, loc, d[2], d[10]));
        v[2]  = arith::SubFOp::create(lb, loc,
            arith::SubFOp::create(lb, loc, d[10], d[2]),
            d1md9);
        v[3]  = arith::AddFOp::create(lb, loc, d1md9,
            arith::SubFOp::create(lb, loc, d[11], d[3]));

        // V[1][j]: BT row [0,1,1,0] -> uses d rows 1 and 2
        v[4]  = arith::SubFOp::create(lb, loc,
            arith::SubFOp::create(lb, loc, d[8], d[4]), d10pd6);
        v[5]  = arith::AddFOp::create(lb, loc,
            arith::AddFOp::create(lb, loc, d[5], d[9]), d10pd6);
        v[6]  = arith::SubFOp::create(lb, loc, d10pd6,
            arith::AddFOp::create(lb, loc, d[5], d[9]));
        v[7]  = arith::SubFOp::create(lb, loc,
            arith::SubFOp::create(lb, loc, d[9], d[11]), d5md9);

        // V[2][j]: BT row [0,-1,1,0] -> uses d rows 1 and 2
        v[8]  = arith::AddFOp::create(lb, loc,
            arith::SubFOp::create(lb, loc, d[8], d[4]), d10md6);
        v[9]  = arith::SubFOp::create(lb, loc,
            arith::SubFOp::create(lb, loc, d[9], d[5]), d10md6);
        v[10] = arith::AddFOp::create(lb, loc, d5md9, d10md6);
        v[11] = arith::SubFOp::create(lb, loc,
            arith::SubFOp::create(lb, loc, d[7], d[11]), d5md9);

        // V[3][j]: BT row [0,1,0,-1] -> uses d rows 1 and 3
        v[12] = arith::SubFOp::create(lb, loc, d6md14,
            arith::SubFOp::create(lb, loc, d[4], d[12]));
        v[13] = arith::SubFOp::create(lb, loc, d5md13,
            arith::SubFOp::create(lb, loc, d[14], d[6]));
        v[14] = arith::AddFOp::create(lb, loc, d6md14, d5md13);
        v[15] = arith::AddFOp::create(lb, loc,
            arith::SubFOp::create(lb, loc, d[15], d[7]), d5md13);

        // Filter transform: G * g * G^T as direct SSA (no temp buffers).
        // Load 9 filter values, compute 16 U values via closed-form F(2,3).
        // Filter layout GKCYX: idx = (((g*K + k)*C + c)*r + fh)*r + fw
        Value gkc_base = arith::MulIOp::create(lb, loc,
            arith::AddIOp::create(lb, loc,
                arith::MulIOp::create(lb, loc, g_idx, idxConst(K)), k_idx),
            idxConst(C));
        gkc_base = arith::AddIOp::create(lb, loc, gkc_base, c_idx);
        gkc_base = arith::MulIOp::create(lb, loc, gkc_base, idxConst(r));

        Value f[9];
        for (int fh = 0; fh < r; fh++) {
          for (int fw = 0; fw < r; fw++) {
            Value fIdx = arith::AddIOp::create(lb, loc,
                arith::MulIOp::create(lb, loc,
                    arith::AddIOp::create(lb, loc, gkc_base, idxConst(fh)),
                    idxConst(r)),
                idxConst(fw));
            f[fh * r + fw] = promote(lb,
                memref::LoadOp::create(lb, loc, filter, fIdx));
          }
        }

        Value half = fpConst(0.5);
        Value quarter = fpConst(0.25);

        // Row sums and diffs: rs_i = f[i][0]+f[i][1]+f[i][2],
        //                     rd_i = f[i][0]-f[i][1]+f[i][2]
        Value rs0 = arith::AddFOp::create(lb, loc,
            arith::AddFOp::create(lb, loc, f[0], f[1]), f[2]);
        Value rd0 = arith::AddFOp::create(lb, loc,
            arith::SubFOp::create(lb, loc, f[0], f[1]), f[2]);
        Value rs1 = arith::AddFOp::create(lb, loc,
            arith::AddFOp::create(lb, loc, f[3], f[4]), f[5]);
        Value rd1 = arith::AddFOp::create(lb, loc,
            arith::SubFOp::create(lb, loc, f[3], f[4]), f[5]);
        Value rs2 = arith::AddFOp::create(lb, loc,
            arith::AddFOp::create(lb, loc, f[6], f[7]), f[8]);
        Value rd2 = arith::AddFOp::create(lb, loc,
            arith::SubFOp::create(lb, loc, f[6], f[7]), f[8]);

        // U[4x4] via G * filter * G^T closed-form for F(2,3)
        Value u[16];
        // Corners
        u[0]  = f[0];
        u[3]  = f[2];
        u[12] = f[6];
        u[15] = f[8];
        // Top/bottom edges
        u[1]  = arith::MulFOp::create(lb, loc, half, rs0);
        u[2]  = arith::MulFOp::create(lb, loc, half, rd0);
        u[13] = arith::MulFOp::create(lb, loc, half, rs2);
        u[14] = arith::MulFOp::create(lb, loc, half, rd2);
        // Left/right edges
        u[4]  = arith::MulFOp::create(lb, loc, half,
            arith::AddFOp::create(lb, loc,
                arith::AddFOp::create(lb, loc, f[0], f[3]), f[6]));
        u[8]  = arith::MulFOp::create(lb, loc, half,
            arith::AddFOp::create(lb, loc,
                arith::SubFOp::create(lb, loc, f[0], f[3]), f[6]));
        u[7]  = arith::MulFOp::create(lb, loc, half,
            arith::AddFOp::create(lb, loc,
                arith::AddFOp::create(lb, loc, f[2], f[5]), f[8]));
        u[11] = arith::MulFOp::create(lb, loc, half,
            arith::AddFOp::create(lb, loc,
                arith::SubFOp::create(lb, loc, f[2], f[5]), f[8]));
        // Center 2x2
        Value sa = arith::AddFOp::create(lb, loc,
            arith::AddFOp::create(lb, loc, rs0, rs1), rs2);
        Value sd = arith::AddFOp::create(lb, loc,
            arith::SubFOp::create(lb, loc, rs0, rs1), rs2);
        Value sb = arith::AddFOp::create(lb, loc,
            arith::AddFOp::create(lb, loc, rd0, rd1), rd2);
        Value sc = arith::AddFOp::create(lb, loc,
            arith::SubFOp::create(lb, loc, rd0, rd1), rd2);
        u[5]  = arith::MulFOp::create(lb, loc, quarter, sa);
        u[9]  = arith::MulFOp::create(lb, loc, quarter, sd);
        u[6]  = arith::MulFOp::create(lb, loc, quarter, sb);
        u[10] = arith::MulFOp::create(lb, loc, quarter, sc);

        // Element-wise MAC: acc[i] += U[i] * V[i]
        for (int i = 0; i < alphaSq; i++) {
          Value av = memref::LoadOp::create(lb, loc, accBuf, idxConst(i));
          Value prod = arith::MulFOp::create(lb, loc, u[i], v[i]);
          memref::StoreOp::create(lb, loc,
              arith::AddFOp::create(lb, loc, av, prod), accBuf, idxConst(i));
        }
      } // end channel loop

      // Output transform: AT * M * A (still in computeType)
      auto outTileMem = MemRefType::get({m * alpha}, computeType, AffineMap{}, privAS);
      Value t2Buf = rock::GpuAllocOp::create(tb, loc, outTileMem);

      for (int i = 0; i < m; i++) {
        for (int j = 0; j < alpha; j++) {
          Value sum = zeroFp;
          for (int k = 0; k < alpha; k++) {
            double at = winograd::AT_2_3[i * alpha + k];
            if (at == 0.0) continue;
            Value mv = memref::LoadOp::create(tb, loc, accBuf, idxConst(k * alpha + j));
            if (at == 1.0) sum = arith::AddFOp::create(tb, loc, sum, mv);
            else if (at == -1.0) sum = arith::SubFOp::create(tb, loc, sum, mv);
            else sum = arith::AddFOp::create(tb, loc, sum,
                arith::MulFOp::create(tb, loc, fpConst(at), mv));
          }
          memref::StoreOp::create(tb, loc, sum, t2Buf, idxConst(i * alpha + j));
        }
      }

      auto yMem = MemRefType::get({m * m}, computeType, AffineMap{}, privAS);
      Value yBuf = rock::GpuAllocOp::create(tb, loc, yMem);
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
          Value sum = zeroFp;
          for (int k = 0; k < alpha; k++) {
            double av = winograd::A_2_3[k * m + j];
            if (av == 0.0) continue;
            Value tv = memref::LoadOp::create(tb, loc, t2Buf, idxConst(i * alpha + k));
            if (av == 1.0) sum = arith::AddFOp::create(tb, loc, sum, tv);
            else if (av == -1.0) sum = arith::SubFOp::create(tb, loc, sum, tv);
            else sum = arith::AddFOp::create(tb, loc, sum,
                arith::MulFOp::create(tb, loc, fpConst(av), tv));
          }
          memref::StoreOp::create(tb, loc, sum, yBuf, idxConst(i * m + j));
        }
      }

      // Write output (flat NGKHW: idx = (((n*G + g)*K + k)*outH + oh)*outW + ow)
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
          Value oh = arith::AddIOp::create(tb, loc,
              arith::MulIOp::create(tb, loc, ty, idxConst(m)), idxConst(i));
          Value ow = arith::AddIOp::create(tb, loc,
              arith::MulIOp::create(tb, loc, tx, idxConst(m)), idxConst(j));

          Value ohOk = arith::CmpIOp::create(tb, loc, arith::CmpIPredicate::slt, oh, idxConst(outH));
          Value owOk = arith::CmpIOp::create(tb, loc, arith::CmpIPredicate::slt, ow, idxConst(outW));
          Value outOk = arith::AndIOp::create(tb, loc, ohOk, owOk);

          scf::IfOp stIf = scf::IfOp::create(tb, loc, outOk, false);
          {
            OpBuilder sb = stIf.getThenBodyBuilder();
            Value yval = memref::LoadOp::create(sb, loc, yBuf, idxConst(i * m + j));
            yval = demote(sb, yval);
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
