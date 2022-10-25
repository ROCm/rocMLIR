//===- RockTuningImpl.cpp - tuning API implementation ----*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file implements the tuning interfaces
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/RockTuning.h"

using namespace mlir;
using namespace mlir::rock;


// Brute-force search in incremental order
void createGemmTuningRangeBF(struct rock::TunableParams *newSpace,
                             rock::RockGemmWrapperInterface gemmOp) {

  OpBuilder b(gemmOp.getContext());
  rock::GemmFeatures currentFeatures = gemmOp.getGemmFeatures();
  if (bitEnumContainsAll(currentFeatures, rock::GemmFeatures::mfma)) {
    // XDLOPS
    // M/block N/block K/block M/wave N/wave kPack aCopyMore bCopyMore
    constexpr std::vector<std::vector<uint32_t>,6> tParams = {{4, 8, 16, 32, 64, 128},
                                                        {16, 32, 64, 128},
                                                        {16, 32, 64, 128},
                                                        {16, 32, 64},
                                                        {16, 32, 64},
                                                        {1, 4}};

    for (uint32_t gemmMPerBlock : tParams[0]) {
      for (uint32_t gemmNPerBlock : tParams[1]) {
        for (uint32_t gemmKPerBlock : tParams[2]) {
          for (uint32_t gemmMPerWave : tParams[3]) {
            for (uint32_t gemmNPerWave : tParams[4]) {
              for (uint32_t gemmKPack : tParams[5]) {
                Attribute gemmParams = b.getAttr<rock::XdlopsGemmParamsAttr>(
                    gemmKPerBlock, gemmMPerBlock, gemmNPerBlock, gemmKPack,
                    gemmMPerWave, gemmNPerWave);
                newSpace->tuningRange.push_back(gemmParams);
              }
            }
          }
        }
      }
    }
  } else {
    // Non-XDLOPS
    // M/block N/block K/block M/thread N/thread
    constexpr std::vector<std::vector<uint32_t>, 5> tParams = {
        {32, 64, 128}, {32, 64, 128}, {4, 8, 16}, {2, 4}, {2, 4}};
    for (uint32_t gemmMPerBlock : tParams[0]) {
      for (uint32_t gemmNPerBlock : tParams[1]) {
        for (uint32_t gemmKPerBlock : tParams[2]) {
          for (uint32_t gemmMPerThread : tParams[3]) {
            for (uint32_t gemmNPerThread : tParams[4]) {
              Attribute gemmParams = b.getAttr<rock::GeneralGemmParamsAttr>(
                  gemmKPerBlock, gemmMPerBlock, gemmNPerBlock,
                  /*kPerThread=*/1, gemmMPerThread, gemmNPerThread,
                  /*kpack=*/1);
              newSpace->tuningRange.push_back(gemmParams);
            }
          }
        }
      }
    }
  }

  newSpace->numHeuristicQuick = 0;
}

/*
void createConv2DFwdTuningRangeMLR(struct rock::TunableParams *newSpace,
                                   rock::Conv2DOp convOp) {
  // Initial point determined by Multi Linear Regression coefficients and
  // iterate further on.

  // These needs to have separate analysis for their own coefficient.
  // For now, NCHW/F32/FWD only.
  uint64_t layout = 0; // NCHW
  uint64_t data_type;
  uint64_t direction;

  uint64_t in_channels;
  uint64_t in_h;
  uint64_t in_w;
  uint64_t fil_h;
  uint64_t fil_w;
  uint64_t out_channels;
  uint64_t batchsize;
  uint64_t group_count;

  auto input = convOp->getOperand(0);
  auto weight = convOp->getOperand(1);
  ShapedType inTy = input.getType().cast<ShapedType>();
  auto inShape = inTy.getShape();
  batchsize = inShape[0];

  ShapedType wtTy = weight.getType().cast<ShapedType>();
  auto wtShape = wtTy.getShape();
  out_channels = wtShape[0];

  if (layout == 0) { // NCHW
    in_channels = inShape[1];
    in_h = inShape[2];
    in_w = inShape[3];
    fil_h = wtShape[2];
    fil_w = wtShape[3];
  } else { // NHWC
    in_channels = inShape[3];
    in_h = inShape[2];
    in_w = inShape[3];
    fil_h = wtShape[2];
    fil_w = wtShape[3];
  }
  group_count = inShape[4];

  rock::ConvolutionContext ctx = rock::populateConvContext(convOp);
  int64_t leftPadH = ctx.getPaddingVal()[0];
  int64_t leftPadW = ctx.getPaddingVal()[2];
  int64_t rightPadH = ctx.getPaddingVal()[1];
  int64_t rightPadW = ctx.getPaddingVal()[3];

  int64_t dilationH = ctx.getDilationVal()[0];
  int64_t dilationW = ctx.getDilationVal()[1];
  int64_t strideH = ctx.getStrideVal()[0];
  int64_t strideW = ctx.getStrideVal()[1];

  std::vector<std::vector<uint32_t>> tParams = {{4, 8, 16, 32, 64, 128},
                                                {16, 32, 64, 128},
                                                {16, 32, 64, 128},
                                                {16, 32, 64},
                                                {16, 32, 64},
                                                {1, 4},
                                                {0, 1},
                                                {0, 1}};
  // std::vector<uint32_t> p7b = {1, 2, 4, 8};

  // Each field of tuning parameter needs one set of coefficient + 1 constant.
  // ((16+1)*8) number of quick set can be also set by MLR

  // clang-format off
  // From gfx90878_1.1.0.udb tuned at 9.Oct.2022  Weekly CI xdlops
  const std::vector<std::vector<float>> coeffs = {
  //  in_ch ;   in_h;   in_w;  fil_h;  fil_w; out_ch;  batch;  lPadH;  lPadW;
rPadH;  rPadW;   strH;   strW;   dilH;   dilW;  group;  const; {
0.0175,-1.1229,-1.1229,-12.153,-12.153,-0.0413,
0.0000,47.5203,47.5203,47.5203,47.5203,55.0025,55.0025, 0.0000, 0.0000, 0.0000,
108.5339}, { -0.014, 0.0725, 0.0725,-4.2654,-4.2654,   0.01,
0,-3.6256,-3.6256,-3.6256,-3.6256,-8.0965,-8.0965,      0,      0,      0,
150.3692}, { 0.0001, 0.0179, 0.0179, 0.0814, 0.0814, 0.0004,
0,-0.0223,-0.0223,-0.0223,-0.0223,-0.4386,-0.4386,      0,      0, 0, 3.866194},
    { 0.0045, -0.132, -0.132,-0.7859,-0.7859, 0.0063,
0,-0.5036,-0.5036,-0.5036,-0.5036,  3.624,  3.624,      0,      0, 0, 70.81350},
    { -0.009,-0.1434,-0.1434,-4.4541,-4.4541, 0.0004,
0, 9.2506, 9.2506, 9.2506, 9.2506, 9.3834, 9.3834,      0,      0, 0, 71.10590},
    {      0,      0,      0,      0,      0,      0,      0,      0,      0, 0,
0,      0,      0,      0,      0,      0,      4.0}, {      0,      0,      0,
0,      0,      0,      0,      0,      0,      0,      0,      0,      0, 0, 0,
0,      1.0}, {      0,      0,      0,      0,      0,      0,      0,      0,
0,      0,      0,      0,      0,      0,      0,      0,      1.0}};
  // clang-format on

  std::vector<float> variables = {
      (float)in_channels, (float)in_h,        (float)in_w,
      (float)fil_h,       (float)fil_w,       (float)out_channels,
      (float)batchsize,   (float)group_count, (float)leftPadH,
      (float)leftPadW,    (float)rightPadH,   (float)rightPadW,
      (float)strideH,     (float)strideW,     (float)dilationH,
      (float)dilationW};

  // calculate prediction of each tuning parameter
  std::vector<float> prediction;
  uint32_t cLen = coeffs.size();
  uint32_t vLen = variables.size();
  for (int j = 0; j < cLen; j++) {
    float pred = 0;
    for (int i = 0; i < vLen; i++) {
      pred += coeffs[j][i] * variables[i];
    }
    pred += coeffs[j][vLen]; // equals to coeffs[*].size - 1
    prediction.push_back(pred);
  }

  // sort each parameter in distance order to the predition
  for (int i = 0; i < prediction.size(); i++) {
    std::sort(tParams[i].begin(), tParams[i].end(),
              [&](uint32_t lhs, uint32_t rhs) {
                float lDis = (float)lhs - prediction[i];
                lDis *= lDis;
                float rDis = (float)rhs - prediction[i];
                rDis *= rDis;
                return lDis < rDis;
              });
  }

  // create space with sorted vectors
  for (uint32_t t0 : tParams[0]) {
    for (uint32_t t1 : tParams[1]) {
      for (uint32_t t2 : tParams[2]) {
        for (uint32_t t3 : tParams[3]) {
          for (uint32_t t4 : tParams[4]) {
            for (uint32_t t5 : tParams[5]) {
              for (uint32_t t6 : tParams[6]) {
                for (uint32_t t7 : tParams[7]) {
                  std::vector<uint32_t> tParam = {t0, t1, t2, t3,
                                                  t4, t5, t6, t7};
                  newSpace->tuningRange.push_back(tParam);
                }
              }
            }
          }
        }
      }
    }
  }
  newSpace->numHeuristicQuick = 1;
}
*/

struct rock::TunableParams *
createTunableParams(rock::RockGemmWrapperInterface op) {
  struct rock::TunableParams *newSpace;
  newSpace = new rock::TunableParams();
  newSpace->primaryOpType =  op.getKernelType();
  // create range and heuristic
  createGemmTuningRangeBF(newSpace, op);
  return newSpace;
}

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//////--------------- FIX! --------- Will be relocaetd  to
/// rockTuningImpl.cpp----------///////////

//////// This one to gemm wrapper interfasce
// Stringfy given param
std::string toPerfConfig(Attribute param) {
  std::string result;
  if (auto paramXAttr = param.dyn_cast<rock::XdlopsGemmParamsAttr>()) {
    result.append(std::to_string(paramXAttr.getMPerBlock()));
    result.append(",");
    result.append(std::to_string(paramXAttr.getNPerBlock()));
    result.append(",");
    result.append(std::to_string(paramXAttr.getKPerBlock()));
    result.append(",");
    result.append(std::to_string(paramXAttr.getMPerWave()));
    result.append(",");
    result.append(std::to_string(paramXAttr.getNPerWave()));
  } else {
    auto paramAttr = param.dyn_cast<rock::GeneralGemmParamsAttr>();
    result.append(std::to_string(paramAttr.getMPerBlock()));
    result.append(",");
    result.append(std::to_string(paramAttr.getNPerBlock()));
    result.append(",");
    result.append(std::to_string(paramAttr.getKPerBlock()));
    result.append(",");
    result.append(std::to_string(paramAttr.getMPerThread()));
    result.append(",");
    result.append(std::to_string(paramAttr.getNPerThread()));
  }
  return result;
}
