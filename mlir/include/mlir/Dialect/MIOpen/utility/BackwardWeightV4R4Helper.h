//===- BackwardWeightV4r4Helper.h - Utility routines for AffineMap
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provide utility routines to check AffineMap instances.
//
//===----------------------------------------------------------------------===//

#ifndef BACKWARD_WEIGHT_V4R4_HELPER_H
#define BACKWARD_WEIGHT_V4R4_HELPER_H

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MIOpen/AffineMapHelper.h"
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/MIOpen/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/MIOpen/XdlopsCodeSelection.h"
#include "mlir/Dialect/MIOpen/utility/common.hpp"
#include "mlir/Dialect/MIOpen/utility/math.hpp"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::miopen;

namespace mlir {
namespace miopen {

inline __int64_t calculateKBlockNum(__int64_t n, __int64_t ho, __int64_t wo) {
  __int64_t gemmK = n * ho * wo;
  __int64_t gemmKBlocks = 1;
  if (gemmK % 16 == 0) {
    auto lcm = math::lcm(ho * wo, (__int64_t)16);
    gemmKBlocks = std::min(gemmK / lcm, n);
  } else if (gemmK % 8 == 0) {
    auto comm = math::lcm(ho * wo, (__int64_t)8);
    gemmKBlocks = std::min(gemmK / comm, n);
  } else if (gemmK % 4 == 0) {
    auto comm = math::lcm(ho * wo, (__int64_t)4);
    gemmKBlocks = std::min(gemmK / comm, n);
  }
  // not more than n
  gemmKBlocks = std::min(n, gemmKBlocks);
  // not less than 1
  gemmKBlocks = std::max((__int64_t)1, gemmKBlocks);

  // llvm::errs() << "\n gemmKBlocks: " << gemmKBlocks << " gemmK: " << gemmK
  //               << " ho: " << ho << " wo: " << wo << "\n";
  return gemmKBlocks;
}
template <typename T>
LogicalResult backwardWeightAtomicAdd(T op, PatternRewriter &b,
                                      const ArgumentFields &fields) {
  auto loc = op.getLoc();
  auto gemmIdAttr = op->template getAttrOfType<IntegerAttr>("gemm_id");
  auto archAttr = op->template getAttrOfType<StringAttr>("arch");
  auto numCuAttr = op->template getAttrOfType<IntegerAttr>("num_cu");

  auto filterLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->template getAttrOfType<ArrayAttr>("input_layout");
  auto outputLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("output_layout");

  auto dilationsAttr = op->template getAttrOfType<ArrayAttr>("dilations");
  auto stridesAttr = op->template getAttrOfType<ArrayAttr>("strides");
  auto paddingAttr = op->template getAttrOfType<ArrayAttr>("padding");

  // Get shape of filter tensor.
  auto filterType = op.filter().getType().template cast<MemRefType>();
  auto filterShape = filterType.getShape();
  auto filterElementType = filterType.getElementType();

  // Get shape of input tensor.
  auto inputType = op.input().getType().template cast<MemRefType>();
  auto inputShape = inputType.getShape();
  auto inputElementType = inputType.getElementType();

  // Get shape of output tensor.
  auto outputType = op.output().getType().template cast<MemRefType>();
  auto outputShape = outputType.getShape();
  auto outputElementType = outputType.getElementType();

  // Obtain convolution parameters: padding / dialtion / stride.
  auto leftPadH =
      paddingAttr.getValue()[0].template cast<IntegerAttr>().getInt();
  auto leftPadW =
      paddingAttr.getValue()[2].template cast<IntegerAttr>().getInt();
  auto rightPadH =
      paddingAttr.getValue()[1].template cast<IntegerAttr>().getInt();
  auto rightPadW =
      paddingAttr.getValue()[3].template cast<IntegerAttr>().getInt();

  auto dilationH =
      dilationsAttr.getValue()[0].template cast<IntegerAttr>().getInt();
  auto dilationW =
      dilationsAttr.getValue()[1].template cast<IntegerAttr>().getInt();
  auto strideH =
      stridesAttr.getValue()[0].template cast<IntegerAttr>().getInt();
  auto strideW =
      stridesAttr.getValue()[1].template cast<IntegerAttr>().getInt();
  // get y, x, ho, wo, hi, wi
  int64_t g, n, k, c, y, x, ho, wo, hi, wi;
  g = n = k = c = y = x = ho = wo = hi = wi = 0;
  for (unsigned i = 0; i < filterLayoutAttr.size(); ++i) {
    auto filterAttr =
        filterLayoutAttr.getValue()[i].template cast<StringAttr>();
    auto inputAttr = inputLayoutAttr.getValue()[i].template cast<StringAttr>();
    auto outputAttr =
        outputLayoutAttr.getValue()[i].template cast<StringAttr>();

    if (filterAttr.getValue() == "g") {
      g = filterShape[i];
    } else if (filterAttr.getValue() == "k") {
      k = filterShape[i];
    } else if (filterAttr.getValue() == "c") {
      c = filterShape[i];
    } else if (filterAttr.getValue() == "y") {
      y = filterShape[i];
    } else if (filterAttr.getValue() == "x") {
      x = filterShape[i];
    }

    if (inputAttr.getValue() == "ni") {
      n = inputShape[i];
    } else if (inputAttr.getValue() == "hi") {
      hi = inputShape[i];
    } else if (inputAttr.getValue() == "wi") {
      wi = inputShape[i];
    }

    if (outputAttr.getValue() == "ho") {
      ho = outputShape[i];
    } else if (outputAttr.getValue() == "wo") {
      wo = outputShape[i];
    }
  }

  // compute padding hi/wi.
  auto hiPadded = hi + leftPadH + rightPadH;
  auto wiPadded = wi + leftPadW + rightPadW;
  // calculate gemmKBlocks

  // static const int64_t MaxSubBlockNum = 2048 / standardBockNum;
  int64_t gemmKBlocks = calculateKBlockNum(n, ho, wo);

  // Transform filter tensor.
  // set layout attribute.
  // Weight tensor transformation for Conv2DOp
  auto getGemmA = [&]() -> Value {
    // key to dim
    std::map<StringRef, int> filterKeyToDim;
    for (unsigned i = 0; i < filterLayoutAttr.size(); ++i) {
      if (auto strAttr =
              filterLayoutAttr.getValue()[i].template cast<StringAttr>()) {
        filterKeyToDim[strAttr.getValue()] = i;
      }
    }
    // add a dimension
    llvm::SmallVector<StringAttr> firtFilterDimName;
    auto getWeiGKblockKCYX = [&]() {
      llvm::SmallVector<StringAttr> &curOutputDimName = firtFilterDimName;
      llvm::SmallVector<int64_t, 6> transformedFilterShape;
      llvm::SmallVector<NamedAttribute, 3> transformedFilterAttrs;
      // g
      curOutputDimName.push_back(b.getStringAttr("g"));
      transformedFilterShape.push_back(g);
      llvm::SmallVector<NamedAttribute, 5> gDimAttr{
          b.getNamedAttr("upper_layer_dimensions",
                         b.getArrayAttr({b.getI32IntegerAttr(0)})),
          b.getNamedAttr("upper_layer_names",
                         b.getArrayAttr({curOutputDimName[0]})),
          b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
          b.getNamedAttr(
              "lower_layer_dimensions",
              b.getArrayAttr({b.getI32IntegerAttr(filterKeyToDim["g"])})),
          b.getNamedAttr("lower_layer_names",
                         b.getArrayAttr({b.getStringAttr("g")}))};

      // kblock
      curOutputDimName.push_back(b.getStringAttr("kblock"));
      transformedFilterShape.push_back(gemmKBlocks);
      llvm::SmallVector<NamedAttribute, 5> kblockDimAttr{
          b.getNamedAttr("upper_layer_dimensions",
                         b.getArrayAttr({b.getI32IntegerAttr(1)})),
          b.getNamedAttr("upper_layer_names",
                         b.getArrayAttr({curOutputDimName[1]})),
          b.getNamedAttr("transformation", b.getStringAttr("AddDim")),
          b.getNamedAttr("lower_layer_dimensions",
                         b.getArrayAttr({b.getI32IntegerAttr(-1)})),
          b.getNamedAttr("lower_layer_names",
                         b.getArrayAttr({b.getStringAttr("")}))};

      // k
      curOutputDimName.push_back(b.getStringAttr("k"));
      transformedFilterShape.push_back(k);
      llvm::SmallVector<NamedAttribute, 5> kDimAttr{
          b.getNamedAttr("upper_layer_dimensions",
                         b.getArrayAttr({b.getI32IntegerAttr(2)})),
          b.getNamedAttr("upper_layer_names",
                         b.getArrayAttr({curOutputDimName[2]})),
          b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
          b.getNamedAttr(
              "lower_layer_dimensions",
              b.getArrayAttr({b.getI32IntegerAttr(filterKeyToDim["k"])})),
          b.getNamedAttr("lower_layer_names",
                         b.getArrayAttr({b.getStringAttr("k")}))};

      // c
      curOutputDimName.push_back(b.getStringAttr("c"));
      transformedFilterShape.push_back(c);
      llvm::SmallVector<NamedAttribute, 5> cDimAttr{
          b.getNamedAttr("upper_layer_dimensions",
                         b.getArrayAttr({b.getI32IntegerAttr(3)})),
          b.getNamedAttr("upper_layer_names",
                         b.getArrayAttr({curOutputDimName[3]})),
          b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
          b.getNamedAttr(
              "lower_layer_dimensions",
              b.getArrayAttr({b.getI32IntegerAttr(filterKeyToDim["c"])})),
          b.getNamedAttr("lower_layer_names",
                         b.getArrayAttr({b.getStringAttr("c")}))};

      // y
      curOutputDimName.push_back(b.getStringAttr("y"));
      transformedFilterShape.push_back(y);
      llvm::SmallVector<NamedAttribute, 6> yDimAttr{
          b.getNamedAttr("upper_layer_dimensions",
                         b.getArrayAttr({b.getI32IntegerAttr(4)})),
          b.getNamedAttr("upper_layer_names",
                         b.getArrayAttr({curOutputDimName[4]})),
          b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
          b.getNamedAttr(
              "lower_layer_dimensions",
              b.getArrayAttr({b.getI32IntegerAttr(filterKeyToDim["y"])})),
          b.getNamedAttr("lower_layer_names",
                         b.getArrayAttr({b.getStringAttr("y")}))};

      // x
      curOutputDimName.push_back(b.getStringAttr("x"));
      transformedFilterShape.push_back(x);
      llvm::SmallVector<NamedAttribute, 6> xDimAttr{
          b.getNamedAttr("upper_layer_dimensions",
                         b.getArrayAttr({b.getI32IntegerAttr(5)})),
          b.getNamedAttr("upper_layer_names",
                         b.getArrayAttr({curOutputDimName[5]})),
          b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
          b.getNamedAttr(
              "lower_layer_dimensions",
              b.getArrayAttr({b.getI32IntegerAttr(filterKeyToDim["x"])})),
          b.getNamedAttr("lower_layer_names",
                         b.getArrayAttr({b.getStringAttr("x")}))};

      transformedFilterAttrs.push_back(b.getNamedAttr(
          "layout",
          b.getArrayAttr(
              {b.getDictionaryAttr(gDimAttr),
               b.getDictionaryAttr(kblockDimAttr),
               b.getDictionaryAttr(kDimAttr), b.getDictionaryAttr(cDimAttr),
               b.getDictionaryAttr(yDimAttr), b.getDictionaryAttr(xDimAttr)})));
      transformedFilterAttrs.push_back(b.getNamedAttr(
          "upper_layer_layout",
          b.getArrayAttr(ArrayRef<Attribute>(curOutputDimName.begin(),
                                             curOutputDimName.end()))));

      transformedFilterAttrs.push_back(
          b.getNamedAttr("lower_layer_layout", filterLayoutAttr));

      auto transformedFilterMemRefType =
          MemRefType::get(transformedFilterShape, filterElementType);
      // set lowest_layer attribute.
      transformedFilterAttrs.push_back(
          b.getNamedAttr("lowest_layer", b.getBoolAttr(true)));
      auto gemm = b.create<miopen::TransformOp>(
          loc, transformedFilterMemRefType, op.filter(), transformedFilterAttrs,
          /*populateBounds=*/true);
      return gemm;
    };

    auto weiGKblockKCYX = getWeiGKblockKCYX();

    // wei_gemmg_gemmm_gemmn
    auto getWeiGemmGGemmMGemmN =
        [&](llvm::SmallVector<StringAttr> &preOutputDimName) {
          llvm::SmallVector<StringAttr, 7> curOutputDimName;
          llvm::SmallVector<int64_t, 6> transformedFilterShape;
          llvm::SmallVector<NamedAttribute, 3> transformedFilterAttrs;
          // gemmG
          curOutputDimName.push_back(b.getStringAttr("gemmG"));
          transformedFilterShape.push_back(g * gemmKBlocks);
          llvm::SmallVector<NamedAttribute, 5> gemmGDimAttr{
              b.getNamedAttr("parameter", b.getI32IntegerAttr(gemmKBlocks)),
              b.getNamedAttr("upper_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(0)})),
              b.getNamedAttr("upper_layer_names",
                             b.getArrayAttr({curOutputDimName[0]})),
              b.getNamedAttr("transformation", b.getStringAttr("Merge")),
              b.getNamedAttr("lower_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(0),
                                             b.getI32IntegerAttr(1)})),
              b.getNamedAttr(
                  "lower_layer_names",
                  b.getArrayAttr({preOutputDimName[0], preOutputDimName[1]}))};

          // GemmM
          curOutputDimName.push_back(b.getStringAttr("gemmM"));
          transformedFilterShape.push_back(k);
          llvm::SmallVector<NamedAttribute, 5> gemmMDimAttr{
              b.getNamedAttr("upper_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(1)})),
              b.getNamedAttr("upper_layer_names",
                             b.getArrayAttr({curOutputDimName[1]})),
              b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
              b.getNamedAttr("lower_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(2)})),
              b.getNamedAttr("lower_layer_names",
                             b.getArrayAttr({preOutputDimName[2]}))};

          // GemmN
          curOutputDimName.push_back(b.getStringAttr("gemmN"));
          transformedFilterShape.push_back(c * y * x);
          llvm::SmallVector<NamedAttribute, 5> gemmNDimAttr{
              b.getNamedAttr("upper_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(2)})),
              b.getNamedAttr("upper_layer_names",
                             b.getArrayAttr({curOutputDimName[2]})),
              b.getNamedAttr("transformation", b.getStringAttr("Merge")),
              b.getNamedAttr("lower_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(3),
                                             b.getI32IntegerAttr(4),
                                             b.getI32IntegerAttr(5)})),
              b.getNamedAttr(
                  "lower_layer_names",
                  b.getArrayAttr({preOutputDimName[3], preOutputDimName[4],
                                  preOutputDimName[5]}))};

          transformedFilterAttrs.push_back(b.getNamedAttr(
              "layout", b.getArrayAttr({b.getDictionaryAttr(gemmGDimAttr),
                                        b.getDictionaryAttr(gemmMDimAttr),
                                        b.getDictionaryAttr(gemmNDimAttr)})));
          transformedFilterAttrs.push_back(b.getNamedAttr(
              "upper_layer_layout",
              b.getArrayAttr(ArrayRef<Attribute>(curOutputDimName.begin(),
                                                 curOutputDimName.end()))));

          transformedFilterAttrs.push_back(b.getNamedAttr(
              "lower_layer_layout",
              b.getArrayAttr(ArrayRef<Attribute>(preOutputDimName.begin(),
                                                 preOutputDimName.end()))));

          auto transformedFilterMemRefType =
              MemRefType::get(transformedFilterShape, filterElementType);
          // set lowest_layer attribute.
          transformedFilterAttrs.push_back(
              b.getNamedAttr("lowest_layer", b.getBoolAttr(true)));
          auto gemm = b.create<miopen::TransformOp>(
              loc, transformedFilterMemRefType, weiGKblockKCYX,
              transformedFilterAttrs, /*populateBounds=*/true);
          return gemm;
        };
    auto gemmA = getWeiGemmGGemmMGemmN(firtFilterDimName);
    return gemmA;
  };

  auto getGemmB = [&]() -> Value {
    // dim of oob check
    llvm::DenseSet<int> inputOobCheckDims;
    // key to dim
    std::map<StringRef, int> currentKeyToDim;
    for (unsigned i = 0; i < inputLayoutAttr.size(); ++i) {
      if (auto strAttr =
              inputLayoutAttr.getValue()[i].template cast<StringAttr>()) {
        currentKeyToDim[strAttr.getValue()] = i;
      }
    }
    llvm::SmallVector<StringAttr> firstOutputDimName;
    auto getInGN0N1CHipWip = [&]() {
      llvm::SmallVector<StringAttr> &curOutputDimName = firstOutputDimName;
      llvm::SmallVector<int64_t, 7> transformedShape;
      llvm::SmallVector<NamedAttribute, 3> transformedAttrs;
      // gi
      curOutputDimName.push_back(b.getStringAttr("gi"));
      transformedShape.push_back(g);
      llvm::SmallVector<NamedAttribute, 5> gDimAttr{
          b.getNamedAttr("upper_layer_dimensions",
                         b.getArrayAttr({b.getI32IntegerAttr(0)})),
          b.getNamedAttr("upper_layer_names",
                         b.getArrayAttr({curOutputDimName[0]})),
          b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
          b.getNamedAttr(
              "lower_layer_dimensions",
              b.getArrayAttr({b.getI32IntegerAttr(currentKeyToDim["gi"])})),
          b.getNamedAttr("lower_layer_names",
                         b.getArrayAttr({b.getStringAttr("gi")}))};
      // ni
      curOutputDimName.push_back(b.getStringAttr("n0"));
      transformedShape.push_back(gemmKBlocks);
      curOutputDimName.push_back(b.getStringAttr("n1"));
      transformedShape.push_back(n / gemmKBlocks);
      llvm::SmallVector<NamedAttribute, 5> nDimAttr{
          b.getNamedAttr(
              "upper_layer_dimensions",
              b.getArrayAttr({b.getI32IntegerAttr(1), b.getI32IntegerAttr(2)})),
          b.getNamedAttr(
              "upper_layer_names",
              b.getArrayAttr({curOutputDimName[1], curOutputDimName[2]})),
          b.getNamedAttr(
              "dimension_lengths",
              b.getArrayAttr({b.getI32IntegerAttr(gemmKBlocks),
                              b.getI32IntegerAttr(n / gemmKBlocks)})),
          b.getNamedAttr("transformation", b.getStringAttr("UnMerge")),
          b.getNamedAttr("parameters", b.getArrayAttr({
                                           b.getI32IntegerAttr(gemmKBlocks),
                                           b.getI32IntegerAttr(n / gemmKBlocks),
                                       })),
          b.getNamedAttr(
              "lower_layer_dimensions",
              b.getArrayAttr({b.getI32IntegerAttr(currentKeyToDim["ni"])})),
          b.getNamedAttr("lower_layer_names",
                         b.getArrayAttr({b.getStringAttr("ni")}))};
      // ci
      curOutputDimName.push_back(b.getStringAttr("ci"));
      transformedShape.push_back(c);
      llvm::SmallVector<NamedAttribute, 5> cDimAttr{
          b.getNamedAttr("upper_layer_dimensions",
                         b.getArrayAttr({b.getI32IntegerAttr(3)})),
          b.getNamedAttr("upper_layer_names",
                         b.getArrayAttr({curOutputDimName[2]})),
          b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
          b.getNamedAttr(
              "lower_layer_dimensions",
              b.getArrayAttr({b.getI32IntegerAttr(currentKeyToDim["ci"])})),
          b.getNamedAttr("lower_layer_names",
                         b.getArrayAttr({b.getStringAttr("ci")}))};

      // hip wip
      curOutputDimName.push_back(b.getStringAttr("hipad"));
      curOutputDimName.push_back(b.getStringAttr("wipad"));
      transformedShape.push_back(hiPadded);
      transformedShape.push_back(wiPadded);
      llvm::SmallVector<NamedAttribute, 6> hwpadDimAttr{
          b.getNamedAttr(
              "upper_layer_dimensions",
              b.getArrayAttr({b.getI32IntegerAttr(4), b.getI32IntegerAttr(5)})),
          b.getNamedAttr(
              "upper_layer_names",
              b.getArrayAttr({curOutputDimName[4], curOutputDimName[5]})),
          b.getNamedAttr("transformation", b.getStringAttr("Pad")),
          b.getNamedAttr("parameters", b.getArrayAttr({
                                           b.getI32IntegerAttr(leftPadH),
                                           b.getI32IntegerAttr(rightPadH),
                                           b.getI32IntegerAttr(leftPadW),
                                           b.getI32IntegerAttr(rightPadW),
                                       })),
          b.getNamedAttr(
              "lower_layer_dimensions",
              b.getArrayAttr({b.getI32IntegerAttr(currentKeyToDim["hi"]),
                              b.getI32IntegerAttr(currentKeyToDim["wi"])})),
          b.getNamedAttr(
              "lower_layer_names",
              b.getArrayAttr({b.getStringAttr("hi"), b.getStringAttr("wi")}))};
      auto isInputHipBoundCheck = [&]() {
        // if pad = 0 , not need oob check
        if (leftPadH == 0 && rightPadH == 0 && leftPadW == 0 && rightPadW == 0)
          return false;
        return true;
      };
      if (isInputHipBoundCheck()) {
        llvm::SmallVector<IntegerAttr, 2> padDim;
        if (leftPadH || rightPadH) {
          inputOobCheckDims.insert(currentKeyToDim["hi"]);
        }
        if (leftPadW || rightPadW) {
          inputOobCheckDims.insert(currentKeyToDim["wi"]);
        }
      }
      transformedAttrs.push_back(b.getNamedAttr(
          "layout", b.getArrayAttr({b.getDictionaryAttr(gDimAttr),
                                    b.getDictionaryAttr(nDimAttr),
                                    b.getDictionaryAttr(cDimAttr),
                                    b.getDictionaryAttr(hwpadDimAttr)})));
      transformedAttrs.push_back(b.getNamedAttr(
          "upper_layer_layout",
          b.getArrayAttr(ArrayRef<Attribute>(curOutputDimName.begin(),
                                             curOutputDimName.end()))));

      transformedAttrs.push_back(
          b.getNamedAttr("lower_layer_layout", inputLayoutAttr));

      auto transformedMemRefType =
          MemRefType::get(transformedShape, inputElementType);
      // set lowest_layer attribute.
      transformedAttrs.push_back(
          b.getNamedAttr("lowest_layer", b.getBoolAttr(true)));
      auto gemm = b.create<miopen::TransformOp>(loc, transformedMemRefType,
                                                op.input(), transformedAttrs,
                                                /*populateBounds=*/true);
      return gemm;
    };
    auto inGN0N1CHipWip = getInGN0N1CHipWip();

    llvm::SmallVector<StringAttr> secondOutputDimName;
    auto getInGN0N1CYHoXWo =
        [&](llvm::SmallVector<StringAttr> &preOutputDimName,
            llvm::SmallVector<StringAttr> &curOutputDimName) {
          llvm::SmallVector<int64_t, 7> transformedShape;
          llvm::SmallVector<NamedAttribute, 3> transformedAttrs;
          // g
          curOutputDimName.push_back(b.getStringAttr("gi"));
          transformedShape.push_back(g);
          llvm::SmallVector<NamedAttribute, 5> gDimAttr{
              b.getNamedAttr("upper_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(0)})),
              b.getNamedAttr("upper_layer_names",
                             b.getArrayAttr({curOutputDimName[0]})),
              b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
              b.getNamedAttr("lower_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(0)})),
              b.getNamedAttr("lower_layer_names",
                             b.getArrayAttr({preOutputDimName[0]}))};
          // n0
          curOutputDimName.push_back(b.getStringAttr("n0"));
          transformedShape.push_back(gemmKBlocks);
          llvm::SmallVector<NamedAttribute, 5> n0DimAttr{
              b.getNamedAttr("upper_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(1)})),
              b.getNamedAttr("upper_layer_names",
                             b.getArrayAttr({curOutputDimName[1]})),
              b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
              b.getNamedAttr("lower_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(1)})),
              b.getNamedAttr("lower_layer_names",
                             b.getArrayAttr({preOutputDimName[1]}))};
          // n1
          curOutputDimName.push_back(b.getStringAttr("n1"));
          transformedShape.push_back(n / gemmKBlocks);
          llvm::SmallVector<NamedAttribute, 5> n1DimAttr{
              b.getNamedAttr("upper_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(2)})),
              b.getNamedAttr("upper_layer_names",
                             b.getArrayAttr({curOutputDimName[2]})),
              b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
              b.getNamedAttr("lower_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(2)})),
              b.getNamedAttr("lower_layer_names",
                             b.getArrayAttr({preOutputDimName[2]}))};

          // c
          curOutputDimName.push_back(b.getStringAttr("ci"));
          transformedShape.push_back(c);
          llvm::SmallVector<NamedAttribute, 5> cDimAttr{
              b.getNamedAttr("upper_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(3)})),
              b.getNamedAttr("upper_layer_names",
                             b.getArrayAttr({curOutputDimName[3]})),
              b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
              b.getNamedAttr("lower_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(3)})),
              b.getNamedAttr("lower_layer_names",
                             b.getArrayAttr({preOutputDimName[3]}))};

          // hi
          curOutputDimName.push_back(b.getStringAttr("y"));
          curOutputDimName.push_back(b.getStringAttr("ho"));
          transformedShape.push_back(y);
          transformedShape.push_back(ho);
          llvm::SmallVector<NamedAttribute, 6> hiDimAttr{
              b.getNamedAttr("upper_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(4),
                                             b.getI32IntegerAttr(5)})),
              b.getNamedAttr(
                  "upper_layer_names",
                  b.getArrayAttr({curOutputDimName[4], curOutputDimName[5]})),
              b.getNamedAttr("transformation", b.getStringAttr("Embed")),
              b.getNamedAttr("parameters", b.getArrayAttr({
                                               b.getI32IntegerAttr(dilationH),
                                               b.getI32IntegerAttr(strideH),
                                           })),
              b.getNamedAttr("lower_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(4)})),
              b.getNamedAttr("lower_layer_names",
                             b.getArrayAttr({preOutputDimName[4]}))};

          // wi
          curOutputDimName.push_back(b.getStringAttr("x"));
          curOutputDimName.push_back(b.getStringAttr("wo"));
          transformedShape.push_back(x);
          transformedShape.push_back(wo);
          llvm::SmallVector<NamedAttribute, 6> wiDimAttr{
              b.getNamedAttr("upper_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(6),
                                             b.getI32IntegerAttr(7)})),
              b.getNamedAttr(
                  "upper_layer_names",
                  b.getArrayAttr({curOutputDimName[6], curOutputDimName[7]})),
              b.getNamedAttr("transformation", b.getStringAttr("Embed")),
              b.getNamedAttr("parameters", b.getArrayAttr({
                                               b.getI32IntegerAttr(dilationW),
                                               b.getI32IntegerAttr(strideW),
                                           })),
              b.getNamedAttr("lower_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(5)})),
              b.getNamedAttr("lower_layer_names",
                             b.getArrayAttr({preOutputDimName[5]}))};

          transformedAttrs.push_back(b.getNamedAttr(
              "layout", b.getArrayAttr({b.getDictionaryAttr(gDimAttr),
                                        b.getDictionaryAttr(n0DimAttr),
                                        b.getDictionaryAttr(n1DimAttr),
                                        b.getDictionaryAttr(cDimAttr),
                                        b.getDictionaryAttr(hiDimAttr),
                                        b.getDictionaryAttr(wiDimAttr)})));
          transformedAttrs.push_back(b.getNamedAttr(
              "upper_layer_layout",
              b.getArrayAttr(ArrayRef<Attribute>(curOutputDimName.begin(),
                                                 curOutputDimName.end()))));

          transformedAttrs.push_back(b.getNamedAttr(
              "lower_layer_layout",
              b.getArrayAttr(ArrayRef<Attribute>(preOutputDimName.begin(),
                                                 preOutputDimName.end()))));

          auto transformedFilterMemRefType =
              MemRefType::get(transformedShape, inputElementType);
          auto gemm =
              b.create<miopen::TransformOp>(loc, transformedFilterMemRefType,
                                            inGN0N1CHipWip, transformedAttrs,
                                            /*populateBounds=*/true);
          return gemm;
        };

    auto inGN0N1CYHoXWo =
        getInGN0N1CYHoXWo(firstOutputDimName, secondOutputDimName);

    auto getInGemmGGemmKGemmN =
        [&](llvm::SmallVector<StringAttr> &preOutputDimName) {
          llvm::SmallVector<StringAttr, 7> curOutputDimName;
          llvm::SmallVector<int64_t, 7> transformedShape;
          // gemmG
          curOutputDimName.push_back(b.getStringAttr("gemmG"));
          transformedShape.push_back(g * gemmKBlocks);
          llvm::SmallVector<NamedAttribute, 5> gemmGDimAttr{
              b.getNamedAttr("upper_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(0)})),
              b.getNamedAttr("upper_layer_names",
                             b.getArrayAttr({curOutputDimName[0]})),
              b.getNamedAttr("transformation", b.getStringAttr("Merge")),
              b.getNamedAttr("lower_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(0),
                                             b.getI32IntegerAttr(1)})),
              b.getNamedAttr(
                  "lower_layer_names",
                  b.getArrayAttr({preOutputDimName[0], preOutputDimName[1]}))};

          // gemmK
          curOutputDimName.push_back(b.getStringAttr("gemmK"));
          transformedShape.push_back(n / gemmKBlocks * ho * wo);
          llvm::SmallVector<NamedAttribute, 5> gemmKDimAttr{
              b.getNamedAttr("upper_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(1)})),
              b.getNamedAttr("upper_layer_names",
                             b.getArrayAttr({curOutputDimName[1]})),
              b.getNamedAttr("transformation", b.getStringAttr("Merge")),
              b.getNamedAttr("lower_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(2),
                                             b.getI32IntegerAttr(5),
                                             b.getI32IntegerAttr(7)})),
              b.getNamedAttr(
                  "lower_layer_names",
                  b.getArrayAttr({preOutputDimName[2], preOutputDimName[5],
                                  preOutputDimName[7]}))};

          // gemmN
          curOutputDimName.push_back(b.getStringAttr("gemmN"));
          transformedShape.push_back(c * y * x);
          llvm::SmallVector<NamedAttribute, 5> gemmNDimAttr{
              b.getNamedAttr("upper_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(2)})),
              b.getNamedAttr("upper_layer_names",
                             b.getArrayAttr({curOutputDimName[2]})),
              b.getNamedAttr("transformation", b.getStringAttr("Merge")),
              b.getNamedAttr("lower_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(3),
                                             b.getI32IntegerAttr(4),
                                             b.getI32IntegerAttr(6)})),
              b.getNamedAttr(
                  "lower_layer_names",
                  b.getArrayAttr({preOutputDimName[3], preOutputDimName[4],
                                  preOutputDimName[6]}))};

          llvm::SmallVector<NamedAttribute, 3> transformedAttrs;
          transformedAttrs.push_back(b.getNamedAttr(
              "layout", b.getArrayAttr({b.getDictionaryAttr(gemmGDimAttr),
                                        b.getDictionaryAttr(gemmKDimAttr),
                                        b.getDictionaryAttr(gemmNDimAttr)})));
          transformedAttrs.push_back(b.getNamedAttr(
              "upper_layer_layout",
              b.getArrayAttr(ArrayRef<Attribute>(curOutputDimName.begin(),
                                                 curOutputDimName.end()))));

          transformedAttrs.push_back(b.getNamedAttr(
              "lower_layer_layout",
              b.getArrayAttr(ArrayRef<Attribute>(preOutputDimName.begin(),
                                                 preOutputDimName.end()))));

          transformedAttrs.push_back(b.getNamedAttr(
              "gridwise_gemm_argument_position", b.getI32IntegerAttr(2)));

          if (inputOobCheckDims.size()) {
            llvm::SmallVector<IntegerAttr, 5> boundDims;
            for (size_t i = 0; i < inputShape.size(); i++) {
              if (inputOobCheckDims.find(i) != inputOobCheckDims.end())
                boundDims.push_back(b.getI32IntegerAttr(1));
              else
                boundDims.push_back(b.getI32IntegerAttr(0));
            }
            transformedAttrs.push_back(b.getNamedAttr(
                "bound_check",
                b.getArrayAttr({boundDims.begin(), boundDims.end()})));
          }

          auto transformedMemRefType =
              MemRefType::get(transformedShape, inputElementType);
          auto gemm = b.create<miopen::TransformOp>(
              loc, transformedMemRefType, inGN0N1CYHoXWo, transformedAttrs,
              /*populateBounds=*/true);
          return gemm;
        };
    auto inGemmGGemmKGemmN = getInGemmGGemmKGemmN(secondOutputDimName);

    return inGemmGGemmKGemmN;
  };

  auto getGemmC = [&]() -> Value {
    // dim of oob check
    llvm::DenseSet<int> outputOobCheckDims;
    // key to dim
    std::map<StringRef, int> currentKeyToDim;
    for (unsigned i = 0; i < outputLayoutAttr.size(); ++i) {
      if (auto strAttr =
              outputLayoutAttr.getValue()[i].template cast<StringAttr>()) {
        currentKeyToDim[strAttr.getValue()] = i;
      }
    }

    llvm::SmallVector<StringAttr> firstOutputDimName;
    auto getOutGN0N1KHoWo = [&]() {
      llvm::SmallVector<StringAttr> &curOutputDimName = firstOutputDimName;
      llvm::SmallVector<int64_t, 7> transformedShape;
      llvm::SmallVector<NamedAttribute, 3> transformedAttrs;
      // go
      curOutputDimName.push_back(b.getStringAttr("go"));
      transformedShape.push_back(g);
      llvm::SmallVector<NamedAttribute, 5> gDimAttr{
          b.getNamedAttr("upper_layer_dimensions",
                         b.getArrayAttr({b.getI32IntegerAttr(0)})),
          b.getNamedAttr("upper_layer_names",
                         b.getArrayAttr({curOutputDimName[0]})),
          b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
          b.getNamedAttr(
              "lower_layer_dimensions",
              b.getArrayAttr({b.getI32IntegerAttr(currentKeyToDim["go"])})),
          b.getNamedAttr("lower_layer_names",
                         b.getArrayAttr({b.getStringAttr("go")}))};
      // no
      curOutputDimName.push_back(b.getStringAttr("n0"));
      transformedShape.push_back(gemmKBlocks);
      curOutputDimName.push_back(b.getStringAttr("n1"));
      transformedShape.push_back(n / gemmKBlocks);
      llvm::SmallVector<NamedAttribute, 5> nDimAttr{
          b.getNamedAttr(
              "upper_layer_dimensions",
              b.getArrayAttr({b.getI32IntegerAttr(1), b.getI32IntegerAttr(2)})),
          b.getNamedAttr(
              "upper_layer_names",
              b.getArrayAttr({curOutputDimName[1], curOutputDimName[2]})),
          b.getNamedAttr(
              "dimension_lengths",
              b.getArrayAttr({b.getI32IntegerAttr(gemmKBlocks),
                              b.getI32IntegerAttr(n / gemmKBlocks)})),
          b.getNamedAttr("transformation", b.getStringAttr("UnMerge")),
          b.getNamedAttr("parameters", b.getArrayAttr({
                                           b.getI32IntegerAttr(gemmKBlocks),
                                           b.getI32IntegerAttr(n / gemmKBlocks),
                                       })),
          b.getNamedAttr(
              "lower_layer_dimensions",
              b.getArrayAttr({b.getI32IntegerAttr(currentKeyToDim["no"])})),
          b.getNamedAttr("lower_layer_names",
                         b.getArrayAttr({b.getStringAttr("no")}))};
      // ko
      curOutputDimName.push_back(b.getStringAttr("ko"));
      transformedShape.push_back(k);
      llvm::SmallVector<NamedAttribute, 5> kDimAttr{
          b.getNamedAttr("upper_layer_dimensions",
                         b.getArrayAttr({b.getI32IntegerAttr(3)})),
          b.getNamedAttr("upper_layer_names",
                         b.getArrayAttr({curOutputDimName[3]})),
          b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
          b.getNamedAttr(
              "lower_layer_dimensions",
              b.getArrayAttr({b.getI32IntegerAttr(currentKeyToDim["ko"])})),
          b.getNamedAttr("lower_layer_names",
                         b.getArrayAttr({b.getStringAttr("ko")}))};

      // ho
      curOutputDimName.push_back(b.getStringAttr("ho"));
      transformedShape.push_back(ho);
      llvm::SmallVector<NamedAttribute, 6> hoDimAttr{
          b.getNamedAttr("upper_layer_dimensions",
                         b.getArrayAttr({b.getI32IntegerAttr(4)})),
          b.getNamedAttr("upper_layer_names",
                         b.getArrayAttr({curOutputDimName[4]})),
          b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
          b.getNamedAttr(
              "lower_layer_dimensions",
              b.getArrayAttr({b.getI32IntegerAttr(currentKeyToDim["ho"])})),
          b.getNamedAttr("lower_layer_names",
                         b.getArrayAttr({b.getStringAttr("ho")}))};

      // wo
      curOutputDimName.push_back(b.getStringAttr("wo"));
      transformedShape.push_back(wo);
      llvm::SmallVector<NamedAttribute, 6> woDimAttr{
          b.getNamedAttr("upper_layer_dimensions",
                         b.getArrayAttr({b.getI32IntegerAttr(5)})),
          b.getNamedAttr("upper_layer_names",
                         b.getArrayAttr({curOutputDimName[5]})),
          b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
          b.getNamedAttr(
              "lower_layer_dimensions",
              b.getArrayAttr({b.getI32IntegerAttr(currentKeyToDim["wo"])})),
          b.getNamedAttr("lower_layer_names",
                         b.getArrayAttr({b.getStringAttr("wo")}))};

      transformedAttrs.push_back(b.getNamedAttr(
          "layout",
          b.getArrayAttr(
              {b.getDictionaryAttr(gDimAttr), b.getDictionaryAttr(nDimAttr),
               b.getDictionaryAttr(kDimAttr), b.getDictionaryAttr(hoDimAttr),
               b.getDictionaryAttr(woDimAttr)})));
      transformedAttrs.push_back(b.getNamedAttr(
          "upper_layer_layout",
          b.getArrayAttr(ArrayRef<Attribute>(curOutputDimName.begin(),
                                             curOutputDimName.end()))));

      transformedAttrs.push_back(
          b.getNamedAttr("lower_layer_layout", outputLayoutAttr));

      auto transformedFilterMemRefType =
          MemRefType::get(transformedShape, outputElementType);
      // set lowest_layer attribute.
      transformedAttrs.push_back(
          b.getNamedAttr("lowest_layer", b.getBoolAttr(true)));
      auto gemm = b.create<miopen::TransformOp>(
          loc, transformedFilterMemRefType, op.output(), transformedAttrs,
          /*populateBounds=*/true);
      return gemm;
    };

    auto outGN0N1KHoWo = getOutGN0N1KHoWo();

    auto getOutGemmGGemmKGemmM =
        [&](llvm::SmallVector<StringAttr> &preOutputDimName) {
          llvm::SmallVector<StringAttr, 7> curOutputDimName;
          llvm::SmallVector<int64_t, 7> transformedShape;
          // gemmG
          curOutputDimName.push_back(b.getStringAttr("gemmG"));
          transformedShape.push_back(g * gemmKBlocks);
          llvm::SmallVector<NamedAttribute, 5> gemmGDimAttr{
              b.getNamedAttr("upper_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(0)})),
              b.getNamedAttr("upper_layer_names",
                             b.getArrayAttr({curOutputDimName[0]})),
              b.getNamedAttr("transformation", b.getStringAttr("Merge")),
              b.getNamedAttr("lower_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(0),
                                             b.getI32IntegerAttr(1)})),
              b.getNamedAttr("lower_layer_names",
                             b.getArrayAttr({preOutputDimName[0]}))};

          // gemmK
          curOutputDimName.push_back(b.getStringAttr("gemmK"));
          transformedShape.push_back(n / gemmKBlocks * ho * wo);
          llvm::SmallVector<NamedAttribute, 5> gemmKDimAttr{
              b.getNamedAttr("upper_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(1)})),
              b.getNamedAttr("upper_layer_names",
                             b.getArrayAttr({curOutputDimName[1]})),
              b.getNamedAttr("transformation", b.getStringAttr("Merge")),
              b.getNamedAttr("lower_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(2),
                                             b.getI32IntegerAttr(4),
                                             b.getI32IntegerAttr(5)})),
              b.getNamedAttr(
                  "lower_layer_names",
                  b.getArrayAttr({preOutputDimName[2], preOutputDimName[4],
                                  preOutputDimName[5]}))};

          // gemmM
          curOutputDimName.push_back(b.getStringAttr("gemmM"));
          transformedShape.push_back(k);
          llvm::SmallVector<NamedAttribute, 5> gemmMDimAttr{
              b.getNamedAttr("upper_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(2)})),
              b.getNamedAttr("upper_layer_names",
                             b.getArrayAttr({curOutputDimName[2]})),
              b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
              b.getNamedAttr("lower_layer_dimensions",
                             b.getArrayAttr({b.getI32IntegerAttr(3)})),
              b.getNamedAttr("lower_layer_names",
                             b.getArrayAttr({preOutputDimName[3]}))};

          llvm::SmallVector<NamedAttribute, 3> transformedAttrs;
          transformedAttrs.push_back(b.getNamedAttr(
              "layout", b.getArrayAttr({b.getDictionaryAttr(gemmGDimAttr),
                                        b.getDictionaryAttr(gemmKDimAttr),
                                        b.getDictionaryAttr(gemmMDimAttr)})));
          transformedAttrs.push_back(b.getNamedAttr(
              "upper_layer_layout",
              b.getArrayAttr(ArrayRef<Attribute>(curOutputDimName.begin(),
                                                 curOutputDimName.end()))));

          transformedAttrs.push_back(b.getNamedAttr(
              "lower_layer_layout",
              b.getArrayAttr(ArrayRef<Attribute>(preOutputDimName.begin(),
                                                 preOutputDimName.end()))));

          transformedAttrs.push_back(b.getNamedAttr(
              "gridwise_gemm_argument_position", b.getI32IntegerAttr(1)));

          if (outputOobCheckDims.size()) {
            llvm::SmallVector<IntegerAttr, 5> boundDims;
            for (size_t i = 0; i < outputShape.size(); i++) {
              if (outputOobCheckDims.find(i) != outputOobCheckDims.end())
                boundDims.push_back(b.getI32IntegerAttr(1));
              else
                boundDims.push_back(b.getI32IntegerAttr(0));
            }
            transformedAttrs.push_back(b.getNamedAttr(
                "bound_check",
                b.getArrayAttr({boundDims.begin(), boundDims.end()})));
          }
          auto transformedMemRefType =
              MemRefType::get(transformedShape, outputElementType);
          auto gemm = b.create<miopen::TransformOp>(
              loc, transformedMemRefType, outGN0N1KHoWo, transformedAttrs,
              /*populateBounds=*/true);
          return gemm;
        };
    auto outGemmGGemmKGemmM = getOutGemmGGemmKGemmM(firstOutputDimName);

    return outGemmGGemmKGemmM;
  };
  Value gemmA = getGemmA();
  Value gemmB = getGemmB();
  Value gemmC = getGemmC();
  // Set attributes for gridwise_gemm op.
  llvm::SmallVector<NamedAttribute, 8> gridwiseGemmAttrs{
      b.getNamedAttr("data_operation", b.getI32IntegerAttr(1)),
      b.getNamedAttr("gemm_id", gemmIdAttr),
      b.getNamedAttr("arch", archAttr),
      b.getNamedAttr("num_cu", numCuAttr),
      b.getNamedAttr("filter_layout", filterLayoutAttr),
      b.getNamedAttr("filter_dimension", b.getI64ArrayAttr(filterShape)),
      b.getNamedAttr("input_layout", inputLayoutAttr),
      b.getNamedAttr("input_dimension", b.getI64ArrayAttr(inputShape)),
      b.getNamedAttr("output_layout", outputLayoutAttr),
      b.getNamedAttr("output_dimension", b.getI64ArrayAttr(outputShape)),
      b.getNamedAttr("dilations", dilationsAttr),
      b.getNamedAttr("strides", stridesAttr),
      b.getNamedAttr("padding", paddingAttr),
  };

  // xdlopsV2.
  auto xdlopsV2Attr = op->template getAttrOfType<BoolAttr>("xdlopsV2");
  if (xdlopsV2Attr && xdlopsV2Attr.getValue() == true)
    gridwiseGemmAttrs.push_back(
        b.getNamedAttr("xdlopsV2", b.getBoolAttr(true)));

  gridwiseGemmAttrs.push_back(b.getNamedAttr(
      "kernel_algorithm", b.getStringAttr("backward_weight_v4r4")));

  // Emit miopen.gridwise_gemm op.
  // Emit miopen.gridwise_gemm_v2 if xdlopsV2 attribute is true.
  auto arguments = SmallVector<Value, 3>{gemmA, gemmB, gemmC};

  if (xdlopsV2Attr && xdlopsV2Attr.getValue() == true) {
    b.create<miopen::GridwiseGemmV2Op>(
        loc, ArrayRef<Type>{},
        ValueRange{arguments[fields.gridwiseGemmArgumentPosition[0]],
                   arguments[fields.gridwiseGemmArgumentPosition[1]],
                   arguments[fields.gridwiseGemmArgumentPosition[2]]},
        gridwiseGemmAttrs);
  } else {
    b.create<miopen::GridwiseGemmOp>(
        loc, ArrayRef<Type>{},
        ValueRange{arguments[fields.gridwiseGemmArgumentPosition[0]],
                   arguments[fields.gridwiseGemmArgumentPosition[1]],
                   arguments[fields.gridwiseGemmArgumentPosition[2]]},
        gridwiseGemmAttrs);
  }
  // Finally, erase the original Conv2D op.
  op.erase();

  return success();
}
} // end namespace miopen
} // end namespace mlir
#endif