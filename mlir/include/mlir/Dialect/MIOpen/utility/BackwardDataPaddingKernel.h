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

#ifndef BACKWARD_DATA_PADDING_KERNEL_HELPER_H
#define BACKWARD_DATA_PADDING_KERNEL_HELPER_H

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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::miopen;

namespace mlir {
namespace miopen {
inline bool isSupportBackwardDataPaddingKernel(bool isXdlops,
                                               bool isStride2Pad1,
                                               int gemmMExtra, int gemmKExtra,
                                               int gemmNExtra) {
  if (gemmNExtra && gemmKExtra) {
    llvm::errs() << "can't support backward data padding kernel when both pad "
                    "gemmN and gemmK due to load issue\n";
    return false;
  }

  if (isXdlops && (gemmMExtra || gemmNExtra)) {
    if (isStride2Pad1) {
      llvm::errs()
          << "can't support backward data padding kernel when xdlops stride 2 "
             "pad_h,pad_w>0 and pad gemmM or gemmN due to store issue\n";
      return false;
    }
  }
  return true;
}

inline Value padFilter(bool isXdlops, int64_t gemmMExtra, int64_t gemmNExtra,
                       int64_t gemmKExtra, Value &gemmAPad, PatternRewriter &b,
                       Location loc, llvm::DenseSet<int> &filterOobCheckDims,
                       llvm::DenseMap<StringRef, int> &nameToDims,
                       ArrayRef<int64_t> filterShape, Type filterElementType,
                       int64_t g, int64_t k, int64_t c, int64_t yDotSlice,
                       int64_t xDotSlice) {
  bool filterCheckPadGemmM = false;
  bool filterCheckPadGemmK = false;

  filterCheckPadGemmM = (gemmMExtra > 0);
  filterCheckPadGemmK = (gemmKExtra > 0);
  if (filterCheckPadGemmM || filterCheckPadGemmK) {
    SmallString<5> arg0TargetLayoutName0("gemmG");
    SmallString<5> arg0TargetLayoutName1("gemmK");
    SmallString<5> arg0TargetLayoutName2("gemmM");

    SmallString<8> gemmKPad_name("gemmKPad");
    SmallString<8> gemmMPad_name("gemmMPad");

    bool isGemmDim1Pad = false;
    bool isGemmDim2Pad = false;
    int64_t nonGemmMSize = k * yDotSlice * xDotSlice;
    int64_t gemmMSize = c;

    StringAttr gemmDim0TargetName = b.getStringAttr(arg0TargetLayoutName0);
    StringAttr gemmDim1TargetName = b.getStringAttr(arg0TargetLayoutName1);
    StringAttr gemmDim2TargetName = b.getStringAttr(arg0TargetLayoutName2);

    llvm::SmallVector<NamedAttribute, 3> paddingFilterAttrs;
    llvm::SmallVector<int64_t, 2> paddingFilterShape;

    paddingFilterAttrs.push_back(
        b.getNamedAttr("gemmMExtra", b.getI32IntegerAttr(gemmMExtra)));

    paddingFilterAttrs.push_back(
        b.getNamedAttr("gemmNExtra", b.getI32IntegerAttr(gemmNExtra)));

    paddingFilterAttrs.push_back(
        b.getNamedAttr("gemmKExtra", b.getI32IntegerAttr(gemmKExtra)));

    paddingFilterAttrs.push_back(
        b.getNamedAttr("extraPad", b.getBoolAttr(true)));

    llvm::SmallVector<NamedAttribute, 0> layoutAttr0;
    llvm::SmallVector<NamedAttribute, 0> layoutAttr1;
    llvm::SmallVector<NamedAttribute, 0> layoutAttr2;

    StringAttr gemmDim0Name = b.getStringAttr(arg0TargetLayoutName0);
    IntegerAttr GemmDim0 = b.getI32IntegerAttr(0);
    StringAttr gemmDim1Name = b.getStringAttr(arg0TargetLayoutName1);
    IntegerAttr GemmDim1 = b.getI32IntegerAttr(1);
    StringAttr gemmDim2Name = b.getStringAttr(arg0TargetLayoutName2);
    IntegerAttr GemmDim2 = b.getI32IntegerAttr(2);

    paddingFilterShape.push_back(g);
    paddingFilterShape.push_back(k * yDotSlice * xDotSlice);
    paddingFilterShape.push_back(c);

    llvm::SmallVector<NamedAttribute, 3> sourceGemmDim0Attr{
        b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
        b.getNamedAttr("lower_layer_dimensions", b.getArrayAttr({GemmDim0})),
        b.getNamedAttr("lower_layer_names", b.getArrayAttr({gemmDim0Name}))};

    llvm::SmallVector<NamedAttribute, 3> sourceGemmDim1Attr{
        b.getNamedAttr("lower_layer_dimensions", b.getArrayAttr({GemmDim1})),
        b.getNamedAttr("lower_layer_names", b.getArrayAttr({gemmDim1Name}))};

    llvm::SmallVector<NamedAttribute, 3> sourceGemmDim2Attr{
        b.getNamedAttr("lower_layer_dimensions", b.getArrayAttr({GemmDim2})),
        b.getNamedAttr("lower_layer_names", b.getArrayAttr({gemmDim2Name}))};

    llvm::SmallVector<NamedAttribute, 3> targetGemmDim0Attr{
        b.getNamedAttr("upper_layer_dimensions", b.getArrayAttr({GemmDim0})),
        b.getNamedAttr("upper_layer_names", b.getArrayAttr({GemmDim0}))};

    llvm::SmallVector<NamedAttribute, 3> targetGemmDim1Attr{
        b.getNamedAttr("upper_layer_dimensions", b.getArrayAttr({GemmDim1}))};

    llvm::SmallVector<NamedAttribute, 3> targetGemmDim2Attr{
        b.getNamedAttr("upper_layer_dimensions", b.getArrayAttr({GemmDim2}))};

    if (filterCheckPadGemmK) {
      isGemmDim1Pad = true;
      gemmDim1TargetName = b.getStringAttr(gemmKPad_name);
      paddingFilterShape[1] = nonGemmMSize + gemmKExtra;
      sourceGemmDim1Attr.push_back(
          b.getNamedAttr("transformation", b.getStringAttr("Pad")));
      if (isXdlops) {
        sourceGemmDim1Attr.push_back(
            b.getNamedAttr("parameters", b.getArrayAttr({
                                             b.getI32IntegerAttr(gemmKExtra),
                                             b.getI32IntegerAttr(0),
                                         })));

      } else {
        sourceGemmDim1Attr.push_back(
            b.getNamedAttr("parameters", b.getArrayAttr({
                                             b.getI32IntegerAttr(0),
                                             b.getI32IntegerAttr(gemmKExtra),
                                         })));
      }
      // we only set k dim intead of k, y,x due to optimization of compiler
      // will mess up
      targetGemmDim1Attr.push_back(
          b.getNamedAttr("upper_layer_names",
                         b.getArrayAttr({b.getStringAttr(gemmKPad_name)})));
      filterOobCheckDims.insert(nameToDims["k"]);
    }

    if (filterCheckPadGemmM) {
      isGemmDim2Pad = true;
      gemmDim2TargetName = b.getStringAttr(gemmMPad_name);
      paddingFilterShape[2] = gemmMSize + gemmMExtra;
      sourceGemmDim2Attr.push_back(
          b.getNamedAttr("transformation", b.getStringAttr("Pad")));
      if (isXdlops) {
        sourceGemmDim2Attr.push_back(
            b.getNamedAttr("parameters", b.getArrayAttr({
                                             b.getI32IntegerAttr(gemmMExtra),
                                             b.getI32IntegerAttr(0),
                                         })));

      } else {
        sourceGemmDim2Attr.push_back(
            b.getNamedAttr("parameters", b.getArrayAttr({
                                             b.getI32IntegerAttr(0),
                                             b.getI32IntegerAttr(gemmMExtra),
                                         })));
      }
      targetGemmDim2Attr.push_back(b.getNamedAttr(
          "names", b.getArrayAttr({b.getStringAttr(gemmMPad_name)})));

      filterOobCheckDims.insert(nameToDims["c"]);
    }

    if (!isGemmDim1Pad) {
      gemmDim1TargetName = gemmDim1Name;
      sourceGemmDim1Attr.push_back(
          b.getNamedAttr("transformation", b.getStringAttr("PassThrough")));
      targetGemmDim1Attr.push_back(
          b.getNamedAttr("upper_layer_names", b.getArrayAttr({gemmDim1Name})));
    } else if (!isGemmDim2Pad) {
      gemmDim2TargetName = gemmDim2Name;
      sourceGemmDim2Attr.push_back(
          b.getNamedAttr("transformation", b.getStringAttr("PassThrough")));
      targetGemmDim2Attr.push_back(
          b.getNamedAttr("upper_layer_names", b.getArrayAttr({gemmDim2Name})));
    }

    layoutAttr0.append(targetGemmDim0Attr.begin(), targetGemmDim0Attr.end());
    layoutAttr0.append(sourceGemmDim0Attr.begin(), sourceGemmDim0Attr.end());
    layoutAttr1.append(targetGemmDim1Attr.begin(), targetGemmDim1Attr.end());
    layoutAttr1.append(sourceGemmDim1Attr.begin(), sourceGemmDim1Attr.end());
    layoutAttr2.append(targetGemmDim2Attr.begin(), targetGemmDim2Attr.end());
    layoutAttr2.append(sourceGemmDim2Attr.begin(), sourceGemmDim2Attr.end());

    paddingFilterAttrs.push_back(b.getNamedAttr(
        "layout", b.getArrayAttr({
                      b.getDictionaryAttr({ArrayRef<NamedAttribute>(
                          layoutAttr0.begin(), layoutAttr0.end())}),
                      b.getDictionaryAttr({ArrayRef<NamedAttribute>(
                          layoutAttr1.begin(), layoutAttr1.end())}),
                      b.getDictionaryAttr({ArrayRef<NamedAttribute>(
                          layoutAttr2.begin(), layoutAttr2.end())}),
                  })));

    paddingFilterAttrs.push_back(
        b.getNamedAttr("upper_layer_layout",
                       b.getArrayAttr({gemmDim0TargetName, gemmDim1TargetName,
                                       gemmDim2TargetName})));

    paddingFilterAttrs.push_back(b.getNamedAttr(
        "lower_layer_layout",
        b.getArrayAttr({gemmDim0Name, gemmDim1Name, gemmDim2Name})));

    if (filterOobCheckDims.size()) {
      llvm::SmallVector<IntegerAttr, 5> boundDims;
      for (size_t i = 0; i < filterShape.size(); i++) {
        if (filterOobCheckDims.find(i) != filterOobCheckDims.end())
          boundDims.push_back(b.getI32IntegerAttr(1));
        else
          boundDims.push_back(b.getI32IntegerAttr(0));
      }

      paddingFilterAttrs.push_back(b.getNamedAttr(
          "bound_check", b.getArrayAttr({boundDims.begin(), boundDims.end()})));
    }

    auto paddingFilterMemRefType =
        MemRefType::get(paddingFilterShape, filterElementType);
    gemmAPad = b.create<miopen::TransformOp>(
        loc, paddingFilterMemRefType, gemmAPad, paddingFilterAttrs, true);
  }
  // if no padding , return original
  return gemmAPad;
}

inline Value padInput(bool isXdlops, int64_t gemmMExtra, int64_t gemmNExtra,
                      int64_t gemmKExtra, Value &gemmBPad, PatternRewriter &b,
                      Location loc, llvm::DenseSet<int> &inputOobCheckDims,
                      llvm::DenseMap<StringRef, int> &nameToDims,
                      ArrayRef<int64_t> transformedShape,
                      ArrayRef<int64_t> inputShape, Type inputElementType) {
  bool inputCheckPadGemmM = (gemmMExtra > 0);
  bool inputCheckPadGemmN = (gemmNExtra > 0);

  if (inputCheckPadGemmM || inputCheckPadGemmN) {
    SmallString<8> gemmMPad_name("gemmMPad");
    SmallString<8> gemmNPad_name("gemmNPad");

    llvm::SmallVector<int64_t, 3> paddingInputShape;
    llvm::SmallVector<NamedAttribute, 3> paddingInputAttrs;

    paddingInputAttrs.push_back(
        b.getNamedAttr("extraPad", b.getBoolAttr(true)));
    paddingInputAttrs.push_back(
        b.getNamedAttr("gemmMExtra", b.getI32IntegerAttr(gemmMExtra)));

    paddingInputAttrs.push_back(
        b.getNamedAttr("gemmKExtra", b.getI32IntegerAttr(gemmKExtra)));
    paddingInputAttrs.push_back(
        b.getNamedAttr("gemmNExtra", b.getI32IntegerAttr(gemmNExtra)));

    llvm::SmallVector<NamedAttribute, 0> layoutAttr0;
    llvm::SmallVector<NamedAttribute, 0> layoutAttr1;
    llvm::SmallVector<NamedAttribute, 0> layoutAttr2;

    SmallString<5> arg1TargetLayoutName0("gemmG");
    SmallString<5> arg1TargetLayoutName1("gemmM");
    SmallString<5> arg1TargetLayoutName2("gemmN");
    StringAttr gemmDim0TargetName = b.getStringAttr(arg1TargetLayoutName0);
    StringAttr gemmDim1TargetName;
    StringAttr gemmDim2TargetName;

    bool isGemmDim1Pad = false;
    bool isGemmDim2Pad = false;

    StringAttr gemmDim0Name = b.getStringAttr(arg1TargetLayoutName0);
    IntegerAttr GemmDim0 = b.getI32IntegerAttr(0);
    StringAttr gemmDim1Name = b.getStringAttr(arg1TargetLayoutName1);
    IntegerAttr GemmDim1 = b.getI32IntegerAttr(1);
    StringAttr gemmDim2Name = b.getStringAttr(arg1TargetLayoutName2);
    IntegerAttr GemmDim2 = b.getI32IntegerAttr(2);

    paddingInputShape.push_back(transformedShape[0]);
    paddingInputShape.push_back(transformedShape[1]);
    paddingInputShape.push_back(transformedShape[2]);

    llvm::SmallVector<NamedAttribute, 3> sourceGemmDim0Attr{
        b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
        b.getNamedAttr("lower_layer_dimensions", b.getArrayAttr({GemmDim0})),
        b.getNamedAttr("lower_layer_names", b.getArrayAttr({gemmDim0Name}))};

    llvm::SmallVector<NamedAttribute, 3> sourceGemmDim1Attr{
        b.getNamedAttr("lower_layer_dimensions", b.getArrayAttr({GemmDim1})),
        b.getNamedAttr("lower_layer_names", b.getArrayAttr({gemmDim1Name}))};

    llvm::SmallVector<NamedAttribute, 3> sourceGemmDim2Attr{
        b.getNamedAttr("lower_layer_dimensions", b.getArrayAttr({GemmDim2})),
        b.getNamedAttr("lower_layer_names", b.getArrayAttr({gemmDim2Name}))};

    llvm::SmallVector<NamedAttribute, 3> targetGemmDim0Attr{
        b.getNamedAttr("upper_layer_dimensions", b.getArrayAttr({GemmDim0})),
        b.getNamedAttr("upper_layer_names", b.getArrayAttr({gemmDim0Name}))};

    llvm::SmallVector<NamedAttribute, 3> targetGemmDim1Attr{
        b.getNamedAttr("upper_layer_dimensions", b.getArrayAttr({GemmDim1}))};

    llvm::SmallVector<NamedAttribute, 3> targetGemmDim2Attr{
        b.getNamedAttr("upper_layer_dimensions", b.getArrayAttr({GemmDim2}))};

    if (inputCheckPadGemmM) {
      isGemmDim1Pad = true;
      gemmDim1TargetName = b.getStringAttr(gemmMPad_name);
      paddingInputShape[1] = paddingInputShape[1] + gemmMExtra;
      sourceGemmDim1Attr.push_back(
          b.getNamedAttr("transformation", b.getStringAttr("Pad")));
      if (isXdlops) {
        sourceGemmDim1Attr.push_back(b.getNamedAttr(
            "parameters", b.getArrayAttr({b.getI32IntegerAttr(gemmMExtra),
                                          b.getI32IntegerAttr(0)})));
      } else {
        sourceGemmDim1Attr.push_back(b.getNamedAttr(
            "parameters", b.getArrayAttr({b.getI32IntegerAttr(0),
                                          b.getI32IntegerAttr(gemmMExtra)})));
      }
      targetGemmDim1Attr.push_back(
          b.getNamedAttr("upper_layer_names",
                         b.getArrayAttr({b.getStringAttr(gemmMPad_name)})));
      // gemmM is c
      inputOobCheckDims.insert(nameToDims["ci"]);
    }

    if (inputCheckPadGemmN) {
      isGemmDim2Pad = true;
      gemmDim2TargetName = b.getStringAttr(gemmNPad_name);

      paddingInputShape[2] = paddingInputShape[2] + gemmNExtra;
      sourceGemmDim2Attr.push_back(
          b.getNamedAttr("transformation", b.getStringAttr("Pad")));
      if (isXdlops) {
        sourceGemmDim2Attr.push_back(b.getNamedAttr(
            "parameters", b.getArrayAttr({b.getI32IntegerAttr(gemmNExtra),
                                          b.getI32IntegerAttr(0)})));

      } else {
        sourceGemmDim2Attr.push_back(b.getNamedAttr(
            "parameters", b.getArrayAttr({b.getI32IntegerAttr(0),
                                          b.getI32IntegerAttr(gemmNExtra)})));
      }
      targetGemmDim2Attr.push_back(
          b.getNamedAttr("upper_layer_names",
                         b.getArrayAttr({b.getStringAttr(gemmNPad_name)})));
      // gemmN is NHW , we use top dim N to do check,
      // if we use all dims (n,h,w), compiler will mess up
      inputOobCheckDims.insert(nameToDims["ni"]);
    }

    if (!isGemmDim1Pad) {
      gemmDim1TargetName = gemmDim1Name;
      sourceGemmDim1Attr.push_back(
          b.getNamedAttr("transformation", b.getStringAttr("PassThrough")));
      targetGemmDim1Attr.push_back(
          b.getNamedAttr("upper_layer_names", b.getArrayAttr({gemmDim1Name})));
    } else if (!isGemmDim2Pad) {
      gemmDim2TargetName = gemmDim2Name;
      sourceGemmDim2Attr.push_back(
          b.getNamedAttr("transformation", b.getStringAttr("PassThrough")));
      targetGemmDim2Attr.push_back(
          b.getNamedAttr("upper_layer_names", b.getArrayAttr({gemmDim2Name})));
    }

    layoutAttr0.append(targetGemmDim0Attr.begin(), targetGemmDim0Attr.end());
    layoutAttr0.append(sourceGemmDim0Attr.begin(), sourceGemmDim0Attr.end());
    layoutAttr1.append(targetGemmDim1Attr.begin(), targetGemmDim1Attr.end());
    layoutAttr1.append(sourceGemmDim1Attr.begin(), sourceGemmDim1Attr.end());
    layoutAttr2.append(targetGemmDim2Attr.begin(), targetGemmDim2Attr.end());
    layoutAttr2.append(sourceGemmDim2Attr.begin(), sourceGemmDim2Attr.end());

    paddingInputAttrs.push_back(b.getNamedAttr(
        "layout",
        b.getArrayAttr({b.getDictionaryAttr({ArrayRef<NamedAttribute>(
                            layoutAttr0.begin(), layoutAttr0.end())}),
                        b.getDictionaryAttr({ArrayRef<NamedAttribute>(
                            layoutAttr1.begin(), layoutAttr1.end())}),
                        b.getDictionaryAttr({ArrayRef<NamedAttribute>(
                            layoutAttr2.begin(), layoutAttr2.end())})})));

    paddingInputAttrs.push_back(
        b.getNamedAttr("upper_layer_layout",
                       b.getArrayAttr({gemmDim0TargetName, gemmDim1TargetName,
                                       gemmDim2TargetName})));

    paddingInputAttrs.push_back(b.getNamedAttr(
        "lower_layer_layout",
        b.getArrayAttr({gemmDim0Name, gemmDim1Name, gemmDim2Name})));

    if (inputOobCheckDims.size()) {
      llvm::SmallVector<IntegerAttr, 5> boundDims;
      for (size_t i = 0; i < inputShape.size(); i++) {
        if (inputOobCheckDims.find(i) != inputOobCheckDims.end())
          boundDims.push_back(b.getI32IntegerAttr(1));
        else
          boundDims.push_back(b.getI32IntegerAttr(0));
      }
      paddingInputAttrs.push_back(b.getNamedAttr(
          "bound_check", b.getArrayAttr({boundDims.begin(), boundDims.end()})));
    }

    auto paddingInputMemRefType =
        MemRefType::get(paddingInputShape, inputElementType);

    gemmBPad = b.create<miopen::TransformOp>(loc, paddingInputMemRefType,
                                             gemmBPad, paddingInputAttrs,
                                             /*populateBounds=*/true);
  }

  // if no padding , return original
  return gemmBPad;
}

inline Value padOutput(bool isXdlops, int64_t gemmMExtra, int64_t gemmNExtra,
                       int64_t gemmKExtra, Value &gemmCPad, PatternRewriter &b,
                       Location loc, llvm::DenseSet<int> &outputOobCheckDims,
                       llvm::DenseMap<StringRef, int> &nameToDims,
                       ArrayRef<int64_t> transformedShape,
                       ArrayRef<int64_t> outputShape, Type outputElementType)

{
  bool outputCheckPadGemmK = (gemmKExtra > 0);
  bool outputCheckPadGemmN = (gemmNExtra > 0);

  if (outputCheckPadGemmK || outputCheckPadGemmN) {
    SmallString<5> arg2TargetLayoutName0("gemmG");
    SmallString<5> arg2TargetLayoutName1("gemmK");
    SmallString<5> arg2TargetLayoutName2("gemmN");

    StringAttr gemmDim0TargetName = b.getStringAttr(arg2TargetLayoutName0);
    StringAttr gemmDim1TargetName;
    StringAttr gemmDim2TargetName;

    bool isGemmDim1Pad = false;
    bool isGemmDim2Pad = false;
    SmallString<8> gemmKPad_name("gemmKPad");
    SmallString<8> gemmMPad_name("gemmMPad");
    SmallString<8> gemmNPad_name("gemmNPad");

    llvm::SmallVector<NamedAttribute, 3> paddingOutputAttrs;
    llvm::SmallVector<int64_t, 2> paddingOutputShape;

    paddingOutputAttrs.push_back(
        b.getNamedAttr("extraPad", b.getBoolAttr(true)));
    paddingOutputAttrs.push_back(
        b.getNamedAttr("gemmKExtra", b.getI32IntegerAttr(gemmKExtra)));
    paddingOutputAttrs.push_back(
        b.getNamedAttr("gemmNExtra", b.getI32IntegerAttr(gemmNExtra)));
    paddingOutputAttrs.push_back(
        b.getNamedAttr("gemmMExtra", b.getI32IntegerAttr(gemmMExtra)));

    llvm::SmallVector<NamedAttribute, 0> layoutAttr0;
    llvm::SmallVector<NamedAttribute, 0> layoutAttr1;
    llvm::SmallVector<NamedAttribute, 0> layoutAttr2;

    StringAttr gemmDim0Name = b.getStringAttr(arg2TargetLayoutName0);
    IntegerAttr GemmDim0 = b.getI32IntegerAttr(0);
    StringAttr gemmDim1Name = b.getStringAttr(arg2TargetLayoutName1);
    IntegerAttr GemmDim1 = b.getI32IntegerAttr(1);
    StringAttr gemmDim2Name = b.getStringAttr(arg2TargetLayoutName2);
    IntegerAttr GemmDim2 = b.getI32IntegerAttr(2);

    paddingOutputShape.push_back(transformedShape[0]);
    paddingOutputShape.push_back(transformedShape[1]);
    paddingOutputShape.push_back(transformedShape[2]);

    llvm::SmallVector<NamedAttribute, 3> sourceGemmDim0Attr{
        b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
        b.getNamedAttr("lower_layer_dimensions", b.getArrayAttr({GemmDim0})),
        b.getNamedAttr("lower_layer_names", b.getArrayAttr({gemmDim0Name}))};

    llvm::SmallVector<NamedAttribute, 3> sourceGemmDim1Attr{
        b.getNamedAttr("lower_layer_dimensions", b.getArrayAttr({GemmDim1})),
        b.getNamedAttr("lower_layer_names", b.getArrayAttr({gemmDim1Name}))};

    llvm::SmallVector<NamedAttribute, 3> sourceGemmDim2Attr{
        b.getNamedAttr("lower_layer_dimensions", b.getArrayAttr({GemmDim2})),
        b.getNamedAttr("lower_layer_names", b.getArrayAttr({gemmDim2Name}))};

    llvm::SmallVector<NamedAttribute, 3> targetGemmDim0Attr{
        b.getNamedAttr("upper_layer_dimensions", b.getArrayAttr({GemmDim0})),
        b.getNamedAttr("upper_layer_names", b.getArrayAttr({GemmDim0}))};

    llvm::SmallVector<NamedAttribute, 3> targetGemmDim1Attr{
        b.getNamedAttr("upper_layer_dimensions", b.getArrayAttr({GemmDim1}))};

    llvm::SmallVector<NamedAttribute, 3> targetGemmDim2Attr{
        b.getNamedAttr("upper_layer_dimensions", b.getArrayAttr({GemmDim2}))};

    if (outputCheckPadGemmK) {
      isGemmDim1Pad = true;
      gemmDim1TargetName = b.getStringAttr(gemmKPad_name);
      paddingOutputShape[1] = paddingOutputShape[1] + gemmKExtra;
      sourceGemmDim1Attr.push_back(
          b.getNamedAttr("transformation", b.getStringAttr("Pad")));
      if (isXdlops) {
        sourceGemmDim1Attr.push_back(b.getNamedAttr(
            "parameters", b.getArrayAttr({b.getI32IntegerAttr(gemmKExtra),
                                          b.getI32IntegerAttr(0)})));

      } else {
        sourceGemmDim1Attr.push_back(b.getNamedAttr(
            "parameters", b.getArrayAttr({b.getI32IntegerAttr(0),
                                          b.getI32IntegerAttr(gemmKExtra)})));
      }
      targetGemmDim1Attr.push_back(
          b.getNamedAttr("upper_layer_names",
                         b.getArrayAttr({b.getStringAttr(gemmKPad_name)})));

      // gemmK is (K, Y, X), we only set top dim K due to if we use all
      // dims compiler can't generate correct code
      outputOobCheckDims.insert(nameToDims["ko"]);
    }

    if (outputCheckPadGemmN) {
      isGemmDim2Pad = true;
      gemmDim2TargetName = b.getStringAttr(gemmNPad_name);

      paddingOutputShape[2] = paddingOutputShape[2] + gemmNExtra;
      sourceGemmDim2Attr.push_back(
          b.getNamedAttr("transformation", b.getStringAttr("Pad")));
      if (isXdlops) {
        sourceGemmDim2Attr.push_back(b.getNamedAttr(
            "parameters", b.getArrayAttr({b.getI32IntegerAttr(gemmNExtra),
                                          b.getI32IntegerAttr(0)})));

      } else {
        sourceGemmDim2Attr.push_back(b.getNamedAttr(
            "parameters", b.getArrayAttr({b.getI32IntegerAttr(0),
                                          b.getI32IntegerAttr(gemmNExtra)})));
      }
      targetGemmDim2Attr.push_back(b.getNamedAttr(
          "names", b.getArrayAttr({b.getStringAttr(gemmNPad_name)})));

      outputOobCheckDims.insert(nameToDims["no"]);
    }

    if (!isGemmDim1Pad) {
      gemmDim1TargetName = gemmDim1Name;
      sourceGemmDim1Attr.push_back(
          b.getNamedAttr("transformation", b.getStringAttr("PassThrough")));
      targetGemmDim1Attr.push_back(
          b.getNamedAttr("upper_layer_names", b.getArrayAttr({gemmDim1Name})));
    } else if (!isGemmDim2Pad) {
      gemmDim2TargetName = gemmDim2Name;
      sourceGemmDim2Attr.push_back(
          b.getNamedAttr("transformation", b.getStringAttr("PassThrough")));
      targetGemmDim2Attr.push_back(
          b.getNamedAttr("upper_layer_names", b.getArrayAttr({gemmDim2Name})));
    }

    layoutAttr0.append(targetGemmDim0Attr.begin(), targetGemmDim0Attr.end());
    layoutAttr0.append(sourceGemmDim0Attr.begin(), sourceGemmDim0Attr.end());
    layoutAttr1.append(targetGemmDim1Attr.begin(), targetGemmDim1Attr.end());
    layoutAttr1.append(sourceGemmDim1Attr.begin(), sourceGemmDim1Attr.end());
    layoutAttr2.append(targetGemmDim2Attr.begin(), targetGemmDim2Attr.end());
    layoutAttr2.append(sourceGemmDim2Attr.begin(), sourceGemmDim2Attr.end());

    paddingOutputAttrs.push_back(b.getNamedAttr(
        "layout", b.getArrayAttr({
                      b.getDictionaryAttr({ArrayRef<NamedAttribute>(
                          layoutAttr0.begin(), layoutAttr0.end())}),
                      b.getDictionaryAttr({ArrayRef<NamedAttribute>(
                          layoutAttr1.begin(), layoutAttr1.end())}),
                      b.getDictionaryAttr({ArrayRef<NamedAttribute>(
                          layoutAttr2.begin(), layoutAttr2.end())}),
                  })));

    paddingOutputAttrs.push_back(
        b.getNamedAttr("upper_layer_layout",
                       b.getArrayAttr({gemmDim0TargetName, gemmDim1TargetName,
                                       gemmDim2TargetName})));

    paddingOutputAttrs.push_back(b.getNamedAttr(
        "lower_layer_layout",
        b.getArrayAttr({gemmDim0Name, gemmDim1Name, gemmDim2Name})));

    if (outputOobCheckDims.size()) {
      llvm::SmallVector<IntegerAttr, 5> boundDims;
      for (size_t i = 0; i < outputShape.size(); i++) {
        if (outputOobCheckDims.find(i) != outputOobCheckDims.end())
          boundDims.push_back(b.getI32IntegerAttr(1));
        else
          boundDims.push_back(b.getI32IntegerAttr(0));
      }
      paddingOutputAttrs.push_back(b.getNamedAttr(
          "bound_check", b.getArrayAttr({boundDims.begin(), boundDims.end()})));
    }

    auto paddingOutputMemRefType =
        MemRefType::get(paddingOutputShape, outputElementType);

    gemmCPad = b.create<miopen::TransformOp>(loc, paddingOutputMemRefType,
                                             gemmCPad, paddingOutputAttrs,
                                             /*populateBounds=*/true);
  }
  return gemmCPad;
}

} // end namespace miopen
} // end namespace mlir
#endif
