//===- RockWinogradConvPass.cpp - Lowering Rock Winograd Conv -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass converts tosa.conv2d operations to Winograd
// convolution decomposition using linalg and rock operations.
// The implementation is based on the paper: Fast Algorithms for Convolutional
// Neural Networks (https://arxiv.org/abs/1509.09308)
//
// The formula of minimal 2D filtering algorithm F(m x m, r x r) is:
// Y = A^T x [ (G x g x G^T) x (B^T x d x B) ] x A
//
// where g is filter and d is input data.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
#define GEN_PASS_DEF_ROCKWINOGRADCONVPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Winograd Convolution Parameters
//===----------------------------------------------------------------------===//

/// Enum representing Winograd convolution F(m, r) configurations.
/// m is the output tile size and r is the filter size.
enum class WinogradFmr {
  F_2_3, // F(2x2, 3x3) - alpha = 4
  F_4_3, // F(4x4, 3x3) - alpha = 6
  F_2_5, // F(2x2, 5x5) - alpha = 6
};

/// Get m and r values from WinogradFmr enum.
static std::pair<int64_t, int64_t> getFmrValues(WinogradFmr fmr) {
  switch (fmr) {
  case WinogradFmr::F_2_3:
    return {2, 3};
  case WinogradFmr::F_4_3:
    return {4, 3};
  case WinogradFmr::F_2_5:
    return {2, 5};
  }
  llvm_unreachable("Unknown WinogradFmr");
}

//===----------------------------------------------------------------------===//
// Winograd Transformation Matrices
//===----------------------------------------------------------------------===//

/// Structure to hold transformation matrix data.
struct TransformMatrix {
  const float *data;
  int64_t rows;
  int64_t cols;
  int64_t scalarFactor;

  TransformMatrix(const float *data, int64_t rows, int64_t cols,
                  int64_t scalarFactor = 1)
      : data(data), rows(rows), cols(cols), scalarFactor(scalarFactor) {}
};

// clang-format off
// Transformation matrices for F(2x2, 3x3)
static constexpr float G_2x2_3x3[] = {
     1,      0,     0,
   0.5f,  0.5f,  0.5f,
   0.5f, -0.5f,  0.5f,
     0,      0,     1
};

static constexpr float GT_2x2_3x3[] = {
     1,   0.5f,  0.5f, 0,
     0,   0.5f, -0.5f, 0,
     0,   0.5f,  0.5f, 1
};

static constexpr float BT_2x2_3x3[] = {
     1,    0,   -1,   0,
     0,    1,    1,   0,
     0,   -1,    1,   0,
     0,    1,    0,  -1
};

static constexpr float B_2x2_3x3[] = {
     1,    0,    0,   0,
     0,    1,   -1,   1,
    -1,    1,    1,   0,
     0,    0,    0,  -1
};

static constexpr float AT_2x2_3x3[] = {
     1,    1,    1,   0,
     0,    1,   -1,  -1
};

static constexpr float A_2x2_3x3[] = {
     1,    0,
     1,    1,
     1,   -1,
     0,   -1
};

// Transformation matrices for F(4x4, 3x3)
static constexpr float G_4x4_3x3[] = {
      0.25f,         0,         0,
   -1.0f/6,   1.0f/6,  -1.0f/6,
   -1.0f/6,  -1.0f/6,  -1.0f/6,
  1.0f/24,  1.0f/12,   1.0f/6,
  1.0f/24, -1.0f/12,   1.0f/6,
        0,         0,         1
};

static constexpr float GT_4x4_3x3[] = {
  0.25f, -1.0f/6, -1.0f/6, 1.0f/24, 1.0f/24, 0,
      0,  1.0f/6, -1.0f/6, 1.0f/12, -1.0f/12, 0,
      0, -1.0f/6, -1.0f/6,  1.0f/6,  1.0f/6, 1
};

static constexpr float BT_4x4_3x3[] = {
      4,    0,   -5,    0,   1,    0,
      0,   -4,   -4,    1,   1,    0,
      0,    4,   -4,   -1,   1,    0,
      0,   -2,   -1,    2,   1,    0,
      0,    2,   -1,   -2,   1,    0,
      0,    4,    0,   -5,   0,    1
};

static constexpr float B_4x4_3x3[] = {
      4,    0,    0,    0,    0,    0,
      0,   -4,    4,   -2,    2,    4,
     -5,   -4,   -4,   -1,   -1,    0,
      0,    1,   -1,    2,   -2,   -5,
      1,    1,    1,    1,    1,    0,
      0,    0,    0,    0,    0,    1
};

static constexpr float AT_4x4_3x3[] = {
      1,    1,    1,    1,    1,    0,
      0,    1,   -1,    2,   -2,    0,
      0,    1,    1,    4,    4,    0,
      0,    1,   -1,    8,   -8,    1
};

static constexpr float A_4x4_3x3[] = {
      1,    0,    0,    0,
      1,    1,    1,    1,
      1,   -1,    1,   -1,
      1,    2,    4,    8,
      1,   -2,    4,   -8,
      0,    0,    0,    1
};

// Transformation matrices for F(2x2, 5x5)
static constexpr float G_2x2_5x5[] = {
       1,        0,        0,        0,        0,
  1.0f/6,  -1.0f/6,   1.0f/6, -1.0f/6,   1.0f/6,
 -1.0f/6,  -1.0f/6,  -1.0f/6, -1.0f/6,  -1.0f/6,
-4.0f/15,  2.0f/15, -1.0f/15, 1.0f/30, -1.0f/60,
 1.0f/60,  1.0f/30,  1.0f/15, 2.0f/15,  4.0f/15,
       0,        0,        0,        0,        1
};

static constexpr float GT_2x2_5x5[] = {
     1,   1.0f/6, -1.0f/6, -4.0f/15, 1.0f/60, 0,
     0,  -1.0f/6, -1.0f/6,  2.0f/15, 1.0f/30, 0,
     0,   1.0f/6, -1.0f/6, -1.0f/15, 1.0f/15, 0,
     0,  -1.0f/6, -1.0f/6,  1.0f/30, 2.0f/15, 0,
     0,   1.0f/6, -1.0f/6, -1.0f/60, 4.0f/15, 1
};

static constexpr float BT_2x2_5x5[] = {
     1,  0.75f,  -1,  -0.75f,  0.5f,     0,
     0,   0.5f, 0.25f, -1.25f,  0.5f,     0,
     0,  -0.5f, -1.25f, -0.25f,  0.5f,     0,
     0,      1,  -0.5f,     -1,  0.5f,     0,
     0,  -0.5f,     -1,   0.5f,     1,     0,
     0,   0.5f,  0.75f,     -1, -0.75f, 0.5f
};

static constexpr float B_2x2_5x5[] = {
      1,      0,      0,      0,      0,      0,
   0.75f,  0.5f,  -0.5f,      1,  -0.5f,   0.5f,
     -1,  0.25f, -1.25f,  -0.5f,     -1,  0.75f,
 -0.75f, -1.25f, -0.25f,     -1,   0.5f,     -1,
   0.5f,   0.5f,   0.5f,   0.5f,      1, -0.75f,
      0,      0,      0,      0,      0,   0.5f
};

static constexpr float AT_2x2_5x5[] = {
      1,    1,    1,    2,    1,    0,
      0,    1,   -1,    1,   -2,    1
};

static constexpr float A_2x2_5x5[] = {
      1,    0,
      1,    1,
      1,   -1,
      2,    1,
      1,   -2,
      0,    1
};
// clang-format on

/// Get G matrix for filter transform.
static TransformMatrix getGMatrix(WinogradFmr fmr) {
  switch (fmr) {
  case WinogradFmr::F_2_3:
    return TransformMatrix(G_2x2_3x3, 4, 3);
  case WinogradFmr::F_4_3:
    return TransformMatrix(G_4x4_3x3, 6, 3);
  case WinogradFmr::F_2_5:
    return TransformMatrix(G_2x2_5x5, 6, 5);
  }
  llvm_unreachable("Unknown WinogradFmr");
}

/// Get G^T matrix for filter transform.
static TransformMatrix getGTMatrix(WinogradFmr fmr) {
  switch (fmr) {
  case WinogradFmr::F_2_3:
    return TransformMatrix(GT_2x2_3x3, 3, 4);
  case WinogradFmr::F_4_3:
    return TransformMatrix(GT_4x4_3x3, 3, 6);
  case WinogradFmr::F_2_5:
    return TransformMatrix(GT_2x2_5x5, 5, 6);
  }
  llvm_unreachable("Unknown WinogradFmr");
}

/// Get B^T matrix for input transform.
static TransformMatrix getBTMatrix(WinogradFmr fmr) {
  switch (fmr) {
  case WinogradFmr::F_2_3:
    return TransformMatrix(BT_2x2_3x3, 4, 4);
  case WinogradFmr::F_4_3:
    return TransformMatrix(BT_4x4_3x3, 6, 6);
  case WinogradFmr::F_2_5:
    return TransformMatrix(BT_2x2_5x5, 6, 6);
  }
  llvm_unreachable("Unknown WinogradFmr");
}

/// Get B matrix for input transform.
static TransformMatrix getBMatrix(WinogradFmr fmr) {
  switch (fmr) {
  case WinogradFmr::F_2_3:
    return TransformMatrix(B_2x2_3x3, 4, 4);
  case WinogradFmr::F_4_3:
    return TransformMatrix(B_4x4_3x3, 6, 6);
  case WinogradFmr::F_2_5:
    return TransformMatrix(B_2x2_5x5, 6, 6);
  }
  llvm_unreachable("Unknown WinogradFmr");
}

/// Get A^T matrix for output transform.
static TransformMatrix getATMatrix(WinogradFmr fmr) {
  switch (fmr) {
  case WinogradFmr::F_2_3:
    return TransformMatrix(AT_2x2_3x3, 2, 4);
  case WinogradFmr::F_4_3:
    return TransformMatrix(AT_4x4_3x3, 4, 6, 32);
  case WinogradFmr::F_2_5:
    return TransformMatrix(AT_2x2_5x5, 2, 6, 16);
  }
  llvm_unreachable("Unknown WinogradFmr");
}

/// Get A matrix for output transform.
static TransformMatrix getAMatrix(WinogradFmr fmr) {
  switch (fmr) {
  case WinogradFmr::F_2_3:
    return TransformMatrix(A_2x2_3x3, 4, 2);
  case WinogradFmr::F_4_3:
    return TransformMatrix(A_4x4_3x3, 6, 4, 32);
  case WinogradFmr::F_2_5:
    return TransformMatrix(A_2x2_5x5, 6, 2, 16);
  }
  llvm_unreachable("Unknown WinogradFmr");
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Create a 2D constant tensor from transformation matrix data.
static Value createTransformMatrix(OpBuilder &builder, Location loc,
                                   const TransformMatrix &matrix,
                                   Type elemType) {
  ArrayRef<float> constVec(matrix.data, matrix.rows * matrix.cols);
  auto tensorType = RankedTensorType::get({matrix.rows, matrix.cols}, elemType);
  return arith::ConstantOp::create(
      builder, loc, DenseFPElementsAttr::get(tensorType, constVec));
}

/// Check if all values in a DenseI64ArrayAttr are 1.
static bool hasAllOnes(DenseI64ArrayAttr attr) {
  return llvm::all_of(attr.asArrayRef(), [](int64_t val) { return val == 1; });
}

/// Create a zero-filled tensor using arith.constant with splat attribute.
static Value createZeroTensor(OpBuilder &builder, Location loc,
                              RankedTensorType tensorType) {
  Type elemType = tensorType.getElementType();
  Attribute zeroAttr;
  if (isa<FloatType>(elemType)) {
    zeroAttr = builder.getFloatAttr(elemType, 0.0);
  } else {
    zeroAttr = builder.getIntegerAttr(elemType, 0);
  }
  auto splatAttr = SplatElementsAttr::get(tensorType, zeroAttr);
  return arith::ConstantOp::create(builder, loc, tensorType, splatAttr);
}

/// Perform 2D matrix multiplication using tosa.matmul.
/// tosa.matmul requires 3D tensors [B, M, K] x [B, K, N] -> [B, M, N],
/// so we expand 2D to 3D with batch=1, compute, and collapse back.
static Value matmul2D(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  auto rhsType = cast<RankedTensorType>(rhs.getType());
  Type elemType = lhsType.getElementType();

  ArrayRef<int64_t> lhsShape = lhsType.getShape(); // [M, K]
  ArrayRef<int64_t> rhsShape = rhsType.getShape(); // [K, N]

  int64_t M = lhsShape[0];
  int64_t K = lhsShape[1];
  int64_t N = rhsShape[1];

  // Expand to 3D: [M, K] -> [1, M, K], [K, N] -> [1, K, N]
  auto lhs3DType = RankedTensorType::get({1, M, K}, elemType);
  auto rhs3DType = RankedTensorType::get({1, K, N}, elemType);

  SmallVector<ReassociationIndices> expandReassoc = {{0, 1}, {2}};
  Value lhs3D = tensor::ExpandShapeOp::create(builder, loc, lhs3DType, lhs,
                                              expandReassoc);
  Value rhs3D = tensor::ExpandShapeOp::create(builder, loc, rhs3DType, rhs,
                                              expandReassoc);

  // Create zero point tensors for tosa.matmul
  auto lhsZp = tosa::createZeroPointTensor(builder, loc, lhs3DType, 0).value();
  auto rhsZp = tosa::createZeroPointTensor(builder, loc, rhs3DType, 0).value();

  // Perform tosa.matmul: [1, M, K] x [1, K, N] -> [1, M, N]
  auto result3DType = RankedTensorType::get({1, M, N}, elemType);
  auto matmulOp = tosa::MatMulOp::create(builder, loc, result3DType, lhs3D,
                                         rhs3D, lhsZp, rhsZp);

  // Collapse back to 2D: [1, M, N] -> [M, N]
  auto result2DType = RankedTensorType::get({M, N}, elemType);
  SmallVector<ReassociationIndices> collapseReassoc = {{0, 1}, {2}};
  return tensor::CollapseShapeOp::create(builder, loc, result2DType,
                                         matmulOp.getResult(), collapseReassoc);
}

/// Extract 2D slice from 4D tensor: tensor[f, :, :, c] for FHWC layout.
/// Returns a tensor of shape [H, W].
static Value extract2DFrom4DFHWC(OpBuilder &builder, Location loc, Value tensor,
                                 Value fIdx, Value cIdx, int64_t height,
                                 int64_t width) {
  auto tensorType = cast<ShapedType>(tensor.getType());
  Type elemType = tensorType.getElementType();
  int64_t rank = tensorType.getRank();

  auto oneIndex = builder.getIndexAttr(1);
  auto zeroIndex = builder.getIndexAttr(0);

  SmallVector<OpFoldResult> offsets(rank, zeroIndex);
  offsets[0] = fIdx;
  offsets[3] = cIdx;

  SmallVector<OpFoldResult> sizes(rank, oneIndex);
  sizes[1] = builder.getIndexAttr(height);
  sizes[2] = builder.getIndexAttr(width);

  SmallVector<OpFoldResult> strides(rank, oneIndex);

  auto resultType = RankedTensorType::get({height, width}, elemType);
  return tensor::ExtractSliceOp::create(builder, loc, resultType, tensor,
                                        offsets, sizes, strides);
}

/// Insert 2D slice into 4D tensor at position [i, :, :, j] for HWCF layout.
static Value insert2DTo4DHWCF(OpBuilder &builder, Location loc, Value slice,
                              Value dest, Value cIdx, Value fIdx,
                              int64_t height, int64_t width) {
  int64_t destRank = cast<ShapedType>(dest.getType()).getRank();
  auto oneIndex = builder.getIndexAttr(1);
  auto zeroIndex = builder.getIndexAttr(0);

  SmallVector<OpFoldResult> offsets(destRank, zeroIndex);
  offsets[2] = cIdx;
  offsets[3] = fIdx;

  SmallVector<OpFoldResult> sizes(destRank, oneIndex);
  sizes[0] = builder.getIndexAttr(height);
  sizes[1] = builder.getIndexAttr(width);

  SmallVector<OpFoldResult> strides(destRank, oneIndex);

  return tensor::InsertSliceOp::create(builder, loc, slice, dest, offsets,
                                       sizes, strides);
}

/// Extract 2D slice from 6D tensor at [h, w, :, :, n, c].
static Value extract2DFrom6D(OpBuilder &builder, Location loc, Value tensor,
                             Value tileHIdx, Value tileWIdx, Value nIdx,
                             Value cIdx, int64_t height, int64_t width) {
  auto tensorType = cast<ShapedType>(tensor.getType());
  Type elemType = tensorType.getElementType();
  int64_t rank = tensorType.getRank();

  auto oneIndex = builder.getIndexAttr(1);
  auto zeroIndex = builder.getIndexAttr(0);

  SmallVector<OpFoldResult> offsets(rank, zeroIndex);
  offsets[2] = tileHIdx;
  offsets[3] = tileWIdx;
  offsets[4] = nIdx;
  offsets[5] = cIdx;

  SmallVector<OpFoldResult> sizes(rank, oneIndex);
  sizes[0] = builder.getIndexAttr(height);
  sizes[1] = builder.getIndexAttr(width);

  SmallVector<OpFoldResult> strides(rank, oneIndex);

  auto resultType = RankedTensorType::get({height, width}, elemType);
  return tensor::ExtractSliceOp::create(builder, loc, resultType, tensor,
                                        offsets, sizes, strides);
}

/// Insert 2D slice into 6D tensor at [h, w, :, :, n, c].
static Value insert2DTo6D(OpBuilder &builder, Location loc, Value slice,
                          Value dest, Value tileHIdx, Value tileWIdx,
                          Value nIdx, Value cIdx, int64_t height,
                          int64_t width) {
  int64_t destRank = cast<ShapedType>(dest.getType()).getRank();
  auto oneIndex = builder.getIndexAttr(1);
  auto zeroIndex = builder.getIndexAttr(0);

  SmallVector<OpFoldResult> offsets(destRank, zeroIndex);
  offsets[2] = tileHIdx;
  offsets[3] = tileWIdx;
  offsets[4] = nIdx;
  offsets[5] = cIdx;

  SmallVector<OpFoldResult> sizes(destRank, oneIndex);
  sizes[0] = builder.getIndexAttr(height);
  sizes[1] = builder.getIndexAttr(width);

  SmallVector<OpFoldResult> strides(destRank, oneIndex);

  return tensor::InsertSliceOp::create(builder, loc, slice, dest, offsets,
                                       sizes, strides);
}

/// Pad tensor to aligned shape using tosa.pad.
static Value padToAlignedTensor(OpBuilder &builder, Location loc, Value tensor,
                                ArrayRef<int64_t> alignedShape) {
  auto tensorType = cast<ShapedType>(tensor.getType());
  Type elemType = tensorType.getElementType();
  ArrayRef<int64_t> inputShape = tensorType.getShape();
  auto alignedType = RankedTensorType::get(alignedShape, elemType);

  // Calculate padding amounts: [pad_before_dim0, pad_after_dim0, ...]
  // tosa.pad expects shape [N * 2] where N is number of dimensions
  int64_t rank = tensorType.getRank();
  SmallVector<int64_t> paddingValues;
  for (int64_t i = 0; i < rank; ++i) {
    int64_t padBefore = 0;
    int64_t padAfter = alignedShape[i] - inputShape[i];
    paddingValues.push_back(padBefore);
    paddingValues.push_back(padAfter);
  }

  // Create padding shape constant using tosa.const_shape
  auto shapeType =
      tosa::shapeType::get(builder.getContext(), paddingValues.size());
  Value paddingShape = tosa::ConstShapeOp::create(
      builder, loc, shapeType, builder.getIndexTensorAttr(paddingValues));

  // Create pad constant value
  Value padConst = tosa::createPadConstTensor(builder, loc, tensor, 0);

  return tosa::PadOp::create(builder, loc, alignedType, tensor, paddingShape,
                             padConst);
}

/// Extract sub-tensor from aligned tensor.
static Value extractFromAlignedTensor(OpBuilder &builder, Location loc,
                                      Value tensor,
                                      RankedTensorType extractedType) {
  OpFoldResult zeroIndex = builder.getIndexAttr(0);
  OpFoldResult oneIndex = builder.getIndexAttr(1);

  SmallVector<OpFoldResult, 4> offsets(4, zeroIndex);
  SmallVector<OpFoldResult, 4> strides(4, oneIndex);

  ArrayRef<int64_t> extractedShape = extractedType.getShape();
  SmallVector<OpFoldResult> sizes;
  for (int64_t dim : extractedShape) {
    sizes.push_back(builder.getIndexAttr(dim));
  }

  return tensor::ExtractSliceOp::create(builder, loc, extractedType, tensor,
                                        offsets, sizes, strides);
}

//===----------------------------------------------------------------------===//
// Winograd Transform Functions
//===----------------------------------------------------------------------===//

/// Transform filter using G and G^T matrices.
/// Filter layout: FHWC -> HWCF (after transform)
/// For each (f, c) pair:
///   transformed_filter[:, :, c, f] = G * filter[f, :, :, c] * G^T
static Value filterTransform(OpBuilder &builder, Location loc, Value filter,
                             Value output, WinogradFmr fmr) {
  auto filterType = cast<ShapedType>(filter.getType());
  Type elemType = filterType.getElementType();
  ArrayRef<int64_t> filterShape = filterType.getShape(); // [F, H, W, C]

  int64_t filterF = filterShape[0];
  int64_t filterH = filterShape[1];
  int64_t filterW = filterShape[2];
  int64_t filterC = filterShape[3];

  auto fmrValues = getFmrValues(fmr);
  int64_t r = fmrValues.second;

  // Verify filter dimensions
  if (filterH != r || filterW != r)
    return Value();

  TransformMatrix GMatrix = getGMatrix(fmr);
  TransformMatrix GTMatrix = getGTMatrix(fmr);

  Value G = createTransformMatrix(builder, loc, GMatrix, elemType);
  Value GT = createTransformMatrix(builder, loc, GTMatrix, elemType);

  // Loop bounds
  Value zeroIdx = arith::ConstantIndexOp::create(builder, loc, 0);
  Value oneStep = arith::ConstantIndexOp::create(builder, loc, 1);
  Value fUpperBound = arith::ConstantIndexOp::create(builder, loc, filterF);
  Value cUpperBound = arith::ConstantIndexOp::create(builder, loc, filterC);

  // Nested loops: for f in [0, F), for c in [0, C)
  auto outerLoop = scf::ForOp::create(
      builder, loc, zeroIdx, fUpperBound, oneStep, ValueRange{output},
      [&](OpBuilder &b, Location loc, Value fIter, ValueRange outerArgs) {
        auto innerLoop = scf::ForOp::create(
            b, loc, zeroIdx, cUpperBound, oneStep, ValueRange{outerArgs[0]},
            [&](OpBuilder &b2, Location loc, Value cIter,
                ValueRange innerArgs) {
              // Extract filter[f, :, :, c]
              Value extractedFilter = extract2DFrom4DFHWC(
                  b2, loc, filter, fIter, cIter, filterH, filterW);

              // temp = G * filter using tosa.matmul
              Value temp = matmul2D(b2, loc, G, extractedFilter);

              // result = temp * G^T using tosa.matmul
              Value result = matmul2D(b2, loc, temp, GT);

              // Insert result into output[h, w, c, f]
              int64_t tempRows = GMatrix.rows;
              int64_t resultCols = GTMatrix.cols;
              Value updated =
                  insert2DTo4DHWCF(b2, loc, result, innerArgs[0], cIter, fIter,
                                   tempRows, resultCols);
              scf::YieldOp::create(b2, loc, ValueRange{updated});
            });
        scf::YieldOp::create(b, loc, innerLoop.getResults());
      });

  return outerLoop.getResult(0);
}

/// Transform input using B^T and B matrices.
/// Input layout: NHWC -> (alphaH, alphaW, tileH, tileW, N, C)
/// For each tile (th, tw) and each (n, c) pair:
///   transformed_input[:, :, th, tw, n, c] = B^T * input_tile * B
static Value inputTransform(OpBuilder &builder, Location loc, Value input,
                            Value output, WinogradFmr fmr, int64_t tileH,
                            int64_t tileW) {
  auto inputType = cast<ShapedType>(input.getType());
  Type elemType = inputType.getElementType();
  ArrayRef<int64_t> inputShape = inputType.getShape(); // [N, H, W, C]

  int64_t inputN = inputShape[0];
  int64_t inputC = inputShape[3];

  auto fmrValues = getFmrValues(fmr);
  int64_t m = fmrValues.first;
  int64_t r = fmrValues.second;
  int64_t alpha = m + r - 1;

  TransformMatrix BTMatrix = getBTMatrix(fmr);
  TransformMatrix BMatrix = getBMatrix(fmr);

  Value BT = createTransformMatrix(builder, loc, BTMatrix, elemType);
  Value B = createTransformMatrix(builder, loc, BMatrix, elemType);

  // Loop bounds
  Value zeroIdx = arith::ConstantIndexOp::create(builder, loc, 0);
  Value oneStep = arith::ConstantIndexOp::create(builder, loc, 1);
  Value tileHBound = arith::ConstantIndexOp::create(builder, loc, tileH);
  Value tileWBound = arith::ConstantIndexOp::create(builder, loc, tileW);
  Value nUpperBound = arith::ConstantIndexOp::create(builder, loc, inputN);
  Value cUpperBound = arith::ConstantIndexOp::create(builder, loc, inputC);

  // 4-level nested loops: tileH, tileW, N, C
  auto tileHLoop = scf::ForOp::create(
      builder, loc, zeroIdx, tileHBound, oneStep, ValueRange{output},
      [&](OpBuilder &b1, Location loc, Value thIter, ValueRange args1) {
        auto tileWLoop = scf::ForOp::create(
            b1, loc, zeroIdx, tileWBound, oneStep, ValueRange{args1[0]},
            [&](OpBuilder &b2, Location loc, Value twIter, ValueRange args2) {
              auto nLoop = scf::ForOp::create(
                  b2, loc, zeroIdx, nUpperBound, oneStep, ValueRange{args2[0]},
                  [&](OpBuilder &b3, Location loc, Value nIter,
                      ValueRange args3) {
                    auto cLoop = scf::ForOp::create(
                        b3, loc, zeroIdx, cUpperBound, oneStep,
                        ValueRange{args3[0]},
                        [&](OpBuilder &b4, Location loc, Value cIter,
                            ValueRange args4) {
                          // Calculate input offset
                          Value mVal =
                              arith::ConstantIndexOp::create(b4, loc, m);
                          Value hOffset =
                              arith::MulIOp::create(b4, loc, thIter, mVal);
                          Value wOffset =
                              arith::MulIOp::create(b4, loc, twIter, mVal);

                          // Extract input tile
                          auto tensorType = cast<ShapedType>(input.getType());
                          int64_t rank = tensorType.getRank();

                          auto oneIndex = b4.getIndexAttr(1);
                          SmallVector<OpFoldResult> offsets(rank,
                                                            b4.getIndexAttr(0));
                          offsets[0] = nIter;
                          offsets[1] = hOffset;
                          offsets[2] = wOffset;
                          offsets[3] = cIter;

                          SmallVector<OpFoldResult> sizes(rank, oneIndex);
                          sizes[1] = b4.getIndexAttr(alpha);
                          sizes[2] = b4.getIndexAttr(alpha);

                          SmallVector<OpFoldResult> strides(rank, oneIndex);

                          auto tileType =
                              RankedTensorType::get({alpha, alpha}, elemType);
                          Value inputTile = tensor::ExtractSliceOp::create(
                              b4, loc, tileType, input, offsets, sizes,
                              strides);

                          // temp = B^T * input_tile using tosa.matmul
                          Value temp = matmul2D(b4, loc, BT, inputTile);

                          // result = temp * B using tosa.matmul
                          Value result = matmul2D(b4, loc, temp, B);

                          // Insert into output[:, :, th, tw, n, c]
                          int64_t tempRows = BTMatrix.rows;
                          int64_t resultCols = BMatrix.cols;
                          Value updated = insert2DTo6D(
                              b4, loc, result, args4[0], thIter, twIter, nIter,
                              cIter, tempRows, resultCols);
                          scf::YieldOp::create(b4, loc, ValueRange{updated});
                        });
                    scf::YieldOp::create(b3, loc, cLoop.getResults());
                  });
              scf::YieldOp::create(b2, loc, nLoop.getResults());
            });
        scf::YieldOp::create(b1, loc, tileWLoop.getResults());
      });

  return tileHLoop.getResult(0);
}

/// Perform batched matrix multiplication in Winograd domain.
/// Input: (alphaH, alphaW, tileH, tileW, N, C)
/// Filter: (alphaH, alphaW, C, F)
/// Output: (alphaH, alphaW, tileH, tileW, N, F)
static Value winogradBatchMatmul(OpBuilder &builder, Location loc,
                                 Value transformedFilter,
                                 Value transformedInput, Type outputElemType) {
  auto filterType = cast<ShapedType>(transformedFilter.getType());
  auto inputType = cast<ShapedType>(transformedInput.getType());

  ArrayRef<int64_t> filterShape =
      filterType.getShape(); // [alphaH, alphaW, C, F]
  ArrayRef<int64_t> inputShape =
      inputType.getShape(); // [alphaH, alphaW, tileH, tileW, N, C]

  int64_t alphaH = filterShape[0];
  int64_t alphaW = filterShape[1];
  int64_t filterC = filterShape[2];
  int64_t filterF = filterShape[3];
  int64_t tileH = inputShape[2];
  int64_t tileW = inputShape[3];
  int64_t inputN = inputShape[4];

  // Collapse filter: [alphaH, alphaW, C, F] -> [alphaH * alphaW, C, F]
  auto filterReassocType = RankedTensorType::get(
      {alphaH * alphaW, filterC, filterF}, filterType.getElementType());
  SmallVector<ReassociationIndices> filterReassoc = {{0, 1}, {2}, {3}};
  Value collapsedFilter = tensor::CollapseShapeOp::create(
      builder, loc, filterReassocType, transformedFilter, filterReassoc);

  // Collapse input: [alphaH, alphaW, tileH, tileW, N, C] -> [alphaH * alphaW,
  // tileH * tileW * N, C]
  auto inputReassocType =
      RankedTensorType::get({alphaH * alphaW, tileH * tileW * inputN, filterC},
                            inputType.getElementType());
  SmallVector<ReassociationIndices> inputReassoc = {{0, 1}, {2, 3, 4}, {5}};
  Value collapsedInput = tensor::CollapseShapeOp::create(
      builder, loc, inputReassocType, transformedInput, inputReassoc);

  // tosa.matmul: [batch, M, K] x [batch, K, N] -> [batch, M, N]
  // Here: [alphaH * alphaW, tileH * tileW * N, C] x [alphaH * alphaW, C, F]
  //    -> [alphaH * alphaW, tileH * tileW * N, F]
  auto matmulType = RankedTensorType::get(
      {alphaH * alphaW, tileH * tileW * inputN, filterF}, outputElemType);

  // Create zero point tensors for tosa.matmul
  auto inputZp =
      tosa::createZeroPointTensor(builder, loc, collapsedInput.getType(), 0)
          .value();
  auto filterZp =
      tosa::createZeroPointTensor(builder, loc, collapsedFilter.getType(), 0)
          .value();

  auto batchMatmul =
      tosa::MatMulOp::create(builder, loc, matmulType, collapsedInput,
                             collapsedFilter, inputZp, filterZp);

  // Expand result: [alphaH * alphaW, tileH * tileW * N, F] -> [alphaH, alphaW,
  // tileH, tileW, N, F]
  SmallVector<ReassociationIndices> outputReassoc = {{0, 1}, {2, 3, 4}, {5}};
  auto outputExpandType = RankedTensorType::get(
      {alphaH, alphaW, tileH, tileW, inputN, filterF}, outputElemType);
  return tensor::ExpandShapeOp::create(builder, loc, outputExpandType,
                                       batchMatmul.getResult(), outputReassoc);
}

/// Transform output using A^T and A matrices.
/// Input: (alphaH, alphaW, tileH, tileW, N, F)
/// Output: NHWF format
static Value outputTransform(OpBuilder &builder, Location loc, Value value,
                             Value output, WinogradFmr fmr) {
  auto valueType = cast<ShapedType>(value.getType());
  Type elemType = valueType.getElementType();
  ArrayRef<int64_t> valueShape =
      valueType.getShape(); // [alphaH, alphaW, tileH, tileW, N, F]

  int64_t alphaH = valueShape[0];
  int64_t alphaW = valueShape[1];
  int64_t tileH = valueShape[2];
  int64_t tileW = valueShape[3];
  int64_t valueN = valueShape[4];
  int64_t valueF = valueShape[5];

  auto fmrValues = getFmrValues(fmr);
  int64_t m = fmrValues.first;

  TransformMatrix ATMatrix = getATMatrix(fmr);
  TransformMatrix AMatrix = getAMatrix(fmr);

  Value AT = createTransformMatrix(builder, loc, ATMatrix, elemType);
  Value A = createTransformMatrix(builder, loc, AMatrix, elemType);

  // Loop bounds
  Value zeroIdx = arith::ConstantIndexOp::create(builder, loc, 0);
  Value oneStep = arith::ConstantIndexOp::create(builder, loc, 1);
  Value tileHBound = arith::ConstantIndexOp::create(builder, loc, tileH);
  Value tileWBound = arith::ConstantIndexOp::create(builder, loc, tileW);
  Value nUpperBound = arith::ConstantIndexOp::create(builder, loc, valueN);
  Value fUpperBound = arith::ConstantIndexOp::create(builder, loc, valueF);

  // 4-level nested loops: tileH, tileW, N, F
  auto tileHLoop = scf::ForOp::create(
      builder, loc, zeroIdx, tileHBound, oneStep, ValueRange{output},
      [&](OpBuilder &b1, Location loc, Value thIter, ValueRange args1) {
        auto tileWLoop = scf::ForOp::create(
            b1, loc, zeroIdx, tileWBound, oneStep, ValueRange{args1[0]},
            [&](OpBuilder &b2, Location loc, Value twIter, ValueRange args2) {
              auto nLoop = scf::ForOp::create(
                  b2, loc, zeroIdx, nUpperBound, oneStep, ValueRange{args2[0]},
                  [&](OpBuilder &b3, Location loc, Value nIter,
                      ValueRange args3) {
                    auto fLoop = scf::ForOp::create(
                        b3, loc, zeroIdx, fUpperBound, oneStep,
                        ValueRange{args3[0]},
                        [&](OpBuilder &b4, Location loc, Value fIter,
                            ValueRange args4) {
                          // Extract value[:, :, th, tw, n, f]
                          Value extracted =
                              extract2DFrom6D(b4, loc, value, thIter, twIter,
                                              nIter, fIter, alphaH, alphaW);

                          // temp = A^T * extracted using tosa.matmul
                          Value temp = matmul2D(b4, loc, AT, extracted);

                          // result = temp * A using tosa.matmul
                          Value result = matmul2D(b4, loc, temp, A);

                          // Calculate output offset
                          Value mVal =
                              arith::ConstantIndexOp::create(b4, loc, m);
                          Value hOffset =
                              arith::MulIOp::create(b4, loc, thIter, mVal);
                          Value wOffset =
                              arith::MulIOp::create(b4, loc, twIter, mVal);

                          // Insert into output[n, h_offset:h_offset+m,
                          // w_offset:w_offset+m, f]
                          int64_t tempRows = ATMatrix.rows;
                          int64_t resultCols = AMatrix.cols;
                          auto outputType =
                              cast<ShapedType>(args4[0].getType());
                          int64_t outRank = outputType.getRank();

                          auto oneIndex = b4.getIndexAttr(1);
                          SmallVector<OpFoldResult> offsets(outRank,
                                                            b4.getIndexAttr(0));
                          offsets[0] = nIter;
                          offsets[1] = hOffset;
                          offsets[2] = wOffset;
                          offsets[3] = fIter;

                          SmallVector<OpFoldResult> sizes(outRank, oneIndex);
                          sizes[1] = b4.getIndexAttr(tempRows);
                          sizes[2] = b4.getIndexAttr(resultCols);

                          SmallVector<OpFoldResult> strides(outRank, oneIndex);

                          Value updated = tensor::InsertSliceOp::create(
                              b4, loc, result, args4[0], offsets, sizes,
                              strides);
                          scf::YieldOp::create(b4, loc, ValueRange{updated});
                        });
                    scf::YieldOp::create(b3, loc, fLoop.getResults());
                  });
              scf::YieldOp::create(b2, loc, nLoop.getResults());
            });
        scf::YieldOp::create(b1, loc, tileWLoop.getResults());
      });

  return tileHLoop.getResult(0);
}

//===----------------------------------------------------------------------===//
// Winograd Conv2D Pattern
//===----------------------------------------------------------------------===//

/// Pattern to convert tosa.conv2d to Winograd convolution.
/// Only handles NHWC input and FHWC filter layouts with 3x3 kernels.
class WinogradConv2DPattern : public OpRewritePattern<tosa::Conv2DOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  WinogradConv2DPattern(MLIRContext *context, WinogradFmr fmr)
      : OpRewritePattern(context), fmr(fmr) {}

  LogicalResult matchAndRewrite(tosa::Conv2DOp convOp,
                                PatternRewriter &rewriter) const override {
    Location loc = convOp.getLoc();

    // Get operands
    Value input = convOp.getInput();
    Value filter = convOp.getWeight();
    Value bias = convOp.getBias();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto filterType = cast<RankedTensorType>(filter.getType());
    auto outputType = cast<RankedTensorType>(convOp.getType());

    // Check for static shapes
    if (!inputType.hasStaticShape() || !filterType.hasStaticShape())
      return rewriter.notifyMatchFailure(convOp, "requires static shapes");

    // Check dilations and strides are all 1
    if (!hasAllOnes(convOp.getDilationAttr()))
      return rewriter.notifyMatchFailure(convOp, "requires unit dilation");

    if (!hasAllOnes(convOp.getStrideAttr()))
      return rewriter.notifyMatchFailure(convOp, "requires unit stride");

    // Get shapes - NHWC for input, FHWC for filter
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> filterShape = filterType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();

    int64_t inputN = inputShape[0];
    int64_t inputH = inputShape[1];
    int64_t inputW = inputShape[2];
    int64_t inputC = inputShape[3];

    int64_t filterF = filterShape[0];
    int64_t filterH = filterShape[1];
    int64_t filterW = filterShape[2];
    int64_t filterC = filterShape[3];

    int64_t outputN = outputShape[0];
    int64_t outputH = outputShape[1];
    int64_t outputW = outputShape[2];
    int64_t outputF = outputShape[3];

    // Verify channel consistency
    if (inputC != filterC)
      return rewriter.notifyMatchFailure(convOp, "channel dimension mismatch");

    // Get Winograd parameters
    auto [m, r] = getFmrValues(fmr);

    // Check filter size matches Winograd configuration
    if (filterH != r || filterW != r)
      return rewriter.notifyMatchFailure(
          convOp, "filter size doesn't match Winograd configuration");

    Type elemType = inputType.getElementType();

    // Calculate tiling parameters
    int64_t alpha = m + r - 1;
    int64_t tileH = llvm::divideCeil(outputH, m);
    int64_t tileW = llvm::divideCeil(outputW, m);

    // --- Filter Transform ---
    // Output shape: [alphaH, alphaW, C, F] (HWCF layout for the transform)
    auto transformedFilterType =
        RankedTensorType::get({alpha, alpha, filterC, filterF}, elemType);
    Value transformedFilterInit =
        tensor::EmptyOp::create(rewriter, loc, transformedFilterType.getShape(),
                                elemType)
            .getResult();

    Value transformedFilter =
        filterTransform(rewriter, loc, filter, transformedFilterInit, fmr);
    if (!transformedFilter)
      return rewriter.notifyMatchFailure(convOp, "filter transform failed");

    // --- Input Transform ---
    // Pad input if necessary
    int64_t alignedInputH = tileH * m + (r - 1);
    int64_t alignedInputW = tileW * m + (r - 1);

    Value paddedInput = input;
    if (alignedInputH != inputH || alignedInputW != inputW) {
      paddedInput = padToAlignedTensor(
          rewriter, loc, input, {inputN, alignedInputH, alignedInputW, inputC});
    }

    // Output shape: [alphaH, alphaW, tileH, tileW, N, C]
    auto transformedInputType = RankedTensorType::get(
        {alpha, alpha, tileH, tileW, inputN, inputC}, elemType);
    Value transformedInputInit =
        tensor::EmptyOp::create(rewriter, loc, transformedInputType.getShape(),
                                elemType)
            .getResult();

    Value transformedInput = inputTransform(
        rewriter, loc, paddedInput, transformedInputInit, fmr, tileH, tileW);
    if (!transformedInput)
      return rewriter.notifyMatchFailure(convOp, "input transform failed");

    // --- Batched Matrix Multiplication ---
    // Input: [alpha, alpha, tileH, tileW, N, C]
    // Filter: [alpha, alpha, C, F]
    // Output: [alpha, alpha, tileH, tileW, N, F]
    Type outputElemType = outputType.getElementType();
    Value matmulResult = winogradBatchMatmul(rewriter, loc, transformedFilter,
                                             transformedInput, outputElemType);

    // --- Output Transform ---
    // Calculate aligned output size
    int64_t alignedOutputH = tileH * m;
    int64_t alignedOutputW = tileW * m;

    Value outputInit;
    RankedTensorType alignedOutputType;
    bool needsExtract =
        (alignedOutputH != outputH || alignedOutputW != outputW);

    if (needsExtract) {
      alignedOutputType = RankedTensorType::get(
          {outputN, alignedOutputH, alignedOutputW, outputF}, outputElemType);
    } else {
      alignedOutputType = outputType;
    }

    outputInit = createZeroTensor(rewriter, loc, alignedOutputType);

    Value transformedOutput =
        outputTransform(rewriter, loc, matmulResult, outputInit, fmr);
    if (!transformedOutput)
      return rewriter.notifyMatchFailure(convOp, "output transform failed");

    // Extract if we padded
    if (needsExtract) {
      transformedOutput = extractFromAlignedTensor(
          rewriter, loc, transformedOutput,
          RankedTensorType::get({outputN, outputH, outputW, outputF},
                                outputElemType));
    }

    // Add bias if non-zero
    Type biasTypeVal = bias.getType();
    if (auto biasRankedType = dyn_cast<RankedTensorType>(biasTypeVal)) {
      // Check if bias is zero
      if (auto constOp = bias.getDefiningOp<arith::ConstantOp>()) {
        if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
          if (denseAttr.isSplat() &&
              denseAttr.getSplatValue<APFloat>().isZero()) {
            // Bias is zero, skip adding
            rewriter.replaceOp(convOp, transformedOutput);
            return success();
          }
        }
      }

      // Add bias - broadcast bias [F] to [N, H, W, F]
      auto addOp = tosa::AddOp::create(rewriter, loc, outputType,
                                       transformedOutput, bias);
      rewriter.replaceOp(convOp, addOp.getResult());
    } else {
      rewriter.replaceOp(convOp, transformedOutput);
    }

    return success();
  }

private:
  WinogradFmr fmr;
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct RockWinogradConv
    : public impl::RockWinogradConvPassBase<RockWinogradConv> {
public:
  void runOnOperation() override {
    auto func = getOperation();
    auto &ctx = getContext();

    RewritePatternSet patterns(&ctx);

    // Add Winograd conv2d pattern for 3x3 filters using F(2, 3)
    patterns.add<WinogradConv2DPattern>(&ctx, WinogradFmr::F_2_3);

    // Apply patterns greedily
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
