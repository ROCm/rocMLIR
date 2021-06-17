#include "PassDetail.h"

#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
struct AffineTransforms : public MIOpenOpsAffineTransformPassBase<AffineTransforms> {
  void runOnFunction() override;

private:
  AffineMap buildIndexAffineMap(miopen::TransformOp);
};
} // anonymous namespace

AffineMap AffineTransforms::buildIndexAffineMap(miopen::TransformOp op) {
  auto inputType = op.input().getType().cast<MemRefType>();
  auto inputShape = inputType.getShape();

  auto layoutAttr = op->template getAttrOfType<ArrayAttr>("layout");

  auto sourceLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("lower_layer_layout");
  auto outputLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("upper_layer_layout");

  llvm::SmallMapVector<int64_t, AffineExpr, 8> affExprsMap;
  for (unsigned i = 0; i < layoutAttr.size(); ++i) {
    if (auto dimLayoutAttr = layoutAttr.getValue()[i].cast<DictionaryAttr>()) {
      auto srcDimAttr =
          dimLayoutAttr.get("lower_layer_dimensions").cast<ArrayAttr>();
      auto destDimAttr =
          dimLayoutAttr.get("upper_layer_dimensions").cast<ArrayAttr>();
      auto transformAttr =
          dimLayoutAttr.get("transformation").cast<StringAttr>();

      if (transformAttr.getValue() == "PassThrough") {
        assert(srcDimAttr.size() == 1);
        assert(destDimAttr.size() == 1);

        auto srcDim = srcDimAttr.getValue()[0].cast<IntegerAttr>().getInt();
        auto destDim = destDimAttr.getValue()[0].cast<IntegerAttr>().getInt();
        auto expr = getAffineDimExpr(destDim, op.getContext());
        affExprsMap.insert({srcDim, expr});
      } else if (transformAttr.getValue() == "Pad") {
        assert(srcDimAttr.size() == destDimAttr.size());

        auto parameters = dimLayoutAttr.get("parameters").cast<ArrayAttr>();
        for (unsigned j = 0; j < srcDimAttr.size(); ++j) {
          // example of h and w pad parameters [0, 2, 3, 1] :
          // leftpadH = 0 rightPadH = 2 leftpadW = 3 rightPadW = 1
          // first run leftPad = 0 rightPad = 2
          // second run leftPad = 3 rightPad = 1
          // if your pad is one dim , example of pad parameters [1,2]
          // leftPad = 1 rightPad = 2
          auto leftPad =
              parameters.getValue()[j * 2].cast<IntegerAttr>().getInt();

          auto srcDim = srcDimAttr.getValue()[j].cast<IntegerAttr>().getInt();
          auto destDim = destDimAttr.getValue()[j].cast<IntegerAttr>().getInt();

          auto expr = getAffineDimExpr(destDim, op.getContext()) + getAffineConstantExpr(-leftPad, op.getContext());

          affExprsMap.insert({srcDim, expr});
        }
      } else if (transformAttr.getValue() == "Merge" ||
                 transformAttr.getValue() == "Unfold") {
        assert(destDimAttr.size() == 1);
        assert(srcDimAttr.size() > 1);

        auto destDim = destDimAttr.getValue()[0].cast<IntegerAttr>().getInt();

        // Find source dimension lengths.
        llvm::SmallVector<int64_t, 4> srcDimLengthVec;
        for (unsigned j = 0; j < srcDimAttr.size(); ++j) {
          auto srcDim = srcDimAttr.getValue()[j].cast<IntegerAttr>().getInt();
          auto srcDimLength = inputShape[srcDim];
          srcDimLengthVec.push_back(srcDimLength);
        }

        // Compute source dimension strides.
        llvm::SmallVector<int64_t, 4> srcDimStrideVec;
        int64_t stride = 1;
        srcDimStrideVec.push_back(stride);
        for (unsigned j = srcDimAttr.size() - 1; j > 0; --j) {
          stride *= srcDimLengthVec[j];
          srcDimStrideVec.push_back(stride);
        }
        std::reverse(srcDimStrideVec.begin(), srcDimStrideVec.end());

        // Build affine transformation expressions.
        auto remainderExpr = getAffineDimExpr(destDim, op.getContext());
        for (unsigned j = 0; j < srcDimAttr.size(); ++j) {
          auto strideExpr = getAffineConstantExpr(srcDimStrideVec[j], op.getContext());
          auto expr = remainderExpr.floorDiv(strideExpr);
          remainderExpr = remainderExpr % strideExpr;

          auto srcDim = srcDimAttr.getValue()[j].cast<IntegerAttr>().getInt();
          affExprsMap.insert({srcDim, expr});
        }
      } else if ((transformAttr.getValue() == "UnMerge2") ||
                 (transformAttr.getValue() == "UnMerge")) {
        assert(srcDimAttr.size() == 1);
        assert(destDimAttr.size() > 1);
        // output data
        auto outputType = op.output().getType().cast<MemRefType>();
        auto outputShape = outputType.getShape();

        auto srcDim = srcDimAttr.getValue()[0].cast<IntegerAttr>().getInt();
        auto destDim = destDimAttr.getValue()[0].cast<IntegerAttr>().getInt();
        auto expr = getAffineDimExpr(destDim, op.getContext());
        auto dimLength =
            dimLayoutAttr.get("dimension_lengths").cast<ArrayAttr>();
        for (unsigned j = 1; j < destDimAttr.size(); ++j) {
          destDim = destDimAttr.getValue()[j].cast<IntegerAttr>().getInt();
          auto length = dimLength.getValue()[j].cast<IntegerAttr>().getInt();
          assert(length == outputShape[destDim]);

          auto lengthExpr = getAffineConstantExpr(length, op.getContext());
          auto partialExpr = getAffineDimExpr(destDim, op.getContext());
          expr = expr * lengthExpr + partialExpr;
        }
        affExprsMap.insert({srcDim, expr});
      } else if (transformAttr.getValue() == "Embed") {
        assert(srcDimAttr.size() == 1);
        assert(destDimAttr.size() > 1);

        auto srcDim = srcDimAttr.getValue()[0].cast<IntegerAttr>().getInt();
        auto parameters = dimLayoutAttr.get("parameters").cast<ArrayAttr>();

        // Build affine transformation expressions.
        AffineExpr expr = getAffineConstantExpr(0, op.getContext());
        for (unsigned j = 0; j < destDimAttr.size(); ++j) {
          auto destDim = destDimAttr.getValue()[j].cast<IntegerAttr>().getInt();
          int64_t param = parameters.getValue()[j].cast<IntegerAttr>().getInt();
          auto partialExpr = getAffineDimExpr(destDim, op.getContext()) * getAffineConstantExpr(param, op.getContext());
          expr = expr + partialExpr;
        }
        affExprsMap.insert({srcDim, expr});
      } else if (transformAttr.getValue() == "Slice") {
        assert(srcDimAttr.size() >= 1);
        assert(srcDimAttr.size() == destDimAttr.size());

        auto begins = dimLayoutAttr.get("begins").cast<ArrayAttr>();
        auto ends = dimLayoutAttr.get("ends").cast<ArrayAttr>();
        assert(begins.size() == ends.size());
        // same dim
        assert(begins.size() == srcDimAttr.size());

        // output data
        auto outputType = op.output().getType().cast<MemRefType>();
        auto outputShape = outputType.getShape();

        for (unsigned j = 0; j < begins.size(); j++) {
          auto begin = begins.getValue()[j].cast<IntegerAttr>().getInt();
          auto end = ends.getValue()[j].cast<IntegerAttr>().getInt();
          auto destDim = destDimAttr.getValue()[j].cast<IntegerAttr>().getInt();
          auto length = outputShape[destDim];
          assert(length == (end - begin));

          auto expr = getAffineDimExpr(destDim, op.getContext()) +
                      getAffineConstantExpr(begin, op.getContext());
          auto srcDim = srcDimAttr.getValue()[j].cast<IntegerAttr>().getInt();
          affExprsMap.insert({srcDim, expr});
        }
      }
    }
  }

  llvm::SmallVector<AffineExpr, 8> affExprsVec;
  for (unsigned i = 0; i < sourceLayoutAttr.size(); ++i) {
    affExprsVec.push_back(affExprsMap[i]);
  }

  auto transformAffineMap =
      AffineMap::get(outputLayoutAttr.size(), 0, affExprsVec, op.getContext());
  OpBuilder b(op.getOperation());
  op->setAttr("map", b.getAffineMapArrayAttr(transformAffineMap));
  return transformAffineMap;
}

void AffineTransforms::runOnFunction() {
  FuncOp func = getFunction();

  func.walk([&](miopen::TransformOp op) {
    AffineMap indexAffineMap = buildIndexAffineMap(op);
    llvm::SmallVector<AffineMap> affineMaps;
    affineMaps.push_back(indexAffineMap);
    auto inputType = op.input().getType().cast<MemRefType>();
    auto inputAffineMaps = inputType.getAffineMaps();
    for (auto &am : inputAffineMaps)
      affineMaps.push_back(am);

    auto outputType = op.output().getType().cast<MemRefType>();
    auto outputShape = outputType.getShape();
    auto transformedOutputType =
        MemRefType::get(outputShape, outputType.getElementType(), affineMaps);

    OpBuilder b(op.getOperation());
    auto loc = op.getLoc();
    auto newOp = b.create<miopen::TransformOp>(loc, transformedOutputType, op.input(), op.getAttrs());

    op.output().replaceAllUsesWith(newOp);
    op.erase();
  });
}

std::unique_ptr<Pass> mlir::miopen::createAffineTransformPass() {
  return std::make_unique<AffineTransforms>();
}

