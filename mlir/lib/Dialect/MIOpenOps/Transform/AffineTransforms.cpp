#include "mlir/Dialect/MIOpenOps/MIOpenOps.h"
#include "mlir/Dialect/MIOpenOps/Passes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
struct AffineTransforms : public FunctionPass<AffineTransforms> {
  void runOnFunction() override;
};
} // anonymous namespace

void AffineTransforms::runOnFunction() {
  FuncOp func = getFunction();

  func.walk([&](miopen::TransformOp op) {
    auto inputType = op.input().getType().dyn_cast<MemRefType>();
    auto inputShape = inputType.getShape();
    auto inputRank = inputType.getRank();
    auto inputElementType = inputType.getElementType();
    auto inputAffineMaps = inputType.getAffineMaps();

    auto layoutAttr = op.template getAttrOfType<ArrayAttr>("layout");

    auto sourceLayoutAttr = op.template getAttrOfType<ArrayAttr>("source_layout");
    if (!sourceLayoutAttr)
      sourceLayoutAttr = op.template getAttrOfType<ArrayAttr>("intermediate_layout");
    auto outputLayoutAttr = op.template getAttrOfType<ArrayAttr>("output_layout");

    llvm::SmallMapVector<int64_t, AffineExpr, 8> affExprsMap;
    for (unsigned i = 0; i < layoutAttr.size(); ++i) {
      if (auto dimLayoutAttr = layoutAttr.getValue()[i].dyn_cast<DictionaryAttr>()) {
        auto srcDimAttr = dimLayoutAttr.get("source_dimensions").dyn_cast<ArrayAttr>();
        auto destDimAttr = dimLayoutAttr.get("dimensions").dyn_cast<ArrayAttr>();
        auto transformAttr = dimLayoutAttr.get("transformation").dyn_cast<StringAttr>();

        if (transformAttr.getValue() == "PassThrough") {
          assert(srcDimAttr.size() == 1);
          assert(destDimAttr.size() == 1);

          auto srcDim = srcDimAttr.getValue()[0].dyn_cast<IntegerAttr>().getInt();
          auto destDim = destDimAttr.getValue()[0].dyn_cast<IntegerAttr>().getInt();
          auto expr = getAffineDimExpr(destDim, op.getContext());
          affExprsMap.insert({srcDim, expr});
        } else if (transformAttr.getValue() == "Pad") {
          assert(srcDimAttr.size() == destDimAttr.size());

          auto parameters = dimLayoutAttr.get("parameters").dyn_cast<ArrayAttr>();
          auto leftPad = parameters.getValue()[0].dyn_cast<IntegerAttr>().getInt();
          auto rightPad = parameters.getValue()[0].dyn_cast<IntegerAttr>().getInt();
          for (unsigned j = 0; j < srcDimAttr.size(); ++j) {
            auto srcDim = srcDimAttr.getValue()[j].dyn_cast<IntegerAttr>().getInt();
            auto destDim = destDimAttr.getValue()[j].dyn_cast<IntegerAttr>().getInt();

            auto expr = getAffineDimExpr(destDim, op.getContext()) + getAffineConstantExpr(-leftPad, op.getContext());
            affExprsMap.insert({srcDim, expr});
          }
        } else if (transformAttr.getValue() == "Merge" ||
                   transformAttr.getValue() == "Unfold") {
          assert(destDimAttr.size() == 1);
          assert(srcDimAttr.size() > 1);

          auto destDim = destDimAttr.getValue()[0].dyn_cast<IntegerAttr>().getInt();

          // Find source dimension lengths.
          llvm::SmallVector<int64_t, 4> srcDimLengthVec;
          for (unsigned j = 0; j < srcDimAttr.size(); ++j) {
            auto srcDim = srcDimAttr.getValue()[j].dyn_cast<IntegerAttr>().getInt();
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

            auto srcDim = srcDimAttr.getValue()[j].dyn_cast<IntegerAttr>().getInt();
            affExprsMap.insert({srcDim, expr});
          }
        } else if (transformAttr.getValue() == "Embed") {
          assert(srcDimAttr.size() == 1);
          assert(destDimAttr.size() > 1);

          auto srcDim = srcDimAttr.getValue()[0].dyn_cast<IntegerAttr>().getInt();
          auto parameters = dimLayoutAttr.get("parameters").dyn_cast<ArrayAttr>();

          // # of parameters would always be 1 more than the # of destDim.
          // populate the initial affine expr.
          auto param = parameters.getValue()[parameters.size() - 1].dyn_cast<IntegerAttr>().getInt();
          auto expr = getAffineConstantExpr(param, op.getContext());

          // Build affine transformation expressions.
          for (unsigned j = 0; j < destDimAttr.size(); ++j) {
            auto destDim = destDimAttr.getValue()[j].dyn_cast<IntegerAttr>().getInt();
            param = parameters.getValue()[j].dyn_cast<IntegerAttr>().getInt();
            auto partialExpr = getAffineDimExpr(destDim, op.getContext()) * getAffineConstantExpr(param, op.getContext());
            expr = expr + partialExpr;
          }
          affExprsMap.insert({srcDim, expr});
        }
      }
    }

    llvm::SmallVector<AffineExpr, 8> affExprsVec;
    for (unsigned i = 0; i < sourceLayoutAttr.size(); ++i) {
      affExprsVec.push_back(affExprsMap[i]);
    }

    auto transformAffineMap = AffineMap::get(outputLayoutAttr.size(), 0, affExprsVec);
    AffineMap outputAffineMap;

    if (inputAffineMaps.size() != 0) {
      auto inputAffineMap = inputAffineMaps[0];
      outputAffineMap = inputAffineMap.compose(transformAffineMap);

      //llvm::errs() << "=========\n";
      //inputAffineMap.dump();
      //transformAffineMap.dump();
      //outputAffineMap.dump();
      //llvm::errs() << "=========\n";
    } else {
      outputAffineMap = transformAffineMap;

      //llvm::errs() << "=========\n";
      //outputAffineMap.dump();
      //llvm::errs() << "=========\n";
    }

    auto outputType = op.output().getType().dyn_cast<MemRefType>();
    auto outputShape = outputType.getShape();
    auto outputElementType = outputType.getElementType();
    auto transformedOutputType = MemRefType::get(outputShape, outputType.getElementType(),
                                                 {outputAffineMap});

    OpBuilder b(op.getOperation());
    auto loc = op.getLoc();
    auto newOp = b.create<miopen::TransformOp>(loc, transformedOutputType, op.input(), op.getAttrs());

    op.output().replaceAllUsesWith(newOp);
    op.erase();
  });
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::miopen::createAffineTransformPass() {
  return std::make_unique<AffineTransforms>();
}

static PassRegistration<AffineTransforms>
  pass("miopen-affine-transform", "Build affine maps from miopen.transform operations");

