#include "PassDetail.h"

#include "mlir/Dialect/MIOpen/AffineMapHelper.h"
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
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
using namespace mlir::miopen;

namespace {
struct TestAffineTransforms
    : public MIOpenOpsTestAffineTransformPassBase<TestAffineTransforms> {
  void runOnFunction() override;
};
} // anonymous namespace

void TestAffineTransforms::runOnFunction() {
  FuncOp func = getFunction();

  func.walk([&](miopen::TransformOp op) {
    OpBuilder b(op.getOperation());
    auto output = op.output();
    auto outputType = output.getType().dyn_cast<MemRefType>();
    auto outputAffineMaps = outputType.getAffineMaps();

    if (testMethod == "hasPadding") {
      bool ret = false;
      for (auto &affineMap : outputAffineMaps)
        ret |= hasPadding(affineMap);
      op.setAttr("hasPadding", b.getBoolAttr(ret));
    }
  });
}

std::unique_ptr<Pass> mlir::miopen::createTestAffineTransformPass() {
  return std::make_unique<TestAffineTransforms>();
}
