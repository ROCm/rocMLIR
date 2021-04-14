#include "PassDetail.h"

#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
struct SplitConv2D : public MIOpenOpsSplitConv2DPassBase<SplitConv2D> {
public:
  SplitConv2D() = default;
  void runOnOperation() override;
};
} // end anonymous namespace

void SplitConv2D::runOnOperation() {
  auto op = getOperation();
  OpBuilder b(op.getContext());
  auto loc = op.getLoc();

  SymbolTable symbolTable(op);

  static int const cloneFactor = 3;

  // Traverse all FuncOps, identify those with kernel attribute.
  for (auto func : op.getOps<FuncOp>()) {
    if (func->getAttrDictionary().get("kernel")) {
      if (func.getName().startswith("cloned"))
        continue;

      // Clone multiple copies of the kernel function.
      for (int i = 0; i < cloneFactor; ++i) {
        // Create a new cloned function.
        auto clonedFunc = func.clone();

        // Set the name to the clone function.
        std::string newName =
            "cloned_" + std::to_string(i) + "_" + func.getName().str();
        clonedFunc.setName(newName);

        // Traverse all conv2d ops, replace with conv2d_dummy ops.
        for (auto conv : clonedFunc.getOps<miopen::Conv2DOp>()) {
          // Create an equivalent conv2d_dummy op.
          b.setInsertionPointToStart(&(clonedFunc.body()).front());
          auto dummy_conv = b.create<miopen::Conv2DDummyOp>(
              loc, ArrayRef<mlir::Type>{},
              ValueRange{clonedFunc.getArgument(0), clonedFunc.getArgument(1),
                         clonedFunc.getArgument(2)},
              conv.getAttrs());
          conv->replaceAllUsesWith(dummy_conv);
          conv.erase();
          break;
        }

        // Insert the cloned function into ModuleOp.
        symbolTable.insert(clonedFunc);
      }
    }
  }
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::miopen::createSplitConv2DPass() {
  return std::make_unique<SplitConv2D>();
}
