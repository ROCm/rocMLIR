//===- LowerMIGraphXOps.cpp - MLIR MIGraphX ops lowering passes ---------------===//
//
// Copyright 2020 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/MIGraphX/LowerMIGraphXOps.h"

#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"
#include "mlir/Dialect/MIGraphX/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {
struct MIGraphXIRDumpPass : public MIGraphXIRDumpPassBase<MIGraphXIRDumpPass> {
  std::string getOpName(Operation &op) {
    auto symbolAttr =
        op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    if (symbolAttr)
      return std::string(symbolAttr.getValue());
    ++unnamedOpCtr;
    return (op.getName().getStringRef() + llvm::utostr(unnamedOpCtr)).str();
  }
  // Print all the ops in a module.
  void processModule(ModuleOp module) {
    for (Operation &op : module) {
      // Modules may actually be nested, recurse on nesting.
      if (auto nestedModule = dyn_cast<ModuleOp>(op)) {
        processModule(nestedModule);
        continue;
      }
      auto opName = getOpName(op);
      //auto opName = op->getName().getStringRef();
      for (Region &region : op.getRegions()) {
        region.walk([&](Operation *op) {
          llvm::errs()<< "visiting op : " << op->getName().getStringRef() << "\n"; 
          //llvm::errs()<< "dialect : " << op->getName().getDialectNamespace() << "\n"; 
          llvm::errs()<< "op : " << op->getName().stripDialect() << "\n"; 
          //llvm::errs()<< "ID : " << op->getName().getIdentifier().strref() << "\n"; 
          llvm::errs()<< "ID : " << op->getName().getIdentifier() << "\n"; 
          for (auto attr : op->getAttrs()) {
            llvm::errs() << '\n' << attr.first << ": ";
          }
        });
      }
    }
  }
  void runOnOperation() override;

  private:
  int unnamedOpCtr = 0;
};
} // end anonymous namespace


void MIGraphXIRDumpPass::runOnOperation() {
  // try module->print() here..
  // try call a function in the headerfile, header file calls json  
  //cout<<"xir dump pass called!\n";

  //llvm::errs() << "xir dump pass called!\n";
  //getOperation()->walk([&](Operation *op) { llvm::errs()<< "visiting op : " << op->getName().getStringRef() << "\n"; });

  processModule(getOperation());
}

std::unique_ptr<Pass> mlir::migraphx::createMIGraphXIRDumpPass() {
  return std::make_unique<MIGraphXIRDumpPass>();
}
