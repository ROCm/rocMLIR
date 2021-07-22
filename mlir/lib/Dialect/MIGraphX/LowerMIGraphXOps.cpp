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
          //llvm::errs()<< "operand0 : " << op->getOperand(0).getDefiningOp()->getName() <<"\n";
          //for (auto attr : op->getAttrs()) {
//            llvm::errs() << '\n' << attr.first << ": ";
          //}
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

static const char kRootOpAttr[] = "__root_op__";
static const char kFusionGroupsAttr[] = "__fused_op__";

using namespace mlir;
//using namespace mlir::migraphx;
using namespace migraphx;
namespace {

static bool isRootOp(Operation *op) {
//  return isa<migraphx::MIGraphX_ConvolutionOp>(op);
}

static unsigned decideFusableMIGOps(FuncOp funcOp){
  unsigned numRootOps = 0;
  MLIRContext *context = funcOp.getContext();
  OpBuilder builder(context);
  for (Block &block : funcOp) {
    auto MIGOps = block.getOps<MIGraphX_Op>();
    for (auto MIGOp : llvm::reverse(MIGOps)) {
      // Start with a root operation and fuse its producers.
      Operation *op = MIGOp.getOperation();
      if (!isRootOp(op)) continue;
      unsigned newGroup = numRootOps++;
      op->setAttr(kRootOpAttr, builder.getI64IntegerAttr(newGroup));

      //appendToFusionGroup(op, newGroup);
    }
}


struct GroupFusablesPass : public GroupFusablesPassBase<GroupFusablesPass> {
  
  void processModule(FuncOp funcOp) {
    MLIRContext *context = funcOp->getContext();
    context->allowUnregisteredDialects(true);

    unsigned numRoots = decideFusableMIGOps(funcOp);


    
  }
  void runOnOperation() override;

  private:
  
};

}


void GroupFusablesPass::runOnOperation() {
  
  processModule(getOperation());
}

std::unique_ptr<Pass> mlir::migraphx::createGroupFusablesPass() {
  return std::make_unique<GroupFusablesPass>();
}


