//===- PopulateArchFeaturesPass.cpp ------------===//
//
// Copyright 2025 Advanced Micro Devices.
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
// ============================================================
//
// This pass modifies rocdl operations by filling in passed in information
// such as arch, num_cus, xdlops, and perf configs
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/UtilityParams.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_POPULATEARCHFEATURESPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "populate-arch-features"

using namespace mlir;
using namespace mlir::rock;

namespace {
class PopulateArchFeaturesPass
    : public rock::impl::PopulateArchFeaturesPassBase<
          PopulateArchFeaturesPass> {
    using rock::impl::PopulateArchFeaturesPassBase<
	PopulateArchFeaturesPass>::PopulateArchFeaturesPassBase;
  void runOnOperation() override;
  void PopulateArchFeaturesImpl(func::FuncOp &func);
    rock::GemmFeatures getArchFeatures(StringAttr archAttr, Type inputType);
};
} // end namespace

rock::GemmFeatures
PopulateArchFeaturesPass::getArchFeatures(StringAttr archAttr, Type inputType) {    
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(archAttr);
  rock::GemmFeatures features = archInfo.getDefaultFeatures(inputType);
  if (xdlopsV2)
    features = rock::bitEnumSet(features, rock::GemmFeatures::mfma, xdlopsV2);
  return features;
}

void PopulateArchFeaturesPass::PopulateArchFeaturesImpl(func::FuncOp &func) {
    llvm::outs() << "Running PopulateArchFeaturesPass\n";
    auto context = func.getContext();
    // repopulate the function
    StringAttr archAttr = StringAttr::get(context, arch);
    IntegerAttr numCUAttr = IntegerAttr::get(IntegerType::get(context, 32), num_cu);
    BoolAttr xdlopsAttr = BoolAttr::get(context, xdlopsV2);
    StringAttr perfConfigAttr = StringAttr::get(context, perf_config);
    
    if(!arch.empty()) {
	func->setAttr("arch", archAttr);
    }

    if(num_cu > 0) {
	func->setAttr("num_cu", numCUAttr);
    }

    if(xdlopsV2) {
	func->setAttr("xdlopsV2", xdlopsAttr);
    }

    if(debug) {
    llvm::outs() << "Printing module\n";
	func.walk([&](mlir::Operation *op) {
	    llvm::outs() << op->getName() << "\n";
	});
    }

    // handle updating gemm ops
    func.walk([&](GemmOp gemmOp) {
	if(debug) {
	    llvm::outs() << "Printing gemm before: ";
	    gemmOp.print(llvm::outs());	
	    llvm::outs() << "\n";
	}
	
	// get inputType
	auto inputType = gemmOp.getA().getType();
	auto features = getArchFeatures(archAttr, inputType);
	auto featuresAttr = rock::GemmFeaturesAttr::get(gemmOp.getContext(), features);
	gemmOp->setAttr("arch", archAttr);
	gemmOp->setAttr("numCU", numCUAttr);
	gemmOp->setAttr("xdlopsV2", xdlopsAttr);
	gemmOp->setAttr("perf_config", perfConfigAttr);
	gemmOp->setAttr("features", featuresAttr);
	if(debug) {
	    llvm::outs() << "Printing gemm after: ";
	    gemmOp.print(llvm::outs());	
	    llvm::outs() << "\n";
	}
    });

    // repeat for every other rock::Op
}
	
void PopulateArchFeaturesPass::runOnOperation() {
  func::FuncOp func = getOperation();
  PopulateArchFeaturesImpl(func);
} // namespace
