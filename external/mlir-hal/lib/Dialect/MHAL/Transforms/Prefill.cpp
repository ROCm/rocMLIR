
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MHAL/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace mhal {
#define GEN_PASS_DEF_MHALPREFILLPASS
#include "mlir/Dialect/MHAL/Transforms/Passes.h.inc"
} // namespace mhal
} // namespace mlir

#define DEBUG_TYPE "mhal-prefill"

using namespace mlir;
namespace {
class MHALPrefillPass
    : public mhal::impl::MHALPrefillPassBase<MHALPrefillPass> {
  void insertPrefillOps(OpBuilder &builder, gpu::LaunchFuncOp &launchOp);

public:
  // Inspects each gpu:: LaunchFuncOp, find arguments which must be prefilled
  // beforehand, and inserts the corresponding gpu::MemsetOp
  void runOnOperation() override;
};
} // namespace

void MHALPrefillPass::insertPrefillOps(OpBuilder &builder,
                                       gpu::LaunchFuncOp &launchOp) {
  auto func = cast<func::FuncOp>(launchOp->getParentOp());
  auto module = cast<ModuleOp>(func->getParentOp());
  auto kernel = launchOp.getKernel();
  auto *callee = module.lookupSymbol(kernel);
  assert(callee != nullptr && "expect to find the function defenition");
  auto llvmFunc = cast<LLVM::LLVMFuncOp>(callee);
  auto gpuModule = cast<gpu::GPUModuleOp>(llvmFunc->getParentOp());

  SmallVector<mhal::PrefillAttr, 4> prefillAttrs;
  if (auto moduleAttr = gpuModule->getAttr(llvmFunc.getSymName())) {
    if (auto arrayAttr = dyn_cast<ArrayAttr>(moduleAttr)) {
      for (auto attr : arrayAttr) {
        if (auto prefillAttr = dyn_cast<mhal::PrefillAttr>(attr)) {
          prefillAttrs.push_back(prefillAttr);
        }
      }
    }
  }

  auto loc = builder.getUnknownLoc();
  auto kernelOperands = launchOp.getKernelOperands();
  for (auto attr : prefillAttrs) {
    auto argIdx = attr.getArgIndex();
    assert(argIdx < kernelOperands.size() &&
           "provided arg index is out of bounds");
    auto arg = kernelOperands[argIdx];
    auto type = arg.getType().cast<MemRefType>();
    auto elementType = type.getElementType();
    builder.setInsertionPoint(launchOp->getBlock(),
                              --Block::iterator(launchOp));
    auto initConstant = builder.create<arith::ConstantOp>(loc, elementType,
                                                          attr.getInitValue());
    builder.create<gpu::MemsetOp>(loc, mlir::Type{}, mlir::ValueRange{}, arg,
                                  initConstant);
  }
}

// Inspects each gpu:: LaunchFuncOp, find arguments which must be prefilled
// beforehand, and inserts the corresponding gpu::MemsetOp
void MHALPrefillPass::runOnOperation() {
  func::FuncOp func = getOperation();
  llvm::SmallVector<gpu::LaunchFuncOp, 4> ops;
  func.walk([&ops](gpu::LaunchFuncOp launchOp) { ops.push_back(launchOp); });
  {
    OpBuilder builder(&getContext());
    OpBuilder::InsertionGuard insertGuard(builder);
    for (gpu::LaunchFuncOp launchOp : ops) {
      insertPrefillOps(builder, launchOp);
    }
  }
}
