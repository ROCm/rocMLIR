
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

  Value createConstantOp(OpBuilder &builder, Type elementType,
                         uint32_t initValue) {
    Value constantOp;
    auto loc = builder.getUnknownLoc();
    if (llvm::isa<FloatType>(elementType)) {
      auto semantics = static_cast<APFloat::Semantics>(-1);
      if (elementType.isF32()) {
        semantics = APFloat::S_IEEEsingle;
      } else if (elementType.isF16()) {
        semantics = APFloat::S_IEEEhalf;
      } else if (elementType.isBF16()) {
        semantics = APFloat::S_BFloat;
      } else if (elementType.isFloat8E4M3FNUZ()) {
        semantics = APFloat::S_Float8E4M3FNUZ;
      } else if (elementType.isFloat8E5M2FNUZ()) {
        semantics = APFloat::S_Float8E5M2FNUZ;
      } else {
        llvm_unreachable("Unexpected float semantics");
      }

      APFloat apValue(*(reinterpret_cast<float *>(&initValue)));
      bool lostInfo = false;
      apValue.convert(APFloat::EnumToSemantics(semantics),
                      APFloat::rmNearestTiesToEven, &lostInfo);

      auto constValue = builder.getFloatAttr(elementType, apValue);
      constantOp =
          builder.create<arith::ConstantOp>(loc, elementType, constValue);
    } else {
      APInt apValue(elementType.getIntOrFloatBitWidth(), initValue, true);
      auto constValue = builder.getIntegerAttr(elementType, apValue);
      constantOp =
          builder.create<arith::ConstantOp>(loc, elementType, constValue);
    }
    return constantOp;
  }

  void insertPrefillOps(OpBuilder &builder, gpu::LaunchFuncOp &launchOp) {
    auto func = llvm::cast<func::FuncOp>(launchOp->getParentOp());
    auto module = llvm::cast<ModuleOp>(func->getParentOp());

    auto kernel = launchOp.getKernel();
    auto *callee = module.lookupSymbol(kernel);
    assert(callee != nullptr && "expect to find the function defenition");

    auto llvmFunc = llvm::cast<LLVM::LLVMFuncOp>(callee);
    auto gpuModule = llvm::cast<gpu::GPUModuleOp>(llvmFunc->getParentOp());
    assert(gpuModule != nullptr &&
           "expect LLVMFuncOp to be inside GPUModuleOp");

    SmallVector<mhal::PrefillAttr, 4> prefillAttrs;
    if (auto moduleAttr = gpuModule->getAttr(llvmFunc.getSymName())) {
      if (auto arrayAttr = llvm::dyn_cast<ArrayAttr>(moduleAttr)) {
        for (auto attr : arrayAttr) {
          if (auto prefillAttr = llvm::dyn_cast<mhal::PrefillAttr>(attr)) {
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

      builder.setInsertionPointToStart(arg.getParentBlock());
      auto initConstant =
          createConstantOp(builder, elementType, attr.getInitValue());

      builder.setInsertionPoint(launchOp->getBlock(),
                                --Block::iterator(launchOp));
      builder.create<gpu::MemsetOp>(loc, mlir::Type{}, mlir::ValueRange{}, arg,
                                    initConstant);
    }
  }

public:
  // Inspects each gpu:: LaunchFuncOp, find arguments which must be prefilled
  // beforehand, and inserts the corresponding gpu::MemsetOp
  void runOnOperation() override {
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
};
} // namespace
