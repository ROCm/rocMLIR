//===- ConvertToLLVMPass.cpp - MLIR LLVM Conversion -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

#define DEBUG_TYPE "convert-to-llvm"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

/// Returns true if the given `func.funcop` can be safely called using the bare
/// pointer calling convention.
static bool canBeCalledWithBarePointers(func::FuncOp func) {
  bool canBeBare = true;
  for (Type type : func.getArgumentTypes())
    if (auto memrefTy = dyn_cast<BaseMemRefType>(type))
      canBeBare &= LLVMTypeConverter::canConvertToBarePtr(memrefTy);
  return canBeBare;
}

namespace {

/// This DialectExtension can be attached to the context, which will invoke the
/// `apply()` method for every loaded dialect. If a dialect implements the
/// `ConvertToLLVMPatternInterface` interface, we load dependent dialects
/// through the interface. This extension is loaded in the context before
/// starting a pass pipeline that involves dialect conversion to LLVM.
class LoadDependentDialectExtension : public DialectExtensionBase {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoadDependentDialectExtension)

  LoadDependentDialectExtension() : DialectExtensionBase(/*dialectNames=*/{}) {}

  void apply(MLIRContext *context,
             MutableArrayRef<Dialect *> dialects) const final {
    LLVM_DEBUG(llvm::dbgs() << "Convert to LLVM extension load\n");
    for (Dialect *dialect : dialects) {
      auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect);
      if (!iface)
        continue;
      LLVM_DEBUG(llvm::dbgs() << "Convert to LLVM found dialect interface for "
                              << dialect->getNamespace() << "\n");
      iface->loadDependentDialects(context);
    }
  }

  /// Return a copy of this extension.
  std::unique_ptr<DialectExtensionBase> clone() const final {
    return std::make_unique<LoadDependentDialectExtension>(*this);
  }
};

/// This is a generic pass to convert to LLVM, it uses the
/// `ConvertToLLVMPatternInterface` dialect interface to delegate to dialects
/// the injection of conversion patterns.
class ConvertToLLVMPass
    : public impl::ConvertToLLVMPassBase<ConvertToLLVMPass> {
  std::shared_ptr<const FrozenRewritePatternSet> patterns;
  std::shared_ptr<const ConversionTarget> target;
  std::shared_ptr<const LLVMTypeConverter> typeConverter;

public:
  using impl::ConvertToLLVMPassBase<ConvertToLLVMPass>::ConvertToLLVMPassBase;
  ConvertToLLVMPass() = default;
  ConvertToLLVMPass(unsigned indexBitwidth, bool useBarePtrCallConv,
                    const std::string &dataLayout) {
    if (this->indexBitwidth.getNumOccurrences() == 0)
      this->indexBitwidth = indexBitwidth;
    if (this->useBarePtrCallConv.getNumOccurrences() == 0)
      this->useBarePtrCallConv = useBarePtrCallConv;
    if (this->dataLayout.getNumOccurrences() == 0)
      this->dataLayout = dataLayout;
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect>();
    registry.addExtensions<LoadDependentDialectExtension>();
  }

  LogicalResult initialize(MLIRContext *context) final { return success(); }

  void runOnOperation() final {
    auto *op = getOperation();
    auto *context = op->getContext();
    RewritePatternSet tempPatterns(context);
    auto target = std::make_shared<ConversionTarget>(*context);
    target->addLegalDialect<LLVM::LLVMDialect>();
    gpu::GPUModuleOp gpuMod;
    for (Region &region : getOperation()->getRegions())
      for (Block &block : region.getBlocks())
        for (auto module : block.getOps<gpu::GPUModuleOp>()) {
          gpuMod = module;
          // Check if the name of the module matches.
          llvm::errs() << "Found GPU Mod\n";
          auto llvmDataLayout = module->getAttrOfType<StringAttr>(
              LLVM::LLVMDialect::getDataLayoutAttrName());
          if (!llvmDataLayout) {
            llvmDataLayout = StringAttr::get(context, dataLayout);
            module->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
                            llvmDataLayout);
          }
        }

    DataLayout dl = DataLayoutAnalysis{getOperation()}.getAtOrAbove(gpuMod);
    LowerToLLVMOptions options(context, dl);
    options.dataLayout = llvm::DataLayout(StringRef{dataLayout});
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    if (useBarePtrCallConv) {
      options.useBarePtrCallConv = true;
      WalkResult canUseBarePointers =
          op->walk([](func::FuncOp func) -> WalkResult {
            if (canBeCalledWithBarePointers(func))
              return WalkResult::advance();
            return WalkResult::interrupt();
          });
      if (canUseBarePointers.wasInterrupted()) {
        emitError(UnknownLoc::get(context),
                  "bare pointer calling convention requires all memrefs to "
                  "have static shape and use the identity map");
        signalPassFailure();
      }
    }
    auto typeConverter = std::make_shared<LLVMTypeConverter>(context, options);
    auto mapping = [](gpu::AddressSpace space) {
      switch (space) {
      case gpu::AddressSpace::Global:
        return 1;
      case gpu::AddressSpace::Workgroup:
        return 3;
      case gpu::AddressSpace::Private:
        return 5;
      }
      llvm_unreachable("unknown address space enum value");
      return 0;
    };
    typeConverter->addTypeAttributeConversion(
        [mapping](BaseMemRefType type, gpu::AddressSpaceAttr memorySpaceAttr) {
          gpu::AddressSpace memorySpace = memorySpaceAttr.getValue();
          unsigned addressSpace = mapping(memorySpace);
          return IntegerAttr::get(
              IntegerType::get(memorySpaceAttr.getContext(), 64), addressSpace);
        });

    if (!filterDialects.empty()) {
      // Test mode: Populate only patterns from the specified dialects. Produce
      // an error if the dialect is not loaded or does not implement the
      // interface.
      for (std::string &dialectName : filterDialects) {
        Dialect *dialect = context->getLoadedDialect(dialectName);
        if (!dialect) {
          emitError(UnknownLoc::get(context))
              << "dialect not loaded: " << dialectName << "\n";
          signalPassFailure();
        }
        auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect);
        if (!iface) {
          emitError(UnknownLoc::get(context))
              << "dialect does not implement ConvertToLLVMPatternInterface: "
              << dialectName << "\n";
          signalPassFailure();
        }

        iface->populateConvertToLLVMConversionPatterns(*target, *typeConverter,
                                                       tempPatterns);
      }
    } else {
      // Normal mode: Populate all patterns from all dialects that implement the
      // interface.
      for (Dialect *dialect : context->getLoadedDialects()) {
        // First time we encounter this dialect: if it implements the interface,
        // let's populate patterns !
        auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect);
        if (!iface)
          continue;
        iface->populateConvertToLLVMConversionPatterns(*target, *typeConverter,
                                                       tempPatterns);
      }
    }
    populateVectorToLLVMConversionPatterns(*typeConverter, tempPatterns);

    this->patterns =
        std::make_unique<FrozenRewritePatternSet>(std::move(tempPatterns));
    this->target = target;
    this->typeConverter = typeConverter;
    if (failed(applyPartialConversion(getOperation(), *target, *patterns)))
      signalPassFailure();
    auto *rocdlDialect = getContext().getLoadedDialect<ROCDL::ROCDLDialect>();
    auto reqdWorkGroupSizeAttrHelper =
        rocdlDialect->getReqdWorkGroupSizeAttrHelper();
    auto flatWorkGroupSizeAttrHelper =
        rocdlDialect->getFlatWorkGroupSizeAttrHelper();
    // Manually rewrite known block size attributes so the LLVMIR translation
    // infrastructure can pick them up.
    gpuMod.walk([&](LLVM::LLVMFuncOp op) {
      if (reqdWorkGroupSizeAttrHelper.isAttrPresent(op)) {
        auto blockSizes = reqdWorkGroupSizeAttrHelper.getAttr(op);
        // Also set up the rocdl.flat_work_group_size attribute to prevent
        // conflicting metadata.
        uint32_t flatSize = 1;
        for (uint32_t size : blockSizes.asArrayRef()) {
          flatSize *= size;
        }
        StringAttr flatSizeAttr =
            StringAttr::get(context, Twine(flatSize) + "," + Twine(flatSize));
        flatWorkGroupSizeAttrHelper.setAttr(op, flatSizeAttr);
      }
    });
  }
};

} // namespace

void mlir::registerConvertToLLVMDependentDialectLoading(
    DialectRegistry &registry) {
  registry.addExtensions<LoadDependentDialectExtension>();
}

std::unique_ptr<Pass> mlir::createConvertToLLVMPass() {
  return std::make_unique<ConvertToLLVMPass>();
}

std::unique_ptr<Pass>
mlir::createConvertToLLVMPass(unsigned indexBitwidth, bool useBarePtrCallConv,
                              const std::string &dataLayout) {
  return std::make_unique<ConvertToLLVMPass>(indexBitwidth, useBarePtrCallConv,
                                             dataLayout);
}
