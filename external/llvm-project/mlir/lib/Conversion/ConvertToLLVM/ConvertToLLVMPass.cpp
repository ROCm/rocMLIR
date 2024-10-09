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
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
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
  ConvertToLLVMPass(unsigned indexBitwidth, bool useBarePtrCallConv) {
    if (this->indexBitwidth.getNumOccurrences() == 0)
      this->indexBitwidth = indexBitwidth;
    if (this->useBarePtrCallConv.getNumOccurrences() == 0)
      this->useBarePtrCallConv = useBarePtrCallConv;
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<ptr::PtrDialect>();
    registry.addExtensions<LoadDependentDialectExtension>();
  }

  LogicalResult initialize(MLIRContext *context) final { return success(); }

  void runOnOperation() final {
    auto *op = getOperation();
    auto *context = op->getContext();
    StringRef dataLayout;
    auto dataLayoutAttr = dyn_cast_or_null<StringAttr>(
        op->getAttr(LLVM::LLVMDialect::getDataLayoutAttrName()));
    if (dataLayoutAttr)
      dataLayout = dataLayoutAttr.getValue();

    if (failed(LLVM::LLVMDialect::verifyDataLayoutString(
            dataLayout, [this](const Twine &message) {
              getOperation()->emitError() << message.str();
            }))) {
      signalPassFailure();
      return;
    }

    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();

    LowerToLLVMOptions options(context,
                               dataLayoutAnalysis.getAtOrAbove(op));
    options.useBarePtrCallConv = useBarePtrCallConv;
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);
    options.dataLayout = llvm::DataLayout(dataLayout);
    if (useBarePtrCallConv) {
      options.useBarePtrCallConv = true;
    }

    RewritePatternSet tempPatterns(context);
    auto target = std::make_shared<ConversionTarget>(*context);
    target->addLegalDialect<LLVM::LLVMDialect>();
    auto typeConverter = std::make_shared<LLVMTypeConverter>(context, options);
    DenseMap<Attribute, uint64_t> addressSpaceMap;
    if (DataLayoutOpInterface iface = dyn_cast<DataLayoutOpInterface>(op)) {
      if (DataLayoutSpecInterface dlSpec = iface.getDataLayoutSpec()) {
        for (DataLayoutEntryInterface entry : dlSpec.getEntries()) {
          if (!entry.isTypeEntry()) {
            continue;
          }
          auto ptrKey = llvm::dyn_cast<mlir::ptr::PtrType>(
              entry.getKey().get<mlir::Type>());
          if (!ptrKey) {
            continue;
          }
          auto addressSpace = ptrKey.getMemorySpace();
          auto value =
              cast<mlir::ptr::SpecAttr>(entry.getValue()).getLlvmAddressSpace();
          addressSpaceMap.insert({addressSpace, value});
        }
      }
      typeConverter->addTypeAttributeConversion(
          [addressSpaceMap](BaseMemRefType type, Attribute memorySpaceAttr) {
            unsigned llvmAddressSpace = 0;
            if (addressSpaceMap.contains(memorySpaceAttr)) {
              llvmAddressSpace = addressSpaceMap.at(memorySpaceAttr);
            }
            return IntegerAttr::get(
                IntegerType::get(memorySpaceAttr.getContext(), 64),
                llvmAddressSpace);
          });
    }

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
    this->patterns =
        std::make_unique<FrozenRewritePatternSet>(std::move(tempPatterns));
    this->target = target;
    this->typeConverter = typeConverter;
    if (failed(applyPartialConversion(getOperation(), *target, *patterns)))
      signalPassFailure();
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

std::unique_ptr<Pass> mlir::createConvertToLLVMPass(unsigned indexBitwidth,
                                                    bool useBarePtrCallConv) {
  return std::make_unique<ConvertToLLVMPass>(indexBitwidth, useBarePtrCallConv);
}
