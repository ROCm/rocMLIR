//===- TestTransformationMapsUtils.cpp - test removeUpperDims utility -----===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//
// test/lib/Dialect/Rock/testTransformationMapsUtilss.cpp

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Casting.h"
#include <tuple>

using namespace mlir;
using namespace mlir::rock;

namespace {
struct TransformMapsUtilsTestPass
    : public PassWrapper<TransformMapsUtilsTestPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransformMapsUtilsTestPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<RockDialect, func::FuncDialect>();
  }

  StringRef getArgument() const final {
    return "rock-transform-maps-utils-test";
  }
  StringRef getDescription() const final {
    return "Tests transformation map utils in Rock";
  }

  void runOnOperation() override;
};
} // end namespace

namespace {
struct ByIndices {
  static std::vector<std::pair<StringRef, SetVector<int64_t>>>
  getTests(func::FuncOp f) {
    std::vector<std::pair<StringRef, SetVector<int64_t>>> tests;
    auto attrMap = f->getAttr("remove_dims_by_indices");
    for (auto &testAttr : attrMap.cast<DictionaryAttr>()) {
      auto attrArray = testAttr.getValue().cast<ArrayAttr>();
      SetVector<int64_t> removeIndices;
      for (auto &attr : attrArray) {
        removeIndices.insert(attr.cast<IntegerAttr>().getInt());
      }
      tests.push_back({testAttr.getName(), removeIndices});
    }
    return tests;
  }
};

struct ByNames {
  static std::vector<std::pair<StringRef, SetVector<StringRef>>>
  getTests(func::FuncOp f) {
    std::vector<std::pair<StringRef, SetVector<StringRef>>> tests;
    auto attrMap = f->getAttr("remove_dims_by_names");
    for (auto &testAttr : attrMap.cast<DictionaryAttr>()) {
      auto attrArray = testAttr.getValue().cast<ArrayAttr>();
      SetVector<StringRef> removeIndices;
      for (auto &attr : attrArray) {
        removeIndices.insert(attr.cast<StringAttr>().getValue());
      }
      tests.push_back({testAttr.getName(), removeIndices});
    }
    return tests;
  }
};
} // namespace

template <typename RequestProcessor>
static LogicalResult testSubDimensions(func::FuncOp f) {
  Value returnValue;
  WalkResult walkResult = f.walk([&](func::ReturnOp op) -> WalkResult {
    if (op.getNumOperands() == 1) {
      returnValue = op->getOperand(0);
      return WalkResult::advance();
    }
    return WalkResult::interrupt();
  });
  if (walkResult.wasInterrupted()) {
    return failure();
  }

  OpBuilder builder(f.getContext());

  ArrayAttr transformAttrs;
  std::tie(std::ignore, transformAttrs, std::ignore) =
      rock::untransform(builder, returnValue);

  auto tests = RequestProcessor::getTests(f);
  for (auto &[testName, removeDims] : tests) {
    auto results = rock::removeUpperDims(builder, transformAttrs, removeDims);
    if (failed(results)) {
      return failure();
    }

    std::string outputString;
    llvm::raw_string_ostream stream(outputString);
    stream << testName << ' ' << f.getSymName() << '\n';
    for (auto item : *results) {
      item.print(stream);
      stream << '\n';
    }
    llvm::outs() << stream.str();
  }
  return success();
}

void TransformMapsUtilsTestPass::runOnOperation() {
  func::FuncOp f = getOperation();
  if (f->getAttr("remove_dims_by_indices")) {
    if (failed(testSubDimensions<ByIndices>(f))) {
      emitError(UnknownLoc::get(f.getContext()), "Pass failure");
      signalPassFailure();
    }
  } else if (f->getAttr("remove_dims_by_names")) {
    if (failed(testSubDimensions<ByNames>(f))) {
      emitError(UnknownLoc::get(f.getContext()), "Pass failure");
      signalPassFailure();
    }
  }
}

namespace mlir {
namespace rock {
void registerTransformMapsUtilsTestPass() {
  PassRegistration<TransformMapsUtilsTestPass>();
}
} // end namespace rock
} // end namespace mlir
