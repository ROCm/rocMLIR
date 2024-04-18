//===- TestBufferDependencyAnalysis.cpp - test correctness of the buffer
// dependency analysis ----===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2024 Advanced Micro Devices Inc.
//===-----------------------------------------------------===//

#include "mlir/Analysis/BufferDependencyAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct BufferDependencyAnalysisTestPass
    : public PassWrapper<BufferDependencyAnalysisTestPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BufferDependencyAnalysisTestPass)

  StringRef getArgument() const final {
    return "rock-buffer-dependency-analysis-test";
  }
  StringRef getDescription() const final {
    return "Tests the buffer dependency analysis in Rock";
  }

  void runOnOperation() override;
};
} // end namespace

struct ExpectedOpNames {
  llvm::SmallSetVector<StringRef, 10> readers;
  llvm::SmallSetVector<StringRef, 10> writers;
};

static void
extractExpectedResults(func::FuncOp func,
                       DenseMap<StringRef, ExpectedOpNames> &expectedResults) {
  auto attrs = func->getAttr("expected").cast<ArrayAttr>();
  for (auto attr : attrs) {
    ExpectedOpNames item;
    auto dictAttr = attr.cast<DictionaryAttr>();
    auto allocName = dictAttr.get("alloc_name").cast<StringAttr>().getValue();

    auto writerListAttr = dictAttr.get("writers").cast<ArrayAttr>();
    for (auto writerAttr : writerListAttr) {
      auto writerName = writerAttr.cast<StringAttr>().getValue();
      item.writers.insert(writerName);
    }

    auto readerListAttr = dictAttr.get("readers").cast<ArrayAttr>();
    for (auto readerAttr : readerListAttr) {
      auto readerName = readerAttr.cast<StringAttr>().getValue();
      item.readers.insert(readerName);
    }

    expectedResults.insert({allocName, item});
  }
}

static LogicalResult analyse(func::FuncOp func) {
  static std::mutex mutex;
  DenseMap<StringRef, ExpectedOpNames> expectedResults;
  extractExpectedResults(func, expectedResults);
  BufferDependencyAnalysis analysis =
      BufferDependencyAnalysis(func.getOperation());

  auto result = func.walk([&](memref::AllocOp allocOp) -> WalkResult {
    auto allocName = allocOp->getAttr("name").cast<StringAttr>().getValue();

    if (!expectedResults.contains(allocName)) {
      return WalkResult::interrupt();
    }

    auto expectedOpNames = expectedResults[allocName];

    // test readers
    auto testReaders = analysis.getReaders(allocOp);
    if (testReaders.has_value()) {
      for (auto *testReaderOp : testReaders.value()) {
        auto testReaderOpName = testReaderOp->getName().getStringRef();
        if (!expectedOpNames.readers.contains(testReaderOpName)) {
          std::lock_guard<std::mutex> guard(mutex);
          llvm::errs() << "failed to find `" << testReaderOpName
                       << "` reader for `" << allocName << "`\n";
          return WalkResult::interrupt();
        }
      }
    }

    // test writers
    auto testWriters = analysis.getWriters(allocOp);
    if (testWriters.has_value()) {
      for (auto *testWriterOp : testWriters.value()) {
        auto testWriterOpName = testWriterOp->getName().getStringRef();
        if (!expectedOpNames.writers.contains(testWriterOpName)) {
          std::lock_guard<std::mutex> guard(mutex);
          llvm::errs() << "failed to find `" << testWriterOpName
                       << "` writer for `" << allocName << "`\n";
          return WalkResult::interrupt();
        }
      }
    }

    return WalkResult::advance();
  });

  OpBuilder builder(func->getContext());
  if (!result.wasInterrupted()) {
    func->setAttr("passed", builder.getUnitAttr());
  }

  return failure(result.wasInterrupted());
}

void BufferDependencyAnalysisTestPass::runOnOperation() {
  func::FuncOp f = getOperation();
  if (failed(analyse(f))) {
    emitError(UnknownLoc::get(f.getContext()), "Pass failure");
    signalPassFailure();
  }
}

namespace mlir {
namespace rock {
void registerBufferDependencyAnalysisTestPass() {
  PassRegistration<BufferDependencyAnalysisTestPass>();
}
} // end namespace rock
} // end namespace mlir
