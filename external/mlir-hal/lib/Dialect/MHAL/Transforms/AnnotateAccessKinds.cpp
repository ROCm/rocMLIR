//===- AnnotateAccessKinds.cpp - Tensor usage annotations ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass to mark mhal.read_acccess and mhal.write_access (conservatively) on
// tensors to ensure that we know what needs to be copied to a device during
// launch formation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MHAL/Transforms/Passes.h"

namespace mlir {
namespace mhal {
#define GEN_PASS_DEF_MHALANNOTATEACCESSKINDSPASS
#include "mlir/Dialect/MHAL/Transforms/Passes.h.inc"
} // namespace mhal
} // namespace mlir

#define DEBUG_TYPE "mhal-annotate-access-kinds"

using namespace mlir;
namespace {
struct MHALAnnotateAccessKindsPass
    : public mhal::impl::MHALAnnotateAccessKindsPassBase<
          MHALAnnotateAccessKindsPass> {
  void runOnOperation() override;
};
} // end namespace

void MHALAnnotateAccessKindsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!func->hasAttr("kernel"))
    return;

  Builder b(&getContext());
  // Add access modes for parameters: read-only, write-only, read-write
  // All MemRef params are marked as 'read-write'
  // Non-MemRef inputs are added as 'read-only'
  auto readAttr = b.getNamedAttr(mhal::MHALDialect::getReadAccessAttrName(),
                                 b.getUnitAttr());
  auto writeAttr = b.getNamedAttr(mhal::MHALDialect::getWriteAccessAttrName(),
                                  b.getUnitAttr());
  auto getAccessAttrs = [&](Type t,
                            bool inputs) -> std::optional<DictionaryAttr> {
    if (isa<VectorType, RankedTensorType, UnrankedTensorType>(t))
      return b.getDictionaryAttr({inputs ? readAttr : writeAttr});
    if (isa<MemRefType>(t))
      return b.getDictionaryAttr({readAttr, writeAttr});
    return {};
  };

  // Non-MemRef inputs are added as 'read-only'
  for (auto [i, arg] : llvm::enumerate(func.getArguments())) {
    if (auto attrs = getAccessAttrs(arg.getType(), true))
      func.setArgAttrs(i, *attrs);
  }
  // Non-MemRef results are added as 'write-only'
  for (auto [i, resTy] : llvm::enumerate(func.getResultTypes())) {
    if (auto attrs = getAccessAttrs(resTy, false))
      func.setResultAttrs(i, *attrs);
  }
}
