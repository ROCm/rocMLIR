//=== Utils.cpp - functions that often come up during lowering
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/MHAL/Utility/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace llvm;

SmallVector<mlir::mhal::PrefillAttr>
mlir::mhal::getStoredPrefillAttributes(mlir::LLVM::LLVMFuncOp func) {
  SmallVector<mhal::PrefillAttr> storedAttrs;
  auto gpuModule = cast<gpu::GPUModuleOp>(func->getParentOp());
  if (auto moduleAttr = gpuModule->getAttr(func.getSymName())) {
    if (auto arrayAttr = dyn_cast<ArrayAttr>(moduleAttr)) {
      for (auto attr : arrayAttr) {
        if (auto prefillAttr = dyn_cast<mhal::PrefillAttr>(attr)) {
          storedAttrs.push_back(prefillAttr);
        }
      }
    }
  }
  return storedAttrs;
}
