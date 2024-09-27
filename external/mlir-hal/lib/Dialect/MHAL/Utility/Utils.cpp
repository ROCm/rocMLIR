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
mlir::mhal::getStoredPrefillAttributes(gpu::BinaryOp binary) {
  SmallVector<mhal::PrefillAttr> storedAttrs;
  auto object = mlir::cast<mlir::gpu::ObjectAttr>(binary.getObjects()[0]);
  DictionaryAttr properties = object.getProperties();
  gpu::KernelTableAttr kernels = object.getKernels();
  // Fail if there are no object properties.
  if (!properties || properties.empty() || !kernels || kernels.size() == 0)
    return storedAttrs;
  ArrayRef<gpu::KernelMetadataAttr> kernelList = kernels.getKernelTable();
  assert(kernelList.size() == 1 &&
         "binaries with multiple kernels are not supported");
  if (auto arrayAttr = properties.getAs<ArrayAttr>(kernelList[0].getName())) {
    for (auto attr : arrayAttr) {
      if (auto prefillAttr = dyn_cast<mhal::PrefillAttr>(attr)) {
        storedAttrs.push_back(prefillAttr);
      }
    }
  }
  return storedAttrs;
}
