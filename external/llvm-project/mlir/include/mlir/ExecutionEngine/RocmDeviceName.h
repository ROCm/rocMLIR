//===- RocmDeviceName.h -  ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares ROCm ISA name string splitter.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_ROCMDEVICENAME_H_
#define MLIR_EXECUTIONENGINE_ROCMDEVICENAME_H_

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallString.h"

namespace mlir {

// RocmDeviceName handles decomposition of gcnArchName
class RocmDeviceName {
public:
  RocmDeviceName(StringRef devName);

  operator bool() const { return succeeded(status); }

  StringRef getChip() const { return chip; }
  StringRef getFeatures() const { return features; }
  StringRef getTriple() const { return triple; }

  SmallString<256> getFullName() const;

private:
  LogicalResult status;
  SmallString<32> chip;
  SmallString<32> features;
  SmallString<32> triple;
};

} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_ROCMDEVICENAME_H_
