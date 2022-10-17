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
#include "llvm/ADT/StringMap.h"

namespace mlir {

// RocmDeviceName handles decomposition of gcnArchName
class RocmDeviceName {
public:
  LogicalResult parse(StringRef devName);

  StringRef getChip() const { return chip; }
  const llvm::StringMap<bool> &getFeatures() const { return features; }
  std::string getFeaturesForBackend() const;
  StringRef getTriple() const { return triple; }

  void getFullName(SmallVectorImpl<char> &out) const;

private:
  SmallString<8> chip;
  llvm::StringMap<bool> features;
  SmallString<32> triple;
};

} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_ROCMDEVICENAME_H_
