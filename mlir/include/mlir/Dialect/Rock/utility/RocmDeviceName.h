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
  LogicalResult parse(llvm::StringRef devName);

  llvm::StringRef getChip() const { return chip; }
  const llvm::StringMap<bool> &getFeatures() const { return features; }
  std::string getFeaturesForBackend() const;
  llvm::StringRef getTriple() const { return triple; }

  void getFullName(llvm::SmallVectorImpl<char> &out) const;

private:
  llvm::SmallString<8> chip;
  llvm::StringMap<bool> features;
  llvm::SmallString<32> triple;
};

} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_ROCMDEVICENAME_H_
