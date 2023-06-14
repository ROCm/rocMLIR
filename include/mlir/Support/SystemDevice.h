//===- SystemDevice.h - ExecutionEngine System Devices ---------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a system device interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_SYSTEMDEVICE_H_
#define MLIR_SUPPORT_SYSTEMDEVICE_H_

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

#include <string>

namespace mlir {
namespace mhal {

/// SystemDevice captures device specifics.
///

struct SystemDevice {
  enum class Type : int32_t { ECPU, EGPU, ENPU, EALT };

  SystemDevice(Type type);

  LogicalResult parse(StringRef arch);
  bool isCompatible(const SystemDevice &that) const;
  std::string getArch() const;
  void dump() const;

private:
  Type type;
  llvm::SmallString<32> llvmTriple = {};
  llvm::SmallString<8> chip = {};
  llvm::StringMap<bool> features = {};
  uint32_t count = 1;
  llvm::StringMap<llvm::SmallString<8>> properties = {};
};

} // namespace mhal
} // namespace mlir

#endif // MLIR_SUPPORT_SYSTEMDEVICE_H_
