//===- SystemDevices.h - ExecutionEngine System Devices --------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a system device interface for the execution engine.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_SYSTEMDEVICES_H_
#define MLIR_EXECUTIONENGINE_SYSTEMDEVICES_H_

#include "mlir/Support/LogicalResult.h"

#include <string>
#include <unordered_map>

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {

/// SystemDevice captures device specifics.
///

struct SystemDevice {
  enum class Type : int32_t { ECPU, EGPU, ENPU, EALT };

  Type type;
  llvm::SmallString<32> llvmTriple = {};
  llvm::SmallString<8> chip = {};
  llvm::StringMap<bool> features = {};
  uint32_t count = 1;
  llvm::StringMap<llvm::SmallString<8>> properties = {};

  LogicalResult parse(llvm::StringRef arch);
  bool isCompatible(const SystemDevice &that) const;
  std::string getArch() const;
  void dump() const;
};

/// SystemDevices captures devices for the current system.
/// - singleton - map references SystemDevice objects from device
///   singletons
class SystemDevices {
public:
  // Singleton per SystemDetect Types
  template <typename... Targs>
  static const SystemDevices &get() {
    static SystemDevices s_systemDevices(types<Targs...>{});
    return s_systemDevices;
  }

  FailureOr<const SystemDevice *const>
  find(SystemDevice::Type type, llvm::StringRef partialArchName = "") const;

  void dump() const;

private:
  template <class...>
  struct types {
    using type = types;
  };

  template <typename... Targs>
  SystemDevices(types<Targs...>) {
    registerSystems(types<Targs...>{});
  }

  template <typename Tsys>
  void registerSystem() {
    for (const auto &dev : Tsys::get())
      devices.push_back(&dev);
  }

  template <typename Tsys, typename... Targs>
  void registerSystems(types<Tsys, Targs...>) {
    registerSystem<Tsys>();
    registerSystems(types<Targs...>{});
  }
  // terminus
  void registerSystems(types<>) {}

  llvm::SmallVector<const SystemDevice *, 0> devices;
};

} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_SYSTEMDEVICES_H_
