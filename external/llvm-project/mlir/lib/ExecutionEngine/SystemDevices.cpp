//===- RocmSystemDetect.cpp - Detect ROCm devices -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the system detection of ROCm devices on the current
// system.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/SystemDevices.h"
#include "mlir/ExecutionEngine/RocmDeviceName.h"
#include "mlir/ExecutionEngine/RocmSystemDetect.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "execution-engine-system-devices"

using namespace mlir;

static const char *getDeviceTypeStr(SystemDevice::Type type) {
  return type == SystemDevice::Type::ECPU
             ? "CPU"
             : type == SystemDevice::Type::EGPU
                   ? "GPU"
                   : type == SystemDevice::Type::ENPU ? "NPU" : "ALT";
}

FailureOr<const SystemDevice *const>
SystemDevices::find(SystemDevice::Type type,
                    llvm::StringRef partialArchName) const {
  llvm::SmallVector<StringRef, 3> parts;

  StringRef maybeTriple, remainder;
  std::tie(maybeTriple, remainder) = partialArchName.split(':');
  if (maybeTriple.contains('-')) {
    // First part is a triple, ignore it
    partialArchName = remainder;
  } else {
    maybeTriple = "";
  }

  StringRef chip, rawFeatures;
  std::tie(chip, rawFeatures) = partialArchName.split(':');

  llvm::SmallVector<StringRef, 1> featureTokens;
  rawFeatures.split(featureTokens, ':');
  llvm::StringMap<bool> features;
  for (StringRef feature : featureTokens) {
    feature = feature.trim();
    if (!feature.empty()) {
      features.insert_or_assign(feature.drop_back(), feature.back() == '+');
    }
  }

  for (const SystemDevice *const device : devices) {
    bool matches = (type == device->type) &&
                   (maybeTriple.empty() || maybeTriple == device->llvmTriple) &&
                   (chip.empty() || chip == device->chip);
    if (matches && !features.empty()) {
      for (const llvm::StringMapEntry<bool> &feature : features) {
        StringRef key = feature.getKey();
        matches &= (device->features.count(key) != 0 &&
                    device->features.lookup(key) == feature.getValue());
      }
    }
    if (matches)
      return device;
  }
  return failure();
}

void SystemDevices::dump() const {
  for (const SystemDevice *device : devices) {
    llvm::errs() << "Device(" << getDeviceTypeStr(device->type) << ") x "
                 << device->count << "\n"
                 << " triple = " << device->llvmTriple
                 << " chip = " << device->chip;
    if (!device->features.empty()) {
      llvm::errs() << " features = ";
      llvm::interleave(
          device->features, llvm::errs(),
          [](const llvm::StringMapEntry<bool> &entry) {
            llvm::errs() << entry.getKey() << (entry.getValue() ? "+" : "-");
          },
          ":");
    }
    llvm::errs() << " {\n";
    for (auto &pair : device->properties) {
      llvm::errs() << "  " << pair.first() << " = " << pair.second << "\n";
    }
    llvm::errs() << "}\n";
  }
}
