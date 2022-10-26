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

// SystemDevice::SystemDevice(Type _type, StringRef _triple, StringRef _chip, llvm::StringMap<bool> _features)
//     : type(_type), llvmTriple(_triple), chip(_chip), features(_features) {}

LogicalResult SystemDevice::parse(StringRef arch) {
  StringRef maybeTriple, remainder;
  std::tie(maybeTriple, remainder) = arch.split(':');
  if (maybeTriple.contains('-')) {
    // First part is a triple, ignore it
    llvmTriple = maybeTriple;
  } else {
    llvmTriple = "";
    remainder = arch;
  }

  StringRef rawFeatures;
  std::tie(chip, rawFeatures) = remainder.split(':');

  if (!rawFeatures.empty()) {
    llvm::SmallVector<StringRef, 1> featureTokens;
    rawFeatures.split(featureTokens, ':'); // check for CSV
    for (StringRef feature : featureTokens) {
      feature = feature.trim();
      if (!feature.empty()) {
        features.insert_or_assign(feature.drop_back(), feature.back() == '+');
      }
    }
  }
  return success();
}

bool SystemDevice::isCompatible(const SystemDevice &that) const {
  bool matches =
      (type == that.type) &&
      (llvmTriple.empty() || that.llvmTriple.empty() || llvmTriple == that.llvmTriple) &&
      (chip.empty() || that.chip.empty() || chip == that.chip);
  if (matches && !features.empty()) {
    for (const llvm::StringMapEntry<bool> &feature : features) {
      StringRef key = feature.getKey();
      matches &= (that.features.count(key) != 0 &&
                  that.features.lookup(key) == feature.getValue());
    }
  }
  return matches;
}

std::string SystemDevice::getArch() const {
  std::string arch;
  if (!llvmTriple.empty()) {
    arch = llvmTriple.str();
    arch += ":";
  }
  if (!chip.empty())
    arch += chip.str();
  
  for (const auto &entry : features ) {
    arch += ":";
    arch += entry.getKey().str();
    arch += (entry.getValue() ? "+" : "-");
  }

  return arch;
}

void SystemDevice::dump() const {
  llvm::errs() << "Device(" << getDeviceTypeStr(type) << ") x "
               << count << "\n"
               << "\n  triple = " << llvmTriple
               << "\n  chip = " << chip;
  if (!features.empty()) {
    llvm::errs() << "\n  features = ";
    llvm::interleave(
        features, llvm::errs(),
        [](const llvm::StringMapEntry<bool> &entry) {
          llvm::errs() << entry.getKey() << (entry.getValue() ? "+" : "-");
        },
        ":");
  }
  if (!properties.empty()) {
    llvm::errs() << "\n  {\n";
    for (const auto &pair : properties) {
      llvm::errs() << "    " << pair.first() << " = " << pair.second << "\n";
    }
    llvm::errs() << "}\n";
  }
}

////////////////////////////////////////////////////////////////////////////
FailureOr<const SystemDevice *const>
SystemDevices::find(SystemDevice::Type type,
                    llvm::StringRef partialArchName) const {
  SystemDevice testDev{type};
  if (failed(testDev.parse(partialArchName))) {
    return failure();
  }
  for (const SystemDevice *const device : devices) {
    if (device->isCompatible(testDev))
      return device;
  }
  return failure();
}

void SystemDevices::dump() const {
  for (const SystemDevice *device : devices) {
    device->dump();
  }
}
