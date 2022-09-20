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

#include "mlir/ExecutionEngine/RocmDeviceName.h"
#include "mlir/ExecutionEngine/RocmSystemDetect.h"

#include "llvm/Support/Error.h"

#define DEBUG_TYPE "execution-engine-system-devices"

using namespace mlir;

static const char *getDeviceTypeStr(SystemDevice::Type type) {
  return type == SystemDevice::Type::ECPU
             ? "CPU"
             : type == SystemDevice::Type::EGPU
                   ? "GPU"
                   : type == SystemDevice::Type::ENPU ? "NPU" : "ALT";
}

void SystemDevices::dump() const {
  for (auto &entry : *this) {
    auto &device = entry.second;
    llvm::errs() << "Device(" << getDeviceTypeStr(device.type)
                 << "): " << device.chip << "(" << device.count << ") {\n";
    for (auto &pair : device.properties) {
      llvm::errs() << "  " << pair.first() << " = " << pair.second << "\n";
    }
    llvm::errs() << "}\n";
  }
}
