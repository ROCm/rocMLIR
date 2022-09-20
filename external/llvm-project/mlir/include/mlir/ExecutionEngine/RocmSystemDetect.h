//===- RocmSystemDetect.h - ExecutionEngine ROCm System Detect --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a ROCm detection utility for the current system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_ROCMSYSTEMDETECT_H_
#define MLIR_EXECUTIONENGINE_ROCMSYSTEMDETECT_H_

#include "mlir/ExecutionEngine/SystemDevices.h"
#include <vector>

namespace mlir {

/// RocmSystemDetect finds ROCm devices on the current system.
///
class RocmSystemDetect : public std::vector<SystemDevice> {
  RocmSystemDetect();
  virtual ~RocmSystemDetect() {}

public:
  static const RocmSystemDetect &get() {
    static RocmSystemDetect s_rocmSystem;
    return s_rocmSystem;
  }
};

} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_ROCMSYSTEMDETECT_H_
