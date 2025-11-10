//===- SystemDevices.h - ExecutionEngine System Devices --------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a system detection interface with default CPU detection
// for the execution engine.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_CPUSYSTEMDETECT_H_
#define MLIR_EXECUTIONENGINE_CPUSYSTEMDETECT_H_

#include "mlir/ExecutionEngine/SystemDevices.h"
#include <vector>

namespace mlir {

/// CpuSystemDetect finds cpus on the current system.
///

class CpuSystemDetect : public std::vector<SystemDevice> {
  CpuSystemDetect();
  virtual ~CpuSystemDetect() {}

public:
  // Singleton
  static const CpuSystemDetect &get() {
    static CpuSystemDetect s_cpuSystem;
    return s_cpuSystem;
  }
};

} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_CPUSYSTEMDETECT_H_
