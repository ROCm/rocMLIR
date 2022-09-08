//===- SystemDetect.cpp - MLIR Execution engine and utils --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the execution engine for MLIR modules based on LLVM Orc
// JIT engine.
//
//===----------------------------------------------------------------------===//
#include "mlir/ExecutionEngine/CpuSystemDetect.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Host.h"

#include <fstream>
#include <iostream>
#include <string>

#define DEBUG_TYPE "execution-engine-system-detect"

using namespace mlir;

///
CpuSystemDetect::CpuSystemDetect() {
  // collect all CPU
  auto targetTriple = llvm::sys::getProcessTriple();
  llvm::SmallString<32> cpu(llvm::sys::getHostCPUName());
  uint32_t count = llvm::sys::getHostNumPhysicalCores();

  // cleanup
  SystemDevice dev{SystemDevice::Type::ECPU,
                   cpu,
                   count,
                   {{"llvm_triple", llvm::StringRef(targetTriple)}}};

  llvm::StringMap<bool> hostFeatures;
  if (llvm::sys::getHostCPUFeatures(hostFeatures)) {
    for (auto &f : hostFeatures) {
      llvm::StringRef val(f.second ? "1" : "0");
      dev.properties.insert({f.first().str(), val});
    }
  }
  push_back(std::move(dev));
}
