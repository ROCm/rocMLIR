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
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Threading.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"

#include <fstream>
#include <iostream>
#include <string>

#define DEBUG_TYPE "execution-engine-system-detect"

using namespace mlir;

///
CpuSystemDetect::CpuSystemDetect() {
  // collect all CPU
  llvm::SmallString<32> targetTriple(llvm::sys::getProcessTriple());
  llvm::SmallString<8> cpu(llvm::sys::getHostCPUName());
  llvm::StringMap<bool> features;
  if (!llvm::sys::getHostCPUFeatures(features)) {
    // System detect can't fail but we somehow missed the features;
    features.clear();
  }
  uint32_t count = llvm::get_physical_cores();

  // cleanup
  SystemDevice dev{
      SystemDevice::Type::ECPU, targetTriple, cpu, features, count, {}};

  push_back(std::move(dev));
}
