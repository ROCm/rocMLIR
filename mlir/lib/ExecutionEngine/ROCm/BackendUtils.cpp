//===- BackendUtils.cpp - MLIR to C++ option parsing ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements mlir rocm backend utils
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/ROCm/BackendUitls.h"
#include "mlir/ExecutionEngine/ROCm/IsaNameParser.h"

#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"

// MC headers.
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"

// If an old rocm_agent_enumerator that has no "-name" option is used, rely on
// the hip runtime function to provide GPU GCN Arch names.
#if __INCLUDE_HIP__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include "hip/hip_runtime.h"
#pragma GCC diagnostic pop
#endif

#include <mutex>
#include <numeric>

using namespace mlir;

static constexpr const char kTargetTriple[] = "amdgcn-amd-amdhsa";

#if __INCLUDE_HIP__
namespace {
void getGpuGCNArchName(hipDevice_t device, std::string &gcnArchName) {
  hipDeviceProp_t props;
  hipError_t result = hipGetDeviceProperties(&props, device);
  if (result != hipSuccess) {
    gcnArchName = "";
    llvm_unreachable("hipGetDeviceProperties() should never fail");
    return;
  }

  const char *pArchName = props.gcnArchName;
  gcnArchName.assign(pArchName);
}
} // namespace
#endif

BackendUtils::BackendUtils(const std::string &defaultTriple,
                           const std::string &defaultChip,
                           const std::string &defaultFeatures)
    : triple(defaultTriple), chip(defaultChip), features(defaultFeatures) {
  if (triple.empty() && chip.empty() && features.empty()) {
    triple = kTargetTriple;
    configTarget(chip, features);
  }
}

// TODO(kdrewnia): Assumes that GPU 0 has the same chipset as the GPU that
// the kernel will be running on
// This is true in our testing environment, but is not true in general
void BackendUtils::configTarget(std::string &targetChip,
                                std::string &features) {
#if __INCLUDE_HIP__
  std::string gcnArchName;
  getGpuGCNArchName(0, gcnArchName);
  auto status = IsaNameParser::parseArchName(gcnArchName, targetChip, features);
  if (status.failed()) {
    llvm_unreachable("HIP ArchName parsing should never fail.");
  }
#else
  llvm_unreachable("BackendUtils.cpp does not include HIP.");
#endif
}
