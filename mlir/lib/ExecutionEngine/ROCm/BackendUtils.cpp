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
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"

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

// lld headers.
#include "lld/Common/Driver.h"

// If an old rocm_agent_enumerator that has no "-name" option is used, rely on
// the hip runtime function to provide GPU GCN Arch names.
#if __USE_ROCM_4_4_OR_OLDER__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include "hip/hip_runtime.h"
#pragma GCC diagnostic pop
#endif

#include <mutex>
#include <numeric>

using namespace mlir;

static constexpr const char kRunnerProgram[] = "mlir-rocm-runner";
static constexpr const char kRocmAgentEnumerator[] = "rocm_agent_enumerator";
static constexpr const char kTargetTriple[] = "amdgcn-amd-amdhsa";

#if __USE_ROCM_4_4_OR_OLDER__
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
                           const std::string &defaultFeatures,
                           const Optional<int> deviceNum, bool systemOverride)
    : triple(defaultTriple), chip(defaultChip), features(defaultFeatures),
      deviceNum(deviceNum) {
  if (systemOverride) {
    triple = kTargetTriple;
    configTarget(chip, features);
  }
}

BackendUtils::BackendUtils() : BackendUtils("", "", "", 0, true) {}

void BackendUtils::configTarget(std::string &targetChip,
                                std::string &features) {
  if (!deviceNum.hasValue()) {
    llvm::WithColor::warning(llvm::errs()
                             << "Runtime architecture detection"
                                "without a device specified, defaulting to 0");
  }
  int deviceId = deviceNum.getValueOr(0);
  // Locate rocm_agent_enumerator.
  llvm::ErrorOr<std::string> rocmAgentEnumerator = llvm::sys::findProgramByName(
      kRocmAgentEnumerator, {__ROCM_PATH__ "/bin"});
  std::error_code ec = rocmAgentEnumerator.getError();
  if (ec) {
    llvm::WithColor::warning(llvm::errs(), kRunnerProgram)
        << kRocmAgentEnumerator << " couldn't be located under "
        << __ROCM_PATH__ << ", set target as " << chip << "\n";
    return;
  }

  // Prepare temp file to hold the outputs.
  int tempFd = -1;
  SmallString<128> tempFilename;
  ec = llvm::sys::fs::createTemporaryFile("rocm_agent", "txt", tempFd,
                                          tempFilename);
  if (ec) {
    llvm::WithColor::warning(llvm::errs(), kRunnerProgram)
        << "temporary file for " << kRocmAgentEnumerator
        << " creation error, set target as " << chip << "\n";
    return;
  }
  llvm::FileRemover cleanup(tempFilename);

  // Invoke rocm_agent_enumerator.
  std::string errorMessage;
#if __USE_ROCM_4_4_OR_OLDER__
  SmallVector<StringRef, 1> args{rocmAgentEnumerator.get()};
#else
  SmallVector<StringRef, 2> args{rocmAgentEnumerator.get(), "-name"};
#endif

  Optional<StringRef> redirects[3] = {{""}, tempFilename.str(), {""}};
  int result =
      llvm::sys::ExecuteAndWait(rocmAgentEnumerator.get(), args, llvm::None,
                                redirects, 0, 0, &errorMessage);
  if (result) {
    llvm::WithColor::warning(llvm::errs(), kRunnerProgram)
        << kRocmAgentEnumerator << " invocation error: " << errorMessage
        << ", set target as " << chip << "\n";
#if __USE_ROCM_4_4_OR_OLDER__
    llvm::WithColor::warning(llvm::errs(), kRunnerProgram)
        << "suggest to use a newer ROCm release and compile with "
           "-DUSE_ROCM_4_4_OR_OLDER=0\n";
#endif
    return;
  }

  // Load and parse the result.
  auto gfxArchList = mlir::openInputFile(tempFilename);
  if (!gfxArchList) {
    llvm::WithColor::error(llvm::errs(), kRunnerProgram)
        << "read ROCm agent list temp file error, set features as " << chip
        << "\n";
    return;
  }
  int lineCount = 0;
#if __USE_ROCM_4_4_OR_OLDER__
  for (llvm::line_iterator lines(*gfxArchList); !lines.is_at_end(); ++lines) {
    // Skip the line with content "gfx000".
    if (*lines == "gfx000") {
      continue;
    }
    if (lineCount++ != deviceId) {
      continue;
    }
    targetChip = lines->str();
    break;
  }
  std::string gcnArchName;
  getGpuGCNArchName(deviceId, gcnArchName);
  std::string chip;
  auto status = IsaNameParser::parseArchName(gcnArchName, chip, features);
  if (status.failed()) {
    llvm_unreachable("HIP ArchName parsing should never fail.");
  }
#else
  int lineCount = 0;
  std::string gcnArchName;
  for (llvm::line_iterator lines(*gfxArchList); !lines.is_at_end(); ++lines) {
    if (lineCount++ != deviceId) {
      continue;
    }
    gcnArchName = lines->str();
    break;
  }
  auto status = IsaNameParser::parseArchName(gcnArchName, targetChip, features);
  if (status.failed()) {
    llvm_unreachable("ArchName parsing failed.");
  }
#endif
}

LogicalResult mlir::switchToDevice(int id) {
#if __USE_ROCM_4_4_OR_OLDER__
  int maxDeviceCount = -1;
  hipGetDeviceCount(&maxDeviceCount);
  if (maxDeviceCount <= id) {
    llvm::WithColor::error(
        llvm::errs()
        << "Error: Requested GPU ID " << id
        << "is larger than the number of devices available on the system ("
        << maxDeviceCount << ")\n");
    if (maxDeviceCount <= 0) {
      llvm::WithColor::note(
          llvm::errs() << "Note: No executors were found. Confirm you have a"
                          "HIP-capable GPU and the permissions to use it\n");
    }
    if (maxDeviceCount == 1 && id == 1) {
      llvm::WithColor::note(llvm::errs()
                            << "Note: the device ID is 0-indexed\n");
    }
    return failure();
  }

  // The device number is per-thread, but the JIT'd code runs on one thread
  // so this is safe to put here
  if (HIP_SUCCESS != hipSetDevice(id)) {
    llvm::WithColor::error(llvm::errs() << "Error: Couldn't switch to device "
                                        << id << "\n");
  }
  return success();

#else // __USE_ROCM_4_4_OR_OLDER__
  llvm_unreachable(
      "switchToDevice() should not be called when MLIR isn't managing"
      "the runtime");
#endif
}
