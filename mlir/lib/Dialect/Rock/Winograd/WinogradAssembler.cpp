//===-- WinogradAssembler.cpp - Assemble Winograd kernels -----------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Winograd/WinogradAssembler.h"
#include "mlir/Dialect/Rock/Winograd/WinogradSolver.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

#define DEBUG_TYPE "rock-winograd-assembler"

using namespace mlir::rock::winograd;

#ifndef ROCMLIR_WINOGRAD_KERNEL_DIR
#define ROCMLIR_WINOGRAD_KERNEL_DIR ""
#endif

static std::string getRocmLlvmBinDir() {
#ifdef _WIN32
  std::string rocmPath = "C:\\Program Files\\AMD\\ROCm";
#else
  std::string rocmPath = "/opt/rocm";
#endif
  if (const char *env = std::getenv("ROCM_PATH"))
    rocmPath = env;
  llvm::SmallString<256> binDir(rocmPath);
  llvm::sys::path::append(binDir, "llvm", "bin");
  return std::string(binDir);
}

static std::string getWinogradKernelDir() {
  if (const char *envDir = std::getenv("ROCMLIR_WINOGRAD_KERNEL_DIR"))
    return std::string(envDir);
  return ROCMLIR_WINOGRAD_KERNEL_DIR;
}

/// Try to find a program by name, first on PATH then under ROCm's llvm/bin.
static llvm::ErrorOr<std::string> findTool(llvm::StringRef name) {
  if (auto path = llvm::sys::findProgramByName(name))
    return path;
  return llvm::sys::findProgramByName(name, {getRocmLlvmBinDir()});
}

std::optional<llvm::SmallVector<char, 0>>
mlir::rock::winograd::assembleWinogradKernel(
    const WinogradKernelSelection &selection, llvm::StringRef chip,
    llvm::StringRef triple, llvm::StringRef features) {

  std::string kernelDir = getWinogradKernelDir();
  if (kernelDir.empty()) {
    llvm::errs() << "Winograd kernel directory not set. "
                 << "Set ROCMLIR_WINOGRAD_KERNEL_DIR or rebuild with "
                 << "-DROCMLIR_WINOGRAD_KERNEL_DIR=...\n";
    return std::nullopt;
  }

  // Build path to the .s file.
  llvm::SmallString<256> srcPath(kernelDir);
  llvm::sys::path::append(srcPath, selection.kernelFile);

  if (!llvm::sys::fs::exists(srcPath)) {
    llvm::errs() << "Winograd kernel source not found: " << srcPath << "\n";
    return std::nullopt;
  }

  // --- Locate tools --------------------------------------------------------

  auto clangPath = findTool("clang");
  if (!clangPath) {
    llvm::errs() << "Could not find clang for Winograd assembly\n";
    return std::nullopt;
  }

  auto ldPath = findTool("ld.lld");
  if (!ldPath) {
    llvm::errs() << "Could not find ld.lld for Winograd linking\n";
    return std::nullopt;
  }

  // --- Create temp files ---------------------------------------------------

  llvm::SmallString<128> objPath;
  if (llvm::sys::fs::createTemporaryFile("winograd", "o", objPath)) {
    llvm::errs() << "Failed to create temp object file\n";
    return std::nullopt;
  }
  llvm::FileRemover objCleaner(objPath);

  llvm::SmallString<128> hsacoPath;
  if (llvm::sys::fs::createTemporaryFile("winograd", "hsaco", hsacoPath)) {
    llvm::errs() << "Failed to create temp HSACO file\n";
    return std::nullopt;
  }
  llvm::FileRemover hsacoCleaner(hsacoPath);

  llvm::SmallString<128> stderrPath;
  if (llvm::sys::fs::createTemporaryFile("winograd_err", "txt", stderrPath)) {
    llvm::errs() << "Failed to create temp stderr file\n";
    return std::nullopt;
  }
  llvm::FileRemover stderrCleaner(stderrPath);

  // --- Step 1: assemble .s -> .o -------------------------------------------

  std::string targetOpt = ("--target=" + triple).str();
  std::string mcpuOpt = ("-mcpu=" + chip).str();
  std::string incOpt = "-I" + kernelDir;
  std::string featOpt;
  if (!features.empty())
    featOpt = ("-mattr=" + features).str();

  llvm::SmallVector<llvm::StringRef> asmArgs = {
      *clangPath,  "-x",    "assembler", targetOpt, mcpuOpt,
      "-mcumode",  "-mwavefrontsize64",
      "-Wa,-defsym,ROCM_METADATA_VERSION=5",
      incOpt,      "-c",    "-o",        objPath,   srcPath};
  if (!featOpt.empty())
    asmArgs.push_back(featOpt);

  // Append any per-kernel compiler options (e.g. -mno-xnack).
  llvm::SmallVector<std::string> extraOpts;
  if (!selection.compOptions.empty()) {
    llvm::SmallVector<llvm::StringRef> parts;
    llvm::StringRef(selection.compOptions).split(parts, ' ', -1, false);
    for (auto &p : parts)
      extraOpts.push_back(p.str());
  }
  for (const auto &opt : extraOpts)
    asmArgs.push_back(opt);

  // Redirect stderr so we can report assembler diagnostics on failure.
  std::optional<llvm::StringRef> redirects[] = {
      std::nullopt, std::nullopt, llvm::StringRef(stderrPath)};

  std::string errMsg;
  int asmRc = llvm::sys::ExecuteAndWait(*clangPath, asmArgs, std::nullopt,
                                        redirects, /*SecondsToWait=*/120,
                                        /*MemoryLimit=*/0, &errMsg);
  if (asmRc != 0) {
    llvm::errs() << "Winograd assembly failed for " << selection.kernelFile;
    if (!errMsg.empty())
      llvm::errs() << ": " << errMsg;
    llvm::errs() << "\n";
    if (auto buf = llvm::MemoryBuffer::getFile(stderrPath, /*IsText=*/true))
      llvm::errs() << (*buf)->getBuffer();
    return std::nullopt;
  }

  // --- Step 2: link .o -> .hsaco -------------------------------------------

  llvm::SmallVector<llvm::StringRef> ldArgs = {*ldPath, "-shared", "-o",
                                                hsacoPath, objPath};

  int ldRc = llvm::sys::ExecuteAndWait(*ldPath, ldArgs, std::nullopt, redirects,
                                       /*SecondsToWait=*/120,
                                       /*MemoryLimit=*/0, &errMsg);
  if (ldRc != 0) {
    llvm::errs() << "Winograd linking failed for " << selection.kernelFile;
    if (!errMsg.empty())
      llvm::errs() << ": " << errMsg;
    llvm::errs() << "\n";
    if (auto buf = llvm::MemoryBuffer::getFile(stderrPath, /*IsText=*/true))
      llvm::errs() << (*buf)->getBuffer();
    return std::nullopt;
  }

  // --- Step 3: read HSACO --------------------------------------------------

  auto hsacoFile =
      llvm::MemoryBuffer::getFile(hsacoPath, /*IsText=*/false);
  if (!hsacoFile) {
    llvm::errs() << "Failed to read HSACO output\n";
    return std::nullopt;
  }

  llvm::StringRef buffer = (*hsacoFile)->getBuffer();
  return llvm::SmallVector<char, 0>(buffer.begin(), buffer.end());
}
