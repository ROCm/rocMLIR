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

extern "C" {

/// Returns the opaque HIP runtime handle owned by `RocmSystemDetect`,
/// or `nullptr` if HIP could not be loaded (or this binary did not
/// link `MLIRRocmExecutionEngineUtils`).
///
/// `RocmSystemDetect` is the canonical owner of the per-process HIP
/// handle. When it loads `libamdhip64`, it uses
/// `dlmopen(LM_ID_NEWLM, ...)` (glibc) to put HIP and its transitive
/// dependencies (libamd_comgr, ROCm's libLLVM) in a private link-map
/// namespace. KFD enforces one HSA session per process; if a second
/// loader (for example mlir-runner's `libmlir_rocm_runtime.so`) opens
/// HIP into a *different* namespace, that second instance receives
/// `hipErrorNoDevice` from every call. To avoid that, all subsequent
/// HIP loaders look up this symbol via `RTLD_DEFAULT` and reuse the
/// returned handle. The recommended way to do that is to call
/// `mlir::rocm_loader::loadRocmLibrary(Library::Hip)` (defined in
/// `mlir/ExecutionEngine/RocmRuntimeLoader.h`), which performs the
/// lookup transparently.
///
/// The function is `extern "C"` and uses an opaque `void *` so it can
/// be safely dlsym-ed from a TU that does not include this header
/// (notably from `RocmRuntimeLoader.cpp` itself, which avoids a
/// link-time dependency on `MLIRRocmExecutionEngineUtils`).
///
/// Visibility: the symbol is exported with default visibility on POSIX
/// and `__declspec(dllexport)` on Windows so it lands in the host
/// process's dynamic symbol table for `RTLD_DEFAULT` lookup.
void *mlirRocmSystemDetectGetHipHandle();

} // extern "C"

#endif // MLIR_EXECUTIONENGINE_ROCMSYSTEMDETECT_H_
