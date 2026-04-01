//===-- MIGraphXExperimentalFlags.h - Hand-toggled experiment flags -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared toggle for local experiments (clone-harness func.call vs mhal.launch,
// MIXR boundary omitting MHALLaunch, etc.). Not a stable API—edit the default
// in this header and rebuild. rocmlir-gen and rocmlir-driver are separate
// binaries: each gets its own copy of the flag storage at runtime.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MIGRAPHXEXPERIMENTALFLAGS_H
#define MLIR_CONVERSION_MIGRAPHXEXPERIMENTALFLAGS_H

namespace mlir::migraphx {

/// When true: clone-harness may use func.call and MIXR boundary logic may omit
/// mhal.launch handling. When false: legacy mhal.launch + MHALLaunchConverter.
inline bool cloneHarnessExperiment = false;

} // namespace mlir::migraphx

#endif // MLIR_CONVERSION_MIGRAPHXEXPERIMENTALFLAGS_H
