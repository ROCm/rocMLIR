//===- RocmArchRuntime.h - C ABI for native AMD GPU arch query ------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stable C ABI exported by the `mlir_rocm_arch_runtime` shared library.
//
// rocMLIR ships an embedded copy of LLVM. The AMD HIP runtime
// (`libamdhip64`) transitively pulls in `amd_comgr` and ROCm's own
// `libLLVM.so`, which collides with rocMLIR's LLVM (duplicate `cl::opt`
// registration, corrupted command-line parser global state, ...). To keep
// `MLIRRockOps` and any binary that links it free of that conflict, all HIP /
// HSA usage is confined to a small standalone shared library which is loaded
// at runtime via `dlmopen(LM_ID_NEWLM, ...)` (Linux/glibc) or `LoadLibraryW`
// (Windows) only when the user asks for native architecture detection (e.g.
// `--arch native:0`). The fresh dlopen namespace keeps ROCm's `libLLVM.so`
// from snapping onto rocMLIR's embedded LLVM. Tools that do not touch the
// `native:` arch path never load it and therefore never see ROCm's LLVM.
//
// The ABI here is intentionally tiny and uses only C primitives so that it
// stays binary-stable across compiler/runtime upgrades. New fields must only
// be appended; existing ones must not be reordered or removed.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_ROCMARCHRUNTIME_H
#define MLIR_EXECUTIONENGINE_ROCMARCHRUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Status codes returned by the runtime entry points. Negative values are
/// reserved for future use; callers should treat any non-zero value as an
/// error.
///
/// ABI stability: codes below are part of the stable C ABI exported by
/// `mlir_rocm_arch_runtime`. New codes must only be appended; existing
/// codes must not be removed or renumbered.
enum MlirRocmArchStatus {
  MLIR_ROCM_ARCH_OK = 0,
  /// Reserved: The HIP runtime (libamdhip64) could not be loaded. The current
  /// implementation never returns this value because the runtime library
  /// itself link-depends on HIP; if HIP cannot be found, the runtime library
  /// fails to load and the loader in `AmdArchDb.cpp` never sees this code.
  /// Kept in the ABI in case a future implementation chooses to dynamically
  /// load HIP from inside the runtime.
  MLIR_ROCM_ARCH_HIP_UNAVAILABLE = 1,
  /// A HIP API call returned an error (e.g. invalid device id).
  MLIR_ROCM_ARCH_HIP_ERROR = 2,
  /// An HSA query failed. Some fields may still be populated.
  MLIR_ROCM_ARCH_HSA_ERROR = 3,
};

/// Maximum length of the `gcnArchName` string, including the terminating NUL.
/// Matches the HIP definition (`HIP_ARCH_NAME_MAX` in `hip_runtime_api.h`).
enum { MLIR_ROCM_ARCH_NAME_MAX = 256 };

/// Plain-old-data view of the subset of HIP/HSA device properties that
/// `MLIRRockOps` consumes. All fields are zero-initialised by the runtime
/// before any query is performed, so consumers may rely on zero meaning
/// "unknown" for unpopulated entries.
struct MlirRocmArchProperties {
  /// Hardware-reported architecture string, e.g. `"gfx942:sramecc+:xnack-"`.
  /// Populated from `hipDeviceProp_t::gcnArchName`.
  char gcnArchName[MLIR_ROCM_ARCH_NAME_MAX];

  /// HIP `multiProcessorCount`.
  uint32_t multiProcessorCount;
  /// HIP `warpSize`.
  uint32_t warpSize;
  /// HIP `maxSharedMemoryPerMultiProcessor`.
  uint64_t sharedMemPerCU;
  /// HIP `sharedMemPerBlock`.
  uint64_t sharedMemPerBlock;

  /// HSA `HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU`. Zero if HSA was unavailable.
  uint32_t simdsPerCU;
  /// HSA `HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU`. Zero if HSA was unavailable.
  uint32_t maxWavesPerCU;
  /// HSA `HSA_AMD_AGENT_INFO_NUM_XCC`. Zero if HSA was unavailable.
  uint32_t numXCC;

  /// Set to non-zero when the HSA query succeeded.
  uint32_t hsaValid;
};

/// ABI version of this header. The runtime library reports its own version
/// via `mlirRocmArchRuntimeAbiVersion`; loaders must refuse to use a runtime
/// whose version differs.
enum { MLIR_ROCM_ARCH_RUNTIME_ABI_VERSION = 1 };

/// Returns the ABI version implemented by the runtime library. Used by the
/// loader to verify compatibility before calling any other entry point.
int32_t mlirRocmArchRuntimeAbiVersion(void);

/// Returns the number of AMD GPUs visible via the HIP runtime, or 0 if the
/// HIP runtime cannot be loaded or returns an error. Never aborts.
uint32_t mlirRocmArchRuntimeDeviceCount(void);

/// Populates `*outProps` with information about device `deviceId`. On success
/// returns `MLIR_ROCM_ARCH_OK`; on failure returns one of the
/// `MLIR_ROCM_ARCH_*` error codes and leaves `*outProps` zero-initialised
/// (HIP errors) or partially populated (HSA errors only -- HIP fields will be
/// valid). `outProps` must not be null.
int32_t
mlirRocmArchRuntimeGetProperties(uint32_t deviceId,
                                 struct MlirRocmArchProperties *outProps);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MLIR_EXECUTIONENGINE_ROCMARCHRUNTIME_H
