//===- RocmRuntimeLoader.h - Lazy ROCm library loading utilities -*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Public API for delay-loading ROCm runtime shared libraries (libamdhip64,
// libhiprtc, libhsa-runtime64) from the MLIR ExecutionEngine and from
// downstream consumers.
//
// Consumers do NOT link the ROCm runtime at build time. Doing so would
// transitively pull `libamd_comgr` and ROCm's `libLLVM.so.<major>` into
// the host process, which collides at static-init time with MLIR's own
// embedded LLVM (duplicate `cl::opt` registration aborts the process
// from `_dl_init` with "Option '...' already exists" or with a
// SmallPtrSet "Bucket < End" assertion). Loading these libraries with
// `dlmopen(LM_ID_NEWLM, ...)` on glibc puts them in a private link-map
// namespace where their LLVM cannot interpose ours.
//
// Design choices that govern this API:
//
//   - Header is platform-agnostic: no `<windows.h>`, no `<dlfcn.h>`, no
//     `_GNU_SOURCE` define. All platform-specific machinery lives in
//     `RocmRuntimeLoader.cpp`. Downstream `add_mlir_library` users can
//     include this header without inheriting Windows-macro pollution
//     (`min`, `max`, `ERROR`, ...) or feature-test-macro surprises.
//
//   - `LoadedLibrary` is an opaque struct rather than a `void *` typedef
//     so a future change can carry extra state (search path used, debug
//     info, ...) without breaking callers.
//
//   - Cross-process coordination: `RocmSystemDetect` exports
//     `mlirRocmSystemDetectGetHipHandle` (declared in
//     `RocmSystemDetect.h`) so subsequent loaders share its HIP handle
//     and the process keeps a single HSA session. KFD enforces one
//     session per process; an independent second `dlmopen` would
//     otherwise return `hipErrorNoDevice` from every call.
//
//   - HIPRTC and HSA load into HIP's link-map namespace via the
//     `relatedHandle` parameter so they share HIP's KFD session even
//     when HIP itself was loaded into a non-default namespace.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_ROCMRUNTIMELOADER_H
#define MLIR_EXECUTIONENGINE_ROCMRUNTIMELOADER_H

#include <cstddef>

namespace mlir::rocm_loader {

/// Identifies which ROCm shared library to delay-load. The enumerator
/// order is internal and may change; do not rely on it.
enum class Library {
  Hip,
  Hiprtc,
  Hsa,
};

/// Opaque handle returned by `loadRocmLibrary`. `handle == nullptr`
/// indicates load failure; callers must treat that as "runtime
/// unavailable" and degrade gracefully.
struct LoadedLibrary {
  void *handle = nullptr;
};

/// How `loadRocmLibrary` should coordinate with other loaders that
/// might already have opened the requested library in this process.
enum class CoordinationPolicy {
  /// Default. For `Library::Hip`, attempt to reuse the HIP handle
  /// owned by `RocmSystemDetect` (looked up via `RTLD_DEFAULT`); for
  /// every other library this is equivalent to `Owned`.
  Auto,

  /// Skip the shared-handle lookup. The caller is the canonical
  /// owner. Used by `RocmSystemDetect.cpp` to avoid recursion.
  Owned,
};

/// Load `lib` into a private link-map namespace and return an opaque
/// handle. On glibc this uses `dlmopen(LM_ID_NEWLM, ...)`; on other
/// POSIX platforms `dlopen(RTLD_LAZY | RTLD_LOCAL)`; on Windows
/// `LoadLibraryW` with UTF-8 -> UTF-16 conversion of the SONAME.
///
/// When `relatedHandle` is non-null, the new library is opened in the
/// same link-map namespace as `relatedHandle` (glibc only; falls back
/// to the default namespace elsewhere). This is how HIPRTC and HSA
/// share HIP's KFD session.
///
/// Returns a `LoadedLibrary` whose `handle` is null on failure.
/// Failures are non-fatal: this function never aborts the process.
LoadedLibrary
loadRocmLibrary(Library lib, void *relatedHandle = nullptr,
                CoordinationPolicy policy = CoordinationPolicy::Auto);

/// Resolve `name` in a previously-loaded library. Returns `nullptr` if
/// the library failed to load or if the symbol is absent. Callers
/// should treat `nullptr` as a soft error and disable the corresponding
/// feature.
void *resolveRocmSymbol(const LoadedLibrary &lib, const char *name);

} // namespace mlir::rocm_loader

#endif // MLIR_EXECUTIONENGINE_ROCMRUNTIMELOADER_H
