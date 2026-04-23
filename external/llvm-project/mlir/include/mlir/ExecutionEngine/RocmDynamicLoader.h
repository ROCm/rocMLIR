//===- RocmDynamicLoader.h - Lazy ROCm library loading utilities *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Header-only helpers for delay-loading `libamdhip64` / `libhiprtc` /
// `libhsa-runtime64` via `dlmopen` (glibc), `dlopen` (other POSIX) or
// `LoadLibraryA` (Windows). Using these keeps consumers from having to
// link the ROCm runtime at build time; ROCm's transitive `libLLVM.so`
// therefore cannot clobber the host process's embedded LLVM at
// static-init time (see the dlopenHip branch write-up for details).
//
// Single source of truth: every translation unit that needs to resolve
// ROCm entry points should include this header and call
// `loadRocmLibrary(...)` + `resolveRocmSymbol(...)` instead of rolling
// its own `dlmopen` loop.
//
// Process-wide coordination: `libMLIRRocmExecutionEngineUtils.so`
// (RocmSystemDetect) exports `mlirRocmSystemDetectGetHipHandle()`. All
// other HIP loaders (mlir-runner's `libmlir_rocm_runtime.so`, rocMLIR's
// `MLIRRockOps`, `rocmlir-tuning-driver`) look that up via `RTLD_DEFAULT`
// first so the process only ever holds *one* HSA session. KFD only
// permits one such session; a second `dlmopen(LM_ID_NEWLM, ...)` would
// otherwise return `hipErrorNoDevice` from every call.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_ROCMDYNAMICLOADER_H
#define MLIR_EXECUTIONENGINE_ROCMDYNAMICLOADER_H

// glibc's `dlmopen` is gated on _GNU_SOURCE. Define it before system
// headers so the first consumer to include this file picks up the
// declaration regardless of include order.
#if !defined(_WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstddef>

namespace mlir::rocm_loader {

/// Which ROCm runtime library to load. The enumerator order determines the
/// search-path preference within `loadRocmLibrary`.
enum class Library {
  Hip,
  Hiprtc,
  Hsa,
};

/// Opaque handle. `handle == nullptr` indicates load failure.
struct LoadedLibrary {
  void *handle = nullptr;
};

namespace detail {

/// SONAME candidates per library, in preference order. ROCm 7 ships
/// `.so.7`; older clusters may still have `.so.6` or an unversioned
/// alias. Windows uses decorated names (`amdhip64_7.dll`, etc.) while
/// keeping the bare name as a final fallback. The tables are
/// `inline constexpr` so every consumer TU gets the same entries.
inline constexpr const char *kHipCandidates[] = {
#ifdef _WIN32
    "amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll",
#else
    "libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so",
#endif
};

inline constexpr const char *kHiprtcCandidates[] = {
#ifdef _WIN32
    "hiprtc0507.dll", "hiprtc.dll",
#else
    "libhiprtc.so.7", "libhiprtc.so.6", "libhiprtc.so",
#endif
};

inline constexpr const char *kHsaCandidates[] = {
#ifdef _WIN32
    // HSA is not shipped on Windows ROCm; the list is empty so
    // `loadRocmLibrary(Library::Hsa)` always returns nullptr on Windows.
#else
    "libhsa-runtime64.so.1", "libhsa-runtime64.so",
#endif
};

inline const char *const *candidatesFor(Library lib, size_t &count) {
  switch (lib) {
  case Library::Hip:
    count = sizeof(kHipCandidates) / sizeof(kHipCandidates[0]);
    return kHipCandidates;
  case Library::Hiprtc:
    count = sizeof(kHiprtcCandidates) / sizeof(kHiprtcCandidates[0]);
    return kHiprtcCandidates;
  case Library::Hsa:
    count = sizeof(kHsaCandidates) / sizeof(kHsaCandidates[0]);
    return kHsaCandidates;
  }
  count = 0;
  return nullptr;
}

/// Open `path` with the strongest loader isolation the platform
/// offers. On glibc that is `dlmopen(LM_ID_NEWLM, ...)` so anything
/// `path` transitively pulls in (`libamd_comgr`, ROCm's `libLLVM.so.*`)
/// lands in a fresh link-map namespace and is invisible to the host.
/// Other POSIX systems fall back to `dlopen(RTLD_LAZY | RTLD_LOCAL)`,
/// which only controls re-exposure; Windows uses `LoadLibraryA`.
inline void *openIsolated(const char *path) {
#ifdef _WIN32
  return reinterpret_cast<void *>(::LoadLibraryA(path));
#elif defined(__GLIBC__)
  return ::dlmopen(LM_ID_NEWLM, path, RTLD_LAZY);
#else
  return ::dlopen(path, RTLD_LAZY | RTLD_LOCAL);
#endif
}

/// Open `path` in the same link-map namespace that `existingHandle`
/// lives in. On glibc this is `dlmopen(<ns>, ...)` using the namespace
/// reported by `dlinfo(RTLD_DI_LMID, ...)`; on other platforms it is
/// equivalent to `openIsolated(path)`. Used to keep HIPRTC / HSA in
/// HIP's namespace so they share the same KFD session.
inline void *openInSameNamespace(const char *path, void *existingHandle) {
#if defined(_WIN32) || !defined(__GLIBC__)
  (void)existingHandle;
  return openIsolated(path);
#else
  if (!existingHandle)
    return openIsolated(path);
  Lmid_t ns = 0;
  if (::dlinfo(existingHandle, RTLD_DI_LMID, &ns) != 0)
    return openIsolated(path);
  return ::dlmopen(ns, path, RTLD_LAZY);
#endif
}

/// Return the HIP handle already opened by `RocmSystemDetect` (if any)
/// or `nullptr`. The coordination symbol
/// `mlirRocmSystemDetectGetHipHandle` is looked up via `RTLD_DEFAULT`
/// so the caller does not need a link-time dependency on
/// `libMLIRRocmExecutionEngineUtils.so`.
inline void *getSharedHipHandle() {
#ifdef _WIN32
  // Windows DLLs have private scopes and do not need cross-library
  // coordination for KFD sessions (ROCm on Windows does not ship HSA).
  return nullptr;
#else
  void *getter = ::dlsym(RTLD_DEFAULT, "mlirRocmSystemDetectGetHipHandle");
  if (!getter)
    return nullptr;
  using GetHandleFn = void *(*)();
  return reinterpret_cast<GetHandleFn>(getter)();
#endif
}

} // namespace detail

/// How `loadRocmLibrary` should coordinate with other loaders in the
/// same process.
enum class CoordinationPolicy {
  /// Default: for `Library::Hip`, first attempt a shared-handle lookup
  /// via `mlirRocmSystemDetectGetHipHandle`; for every other library
  /// this is equivalent to `Owned`.
  Auto,
  /// Never look up a shared handle. The caller is willing to own an
  /// independent copy of the library (typically because it *is* the
  /// canonical owner). Used by `RocmSystemDetect.cpp`.
  Owned,
};

/// Load `lib` into an isolated link-map namespace. When `relatedHandle`
/// is non-null, tries to reuse its namespace so transitive dependencies
/// are shared (e.g. HIPRTC and HSA in HIP's namespace). Returns a
/// `LoadedLibrary` whose `handle` is null on failure; the caller should
/// treat null as "runtime unavailable" rather than a fatal error.
inline LoadedLibrary
loadRocmLibrary(Library lib, void *relatedHandle = nullptr,
                CoordinationPolicy policy = CoordinationPolicy::Auto) {
  LoadedLibrary out;
  if (lib == Library::Hip && policy == CoordinationPolicy::Auto) {
    if (void *shared = detail::getSharedHipHandle()) {
      out.handle = shared;
      return out;
    }
  }
  size_t count = 0;
  const char *const *candidates = detail::candidatesFor(lib, count);
  for (size_t i = 0; i < count; ++i) {
    void *h = relatedHandle
                  ? detail::openInSameNamespace(candidates[i], relatedHandle)
                  : detail::openIsolated(candidates[i]);
    if (h) {
      out.handle = h;
      return out;
    }
  }
  return out;
}

/// Resolve `name` in a previously-loaded library. Returns `nullptr` if
/// the symbol is absent; callers should treat that as a soft error and
/// disable the corresponding feature.
inline void *resolveRocmSymbol(const LoadedLibrary &lib, const char *name) {
  if (!lib.handle)
    return nullptr;
#ifdef _WIN32
  return reinterpret_cast<void *>(
      ::GetProcAddress(static_cast<HMODULE>(lib.handle), name));
#else
  return ::dlsym(lib.handle, name);
#endif
}

} // namespace mlir::rocm_loader

#endif // MLIR_EXECUTIONENGINE_ROCMDYNAMICLOADER_H
