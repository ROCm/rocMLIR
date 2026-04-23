//===- RocmRuntimeLoader.cpp - Lazy ROCm library loading utilities --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// glibc's `dlmopen` is gated on `_GNU_SOURCE`. Define it before any
// system header is transitively included so the declaration is visible
// regardless of how the build picks compile flags. The upstream LLVM
// build sets `_GNU_SOURCE` repo-wide via `cmake/modules/AddLLVM.cmake`,
// but downstream consumers compiling this TU through their own build
// system may not, so we define it defensively here. This define is
// confined to the implementation file and never leaks through the
// public header.
#if !defined(_WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "mlir/ExecutionEngine/RocmRuntimeLoader.h"

#include "llvm/Support/Compiler.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define DEBUG_TYPE "rocm-runtime-loader"

using namespace mlir;

namespace {

/// Platform-specific SONAME candidates per library, in preference
/// order. ROCm 7 ships `.so.7`; older clusters may still have `.so.6`
/// or an unversioned alias. Windows uses decorated names
/// (`amdhip64_7.dll`, etc.) with the bare name as a final fallback.
///
/// We deliberately ship the empty-list case via a sentinel `nullptr`
/// rather than a zero-element array, because zero-element C arrays are
/// ill-formed in standard C++ (MSVC `error C2466`). This keeps the
/// translation unit MSVC-clean.
constexpr const char *kHipCandidates[] = {
#ifdef _WIN32
    "amdhip64_7.dll",
    "amdhip64_6.dll",
    "amdhip64.dll",
#else
    "libamdhip64.so.7",
    "libamdhip64.so.6",
    "libamdhip64.so",
#endif
};

constexpr const char *kHiprtcCandidates[] = {
#ifdef _WIN32
    "hiprtc.dll", // ROCm renames the decorated DLL each major release
                  // (e.g. hiprtc0507.dll on ROCm 5.7); the bare alias
                  // is the safest fallback.
    "hiprtc0700.dll",
    "hiprtc0600.dll",
#else
    "libhiprtc.so.7",
    "libhiprtc.so.6",
    "libhiprtc.so",
#endif
};

#ifndef _WIN32
constexpr const char *kHsaCandidates[] = {
    "libhsa-runtime64.so.1",
    "libhsa-runtime64.so",
};
#endif

/// Returns the candidate list for `lib`. On Windows, `Library::Hsa`
/// returns `(nullptr, 0)` because ROCm-on-Windows does not ship HSA;
/// callers will then degrade to "HSA unavailable".
const char *const *candidatesFor(rocm_loader::Library lib, size_t &count) {
  switch (lib) {
  case rocm_loader::Library::Hip:
    count = std::size(kHipCandidates);
    return kHipCandidates;
  case rocm_loader::Library::Hiprtc:
    count = std::size(kHiprtcCandidates);
    return kHiprtcCandidates;
  case rocm_loader::Library::Hsa:
#ifdef _WIN32
    count = 0;
    return nullptr;
#else
    count = std::size(kHsaCandidates);
    return kHsaCandidates;
#endif
  }
  llvm_unreachable("unknown rocm_loader::Library enumerator");
}

#ifdef _WIN32

void *windowsLoadLibrary(const char *path) {
  // Convert the UTF-8 SONAME to UTF-16 and call LoadLibraryW. This
  // mirrors `llvm/lib/Support/Windows/DynamicLibrary.inc`. While our
  // SONAMEs are pure ASCII today, a downstream caller might extend the
  // candidate list with a localized path, so we use the wide form
  // unconditionally.
  llvm::SmallVector<llvm::UTF16, 64> wide;
  if (!llvm::convertUTF8ToUTF16String(llvm::StringRef(path), wide)) {
    LLVM_DEBUG(llvm::dbgs() << "rocm-runtime-loader: bad UTF-8 in SONAME '"
                            << path << "'\n");
    return nullptr;
  }
  HMODULE h = ::LoadLibraryW(reinterpret_cast<LPCWSTR>(wide.data()));
  if (!h) {
    LLVM_DEBUG(llvm::dbgs() << "rocm-runtime-loader: LoadLibraryW(" << path
                            << ") failed (error " << ::GetLastError() << ")\n");
  }
  return reinterpret_cast<void *>(h);
}

#else // !_WIN32

void *posixOpenIsolated(const char *path) {
#if defined(__GLIBC__)
  void *h = ::dlmopen(LM_ID_NEWLM, path, RTLD_LAZY);
#else
  void *h = ::dlopen(path, RTLD_LAZY | RTLD_LOCAL);
#endif
  if (!h) {
    LLVM_DEBUG(llvm::dbgs() << "rocm-runtime-loader: load failed for '" << path
                            << "': " << ::dlerror() << "\n");
  }
  return h;
}

void *posixOpenInSameNamespace(const char *path, void *existingHandle) {
#if defined(__GLIBC__)
  Lmid_t ns = LM_ID_NEWLM;
  if (existingHandle && ::dlinfo(existingHandle, RTLD_DI_LMID, &ns) != 0)
    ns = LM_ID_NEWLM;
  void *h = ::dlmopen(ns, path, RTLD_LAZY);
  if (!h) {
    LLVM_DEBUG(llvm::dbgs() << "rocm-runtime-loader: dlmopen(ns=" << ns << ", "
                            << path << ") failed: " << ::dlerror() << "\n");
  }
  return h;
#else
  (void)existingHandle;
  return posixOpenIsolated(path);
#endif
}

/// Emit a one-time advisory when running on a POSIX platform that lacks
/// `dlmopen`. Without namespace isolation, ROCm's libLLVM may still
/// interpose the host process's LLVM symbols. The fallback is
/// best-effort and depends on the host having hidden its LLVM exports
/// (e.g. via `-Wl,--exclude-libs,ALL` or visibility="hidden").
void warnIfWeakIsolation() {
#if !defined(_WIN32) && !defined(__GLIBC__)
  static bool warned = []() {
    LLVM_DEBUG(llvm::dbgs()
               << "rocm-runtime-loader: this libc lacks `dlmopen`; ROCm "
                  "runtime libraries cannot be placed in a private "
                  "link-map namespace. Process-wide static-init "
                  "collisions between ROCm's libLLVM and the host's "
                  "embedded LLVM are possible if the host does not also "
                  "hide its LLVM symbols at link time.\n");
    return true;
  }();
  (void)warned;
#endif
}

#endif // _WIN32

void *openIsolatedImpl(const char *path) {
#ifdef _WIN32
  return windowsLoadLibrary(path);
#else
  warnIfWeakIsolation();
  return posixOpenIsolated(path);
#endif
}

void *openInSameNamespaceImpl(const char *path, void *existingHandle) {
#ifdef _WIN32
  // Windows DLLs have private scopes per-DLL; there is nothing to
  // share, so loading-in-the-same-namespace is just a normal load.
  (void)existingHandle;
  return windowsLoadLibrary(path);
#else
  if (!existingHandle)
    return openIsolatedImpl(path);
  return posixOpenInSameNamespace(path, existingHandle);
#endif
}

/// Look up the HIP handle owned by `RocmSystemDetect`, if it has been
/// loaded into this process. Returns `nullptr` when the symbol is
/// absent (typical for binaries that do not link
/// `MLIRRocmExecutionEngineUtils`) or when `RocmSystemDetect` itself
/// failed to load HIP.
///
/// The lookup goes through `RTLD_DEFAULT` so we do not need a
/// link-time dependency on `MLIRRocmExecutionEngineUtils`. Windows DLLs
/// have private scopes so no equivalent coordination is needed there;
/// also, ROCm-on-Windows does not ship HSA, so KFD's session limit
/// does not apply.
void *getSharedHipHandle() {
#ifdef _WIN32
  return nullptr;
#else
  void *getter = ::dlsym(RTLD_DEFAULT, "mlirRocmSystemDetectGetHipHandle");
  if (!getter)
    return nullptr;
  using GetHandleFn = void *(*)();
  return reinterpret_cast<GetHandleFn>(getter)();
#endif
}

} // namespace

namespace mlir::rocm_loader {

LoadedLibrary loadRocmLibrary(Library lib, void *relatedHandle,
                              CoordinationPolicy policy) {
  LoadedLibrary out;
  if (lib == Library::Hip && policy == CoordinationPolicy::Auto) {
    if (void *shared = getSharedHipHandle()) {
      out.handle = shared;
      return out;
    }
  }
  size_t count = 0;
  const char *const *candidates = candidatesFor(lib, count);
  for (size_t i = 0; i < count; ++i) {
    void *h = relatedHandle
                  ? openInSameNamespaceImpl(candidates[i], relatedHandle)
                  : openIsolatedImpl(candidates[i]);
    if (h) {
      out.handle = h;
      return out;
    }
  }
  return out;
}

void *resolveRocmSymbol(const LoadedLibrary &lib, const char *name) {
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
