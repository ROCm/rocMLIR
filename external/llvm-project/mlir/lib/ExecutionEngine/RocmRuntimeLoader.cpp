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

#include <cstdio>
#include <string>
#include <vector>

// We deliberately do NOT use `llvm::sys::DynamicLibrary` here even
// though it is the standard upstream wrapper for `dlopen`/`dlsym` and
// `LoadLibraryW`/`GetProcAddress`. The reason: on POSIX it always
// passes `RTLD_LAZY | RTLD_GLOBAL` to `dlopen`, which is exactly the
// thing we need to avoid -- `RTLD_GLOBAL` lets the dynamic linker
// unify ROCm's `libLLVM.so` symbols with the host's embedded LLVM,
// which is the original `cl::opt` collision we are fixing. There is
// no public knob in `sys::DynamicLibrary` to swap in `RTLD_LOCAL` or
// `dlmopen(LM_ID_NEWLM, ...)`, and the namespace-isolation guarantee
// is the entire point of this loader. The implementation below uses
// the same OS APIs as `lib/Support/{Unix,Windows}/DynamicLibrary.inc`
// (`dlopen`, `dlsym`, `LoadLibraryW`, `GetProcAddress`) but with the
// flags we actually need.
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define DEBUG_TYPE "rocm-runtime-loader"

using namespace mlir;

namespace {

/// Highest ROCm major version the loader will probe when iterating
/// numeric SONAME suffixes (e.g. `libamdhip64.so.<N>`). Picked
/// generously so the loader continues working on future ROCm releases
/// without code changes; bumping it has zero functional cost (a missing
/// SONAME returns from `dlopen` in O(microseconds) on every modern
/// libc, paid once at startup). The lower bound is `1` -- ROCm has
/// never shipped a `.so.0`. Adjust upward if AMD ever reaches a major
/// version above this constant.
constexpr unsigned kMaxProbedRocmMajor = 99;

/// Build the candidate SONAME list for `lib`, in preference order:
///
///   1. Bare SONAME (e.g. `libamdhip64.so` / `amdhip64.dll`). This
///      matches what `find_package(hip)`, IREE's HIP HAL, and Triton's
///      HIP loader do, and resolves through `LD_LIBRARY_PATH` /
///      `RPATH` / `RUNPATH` / `/etc/ld.so.cache` on glibc -- the
///      user's expected policy. Standard ROCm installs always ship
///      the unversioned symlink (e.g. `libamdhip64.so` ->
///      `libamdhip64.so.<MAJOR>` -> `libamdhip64.so.<MAJOR>.<MINOR>...`).
///
///   2. `<base>.so.<MAJOR>` for MAJOR descending from
///      `kMaxProbedRocmMajor` to `1`. Covers runtime-only installs
///      where the unversioned symlink has been stripped and the
///      versioned SONAME is the only file present.
///
/// Windows HIPRTC has a quirk: AMD decorates the DLL name with the
/// ROCm major+minor (`hiprtc<MM><mm>.dll`, e.g. `hiprtc0507.dll` on
/// ROCm 5.7, `hiprtc0700.dll` on ROCm 7.0). For each candidate major
/// we therefore probe `hiprtc<MM>00.dll` -- AMD has only ever shipped
/// the `.0` minor decoration in practice; downstream consumers
/// shipping a non-`.0` minor must put the DLL on `PATH` so the bare
/// `hiprtc.dll` lookup picks it up.
///
/// On Windows, `Library::Hsa` returns an empty list because ROCm on
/// Windows ships no HSA runtime; callers must treat
/// `loadRocmLibrary(Hsa)` as "HSA unavailable" there.
std::vector<std::string> candidatesFor(rocm_loader::Library lib) {
  std::vector<std::string> out;
  out.reserve(1 + kMaxProbedRocmMajor);
  switch (lib) {
  case rocm_loader::Library::Hip:
#ifdef _WIN32
    out.emplace_back("amdhip64.dll");
    for (unsigned m = kMaxProbedRocmMajor; m >= 1; --m)
      out.emplace_back("amdhip64_" + std::to_string(m) + ".dll");
#else
    out.emplace_back("libamdhip64.so");
    for (unsigned m = kMaxProbedRocmMajor; m >= 1; --m)
      out.emplace_back("libamdhip64.so." + std::to_string(m));
#endif
    return out;
  case rocm_loader::Library::Hiprtc:
#ifdef _WIN32
    out.emplace_back("hiprtc.dll");
    for (unsigned m = kMaxProbedRocmMajor; m >= 1; --m) {
      // `hiprtc<MM>00.dll` -- e.g. `hiprtc0700.dll` for ROCm 7.0.
      // The 4-digit zero-padded form is what AMD's installer ships.
      char buf[16];
      std::snprintf(buf, sizeof(buf), "hiprtc%02u00.dll", m);
      out.emplace_back(buf);
    }
#else
    out.emplace_back("libhiprtc.so");
    for (unsigned m = kMaxProbedRocmMajor; m >= 1; --m)
      out.emplace_back("libhiprtc.so." + std::to_string(m));
#endif
    return out;
  case rocm_loader::Library::Hsa:
#ifdef _WIN32
    return out; // empty: no HSA on Windows
#else
    // HSA's SONAME has stayed on `.so.1` for the entire ROCm 4.x-7.x
    // window, but we still iterate to be future-safe in case AMD ever
    // bumps it.
    out.emplace_back("libhsa-runtime64.so");
    for (unsigned m = kMaxProbedRocmMajor; m >= 1; --m)
      out.emplace_back("libhsa-runtime64.so." + std::to_string(m));
    return out;
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
  for (const std::string &cand : candidatesFor(lib)) {
    void *h = relatedHandle
                  ? openInSameNamespaceImpl(cand.c_str(), relatedHandle)
                  : openIsolatedImpl(cand.c_str());
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
