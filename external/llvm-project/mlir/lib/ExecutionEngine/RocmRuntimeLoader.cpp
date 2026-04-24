//===- RocmRuntimeLoader.cpp - Lazy ROCm library loading utilities --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// glibc's `dlmopen` is gated on `_GNU_SOURCE`. Define it before any system
// header is transitively included so the declaration is visible regardless of
// how the build picks compile flags. The upstream LLVM build sets `_GNU_SOURCE`
// repo-wide via `cmake/config-ix.cmake`, but downstream consumers compiling
// this TU through their own build system may not, so we define it defensively
// here. This define is confined to the implementation file and never leaks
// through the public header.
#if !defined(_WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "mlir/ExecutionEngine/RocmRuntimeLoader.h"

#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <string>
#include <vector>

// We deliberately do NOT use `llvm::sys::DynamicLibrary` here even though it is
// the standard upstream wrapper for `dlopen`/`dlsym` and
// `LoadLibraryW`/`GetProcAddress`. The reason: on POSIX it always passes
// `RTLD_LAZY | RTLD_GLOBAL` to `dlopen` (see
// `lib/Support/Unix/DynamicLibrary.inc`), which is exactly the thing we need
// to avoid -- `RTLD_GLOBAL` lets the dynamic linker unify ROCm's `libLLVM.so`
// symbols with the host's embedded LLVM, which is the original `cl::opt`
// collision we are fixing. There is no public knob to swap in `RTLD_LOCAL` or
// `dlmopen(LM_ID_NEWLM, ...)`, and the namespace-isolation guarantee is the
// entire point of this loader. We therefore drop to the same OS APIs as
// `lib/Support/{Unix,Windows}/DynamicLibrary.inc` but with the flags we
// actually need.
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define DEBUG_TYPE "rocm-runtime-loader"

using namespace mlir;
using namespace mlir::rocm_loader;

namespace {

// Highest ROCm major version the loader will probe when iterating numeric
// SONAME suffixes (e.g. `libamdhip64.so.<N>`). Picked generously so the loader
// continues working on future ROCm releases without code changes; bumping it
// has zero functional cost (a missing SONAME returns from `dlopen` in
// O(microseconds) on every modern libc, paid once at startup). The lower bound
// is `1` -- ROCm has never shipped a `.so.0`. Adjust upward if AMD ever reaches
// a major version above this constant.
constexpr unsigned kMaxProbedRocmMajor = 99;

// Append `bare` (the unversioned alias, preferred when present), then
// `joiner(MAJOR)` for descending MAJOR, to `out`. The unversioned alias
// resolves through `LD_LIBRARY_PATH` / `RPATH` / `RUNPATH` /
// `/etc/ld.so.cache` on glibc (the user's expected policy, also what
// `find_package(hip)`, IREE and Triton do); the numeric fallback covers
// runtime-only installs where the symlink has been stripped.
//
// `joiner` produces the platform-decorated versioned name -- for HIP on POSIX
// it returns `libamdhip64.so.<N>`, on Windows `amdhip64_<N>.dll`, etc.
template <typename Joiner>
void appendCandidates(std::vector<std::string> &out, llvm::StringRef bare,
                      Joiner joiner) {
  out.emplace_back(bare.str());
  for (unsigned m = kMaxProbedRocmMajor; m >= 1; --m)
    out.emplace_back(joiner(m));
}

// Build the SONAME candidate list for `lib`. On Windows, `Library::Hsa`
// returns an empty list because ROCm on Windows ships no HSA runtime; callers
// must treat `loadRocmLibrary(Hsa)` as "HSA unavailable" there.
//
// Windows HIPRTC has a quirk: AMD decorates the DLL name with the ROCm major
// AND minor (`hiprtc<MM><mm>.dll`, e.g. `hiprtc0700.dll` for ROCm 7.0). For
// each candidate major we therefore probe `hiprtc<MM>00.dll`; AMD has only
// ever shipped the `.0` minor decoration in practice. Downstream consumers
// shipping a non-`.0` minor must put the DLL on `PATH` so the bare
// `hiprtc.dll` lookup picks it up.
std::vector<std::string> candidatesFor(Library lib) {
  std::vector<std::string> out;
  out.reserve(1 + kMaxProbedRocmMajor);
  switch (lib) {
  case Library::Hip:
#ifdef _WIN32
    appendCandidates(out, "amdhip64.dll", [](unsigned m) {
      return "amdhip64_" + std::to_string(m) + ".dll";
    });
#else
    appendCandidates(out, "libamdhip64.so", [](unsigned m) {
      return "libamdhip64.so." + std::to_string(m);
    });
#endif
    return out;
  case Library::Hiprtc:
#ifdef _WIN32
    appendCandidates(out, "hiprtc.dll", [](unsigned m) {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "hiprtc%02u00.dll", m);
      return std::string(buf);
    });
#else
    appendCandidates(out, "libhiprtc.so", [](unsigned m) {
      return "libhiprtc.so." + std::to_string(m);
    });
#endif
    return out;
  case Library::Hsa:
#ifdef _WIN32
    return out; // empty: no HSA on Windows.
#else
    // HSA's SONAME has been `.so.1` for the entire ROCm 4.x-7.x window, but
    // we still iterate to be future-safe in case AMD ever bumps it.
    appendCandidates(out, "libhsa-runtime64.so", [](unsigned m) {
      return "libhsa-runtime64.so." + std::to_string(m);
    });
    return out;
#endif
  }
  llvm_unreachable("unknown rocm_loader::Library enumerator");
}

// Open `path` so its symbols cannot interpose anything in the host process.
// On glibc this means `dlmopen(LM_ID_NEWLM, ...)` (a fresh link-map
// namespace); on other POSIX systems `dlopen(RTLD_LAZY | RTLD_LOCAL)`; on
// Windows `LoadLibraryW` (DLLs have private scopes per-DLL there).
//
// Returns null on failure -- never aborts.
void *openIsolated(const char *path) {
#ifdef _WIN32
  // Convert the UTF-8 SONAME to UTF-16 and call LoadLibraryW. This mirrors
  // `llvm/lib/Support/Windows/DynamicLibrary.inc`. Our SONAMEs are pure ASCII
  // today, but a downstream caller might extend the candidate list with a
  // localized path, so we use the wide form unconditionally.
  llvm::SmallVector<llvm::UTF16, 64> wide;
  if (!llvm::convertUTF8ToUTF16String(llvm::StringRef(path), wide)) {
    LLVM_DEBUG(llvm::dbgs()
               << DEBUG_TYPE ": bad UTF-8 in SONAME '" << path << "'\n");
    return nullptr;
  }
  HMODULE h = ::LoadLibraryW(reinterpret_cast<LPCWSTR>(wide.data()));
  if (!h) {
    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE ": LoadLibraryW(" << path
                            << ") failed (error " << ::GetLastError() << ")\n");
  }
  return reinterpret_cast<void *>(h);
#else
  // On glibc we open into a fresh link-map namespace so the loaded library's
  // symbols cannot interpose the host's. Other POSIX libcs (musl, ...) lack
  // `dlmopen`, so we settle for `RTLD_LOCAL`; isolation there is incomplete
  // and depends on the host having hidden its own LLVM exports at link time
  // (`-Wl,--exclude-libs,ALL`, visibility=hidden, ...).
#if defined(__GLIBC__)
  void *h = ::dlmopen(LM_ID_NEWLM, path, RTLD_LAZY);
#else
  void *h = ::dlopen(path, RTLD_LAZY | RTLD_LOCAL);
#endif
  if (!h) {
    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE ": load failed for '" << path
                            << "': " << ::dlerror() << "\n");
  }
  return h;
#endif
}

// Open `path` into the SAME link-map namespace as `existingHandle`, so the
// new library shares state (most importantly KFD's per-process HSA session)
// with the previously-loaded HIP runtime. On glibc we look up
// `existingHandle`'s namespace via `dlinfo()` and pass it back to `dlmopen()`.
// On Windows / non-glibc POSIX, where namespaces don't exist, this is just a
// regular load. (`__GLIBC__` is never defined on Windows, so the single guard
// suffices.)
//
// Precondition: `existingHandle` is non-null. Caller routes the null case to
// `openIsolated`.
void *openInRelatedNamespace(const char *path, void *existingHandle) {
#if defined(__GLIBC__)
  Lmid_t ns = LM_ID_NEWLM;
  if (::dlinfo(existingHandle, RTLD_DI_LMID, &ns) != 0)
    ns = LM_ID_NEWLM;
  void *h = ::dlmopen(ns, path, RTLD_LAZY);
  if (!h) {
    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE ": dlmopen(ns=" << ns << ", " << path
                            << ") failed: " << ::dlerror() << "\n");
  }
  return h;
#else
  (void)existingHandle;
  return openIsolated(path);
#endif
}

// Look up the HIP handle owned by `RocmSystemDetect`, if it has been loaded
// into this process. Returns null when the symbol is absent (typical for
// binaries that do not link `MLIRRocmExecutionEngineUtils`) or when
// `RocmSystemDetect` itself failed to load HIP.
//
// The lookup goes through `RTLD_DEFAULT` so we do not need a link-time
// dependency on `MLIRRocmExecutionEngineUtils`. On Windows, DLLs have private
// scopes so there is no equivalent coordination; ROCm-on-Windows also ships
// no HSA, so KFD's session limit does not apply.
void *getSharedHipHandle() {
#ifdef _WIN32
  return nullptr;
#else
  using GetHandleFn = void *(*)();
  if (auto *fn = reinterpret_cast<GetHandleFn>(
          ::dlsym(RTLD_DEFAULT, "mlirRocmSystemDetectGetHipHandle")))
    return fn();
  return nullptr;
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
    out.handle = relatedHandle
                     ? openInRelatedNamespace(cand.c_str(), relatedHandle)
                     : openIsolated(cand.c_str());
    if (out.handle)
      return out;
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
