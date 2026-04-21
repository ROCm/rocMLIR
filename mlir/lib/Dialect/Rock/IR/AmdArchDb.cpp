//===- AmdArchDb.cpp - Dtabase of AMD GPU features ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// `_GNU_SOURCE` is required for glibc's `dlmopen` (used by the AMD GPU arch
// runtime loader below). It must be defined before any system header is
// included.
#if !defined(_WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/ExecutionEngine/RocmArchRuntime.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "rock-amd-arch-db"

using namespace mlir;
using namespace mlir::rock;

static constexpr AmdArchInfo
    gcnInfo(GemmFeatures::none, /*waveSize=*/64,
            /*maxWavesPerEU*/ 10, /*totalSGPRPerEU*/ 512,
            /*totalVGPRPerEU*/ 256, /*totalSharedMemPerCU*/ 65536,
            /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/80,
            /*hasFp8ConversionInstrs=*/false,
            /*hasOcpFp8ConversionInstrs=*/false, /*hasScaledGemm=*/false,
            /*maxNumXCC=*/1, /*hasLdsTransposeLoad=*/false),
    cdna50Info(GemmFeatures::dot, /*waveSize=*/64, /*maxWavesPerEU*/ 8,
               /*totalSGPRPerEU*/ 512, /*totalVGPRPerEU*/ 256,
               /*totalSharedMemPerCU*/ 65536, /*maxSharedMemPerWG*/ 65536,
               /*numEUPerCU=*/4, /*minNumCU=*/10,
               /*hasFp8ConversionInstrs=*/false,
               /*hasOcpFp8ConversionInstrs=*/false, /*hasScaledGemm=*/false,
               /*maxNumXCC=*/1, /*hasLdsTransposeLoad=*/false),
    cdnaInfo(GemmFeatures::mfma | GemmFeatures::dot | GemmFeatures::atomic_add |
                 GemmFeatures::atomic_add_f16,
             /*waveSize=*/64, /*maxWavesPerEU*/ 10, /*totalSGPRPerEU*/ 800,
             /*totalVGPRPerEU*/ 256, /*totalSharedMemPerCU*/ 65536,
             /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/120,
             /*hasFp8ConversionInstrs=*/false,
             /*hasOcpFp8ConversionInstrs=*/false, /*hasScaledGemm=*/false,
             /*maxNumXCC=*/1, /*hasLdsTransposeLoad=*/false),
    cdna2Info(GemmFeatures::mfma | GemmFeatures::dot |
                  GemmFeatures::atomic_add | GemmFeatures::atomic_add_f16,
              /*waveSize=*/64, /*maxWavesPerEU*/ 8, /*totalSGPRPerEU*/ 800,
              /*totalVGPRPerEU*/ 512, /*totalSharedMemPerCU*/ 65536,
              /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/104,
              /*hasFp8ConversionInstrs=*/false,
              /*hasOcpFp8ConversionInstrs=*/false, /*hasScaledGemm=*/false,
              /*maxNumXCC=*/1, /*hasLdsTransposeLoad=*/false),
    cdna3Info(GemmFeatures::mfma | GemmFeatures::dot |
                  GemmFeatures::atomic_add | GemmFeatures::atomic_add_f16 |
                  GemmFeatures::direct_to_lds_32b,
              /*waveSize=*/64, /*maxWavesPerEU*/ 8, /*totalSGPRPerEU*/ 800,
              /*totalVGPRPerEU*/ 512, /*totalSharedMemPerCU*/ 65536,
              /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/20,
              /*hasFp8ConversionInstrs=*/true,
              /*hasOcpFp8ConversionInstrs=*/false, /*hasScaledGemm=*/false,
              /*maxNumXCC=*/8, /*hasLdsTransposeLoad=*/false),
    cdna40Info(GemmFeatures::mfma | GemmFeatures::dot |
                   GemmFeatures::atomic_add | GemmFeatures::atomic_add_f16 |
                   GemmFeatures::atomic_add_bf16 |
                   GemmFeatures::direct_to_lds_32b |
                   GemmFeatures::direct_to_lds_128b,
               /*waveSize=*/64, /*maxWavesPerEU*/ 8, /*totalSGPRPerEU*/ 800,
               /*totalVGPRPerEU*/ 512, /*totalSharedMemPerCU*/ 163840,
               /*maxSharedMemPerWG*/ 163840, /*numEUPerCU=*/4, /*minNumCU=*/256,
               /*hasFp8ConversionInstrs=*/false,
               /*hasOcpFp8ConversionInstrs=*/true, /*hasScaledGemm=*/true,
               /*maxNumXCC=*/8, /*hasLdsTransposeLoad=*/true),
    // amdgpu target builds all RDNA in WGP Mode
    rdnaNoDotInfo(GemmFeatures::atomic_fmax_f32, /*waveSize=*/32,
                  /*maxWavesPerEU*/ 16, /*totalSGPRPerEU*/ 512,
                  /*totalVGPRPerEU*/ 1024, /*totalSharedMemPerCU*/ 131072,
                  /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4,
                  /*minNumCU=*/30,
                  /*hasFp8ConversionInstrs=*/false,
                  /*hasOcpFp8ConversionInstrs=*/false, /*hasScaledGemm=*/false,
                  /*maxNumXCC=*/1, /*hasLdsTransposeLoad=*/false),
    rdnaInfo(GemmFeatures::dot | GemmFeatures::atomic_fmax_f32,
             /*waveSize=*/32, /*maxWavesPerEU*/ 16, /*totalSGPRPerEU*/ 512,
             /*totalVGPRPerEU*/ 1024, /*totalSharedMemPerCU*/ 131072,
             /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/2,
             /*hasFp8ConversionInstrs=*/false,
             /*hasOcpFp8ConversionInstrs=*/false, /*hasScaledGemm=*/false,
             /*maxNumXCC=*/1, /*hasLdsTransposeLoad=*/false),
    rdna3Info(GemmFeatures::dot | GemmFeatures::atomic_add |
                  GemmFeatures::atomic_fmax_f32 | GemmFeatures::wmma,
              /*waveSize=*/32, /*maxWavesPerEU*/ 16, /*totalSGPRPerEU*/ 800,
              /*totalVGPRPerEU*/ 1536, /*totalSharedMemPerCU*/ 131072,
              /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/2,
              /*hasFp8ConversionInstrs=*/false,
              /*hasOcpFp8ConversionInstrs=*/false, /*hasScaledGemm=*/false,
              /*maxNumXCC=*/1, /*hasLdsTransposeLoad=*/false),
    rdna4Info(GemmFeatures::dot | GemmFeatures::atomic_add |
                  GemmFeatures::atomic_fmax_f32 | GemmFeatures::wmma |
                  GemmFeatures::atomic_add_f16 | GemmFeatures::atomic_add_bf16,
              /*waveSize=*/32, /*maxWavesPerEU*/ 16, /*totalSGPRPerEU*/ 800,
              /*totalVGPRPerEU*/ 1536, /*totalSharedMemPerCU*/ 131072,
              /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/12,
              /*hasFp8ConversionInstrs=*/false,
              /*hasOcpFp8ConversionInstrs=*/true, /*hasScaledGemm=*/false,
              /*maxNumXCC=*/1, /*hasLdsTransposeLoad=*/false),
    // TODO: update with right information
    gfx1250Info(GemmFeatures::dot | GemmFeatures::atomic_add |
                    GemmFeatures::atomic_fmax_f32 | GemmFeatures::wmma |
                    GemmFeatures::atomic_add_f16 |
                    GemmFeatures::atomic_add_bf16,
                /*waveSize=*/32, /*maxWavesPerEU*/ 16, /*totalSGPRPerEU*/ 800,
                /*totalVGPRPerEU*/ 1536, /*totalSharedMemPerCU*/ 131072,
                /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/12,
                /*hasFp8ConversionInstrs=*/false,
                /*hasOcpFp8ConversionInstrs=*/true, /*hasScaledGemm=*/false,
                /*maxNumXCC=*/1, /*hasLdsTransposeLoad=*/false);

static std::tuple<StringRef, unsigned> parseArchString(StringRef arch) {
  std::tuple<StringRef, unsigned> ret("", 0);

  StringRef firstPart, remainingParts;
  std::tie(firstPart, remainingParts) = arch.split(':');
  if (firstPart == "native") {
    std::get<0>(ret) = firstPart;
    if (unsigned long long deviceId;
        !llvm::getAsUnsignedInteger(remainingParts, 0, deviceId)) {
      std::get<1>(ret) = deviceId;
    }
  } else {
    auto chipPos = firstPart.find("gfx");
    if (chipPos != StringRef::npos) {
      firstPart = firstPart.substr(chipPos);
    } else {
      std::tie(firstPart, remainingParts) = remainingParts.split(':');
    }
    std::get<0>(ret) = firstPart;
  }

  return ret;
}

namespace {

/// Platform-specific shared-library file name of `mlir_rocm_arch_runtime`.
/// Kept as a single source of truth so the loader's search paths and the
/// runtime library's installed name cannot drift apart.
constexpr StringRef kRocmArchRuntimeLibName =
#if defined(_WIN32)
    "mlir_rocm_arch_runtime.dll";
#elif defined(__APPLE__)
    "libmlir_rocm_arch_runtime.dylib";
#else
    "libmlir_rocm_arch_runtime.so";
#endif

/// Function-pointer table resolved from the runtime library. A null
/// `getProperties` means the runtime is not loaded; callers must check before
/// use. The OS handle is intentionally leaked: the runtime (and everything it
/// transitively loaded, including ROCm's libLLVM.so) must stay mapped while
/// any pointer it returned is still in flight.
struct RocmArchRuntimeFns {
  uint32_t (*deviceCount)(void) = nullptr;
  int32_t (*getProperties)(uint32_t, MlirRocmArchProperties *) = nullptr;
};

/// Open `path` in a way that keeps its transitive `libLLVM.so` from
/// colliding with rocMLIR's embedded LLVM. On glibc that means a fresh
/// link-map namespace via `dlmopen(LM_ID_NEWLM, ...)`; on Windows every DLL
/// already has its own scope, so plain `LoadLibraryW` suffices. The
/// non-glibc POSIX fallback uses `RTLD_LOCAL` only, which provides weaker
/// isolation -- such environments (e.g. musl) are not officially supported.
void *osOpen(const char *path) {
#ifdef _WIN32
  SmallVector<UTF16, 256> wide;
  if (!convertUTF8ToUTF16String(StringRef(path), wide)) {
    LLVM_DEBUG(llvm::dbgs() << "rock-amd-arch-db: bad UTF-8 in runtime path '"
                            << path << "'\n");
    return nullptr;
  }
  HMODULE h = ::LoadLibraryW(reinterpret_cast<LPCWSTR>(wide.data()));
  if (!h)
    LLVM_DEBUG(llvm::dbgs() << "rock-amd-arch-db: LoadLibraryW(" << path
                            << ") failed (error " << ::GetLastError() << ")\n");
  return h;
#elif defined(__GLIBC__)
  void *h = ::dlmopen(LM_ID_NEWLM, path, RTLD_LAZY);
  if (!h)
    LLVM_DEBUG(llvm::dbgs() << "rock-amd-arch-db: dlmopen(" << path
                            << ") failed: " << ::dlerror() << "\n");
  return h;
#else
  void *h = ::dlopen(path, RTLD_LAZY | RTLD_LOCAL);
  if (!h)
    LLVM_DEBUG(llvm::dbgs() << "rock-amd-arch-db: dlopen(" << path
                            << ") failed: " << ::dlerror() << "\n");
  return h;
#endif
}

void *osSym(void *h, const char *name) {
#ifdef _WIN32
  return reinterpret_cast<void *>(
      ::GetProcAddress(static_cast<HMODULE>(h), name));
#else
  return ::dlsym(h, name);
#endif
}

/// Locate and load the `mlir_rocm_arch_runtime` shared library. Search order:
///   1. `<exe-dir>` (Windows install / build layout: DLLs next to consumers).
///   2. `<exe-dir>/../lib/` (POSIX install / build layout).
///   3. `ROCMLIR_BUILD_RUNTIME_DIR` (compile-time path; lets nested test
///      binaries find the runtime without `LD_LIBRARY_PATH`).
///   4. The platform loader's default search path
///      (`LD_LIBRARY_PATH`/`RPATH`/`PATH`).
void *loadRocmArchRuntime() {
  SmallVector<SmallString<256>, 4> searchDirs;
  std::string mainExe = llvm::sys::fs::getMainExecutable(nullptr, nullptr);
  if (!mainExe.empty()) {
    StringRef binDir = llvm::sys::path::parent_path(mainExe);
    searchDirs.emplace_back(binDir);
    SmallString<256> &siblingLib = searchDirs.emplace_back(binDir);
    llvm::sys::path::append(siblingLib, "..", "lib");
  }
#ifdef ROCMLIR_BUILD_RUNTIME_DIR
  searchDirs.emplace_back(StringRef(ROCMLIR_BUILD_RUNTIME_DIR));
#endif

  for (const auto &dir : searchDirs) {
    SmallString<256> candidate(dir);
    llvm::sys::path::append(candidate, kRocmArchRuntimeLibName);
    if (!llvm::sys::fs::exists(candidate))
      continue;
    if (void *h = osOpen(candidate.c_str())) {
      LLVM_DEBUG(llvm::dbgs() << "rock-amd-arch-db: loaded runtime from "
                              << candidate << "\n");
      return h;
    }
  }
  // Fall back to the platform's default loader search path.
  SmallString<256> leafName(kRocmArchRuntimeLibName);
  return osOpen(leafName.c_str());
}

/// Resolve every entry point we need from `h`. Returns an empty table on
/// any failure (missing symbol or ABI version mismatch).
RocmArchRuntimeFns resolveRuntimeFns(void *h) {
  RocmArchRuntimeFns fns;
  if (!h)
    return fns;

  auto *abi = reinterpret_cast<int32_t (*)(void)>(
      osSym(h, "mlirRocmArchRuntimeAbiVersion"));
  auto *deviceCount = reinterpret_cast<uint32_t (*)(void)>(
      osSym(h, "mlirRocmArchRuntimeDeviceCount"));
  auto *getProperties =
      reinterpret_cast<int32_t (*)(uint32_t, MlirRocmArchProperties *)>(
          osSym(h, "mlirRocmArchRuntimeGetProperties"));
  if (!abi || !deviceCount || !getProperties) {
    LLVM_DEBUG(llvm::dbgs() << "rock-amd-arch-db: missing entry-point symbol "
                               "in AMD GPU arch runtime\n");
    return fns;
  }
  if (int32_t version = abi(); version != MLIR_ROCM_ARCH_RUNTIME_ABI_VERSION) {
    LLVM_DEBUG(llvm::dbgs()
               << "rock-amd-arch-db: ABI mismatch (got " << version
               << ", expected " << MLIR_ROCM_ARCH_RUNTIME_ABI_VERSION << ")\n");
    return fns;
  }

  fns.deviceCount = deviceCount;
  fns.getProperties = getProperties;
  return fns;
}

/// Process-wide singleton accessor.
const RocmArchRuntimeFns &getRocmArchRuntime() {
  static RocmArchRuntimeFns fns = resolveRuntimeFns(loadRocmArchRuntime());
  return fns;
}

template <typename LHS, typename RHS>
std::enable_if_t<std::is_assignable_v<LHS &, RHS &&>, void>
checkAndSetInfo(StringRef name, LHS &lhs, RHS &&rhs) {
  if (lhs != static_cast<LHS>(rhs)) {
    LLVM_DEBUG(llvm::dbgs() << "NOTE: Value discrepancy for " << name << ": "
                            << lhs << " (old) != " << rhs
                            << " (new). Proceeding with " << rhs << ".\n");
    lhs = std::forward<RHS>(rhs);
  }
}

AmdArchInfo fetchNativeArchInfo(const MlirRocmArchProperties &props) {
  auto ret = lookupArchInfo(props.gcnArchName); // get baseline

  checkAndSetInfo("(HIP) minNumCU", ret.minNumCU, props.multiProcessorCount);
  checkAndSetInfo("(HIP) waveSize", ret.waveSize, props.warpSize);
  checkAndSetInfo("(HIP) totalSharedMemPerCU", ret.totalSharedMemPerCU,
                  props.sharedMemPerCU);
  checkAndSetInfo("(HIP) maxSharedMemPerWG", ret.maxSharedMemPerWG,
                  props.sharedMemPerBlock);

  if (props.hsaValid && props.simdsPerCU != 0) {
    checkAndSetInfo("(HSA) numEUPerCU", ret.numEUPerCU, props.simdsPerCU);
    checkAndSetInfo("(HSA) maxWavesPerEU", ret.maxWavesPerEU,
                    props.maxWavesPerCU / props.simdsPerCU);
    checkAndSetInfo("(HSA) maxNumXCC", ret.maxNumXCC, props.numXCC);
  }

  // NOTE: the following AmdArchInfo fields are not yet sourced from hardware
  // and therefore keep their static-preset values from `lookupArchInfo` above:
  //   - totalSGPRPerEU
  //   - totalVGPRPerEU
  //   - defaultFeatures
  //   - hasOcpFp8ConversionInstrs
  // Adding HIP/HSA queries for these is tracked as part of the original
  // native-arch work (PR #1790).
  return ret;
}

AmdArchInfo nativeArchInfo(unsigned deviceId) {
  static std::mutex m;
  static std::unordered_map<std::string, AmdArchInfo> cache;

  LLVM_DEBUG(llvm::dbgs() << "Retrieving native arch info for device "
                          << deviceId << "...\n");

  const RocmArchRuntimeFns &fns = getRocmArchRuntime();
  if (!fns.getProperties)
    llvm::report_fatal_error(
        llvm::Twine("Failed to load AMD GPU arch runtime (") +
        kRocmArchRuntimeLibName +
        "): native architecture detection is unavailable. Ensure the runtime "
        "is installed alongside the executable or on the dynamic-loader "
        "search path.");

  MlirRocmArchProperties props{};
  int32_t status = fns.getProperties(deviceId, &props);
  if (status == MLIR_ROCM_ARCH_HIP_ERROR)
    llvm::report_fatal_error(
        llvm::Twine("AMD GPU arch runtime: HIP query failed for device ") +
        llvm::Twine(deviceId));

  LLVM_DEBUG(llvm::dbgs() << "gcnArchName: " << props.gcnArchName << "\n");

  std::lock_guard<std::mutex> lock(m);
  std::string archKey(props.gcnArchName);
  auto it = cache.find(archKey);
  if (it == cache.end()) {
    LLVM_DEBUG(llvm::dbgs() << "Cache miss! Fetching native arch info...\n");
    it = cache.emplace(std::move(archKey), fetchNativeArchInfo(props)).first;
  }
  return it->second;
}

} // anonymous namespace

AmdArchInfo mlir::rock::lookupArchInfo(StringRef arch) {
  // Keep this implementation in sync with
  // mlir/test/lit.site.cfg.py.in:set_arch_features()
  auto [chip, deviceId] = parseArchString(arch);
  if (chip == "native")
    return nativeArchInfo(deviceId);
  StringRef minor = chip.take_back(2);
  StringRef major = chip.slice(0, chip.size() - 2);
  if (major == "gfx9") {
    return llvm::StringSwitch<AmdArchInfo>(minor)
        .Case("08", cdnaInfo)
        .Case("0a", cdna2Info)
        .Case("42", cdna3Info)
        .Case("50", cdna40Info)
        // gfx906 has the dot product instructions, uniquely
        .Case("06", cdna50Info)
        .Default(gcnInfo);
  }
  if (major == "gfx10") {
    return llvm::StringSwitch<AmdArchInfo>(minor)
        .Cases({"11", "13"}, rdnaNoDotInfo)
        .Cases({"10", "12"}, rdnaInfo)
        // All gfx103x are the same for us
        .StartsWith("3", rdnaInfo)
        .Default(rdnaNoDotInfo);
  }
  if (major == "gfx11") {
    // We know these chips have common features per backend
    return rdna3Info;
  }
  if (major == "gfx12") {
    return llvm::StringSwitch<AmdArchInfo>(minor)
        .Case("50", gfx1250Info)
        .Default(rdna4Info);
  }
  auto msg = "Unsupported architecture: " + arch.str();
  llvm_unreachable(msg.c_str());
}

unsigned mlir::rock::nativeDeviceCount() {
  const RocmArchRuntimeFns &fns = getRocmArchRuntime();
  if (!fns.deviceCount)
    return 0;
  return fns.deviceCount();
}

std::string mlir::rock::nativeArchName(unsigned deviceId) {
  const RocmArchRuntimeFns &fns = getRocmArchRuntime();
  if (!fns.getProperties)
    return std::string();
  MlirRocmArchProperties props{};
  if (fns.getProperties(deviceId, &props) != MLIR_ROCM_ARCH_OK)
    return std::string();
  return std::string(props.gcnArchName);
}

GemmFeatures mlir::rock::AmdArchInfo::getDefaultFeatures(Type dataType) {
  GemmFeatures theseFeatures = defaultFeatures;
  bool isWmma = bitEnumContainsAll(theseFeatures, GemmFeatures::wmma);

  // Get the underlying element type of the dataType. We may have to do this
  // recursively if the initial dataType is a nested vector.
  Type elementType = getElementTypeOrSelf(dataType);
  while (isa<ShapedType>(elementType)) {
    elementType = getElementTypeOrSelf(elementType);
  }

  if (isWmma) {
    if (!(isa<Float16Type, BFloat16Type>(elementType) ||
          elementType.isInteger(8) ||
          (hasFp8ConversionInstrs &&
           isa<Float8E5M2FNUZType, Float8E4M3FNUZType>(elementType)) ||
          (hasOcpFp8ConversionInstrs &&
           isa<Float8E5M2Type, Float8E4M3FNType>(elementType)))) {
      theseFeatures = bitEnumClear(theseFeatures, GemmFeatures::wmma);
    }
  }
  bool isMfma = bitEnumContainsAll(theseFeatures, GemmFeatures::mfma);

  if (isMfma && !hasFp8ConversionInstrs) {
    if (isa<Float8E4M3FNUZType>(elementType) ||
        isa<Float8E5M2FNUZType>(elementType))
      theseFeatures = bitEnumClear(theseFeatures, GemmFeatures::mfma);
  }
  if (isMfma && !hasOcpFp8ConversionInstrs) {
    if (isa<Float8E4M3FNType>(elementType) || isa<Float8E5M2Type>(elementType))
      theseFeatures = bitEnumClear(theseFeatures, GemmFeatures::mfma);
  }
  if (isMfma && !hasScaledGemm) {
    if (isa<Float4E2M1FNType>(elementType) ||
        isa<Float8E8M0FNUType>(elementType)) {
      theseFeatures = bitEnumClear(theseFeatures, GemmFeatures::mfma);
      LLVM_DEBUG(
          llvm::dbgs()
          << "Disabling mfma accel for Float4E2M1FN or Float8E8M0FNU type: "
          << elementType << "\n");
    }
  }
  return theseFeatures;
}

GemmFeatures mlir::rock::AmdArchInfo::getDefaultFeatures(ArrayRef<Type> types) {
  if (types.empty())
    return GemmFeatures::none;

  std::optional<GemmFeatures> features = std::nullopt;
  for (Type ty : types) {
    auto newFeatures = getDefaultFeatures(ty);
    if (!features.has_value()) {
      features = newFeatures;
      continue;
    }
    // Intersect features from all types
    features = features.value() & newFeatures;
  }

  // Disable accel for unsupported mixed types
  if (types.size() == 2) {
    Type elemTypeA = getElementTypeOrSelf(types[0]);
    while (isa<ShapedType>(elemTypeA)) {
      elemTypeA = getElementTypeOrSelf(elemTypeA);
    }
    Type elemTypeB = getElementTypeOrSelf(types[1]);
    while (isa<ShapedType>(elemTypeB)) {
      elemTypeB = getElementTypeOrSelf(elemTypeB);
    }
    if (elemTypeA != elemTypeB) {
      bool validMixedTypesWmma = false;
      bool validMixedTypesMfma = false;

      // Keep in sync with convertTypesToId in WmmaInsnGroup.cpp
      if (isa<Float8E4M3FNType>(elemTypeA) && isa<Float8E4M3FNType>(elemTypeB))
        validMixedTypesWmma = true;
      if (isa<Float8E4M3FNType>(elemTypeA) && isa<Float8E5M2Type>(elemTypeB))
        validMixedTypesWmma = true;
      if (isa<Float8E5M2Type>(elemTypeA) && isa<Float8E4M3FNType>(elemTypeB))
        validMixedTypesWmma = true;
      if (isa<Float8E5M2Type>(elemTypeA) && isa<Float8E5M2Type>(elemTypeB))
        validMixedTypesWmma = true;

      if (!validMixedTypesWmma) {
        LLVM_DEBUG(llvm::dbgs() << "Disabling wmma accel for mixed types: "
                                << elemTypeA << " and " << elemTypeB << "\n");
        features = bitEnumClear(features.value(), GemmFeatures::wmma);
      }

      // Keep in sync with convertTypesToId in MfmaInsnGroup.cpp
      if (isa<Float8E4M3FNUZType>(elemTypeA) &&
          isa<Float8E5M2FNUZType>(elemTypeB)) {
        validMixedTypesMfma = true;
      }
      if (isa<Float8E5M2FNUZType>(elemTypeA) &&
          isa<Float8E4M3FNUZType>(elemTypeB)) {
        validMixedTypesMfma = true;
      }
      if (isa<Float8E4M3FNType>(elemTypeA) && isa<Float8E5M2Type>(elemTypeB)) {
        validMixedTypesMfma = true;
      }
      if (isa<Float8E5M2Type>(elemTypeA) && isa<Float8E4M3FNType>(elemTypeB)) {
        validMixedTypesMfma = true;
      }

      if (!validMixedTypesMfma) {
        LLVM_DEBUG(llvm::dbgs() << "Disabling mfma accel for mixed types: "
                                << elemTypeA << " and " << elemTypeB << "\n");
        features = bitEnumClear(features.value(), GemmFeatures::mfma);
      }
    }
  }

  return features.value();
}

GemmFeatures
mlir::rock::AmdArchInfo::getFeaturesFromAttr(ArrayRef<Type> types,
                                             GemmFeaturesAttr featuresAttr) {
  LLVM_DEBUG(llvm::dbgs() << "getFeaturesFromAttr: types=" << types
                          << ", featuresAttr=" << featuresAttr << "\n");
  // The attribute has precedence over the types. If it is present, use it.
  // Otherwise, use the default features.
  if (featuresAttr)
    return featuresAttr.getValue();
  return getDefaultFeatures(types);
}

bool mlir::rock::AmdArchInfo::isAccel(Type dataTypeA, Type dataTypeB,
                                      GemmFeaturesAttr featuresAttr) {
  GemmFeatures features =
      getFeaturesFromAttr({dataTypeA, dataTypeB}, featuresAttr);
  LLVM_DEBUG(llvm::dbgs() << "isAccel: features=" << features << "\n");
  return bitEnumContainsAny(features, GemmFeatures::wmma | GemmFeatures::mfma);
}

bool mlir::rock::AmdArchInfo::isMfma(Type dataTypeA, Type dataTypeB,
                                     GemmFeaturesAttr featuresAttr) {
  GemmFeatures features =
      getFeaturesFromAttr({dataTypeA, dataTypeB}, featuresAttr);
  LLVM_DEBUG(llvm::dbgs() << "isMfma: features=" << features << "\n");
  return bitEnumContainsAll(features, GemmFeatures::mfma);
}

bool mlir::rock::AmdArchInfo::isAccel(RockGemmWrapperInterface op) {
  return isAccel(op.getAType(), op.getBType(), op.getGemmFeaturesAttr());
}

bool mlir::rock::AmdArchInfo::isAccel(RockGemmGemmWrapperInterface op) {
  return isAccel(op.getAType(), op.getBType(), op.getGemmFeaturesAttr());
}

bool mlir::rock::AmdArchInfo::isMfma(RockGemmWrapperInterface op) {
  return isMfma(op.getAType(), op.getBType(), op.getGemmFeaturesAttr());
}

bool mlir::rock::AmdArchInfo::isMfma(RockGemmGemmWrapperInterface op) {
  return isMfma(op.getAType(), op.getBType(), op.getGemmFeaturesAttr());
}

bool mlir::rock::AmdArchInfo::isWmma(Type dataTypeA, Type dataTypeB,
                                     GemmFeaturesAttr featuresAttr) {
  GemmFeatures features =
      getFeaturesFromAttr({dataTypeA, dataTypeB}, featuresAttr);
  LLVM_DEBUG(llvm::dbgs() << "isWmma: features=" << features << "\n");
  return bitEnumContainsAll(features, GemmFeatures::wmma);
}

bool mlir::rock::AmdArchInfo::isWmma(RockGemmWrapperInterface op) {
  return isWmma(op.getAType(), op.getBType(), op.getGemmFeaturesAttr());
}

bool mlir::rock::AmdArchInfo::isWmma(RockGemmGemmWrapperInterface op) {
  return isWmma(op.getAType(), op.getBType(), op.getGemmFeaturesAttr());
}

bool mlir::rock::AmdArchInfo::hasAtomicAdd(Type dataType) const {
  // Get the underlying element type. We may have to do this recursively if the
  // initial dataType is a nested vector.
  Type elementType = getElementTypeOrSelf(dataType);
  while (isa<ShapedType>(elementType)) {
    elementType = getElementTypeOrSelf(elementType);
  }

  // Check based on the element type
  if (elementType.isF32()) {
    return bitEnumContainsAll(defaultFeatures, GemmFeatures::atomic_add);
  } else if (elementType.isF16()) {
    return bitEnumContainsAll(defaultFeatures, GemmFeatures::atomic_add_f16);
  } else if (elementType.isBF16()) {
    return bitEnumContainsAll(defaultFeatures, GemmFeatures::atomic_add_bf16);
  }
  llvm_unreachable("Unsupported element type for atomic add");
  return false;
}

bool mlir::rock::AmdArchInfo::hasAtomicFmaxF32() const {
  return bitEnumContainsAll(defaultFeatures, GemmFeatures::atomic_fmax_f32);
}

bool mlir::rock::isDirectToLDSSupported(GemmFeatures features) {
  return bitEnumContainsAll(features, GemmFeatures::direct_to_lds_128b) ||
         bitEnumContainsAll(features, GemmFeatures::direct_to_lds_32b);
}

bool mlir::rock::isAsyncDirectToLDSSupported(StringRef arch) {
  return arch.contains("gfx1250");
}

int64_t
mlir::rock::AmdArchInfo::getMaxLDSVectorLength(int64_t elementBitWidth) {
  int64_t maxGlobalToLDSVectorLen = std::numeric_limits<int64_t>::max();
  assert(elementBitWidth > 0 && "elementBitWidth must be greater than 0");
  if (bitEnumContainsAll(defaultFeatures, GemmFeatures::direct_to_lds_128b)) {
    maxGlobalToLDSVectorLen = 128 / elementBitWidth;
  } else if (bitEnumContainsAll(defaultFeatures,
                                GemmFeatures::direct_to_lds_32b)) {
    maxGlobalToLDSVectorLen = 32 / elementBitWidth;
  }

  return maxGlobalToLDSVectorLen;
}

bool mlir::rock::isGlobalPrefetchSupported(StringRef arch) {
  return arch.contains("gfx1250");
}

bool mlir::rock::AmdArchInfo::isWrWAtomicKernel(GemmFeaturesAttr featuresAttr,
                                                Type dataType,
                                                bool requiredPadding) {
  // We check only for GemmFeatures::atomic_add (f32) even though we accept
  // dataType to be either f32 or f16. This is because f16 WrW atomic uses f32
  // workspace, computing atomic adds in f32 and later a second kernel converts
  // from f32 to f16.
  return isAccel(dataType, dataType, featuresAttr) &&
         bitEnumContainsAll(defaultFeatures, GemmFeatures::atomic_add) &&
         (dataType.isF32() || dataType.isF16()) && !requiredPadding;
}
