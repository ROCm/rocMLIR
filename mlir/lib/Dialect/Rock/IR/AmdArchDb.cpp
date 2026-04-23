//===- AmdArchDb.cpp - Database of AMD GPU features -----------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `MLIRRockOps` does not link `libamdhip64` / `libhsa-runtime64` at build
// time. Doing so would transitively pull in ROCm's `libamd_comgr` and its
// embedded `libLLVM.so.*`, which collides with rocMLIR's own LLVM at load
// time (duplicate `cl::opt` registration, corrupted global command-line
// parser state, ...). The HIP and HSA symbols used by
// `rock.arch = "native[:N]"` are resolved on demand via the shared
// `mlir::rocm_loader` helpers; see
// `external/llvm-project/mlir/include/mlir/ExecutionEngine/RocmDynamicLoader.h`
// for the full rationale, including how we share a single HSA session
// across all consumers of libamdhip64 in the process.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/ExecutionEngine/RocmRuntimeLoader.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include <mutex>
#include <optional>

// HIP / HSA headers are pulled in for their POD types (`hipDeviceProp_t`,
// `hsa_agent_t`, ...); the shared libraries themselves are loaded at
// runtime by `mlir::rocm_loader`, not linked at build time.
// `__HIP_PLATFORM_AMD__` picks the AMD variant of `hipDeviceProp_t`.
#define __HIP_PLATFORM_AMD__ 1
#include "hip/hip_runtime_api.h"

#ifndef _WIN32
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#endif

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

//===----------------------------------------------------------------------===//
// HIP delay-load.
//===----------------------------------------------------------------------===//

/// Function-pointer table resolved from `libamdhip64`. A null `handle`
/// means HIP was not loaded; callers must check before use. The handle
/// is intentionally leaked at process teardown: anything HIP pulled in
/// (most notably `libamd_comgr` and ROCm's `libLLVM.so`) must stay
/// mapped while any pointer HIP returned is still in flight.
struct HipRuntime {
  rocm_loader::LoadedLibrary lib;
  hipError_t (*getDeviceCount)(int *) = nullptr;
  hipError_t (*getDeviceProperties)(hipDeviceProp_t *, int) = nullptr;
};

HipRuntime loadHipRuntime() {
  HipRuntime rt;
  rt.lib = rocm_loader::loadRocmLibrary(rocm_loader::Library::Hip);
  if (!rt.lib.handle) {
    LLVM_DEBUG(llvm::dbgs()
               << "rock-amd-arch-db: libamdhip64 not found on the loader "
                  "search path; disabling native-arch detection\n");
    return rt;
  }

  rt.getDeviceCount = reinterpret_cast<hipError_t (*)(int *)>(
      rocm_loader::resolveRocmSymbol(rt.lib, "hipGetDeviceCount"));
  // HIP 6+ renamed the struct-stable form to `...R0600`. Prefer that if
  // available; fall back to the legacy symbol for older ROCm installs.
  rt.getDeviceProperties =
      reinterpret_cast<hipError_t (*)(hipDeviceProp_t *, int)>(
          rocm_loader::resolveRocmSymbol(rt.lib,
                                         "hipGetDevicePropertiesR0600"));
  if (!rt.getDeviceProperties)
    rt.getDeviceProperties =
        reinterpret_cast<hipError_t (*)(hipDeviceProp_t *, int)>(
            rocm_loader::resolveRocmSymbol(rt.lib, "hipGetDeviceProperties"));

  if (!rt.getDeviceCount || !rt.getDeviceProperties) {
    LLVM_DEBUG(llvm::dbgs()
               << "rock-amd-arch-db: HIP loaded but required symbols are "
                  "missing; disabling native-arch detection\n");
    rt.lib.handle = nullptr;
  }
  return rt;
}

const HipRuntime &getHipRuntime() {
  static HipRuntime rt = loadHipRuntime();
  return rt;
}

//===----------------------------------------------------------------------===//
// HSA delay-load.
//===----------------------------------------------------------------------===//

#ifndef _WIN32
/// HSA is used to obtain the per-agent properties that HIP does not expose
/// directly (SIMDs per CU, max waves per CU, XCC count). An HSA failure is
/// non-fatal: we fall back to the static `AmdArchDb` presets for those
/// fields. HSA is loaded into HIP's link-map namespace so both share a
/// single KFD session (HIP initialises HSA internally).
struct HsaRuntime {
  rocm_loader::LoadedLibrary lib;
  hsa_status_t (*init)(void) = nullptr;
  hsa_status_t (*iterateAgents)(hsa_status_t (*)(hsa_agent_t, void *),
                                void *) = nullptr;
  hsa_status_t (*agentGetInfo)(hsa_agent_t, hsa_agent_info_t, void *) = nullptr;
};

HsaRuntime loadHsaRuntime(void *hipHandle) {
  HsaRuntime rt;
  rt.lib = rocm_loader::loadRocmLibrary(rocm_loader::Library::Hsa, hipHandle);
  if (!rt.lib.handle) {
    LLVM_DEBUG(llvm::dbgs() << "rock-amd-arch-db: libhsa-runtime64 not "
                               "found; HSA-derived fields will use presets\n");
    return rt;
  }

  rt.init = reinterpret_cast<hsa_status_t (*)(void)>(
      rocm_loader::resolveRocmSymbol(rt.lib, "hsa_init"));
  rt.iterateAgents = reinterpret_cast<hsa_status_t (*)(
      hsa_status_t (*)(hsa_agent_t, void *), void *)>(
      rocm_loader::resolveRocmSymbol(rt.lib, "hsa_iterate_agents"));
  rt.agentGetInfo =
      reinterpret_cast<hsa_status_t (*)(hsa_agent_t, hsa_agent_info_t, void *)>(
          rocm_loader::resolveRocmSymbol(rt.lib, "hsa_agent_get_info"));

  if (!rt.init || !rt.iterateAgents || !rt.agentGetInfo) {
    LLVM_DEBUG(llvm::dbgs()
               << "rock-amd-arch-db: HSA loaded but required symbols are "
                  "missing; HSA-derived fields will fall back to presets\n");
    rt.lib.handle = nullptr;
    return rt;
  }

  // HIP indirectly calls `hsa_init()`; do it explicitly so we can use HSA
  // without HIP (e.g. future tests that want HSA-only queries). The runtime
  // is reference-counted internally, so double-initialisation is harmless.
  if (rt.init() != HSA_STATUS_SUCCESS) {
    LLVM_DEBUG(llvm::dbgs() << "rock-amd-arch-db: hsa_init() failed\n");
    rt.lib.handle = nullptr;
  }
  return rt;
}

const HsaRuntime &getHsaRuntime() {
  static HsaRuntime rt = loadHsaRuntime(getHipRuntime().lib.handle);
  return rt;
}

/// Agent-iteration callback and per-device state for HSA queries.
struct AgentQuery {
  const HsaRuntime *hsa;
  uint32_t targetDeviceId;
  int numCpus;
  uint32_t simdsPerCU;
  uint32_t maxWavesPerCU;
  uint32_t numXCC;
  bool found;
};

// Adapted from `rocminfo.cc`; see
// https://github.com/ROCm/rocm-systems/blob/develop/projects/rocminfo/rocminfo.cc
hsa_status_t acquireAgentInfo(hsa_agent_t agent, void *data) {
  auto *q = static_cast<AgentQuery *>(data);
  const HsaRuntime &hsa = *q->hsa;

  hsa_device_type_t deviceType;
  if (hsa_status_t err =
          hsa.agentGetInfo(agent, HSA_AGENT_INFO_DEVICE, &deviceType);
      err != HSA_STATUS_SUCCESS)
    return err;

  if (deviceType != HSA_DEVICE_TYPE_GPU) {
    ++q->numCpus;
    return HSA_STATUS_SUCCESS;
  }

  uint32_t internalNodeId = 0;
  if (hsa_status_t err = hsa.agentGetInfo(
          agent,
          static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID),
          &internalNodeId);
      err != HSA_STATUS_SUCCESS)
    return err;

  if (internalNodeId < static_cast<uint32_t>(q->numCpus))
    return HSA_STATUS_SUCCESS;

  uint32_t gpuId = internalNodeId - static_cast<uint32_t>(q->numCpus);
  if (gpuId != q->targetDeviceId)
    return HSA_STATUS_SUCCESS;

  if (hsa_status_t err = hsa.agentGetInfo(
          agent,
          static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU),
          &q->simdsPerCU);
      err != HSA_STATUS_SUCCESS)
    return err;
  if (hsa_status_t err = hsa.agentGetInfo(
          agent,
          static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU),
          &q->maxWavesPerCU);
      err != HSA_STATUS_SUCCESS)
    return err;
  if (hsa_status_t err = hsa.agentGetInfo(
          agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_XCC),
          &q->numXCC);
      err != HSA_STATUS_SUCCESS)
    return err;

  q->found = true;
  return HSA_STATUS_SUCCESS;
}
#endif // _WIN32

//===----------------------------------------------------------------------===//
// Native arch inference.
//===----------------------------------------------------------------------===//

/// Apply the WGP-as-CU correction for Navi-class GPUs. HIP/HSA report
/// wavefront-32 GPUs in WGP mode, which halves several per-CU metrics
/// compared to our rocMLIR convention; scale them back up. The
/// `sharedMemPerCU` out-parameter is only updated when the caller has
/// matching HSA data; HIP-only paths leave it alone.
void applyNaviCorrection(uint32_t warpSize, uint32_t &simdsPerCU,
                         uint32_t &maxWavesPerCU, int64_t &sharedMemPerCU) {
  if (warpSize != 32)
    return;
  simdsPerCU *= 2;
  maxWavesPerCU *= 2;
  sharedMemPerCU *= 2;
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

/// Query HIP (+ HSA where available) for the running device and adjust the
/// static preset that matches `gcnArchName`. Returns `std::nullopt` if HIP
/// cannot be loaded or the query fails.
std::optional<AmdArchInfo> tryQueryNativeArchInfo(unsigned deviceId,
                                                  std::string &gcnArchName) {
  const HipRuntime &hip = getHipRuntime();
  if (!hip.lib.handle)
    return std::nullopt;

  hipDeviceProp_t prop{};
  if (hip.getDeviceProperties(&prop, static_cast<int>(deviceId)) != hipSuccess)
    return std::nullopt;
  gcnArchName = prop.gcnArchName;
  LLVM_DEBUG(llvm::dbgs() << "gcnArchName: " << gcnArchName << "\n");

  // Query HSA up front so we can apply the Navi WGP-as-CU correction to the
  // shared-memory figure before it reaches `checkAndSetInfo`. This matches
  // the shim's semantics exactly (@31416e7): on Navi-with-HSA the per-CU
  // shared memory reported by HIP is doubled to account for WGP mode. If
  // HSA is unavailable we skip the correction rather than silently emitting
  // a value that differs from the static preset.
  uint32_t simdsPerCU = 0;
  uint32_t maxWavesPerCU = 0;
  uint32_t numXCC = 0;
  bool hsaValid = false;
  int64_t sharedMemPerCU =
      static_cast<int64_t>(prop.maxSharedMemoryPerMultiProcessor);

#ifndef _WIN32
  const HsaRuntime &hsa = getHsaRuntime();
  if (hsa.lib.handle) {
    AgentQuery q{};
    q.hsa = &hsa;
    q.targetDeviceId = deviceId;
    if (hsa.iterateAgents(acquireAgentInfo, &q) == HSA_STATUS_SUCCESS &&
        q.found && q.simdsPerCU != 0) {
      simdsPerCU = q.simdsPerCU;
      maxWavesPerCU = q.maxWavesPerCU;
      numXCC = q.numXCC;
      hsaValid = true;
      applyNaviCorrection(static_cast<uint32_t>(prop.warpSize), simdsPerCU,
                          maxWavesPerCU, sharedMemPerCU);
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "rock-amd-arch-db: HSA agent query for device " << deviceId
                 << " failed; keeping preset values\n");
    }
  }
#endif

  AmdArchInfo ret = lookupArchInfo(gcnArchName);
  checkAndSetInfo("(HIP) minNumCU", ret.minNumCU,
                  static_cast<int64_t>(prop.multiProcessorCount));
  checkAndSetInfo("(HIP) waveSize", ret.waveSize,
                  static_cast<int64_t>(prop.warpSize));
  checkAndSetInfo("(HIP) totalSharedMemPerCU", ret.totalSharedMemPerCU,
                  sharedMemPerCU);
  checkAndSetInfo("(HIP) maxSharedMemPerWG", ret.maxSharedMemPerWG,
                  static_cast<int64_t>(prop.sharedMemPerBlock));

  if (hsaValid) {
    checkAndSetInfo("(HSA) numEUPerCU", ret.numEUPerCU,
                    static_cast<int64_t>(simdsPerCU));
    checkAndSetInfo("(HSA) maxWavesPerEU", ret.maxWavesPerEU,
                    static_cast<int64_t>(maxWavesPerCU / simdsPerCU));
    checkAndSetInfo("(HSA) maxNumXCC", ret.maxNumXCC,
                    static_cast<int64_t>(numXCC));
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
  static llvm::StringMap<AmdArchInfo> cache;

  LLVM_DEBUG(llvm::dbgs() << "Retrieving native arch info for device "
                          << deviceId << "...\n");

  std::string gcnArchName;
  std::optional<AmdArchInfo> queried =
      tryQueryNativeArchInfo(deviceId, gcnArchName);
  if (!queried)
    llvm::report_fatal_error(
        "Failed to query AMD GPU arch runtime for native architecture "
        "detection. Ensure a ROCm installation with libamdhip64 is visible "
        "via LD_LIBRARY_PATH / RPATH, and that the requested device id is "
        "valid.");

  std::lock_guard<std::mutex> lock(m);
  auto [it, inserted] = cache.try_emplace(gcnArchName, *queried);
  if (inserted) {
    LLVM_DEBUG(llvm::dbgs() << "Cache miss! Caching native arch info for "
                            << gcnArchName << "\n");
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
  const HipRuntime &hip = getHipRuntime();
  if (!hip.lib.handle)
    return 0;
  int count = 0;
  if (hip.getDeviceCount(&count) != hipSuccess)
    return 0;
  if (count < 0)
    return 0;
  return static_cast<unsigned>(count);
}

std::string mlir::rock::nativeArchName(unsigned deviceId) {
  const HipRuntime &hip = getHipRuntime();
  if (!hip.lib.handle)
    return std::string();
  hipDeviceProp_t prop{};
  if (hip.getDeviceProperties(&prop, static_cast<int>(deviceId)) != hipSuccess)
    return std::string();
  return std::string(prop.gcnArchName);
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
