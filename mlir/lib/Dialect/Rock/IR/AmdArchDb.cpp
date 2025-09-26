//===- AmdArchDb.cpp - Dtabase of AMD GPU features ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"

#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetSelect.h"

#include "hip/hip_runtime_api.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#define DEBUG_TYPE "rock-amd-arch-db"

using namespace mlir;
using namespace mlir::rock;

static constexpr AmdArchInfo
    gcnInfo(GemmFeatures::none, /*waveSize=*/64,
            /*maxWavesPerEU*/ 10, /*totalSGPRPerEU*/ 512,
            /*totalVGPRPerEU*/ 256, /*totalSharedMemPerCU*/ 65536,
            /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/80,
            /*hasFp8ConversionInstrs=*/false,
            /*hasOcpFp8ConversionInstrs=*/false, /*maxNumXCC=*/1),
    cdna50Info(GemmFeatures::dot, /*waveSize=*/64, /*maxWavesPerEU*/ 8,
               /*totalSGPRPerEU*/ 512, /*totalVGPRPerEU*/ 256,
               /*totalSharedMemPerCU*/ 65536, /*maxSharedMemPerWG*/ 65536,
               /*numEUPerCU=*/4, /*minNumCU=*/10,
               /*hasFp8ConversionInstrs=*/false,
               /*hasOcpFp8ConversionInstrs=*/false, /*maxNumXCC=*/1),
    cdnaInfo(GemmFeatures::mfma | GemmFeatures::dot | GemmFeatures::atomic_add |
                 GemmFeatures::atomic_add_f16,
             /*waveSize=*/64, /*maxWavesPerEU*/ 10, /*totalSGPRPerEU*/ 800,
             /*totalVGPRPerEU*/ 256, /*totalSharedMemPerCU*/ 65536,
             /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/120,
             /*hasFp8ConversionInstrs=*/false,
             /*hasOcpFp8ConversionInstrs=*/false, /*maxNumXCC=*/1),
    cdna2Info(GemmFeatures::mfma | GemmFeatures::dot |
                  GemmFeatures::atomic_add | GemmFeatures::atomic_add_f16,
              /*waveSize=*/64, /*maxWavesPerEU*/ 8, /*totalSGPRPerEU*/ 800,
              /*totalVGPRPerEU*/ 512, /*totalSharedMemPerCU*/ 65536,
              /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/104,
              /*hasFp8ConversionInstrs=*/false,
              /*hasOcpFp8ConversionInstrs=*/false, /*maxNumXCC=*/1),
    cdna3Info(GemmFeatures::mfma | GemmFeatures::dot |
                  GemmFeatures::atomic_add | GemmFeatures::atomic_add_f16 |
                  GemmFeatures::direct_to_lds_32b,
              /*waveSize=*/64, /*maxWavesPerEU*/ 8, /*totalSGPRPerEU*/ 800,
              /*totalVGPRPerEU*/ 512, /*totalSharedMemPerCU*/ 65536,
              /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/80,
              /*hasFp8ConversionInstrs=*/true,
              /*hasOcpFp8ConversionInstrs=*/false, /*maxNumXCC=*/8),
    cdna40Info(GemmFeatures::mfma | GemmFeatures::dot |
                   GemmFeatures::atomic_add | GemmFeatures::atomic_add_f16 |
                   GemmFeatures::atomic_add_bf16 |
                   GemmFeatures::direct_to_lds_32b |
                   GemmFeatures::direct_to_lds_128b,
               /*waveSize=*/64, /*maxWavesPerEU*/ 8, /*totalSGPRPerEU*/ 800,
               /*totalVGPRPerEU*/ 512, /*totalSharedMemPerCU*/ 163840,
               /*maxSharedMemPerWG*/ 163840, /*numEUPerCU=*/4, /*minNumCU=*/256,
               /*hasFp8ConversionInstrs=*/false,
               /*hasOcpFp8ConversionInstrs=*/true, /*maxNumXCC=*/8),
    // amdgpu target builds all RDNA in WGP Mode
    rdnaNoDotInfo(GemmFeatures::atomic_fmax_f32, /*waveSize=*/32,
                  /*maxWavesPerEU*/ 16, /*totalSGPRPerEU*/ 512,
                  /*totalVGPRPerEU*/ 1024, /*totalSharedMemPerCU*/ 131072,
                  /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4,
                  /*minNumCU=*/36,
                  /*hasFp8ConversionInstrs=*/false,
                  /*hasOcpFp8ConversionInstrs=*/false, /*maxNumXCC=*/1),
    rdnaInfo(GemmFeatures::dot | GemmFeatures::atomic_fmax_f32,
             /*waveSize=*/32, /*maxWavesPerEU*/ 16, /*totalSGPRPerEU*/ 512,
             /*totalVGPRPerEU*/ 1024, /*totalSharedMemPerCU*/ 131072,
             /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/36,
             /*hasFp8ConversionInstrs=*/false,
             /*hasOcpFp8ConversionInstrs=*/false, /*maxNumXCC=*/1),
    rdna3Info(GemmFeatures::dot | GemmFeatures::atomic_add |
                  GemmFeatures::atomic_fmax_f32 | GemmFeatures::wmma,
              /*waveSize=*/32, /*maxWavesPerEU*/ 16, /*totalSGPRPerEU*/ 800,
              /*totalVGPRPerEU*/ 1536, /*totalSharedMemPerCU*/ 131072,
              /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/12,
              /*hasFp8ConversionInstrs=*/false,
              /*hasOcpFp8ConversionInstrs=*/false, /*maxNumXCC=*/1),
    rdna4Info(GemmFeatures::dot | GemmFeatures::atomic_add |
                  GemmFeatures::atomic_fmax_f32 | GemmFeatures::wmma |
                  GemmFeatures::atomic_add_f16 | GemmFeatures::atomic_add_bf16,
              /*waveSize=*/32, /*maxWavesPerEU*/ 16, /*totalSGPRPerEU*/ 800,
              /*totalVGPRPerEU*/ 1536, /*totalSharedMemPerCU*/ 131072,
              /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/12,
              /*hasFp8ConversionInstrs=*/false,
              /*hasOcpFp8ConversionInstrs=*/true, /*maxNumXCC=*/1);

namespace {

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

std::tuple<StringRef, unsigned> parseArchString(StringRef arch) {
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

#define RET_IF_HSA_ERR(err)                                                    \
  {                                                                            \
    if ((err) != HSA_STATUS_SUCCESS) {                                         \
      return err;                                                              \
    }                                                                          \
  }

struct AgentInfo {
  // Input fields:
  //   The ID of the GPU device that we are looking for.
  unsigned deviceId;
  //   Used in acquireAgentInfo, to compute GPU internal IDs.
  int numCpus;
  // Output fields:
  uint32_t simdsPerCU;
  uint32_t maxWavesPerCU;
};

AmdArchInfo fetchNativeArchInfo(const hipDeviceProp_t &prop,
                                AgentInfo &agent_info) {
  auto ret = lookupArchInfo(prop.gcnArchName); // get baseline

  checkAndSetInfo("(HIP) minNumCU", ret.minNumCU, prop.multiProcessorCount);
  checkAndSetInfo("(HIP) waveSize", ret.waveSize, prop.warpSize);
  checkAndSetInfo("(HIP) totalSharedMemPerCU", ret.totalSharedMemPerCU,
                  prop.maxSharedMemoryPerMultiProcessor);
  checkAndSetInfo("(HIP) maxSharedMemPerWG", ret.maxSharedMemPerWG,
                  prop.sharedMemPerBlock);

  checkAndSetInfo("(HSA) numEUPerCU", ret.numEUPerCU, agent_info.simdsPerCU);
  checkAndSetInfo("(HSA) maxWavesPerEU", ret.maxWavesPerEU,
                  agent_info.maxWavesPerCU / agent_info.simdsPerCU);

  // TODO: Add missing fields:
  // - totalSGPRPerEU
  // - totalVGPRPerEU
  // - defaultFeatures
  // - hasOcpFp8ConversionInstrs
  return ret;
}

// hsa_iterate_agents expects a callback function (acquireAgentInfo in this
// case) with one void* argument which contains arbitrary data to be used by the
// called function. Each time the callback is invoked, it is called with a
// different HSA agent and the pointer (i.e., the void* argument is shared
// across all calls). That is also why we count the number of CPUs, since we
// need to match the HIP deviceId with the HSA agent index.
//
// See hsa_iterate_agents documentation in
// https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/api-reference/api.html
// for more information.
static hsa_status_t acquireAgentInfo(hsa_agent_t agent, void *data) {
  // Use HSA to get data not exposed by HIP.
  // Based on:
  // https://github.com/ROCm/rocm-systems/blob/develop/projects/rocminfo/rocminfo.cc
  hsa_status_t err;
  AgentInfo *agent_i = reinterpret_cast<AgentInfo *>(data);

  hsa_device_type_t device_type;
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  RET_IF_HSA_ERR(err);

  if (HSA_DEVICE_TYPE_GPU == device_type) {
    // This a GPU, check if its the GPU that we are looking for.
    uint32_t internal_node_id;
    err = hsa_agent_get_info(
        agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DRIVER_NODE_ID,
        &internal_node_id);
    RET_IF_HSA_ERR(err);

    unsigned gpuDeviceId = internal_node_id - agent_i->numCpus;

    if (gpuDeviceId == agent_i->deviceId) {
      // This is the GPU that we want to check.
      err = hsa_agent_get_info(
          agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU,
          &agent_i->simdsPerCU);
      RET_IF_HSA_ERR(err);

      err = hsa_agent_get_info(
          agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU,
          &agent_i->maxWavesPerCU);
      RET_IF_HSA_ERR(err);
    }
  } else {
    agent_i->numCpus++;
  }

  return HSA_STATUS_SUCCESS;
}

void fixNaviProperties(AgentInfo *agent_i, hipDeviceProp_t *prop) {
  // Fix per CU metrics in Navi GPUs due to WGPs.
  // I wonder why we have to implement this logic instead of relying
  // on HIP to do this.
  //
  // Navi AMD docs define a CU as "One half of a WGP. Contains 2 SIMD32’s that
  // share one path to memory" In this context we treat a WGP as CU, so we need
  // to double simdsPerCU, totalSharedMemPerCU and
  // maxSharedMemoryPerMultiProcessor. This is consistent with the behavior of
  // amdgpu target in LLVM. They say: "Per CU" really means "per whatever
  // functional block the waves of a workgroup must share" This is also
  // mentioned on HIP multiProcessorCount field: "When the GPU works in Compute
  // Unit (CU) mode, this value equals the number of CUs; when in Workgroup
  // Processor (WGP) mode, this value equels half of CUs, because a single WGP
  // contains two CUs"
  //
  // References:
  // -
  // https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.2/user_guide/hip_rtc.html#cu-mode-vs-wgp-mode
  // -
  // https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf
  // -
  // https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp

  // TODO: Can we check WGP mode in a better way instead of checking warp size?
  if (prop->warpSize == 32) {
    agent_i->simdsPerCU *= 2;
    agent_i->maxWavesPerCU *= 2;
    prop->maxSharedMemoryPerMultiProcessor *= 2;
  }
}

AmdArchInfo nativeArchInfo(unsigned deviceId = 0) {
  static std::mutex m;
  static std::unordered_map<std::string, AmdArchInfo> cache;

  LLVM_DEBUG(llvm::dbgs() << "Retrieving native arch info for device "
                          << deviceId << "...\n");

  hipDeviceProp_t prop;
  if (auto err = hipGetDeviceProperties(&prop, deviceId); err != hipSuccess) {
    auto reason = "hipGetDeviceProperties failed with error: " +
                  std::string(hipGetErrorString(err));
    llvm::report_fatal_error(reason.c_str());
  }

  LLVM_DEBUG(llvm::dbgs() << "gcnArchName: " << prop.gcnArchName << "\n");

  AgentInfo agent_info;
  agent_info.numCpus = 0;
  agent_info.deviceId = deviceId;
  hsa_status_t err = hsa_iterate_agents(acquireAgentInfo, &agent_info);
  if (err != HSA_STATUS_SUCCESS) {
    char err_val[12];
    char *err_str = NULL;
    if (hsa_status_string(err, (const char **)&err_str) != HSA_STATUS_SUCCESS) {
      snprintf(&(err_val[0]), sizeof(err_val), "%#x", (uint32_t)err);
      err_str = &(err_val[0]);
    }
    llvm::report_fatal_error(err_str);
  }

  fixNaviProperties(&agent_info, &prop);

  std::lock_guard<std::mutex> lock(m);

  auto it = cache.find(prop.gcnArchName);
  if (it == cache.end()) {
    LLVM_DEBUG(llvm::dbgs() << "Cache miss! Fetching native arch info...\n");
    it = cache.emplace(prop.gcnArchName, fetchNativeArchInfo(prop, agent_info))
             .first;
  }

  return it->second;
}

} // anonymous namespace

AmdArchInfo mlir::rock::lookupArchInfo(StringRef arch) {
  // Keep this implementation in sync with
  // mlir/test/lit.site.cfg.py.in:set_arch_features()
  auto [chip, deviceId] = parseArchString(arch);
  if (chip == "native") {
    return nativeArchInfo(deviceId);
  }
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
        .Cases("11", "13", rdnaNoDotInfo)
        .Cases("10", "12", rdnaInfo)
        // All gfx103x are the same for us
        .StartsWith("3", rdnaInfo)
        .Default(rdnaNoDotInfo);
  }
  if (major == "gfx11") {
    // We know these chips have common features per backend
    return rdna3Info;
  }
  if (major == "gfx12") {
    return rdna4Info;
  }
  auto msg = "Unsupported architecture: " + arch.str();
  llvm_unreachable(msg.c_str());
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
  return theseFeatures;
}
