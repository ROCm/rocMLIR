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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetSelect.h"

#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"

#include "hip/hip_runtime_api.h"

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
             /*waveSize=*/64, /*maxWavesPerEU*/ 8, /*totalSGPRPerEU*/ 512,
             /*totalVGPRPerEU*/ 512, /*totalSharedMemPerCU*/ 65536,
             /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/120,
             /*hasFp8ConversionInstrs=*/false,
             /*hasOcpFp8ConversionInstrs=*/false, /*maxNumXCC=*/1),
    cdna2Info(GemmFeatures::mfma | GemmFeatures::dot |
                  GemmFeatures::atomic_add | GemmFeatures::atomic_add_f16,
              /*waveSize=*/64, /*maxWavesPerEU*/ 8, /*totalSGPRPerEU*/ 512,
              /*totalVGPRPerEU*/ 512, /*totalSharedMemPerCU*/ 65536,
              /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/104,
              /*hasFp8ConversionInstrs=*/false,
              /*hasOcpFp8ConversionInstrs=*/false, /*maxNumXCC=*/1),
    cdna3Info(GemmFeatures::mfma | GemmFeatures::dot |
                  GemmFeatures::atomic_add | GemmFeatures::atomic_add_f16,
              /*waveSize=*/64, /*maxWavesPerEU*/ 8, /*totalSGPRPerEU*/ 800,
              /*totalVGPRPerEU*/ 512, /*totalSharedMemPerCU*/ 65536,
              /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/80,
              /*hasFp8ConversionInstrs=*/true,
              /*hasOcpFp8ConversionInstrs=*/false, /*maxNumXCC=*/8),
    cdna35Info(GemmFeatures::mfma | GemmFeatures::dot |
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
              /*waveSize=*/32, /*maxWavesPerEU*/ 20, /*totalSGPRPerEU*/ 512,
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
    llvm::outs() << "NOTE: Value discrepancy for " << name << ": " << lhs
                 << " (old) != " << rhs << " (new). Proceeding with " << rhs
                 << ".\n";
    lhs = std::forward<RHS>(rhs);
  }
}

GemmFeatures &operator|=(GemmFeatures &lhs, const GemmFeatures &rhs) {
  lhs = lhs | rhs;
  return lhs;
}

GemmFeatures &operator&=(GemmFeatures &lhs, const GemmFeatures &rhs) {
  lhs = lhs & rhs;
  return lhs;
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

std::unique_ptr<const GCNTargetMachine>
createTargetMachine(StringRef chip, StringRef featureString = "") {
  static std::once_flag flag;
  std::call_once(flag, [] {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
  });

  Triple triple("amdgcn-amd-amdhsa");
  std::string error;
  const Target *target = TargetRegistry::lookupTarget(triple, error);
  if (!target) {
    llvm::errs() << "WARNING: Target registry lookup failed with error: "
                 << error << ".\n";
    return nullptr;
  }

  return std::unique_ptr<GCNTargetMachine>(static_cast<GCNTargetMachine *>(
      target->createTargetMachine(triple, chip, featureString, {}, {}, {})));
}

AmdArchInfo fetchNativeArchInfo(unsigned deviceId = 0) {
  llvm::outs() << "Fetching native arch info for device " << deviceId
               << "...\n";

  hipDeviceProp_t prop;
  if (auto err = hipGetDeviceProperties(&prop, deviceId); err != hipSuccess) {
    auto reason = "hipGetDeviceProperties failed with error: " +
                  std::string(hipGetErrorString(err));
    llvm::report_fatal_error(reason.c_str());
  }

  llvm::outs() << "gcnArchName: " << prop.gcnArchName << "\n";
  auto ret = lookupArchInfo(prop.gcnArchName); // get baseline

  checkAndSetInfo("(HIP) minNumCU", ret.minNumCU, prop.multiProcessorCount);
  checkAndSetInfo("(HIP) waveSize", ret.waveSize, prop.warpSize);
  checkAndSetInfo("(HIP) totalSharedMemPerCU", ret.totalSharedMemPerCU,
                  prop.maxSharedMemoryPerMultiProcessor);
  checkAndSetInfo("(HIP) maxSharedMemPerWG", ret.maxSharedMemPerWG,
                  prop.sharedMemPerBlock);

  auto chip = std::get<0>(parseArchString(prop.gcnArchName));
  if (auto tm = createTargetMachine(chip); !tm) {
    llvm::errs() << "WARNING: Couldn't create target machine. Proceeding with "
                    "HIP values.\n";
  } else {
    GCNSubtarget st(tm->getTargetTriple(), std::string(tm->getTargetCPU()),
                    std::string(tm->getTargetFeatureString()), *tm);

    checkAndSetInfo("(LLVM) numEUPerCU", ret.numEUPerCU, st.getEUsPerCU());
    checkAndSetInfo("(LLVM) maxWavesPerEU", ret.maxWavesPerEU,
                    st.getMaxWavesPerEU());
    checkAndSetInfo("(LLVM) totalSGPRPerEU", ret.totalSGPRPerEU,
                    st.getTotalNumSGPRs());
    checkAndSetInfo("(LLVM) totalVGPRPerEU", ret.totalVGPRPerEU,
                    st.getTotalNumVGPRs());
    checkAndSetInfo("(LLVM) waveSize", ret.waveSize, st.getWavefrontSize());
    checkAndSetInfo("(LLVM) totalSharedMemPerCU", ret.totalSharedMemPerCU,
                    st.getLocalMemorySize());
    checkAndSetInfo("(LLVM) maxSharedMemPerWG", ret.maxSharedMemPerWG,
                    st.getAddressableLocalMemorySize());

    auto features = ret.defaultFeatures;
    if (st.hasAtomicFaddInsts()) {
      features |= GemmFeatures::atomic_add;
    } else {
      features &= ~GemmFeatures::atomic_add;
    }
    if (st.hasAtomicBufferGlobalPkAddF16Insts() ||
        st.hasAtomicBufferGlobalPkAddF16NoRtnInsts()) {
      features |= GemmFeatures::atomic_add_f16;
    } else {
      features &= ~GemmFeatures::atomic_add_f16;
    }
    if (st.hasAtomicBufferPkAddBF16Inst()) {
      features |= GemmFeatures::atomic_add_bf16;
    } else {
      features &= ~GemmFeatures::atomic_add_bf16;
    }
    if (st.hasAtomicFMinFMaxF32GlobalInsts()) {
      features |= GemmFeatures::atomic_fmax_f32;
    } else {
      features &= ~GemmFeatures::atomic_fmax_f32;
    }
    checkAndSetInfo("(LLVM) defaultFeatures", ret.defaultFeatures, features);

    if (st.hasFP8ConversionInsts()) {
      if (auto maybeChipset = amdgpu::Chipset::parse(chip);
          failed(maybeChipset)) {
        llvm::errs()
            << "WARNING: Failed parsing chipset. Proceeding with preset "
               "values.\n";
      } else {
        checkAndSetInfo("(LLVM) hasOcpFp8ConversionInstrs",
                        ret.hasOcpFp8ConversionInstrs,
                        amdgpu::hasOcpFp8(maybeChipset.value()));
        checkAndSetInfo("(LLVM) hasFp8ConversionInstrs",
                        ret.hasFp8ConversionInstrs,
                        !ret.hasOcpFp8ConversionInstrs);
      }
    } else {
      checkAndSetInfo("(LLVM) hasFp8ConversionInstrs",
                      ret.hasFp8ConversionInstrs, false);
      checkAndSetInfo("(LLVM) hasOcpFp8ConversionInstrs",
                      ret.hasOcpFp8ConversionInstrs, false);
    }
  }

  // TODO check and set maxNumXCC

  return ret;
}

} // anonymous namespace

AmdArchInfo mlir::rock::lookupArchInfo(StringRef arch) {
  // Keep this implementation in sync with
  // mlir/test/lit.site.cfg.py.in:set_arch_features()
  if (arch.empty()) {
    return gcnInfo;
  }
  auto [chip, deviceId] = parseArchString(arch);
  if (chip == "native") {
    return fetchNativeArchInfo(deviceId);
  }
  StringRef minor = chip.take_back(2);
  StringRef major = chip.slice(0, chip.size() - 2);
  if (major == "gfx9") {
    return llvm::StringSwitch<AmdArchInfo>(minor)
        .Case("08", cdnaInfo)
        .Case("0a", cdna2Info)
        .Case("42", cdna3Info)
        .Case("50", cdna35Info)
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
