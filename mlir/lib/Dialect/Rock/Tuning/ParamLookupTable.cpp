#include "mlir/Dialect/Rock/Tuning/ParamLookupTable.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#include <mutex>

#define DEBUG_TYPE "rock-tuning-parameter"

using namespace mlir;
using namespace mlir::rock;

template <typename ParamsType>
ArrayRef<StringRef> ParamLookupTable<ParamsType>::lookup(StringRef arch,
                                                         KernelType op,
                                                         Type dataType) {
  if (dataType.getIntOrFloatBitWidth() == 4 && dataType.isFloat() &&
      op == KernelType::Gemm && !lookupArchInfo(arch).hasScaledGemm)
    llvm::report_fatal_error(Twine("fp4 gemm is not supported on ") + arch);

  arch = normalizeArch(arch);
  auto key = makeKey(arch, op, dataType);
  LLVM_DEBUG(llvm::dbgs() << "Lookup for tuning parameters with key " << key
                          << "\n");

  static const auto &table = getTable();
  auto it = table.find(key);
  if (it != table.end())
    return it->second;

  auto fallbackKey = findFallback(key);
  if (!fallbackKey.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Falling back to tuning parameters with key "
                            << fallbackKey << "\n");
    return table.at(fallbackKey);
  }

  llvm::report_fatal_error(Twine("Tuning parameters not found for key ") + key);
}

template <typename ParamsType>
StringRef ParamLookupTable<ParamsType>::findFallback(StringRef target) {
  const auto relatives = getRelatives(target);
  if (relatives.empty())
    return StringRef();

  auto it = std::lower_bound(relatives.begin(), relatives.end(), target);
  if (it == relatives.end())
    return relatives.back();
  if (it == relatives.begin())
    return relatives.front();

  auto prev = std::prev(it);
  auto mismatchNext = std::mismatch(target.begin(), target.end(), it->begin());
  auto mismatchPrev =
      std::mismatch(target.begin(), target.end(), prev->begin());

  if (mismatchNext.first < mismatchPrev.first)
    return *prev;
  else
    // If the mismatches are equal, prefer the larger (newer) candidate
    return *it;
}

template <typename ParamsType>
SmallVector<StringRef, 12>
ParamLookupTable<ParamsType>::getRelatives(StringRef target) {
  // For non-accel params, fall back to any gfx
  constexpr auto fallbackArchPrefixLen =
      std::is_same_v<ParamsType, GeneralGemmParamsAttr> ? 3 : 4;
  const auto suffixLen = target.size() - target.find(separator);

  SmallVector<StringRef, 12> relatives;

  static const auto &table = getTable();
  for (const auto &entry : table) {
    StringRef candidate = entry.first;
    // If suffix and prefix match, then they are relatives
    if (target.ends_with(candidate.substr(candidate.size() - suffixLen)) &&
        target.starts_with(candidate.substr(0, fallbackArchPrefixLen))) {
      relatives.push_back(candidate);
    }
  }

  return relatives;
}

template <typename ParamsType>
StringRef ParamLookupTable<ParamsType>::normalizeArch(StringRef arch) {
  // Resolve `native[:N]` to the hardware-reported arch name via the same
  // runtime that `lookupArchInfo` uses. The resolved string is cached in a
  // process-wide map so the returned `StringRef` stays valid for the lifetime
  // of the process (required by the `static const std::map<StringRef, ...>`
  // lookup table this value feeds into).
  if (arch.consume_front("native")) {
    static std::mutex m;
    static llvm::StringMap<std::string> resolvedCache;
    std::lock_guard<std::mutex> lock(m);
    // Re-form the original key for the cache lookup.
    std::string cacheKey = ("native" + arch).str();
    auto [it, inserted] = resolvedCache.try_emplace(cacheKey);
    if (inserted) {
      unsigned deviceId = 0;
      if (arch.consume_front(":")) {
        unsigned long long parsed = 0;
        if (!llvm::getAsUnsignedInteger(arch, 0, parsed))
          deviceId = static_cast<unsigned>(parsed);
      }
      it->second = nativeArchName(deviceId);
      if (it->second.empty())
        llvm::report_fatal_error(
            Twine("Failed to resolve `") + cacheKey +
            "`: AMD GPU arch runtime unavailable or device not present");
    }
    return normalizeArch(it->second);
  }

  auto gfxPos = arch.find("gfx");
  if (gfxPos == StringRef::npos)
    llvm::report_fatal_error(Twine("Invalid architecture string: ") + arch);
  StringRef remaining = arch.substr(gfxPos);
  auto endPos =
      remaining.find_if_not([](char c) { return llvm::isAlnum(c); }, 3);
  return remaining.substr(0, endPos);
}

template <typename ParamsType>
std::string
ParamLookupTable<ParamsType>::getKernelTypeString(KernelType kernelType) {
  switch (kernelType) {
  case KernelType::ConvBwdData:
  case KernelType::ConvBwdWeight:
    // We use the same suffix for all convolution types
    return stringifyEnum(KernelType::Conv).lower();
  default:
    return stringifyEnum(kernelType).lower();
  }
}

template <typename ParamsType>
std::string ParamLookupTable<ParamsType>::getDataTypeString(Type dataType) {
  if constexpr (std::is_same_v<ParamsType, GeneralGemmParamsAttr>) {
    // For non-accel params, we only support f32
    return "f32";
  } else if (dataType.isBF16()) {
    // Special case for bf16, we don't want to normalize it to f16
    return "bf16";
  } else if (dataType.isFloat()) {
    // Normalize other float types by bitwidth
    unsigned bitwidth = dataType.getIntOrFloatBitWidth();
    switch (bitwidth) {
    case 4:
    case 8:
      return "fp" + std::to_string(bitwidth);
    case 16:
    case 32:
      return "f" + std::to_string(bitwidth);
    default:
      llvm::report_fatal_error("Unsupported float bitwidth: " +
                               Twine(bitwidth));
    }
  } else if (dataType.isInteger()) {
    // Normalize integer types by bitwidth
    unsigned bitwidth = dataType.getIntOrFloatBitWidth();
    switch (bitwidth) {
    case 8:
      return "i" + std::to_string(bitwidth);
    default:
      llvm::report_fatal_error("Unsupported integer bitwidth: " +
                               Twine(bitwidth));
    }
  } else {
    llvm::report_fatal_error("Unsupported data type");
  }
}

template <>
std::map<StringRef, ArrayRef<StringRef>>
ParamLookupTable<GeneralGemmParamsAttr>::buildTable() {
  return {
#define NonAccel_LOOKUP_TABLE_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef NonAccel_LOOKUP_TABLE_GEN
  };
}

template <>
std::map<StringRef, ArrayRef<StringRef>>
ParamLookupTable<AccelGemmParamsAttr>::buildTable() {
  return {
#define Accel_LOOKUP_TABLE_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef Accel_LOOKUP_TABLE_GEN
  };
}

template <>
std::map<StringRef, ArrayRef<StringRef>>
ParamLookupTable<GemmGemmParamsAttr>::buildTable() {
  return {
#define GemmGemm_LOOKUP_TABLE_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef GemmGemm_LOOKUP_TABLE_GEN
  };
}

template class mlir::rock::ParamLookupTable<GeneralGemmParamsAttr>;
template class mlir::rock::ParamLookupTable<AccelGemmParamsAttr>;
template class mlir::rock::ParamLookupTable<GemmGemmParamsAttr>;
