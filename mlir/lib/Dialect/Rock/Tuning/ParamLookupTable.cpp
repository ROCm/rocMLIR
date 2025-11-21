#include "mlir/Dialect/Rock/Tuning/ParamLookupTable.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "rock-tuning-parameter"

using namespace mlir;
using namespace mlir::rock;

template <typename ParamsType>
ArrayRef<ParamsType> ParamLookupTable<ParamsType>::lookup(StringRef arch,
                                                          KernelType op,
                                                          Type dataType) {
  if (dataType.getIntOrFloatBitWidth() == 4 && isa<FloatType>(dataType) &&
      op == KernelType::Gemm && !lookupArchInfo(arch).hasScaledGemm)
    llvm::report_fatal_error("Unsupported arch for f4 kernels");

  arch = getArchName(arch);
  auto key = makeKey(arch, op, dataType);
  LLVM_DEBUG(llvm::dbgs() << "Lookup for tuning parameters with key " << key
                          << "\n");

  static const auto &table = getTable();
  auto it = table.find(key);
  if (it != table.end()) {
    return ArrayRef<ParamsType>(it->second.first, it->second.second);
  }

  auto fallbackKey = findFallback(key);
  if (!fallbackKey.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Falling back to tuning parameters with key "
                            << fallbackKey << "\n");
    return ArrayRef<ParamsType>(table.at(fallbackKey).first,
                                table.at(fallbackKey).second);
  }

  llvm::report_fatal_error(llvm::Twine("Tuning parameters not found for key ") +
                           key);
}

template <typename ParamsType>
std::string
ParamLookupTable<ParamsType>::findFallback(const std::string &target) {
  const auto relatives = getRelatives(target);
  if (relatives.empty())
    return "";

  auto it = std::lower_bound(relatives.begin(), relatives.end(), target);
  if (it == relatives.end())
    return relatives.back();
  else if (it == relatives.begin())
    return relatives.front();
  else {
    auto mismatchNext = target.end();
    std::tie(mismatchNext, std::ignore) =
        std::mismatch(target.begin(), target.end(), it->begin());

    auto mismatchPrev = target.end();
    std::tie(mismatchPrev, std::ignore) =
        std::mismatch(target.begin(), target.end(), std::prev(it)->begin());

    if (mismatchNext < mismatchPrev)
      return *std::prev(it);
    else
      // If the mismatches are equal, prefer the larger (newer) candidate
      return *it;
  }
}

template <typename ParamsType>
std::vector<std::string>
ParamLookupTable<ParamsType>::getRelatives(const std::string &target) {
  // For non-accel params, fall back to any gfx
  constexpr auto fallbackArchPrefixLen =
      std::is_same_v<ParamsType, InitParamsNonAccel> ? 3 : 4;
  const auto suffixLen = target.size() - target.find(separator);

  std::vector<std::string> relatives;

  static const auto &table = getTable();
  for (const auto &entry : table) {
    const auto &candidate = entry.first;
    // If suffix and prefix match, then they are relatives
    if (std::equal(target.rbegin(), target.rbegin() + suffixLen,
                   candidate.rbegin()) &&
        std::equal(target.begin(), target.begin() + fallbackArchPrefixLen,
                   candidate.begin())) {
      relatives.push_back(candidate);
    }
  }

  return relatives;
}

template <typename ParamsType>
StringRef ParamLookupTable<ParamsType>::getArchName(StringRef arch) {
  auto gfxPos = arch.find("gfx");
  if (gfxPos == StringRef::npos) {
    llvm_unreachable("Invalid architecture string");
  }
  auto remaining = arch.substr(gfxPos);
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
  std::string dataTypeStr;
  if constexpr (std::is_same_v<ParamsType, InitParamsNonAccel>) {
    // For non-accel params, we only support f32
    dataTypeStr = "f32";
  } else if (dataType.getIntOrFloatBitWidth() == 4 &&
             isa<FloatType>(dataType)) {
    // We usa simplified "f4" for all 4-bit float types
    dataTypeStr = "f4";
  } else if (dataType.getIntOrFloatBitWidth() == 8 &&
             isa<FloatType>(dataType)) {
    // There are several 8-bit float types, but we use "f8" generically
    dataTypeStr = "f8";
  } else if (dataType.getIntOrFloatBitWidth() == 16 &&
             isa<FloatType>(dataType)) {
    // We use "f16" for bf16 and f16 generically
    dataTypeStr = "f16";
  } else {
    llvm::raw_string_ostream os(dataTypeStr);
    os << dataType;
    if (dataType.isInteger() &&
        (dataTypeStr.at(0) == 's' || dataTypeStr.at(0) == 'u')) {
      // Integer types can be printed as "sint" or "uint"
      dataTypeStr = dataTypeStr.substr(1);
    }
  }
  return dataTypeStr;
}

template <>
std::map<std::string, ParamLookupTable<InitParamsNonAccel>::ParamArray>
ParamLookupTable<InitParamsNonAccel>::buildTable() {
  return {
#define NonAccel_LOOKUP_TABLE_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef NonAccel_LOOKUP_TABLE_GEN
  };
}

// Specialization for Accel (XDL/WMMA) parameters
template <>
std::map<std::string, ParamLookupTable<InitParamsAccel>::ParamArray>
ParamLookupTable<InitParamsAccel>::buildTable() {
  return {
#define Accel_LOOKUP_TABLE_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef Accel_LOOKUP_TABLE_GEN
  };
}

template class mlir::rock::ParamLookupTable<InitParamsNonAccel>;
template class mlir::rock::ParamLookupTable<InitParamsAccel>;
