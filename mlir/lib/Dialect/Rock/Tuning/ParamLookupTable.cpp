#include "mlir/Dialect/Rock/Tuning/ParamLookupTable.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "rock-tuning-parameter"

using namespace mlir;
using namespace mlir::rock;

ArrayRef<StringRef> ParamLookupTable::lookup(StringRef arch, KernelType op,
                                             Type dataType) {
  if (dataType.getIntOrFloatBitWidth() == 4 && isa<FloatType>(dataType) &&
      op == KernelType::Gemm && !lookupArchInfo(arch).hasScaledGemm)
    llvm::report_fatal_error("Unsupported arch for f4 kernels");

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

StringRef ParamLookupTable::findFallback(StringRef target) {
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

SmallVector<StringRef, 12> ParamLookupTable::getRelatives(StringRef target) {
  // For non-accel params, fall back to any gfx
  constexpr auto fallbackArchPrefixLen = 4; // TODO NonAccel
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

StringRef ParamLookupTable::normalizeArch(StringRef arch) {
  auto gfxPos = arch.find("gfx");
  if (gfxPos == StringRef::npos) {
    llvm_unreachable("Invalid architecture string");
  }
  auto remaining = arch.substr(gfxPos);
  auto endPos =
      remaining.find_if_not([](char c) { return llvm::isAlnum(c); }, 3);
  return remaining.substr(0, endPos);
}

std::string ParamLookupTable::getKernelTypeString(KernelType kernelType) {
  switch (kernelType) {
  case KernelType::ConvBwdData:
  case KernelType::ConvBwdWeight:
    // We use the same suffix for all convolution types
    return stringifyEnum(KernelType::Conv).lower();
  default:
    return stringifyEnum(kernelType).lower();
  }
}

std::string ParamLookupTable::getDataTypeString(Type dataType) {
  if constexpr (         /*std::is_same_v<StringRef, InitParamsNonAccel>*/
                false) { // TODO
    // For non-accel params, we only support f32
    return "f32";
  } else if (dataType.getIntOrFloatBitWidth() == 4 &&
             isa<FloatType>(dataType)) {
    // We usa simplified "f4" for all 4-bit float types
    return "f4";
  } else if (dataType.getIntOrFloatBitWidth() == 8 &&
             isa<FloatType>(dataType)) {
    // There are several 8-bit float types, but we use "fp8" generically
    return "fp8";
  } else if (dataType.getIntOrFloatBitWidth() == 16 &&
             isa<FloatType>(dataType)) {
    // We use "f16" for bf16 and f16 generically
    return "f16";
  } else {
    std::string result;
    llvm::raw_string_ostream os(result);
    os << dataType;
    if (dataType.isInteger() && (result.at(0) == 's' || result.at(0) == 'u')) {
      // Integer types can be printed as "sint" or "uint"
      result.erase(result.begin());
    }
    return result;
  }
}

std::map<StringRef, ArrayRef<StringRef>> ParamLookupTable::buildTable() {
  return {
#define PARAM_LOOKUP_TABLE_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef PARAM_LOOKUP_TABLE_GEN
  };
}
