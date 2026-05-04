//===- QuickTuningDb.cpp - MLIR tuning parameter lookup -------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/QuickTuningDb.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Process.h"

#define DEBUG_TYPE "rock-tuning-parameter"

using namespace mlir;
using namespace mlir::rock;

namespace {

// Maps a problem key hash to a range within `indices`.
struct ProblemRef {
  uint64_t hash;
  unsigned offset;
  unsigned count;
};

struct QuickTuningDbEntry {
  const char *key;
  const StringRef *setCover;
  unsigned setCoverSize;
  const unsigned *indices;
  unsigned indicesSize;
  const ProblemRef *problemMap;
  unsigned problemMapSize;
};

#define QUICK_TUNING_DB_ARRAYS
#include "QuickTuningDb.inc"
#undef QUICK_TUNING_DB_ARRAYS

const QuickTuningDbEntry kQuickTuningDb[] = {
#define QUICK_TUNING_DB_ENTRIES
#include "QuickTuningDb.inc"
#undef QUICK_TUNING_DB_ENTRIES
};

constexpr const char *kSeparator = "_";

// Caps the number of perfconfigs returned by lookup(). Read once from the
// environment; defaults to 30 when unset or unparseable.
unsigned getMaxListLength() {
  constexpr unsigned kDefaultMaxListLength = 30;
  constexpr const char *kMaxListLengthEnvVar = "ROCMLIR_QUICK_TUNING_LIST_MAX";

  static const unsigned cached = []() -> unsigned {
    auto envVal = llvm::sys::Process::GetEnv(kMaxListLengthEnvVar);
    unsigned parsed;
    if (!envVal || StringRef(*envVal).getAsInteger(10, parsed))
      return kDefaultMaxListLength;
    return parsed;
  }();

  return cached;
}

// Key accessors so lowerBoundByKey can iterate over both raw entries
// and pointers to entries.
StringRef entryKey(const QuickTuningDbEntry &e) { return e.key; }
StringRef entryKey(const QuickTuningDbEntry *e) { return e->key; }

// Returns the first element of `range` whose key is >= target (or
// end() when none). Works on any sorted range whose elements satisfy
// the entryKey accessor above.
template <typename Range>
auto lowerBoundByKey(Range &&range, StringRef target) {
  return std::lower_bound(
      std::begin(range), std::end(range), target,
      [](const auto &e, StringRef k) { return entryKey(e) < k; });
}

// Returns the per-problem slice of `entry`'s set-cover for `problemKeyHash`,
// or an empty vector if the hash is absent. problemMap is sorted by hash.
SmallVector<StringRef> lookupProblem(const QuickTuningDbEntry &entry,
                                     uint64_t problemKeyHash) {
  SmallVector<StringRef> result;

  if (entry.problemMap == nullptr)
    return result;

  const ProblemRef *begin = entry.problemMap;
  const ProblemRef *end = entry.problemMap + entry.problemMapSize;
  const ProblemRef *it = std::lower_bound(
      begin, end, problemKeyHash,
      [](const ProblemRef &ref, uint64_t hash) { return ref.hash < hash; });
  if (it == end || it->hash != problemKeyHash)
    return result;

  // Walk `it->count` entries starting at indices[it->offset]; each is a
  // position in setCover.
  result.reserve(it->count);
  for (unsigned j = 0; j < it->count; ++j)
    result.push_back(entry.setCover[entry.indices[it->offset + j]]);

  return result;
}

// Extracts the bare "gfx<N>" identifier from a possibly-decorated arch string.
StringRef normalizeArch(StringRef arch) {
  auto gfxPos = arch.find("gfx");
  if (gfxPos == StringRef::npos)
    llvm::report_fatal_error(Twine("Invalid architecture string: ") + arch);

  auto remaining = arch.substr(gfxPos);
  auto endPos =
      remaining.find_if_not([](char c) { return llvm::isAlnum(c); }, 3);
  return remaining.substr(0, endPos);
}

// Maps a kernel type to its key suffix. All conv directions collapse to
// "conv"; the rest get their own dedicated suffix.
StringRef getKernelTypeString(KernelType kernelType) {
  switch (kernelType) {
  case KernelType::Conv:
  case KernelType::ConvBwdData:
  case KernelType::ConvBwdWeight:
    return "conv";
  case KernelType::Gemm:
    return "gemm";
  case KernelType::Attention:
    return "attention";
  case KernelType::GemmElementwiseGemm:
    return "gemmelementwisegemm";
  case KernelType::ConvElementwiseGemm:
    return "convelementwisegemm";
  }
  llvm_unreachable("Unknown KernelType");
}

// bf16 has its own suffix; other types are keyed by bitwidth.
StringRef getDataTypeString(Type dataType) {
  if (dataType.isBF16())
    return "bf16";
  if (dataType.isFloat()) {
    switch (dataType.getIntOrFloatBitWidth()) {
    case 4:
      return "fp4";
    case 8:
      return "fp8";
    case 16:
      return "f16";
    case 32:
      return "f32";
    default:
      llvm::report_fatal_error("Unsupported float bitwidth: " +
                               Twine(dataType.getIntOrFloatBitWidth()));
    }
  }
  if (dataType.isInteger()) {
    switch (dataType.getIntOrFloatBitWidth()) {
    case 8:
      return "i8";
    default:
      llvm::report_fatal_error("Unsupported integer bitwidth: " +
                               Twine(dataType.getIntOrFloatBitWidth()));
    }
  }
  llvm::report_fatal_error("Unsupported data type");
}

std::string makeKey(StringRef arch, KernelType op, Type dataType) {
  return (Twine(arch) + kSeparator + getKernelTypeString(op) + kSeparator +
          getDataTypeString(dataType))
      .str();
}

// Returns pointers to all DB entries whose arch prefix and "_<op>_<dtype>"
// suffix match the query:
//   - Accel:     same gfx family (arch's first 4 chars), suffix verbatim.
//   - Non-accel: gfx1* candidates, suffix forced to "_<op>_f32".
// Entries sharing an arch prefix form a contiguous range in the sorted
// DB, so we binary-search to the first one and walk forward.
SmallVector<const QuickTuningDbEntry *, 16>
getRelatives(StringRef arch, KernelType op, Type dataType, bool isAccel) {
  constexpr size_t kAccelArchPrefixLen = 4;
  // Non-accel queries are answered exclusively by gfx1*+f32 entries; the
  // rest of the DB carries entries that only an accel query can consume.
  constexpr StringRef kNonAccelArchPrefix = "gfx1";
  constexpr StringRef kNonAccelDataType = "f32";

  if (isAccel && arch.size() < kAccelArchPrefixLen)
    llvm_unreachable("Invalid arch");

  StringRef archPrefix =
      isAccel ? arch.substr(0, kAccelArchPrefixLen) : kNonAccelArchPrefix;
  StringRef dataTypeString =
      isAccel ? getDataTypeString(dataType) : kNonAccelDataType;
  std::string keySuffix = (Twine(kSeparator) + getKernelTypeString(op) +
                           kSeparator + dataTypeString)
                              .str();

  SmallVector<const QuickTuningDbEntry *, 16> relatives;

  for (const auto *it = lowerBoundByKey(kQuickTuningDb, archPrefix);
       it != std::end(kQuickTuningDb); ++it) {
    StringRef candidate = entryKey(*it);
    if (!candidate.starts_with(archPrefix))
      break;
    if (candidate.ends_with(keySuffix))
      relatives.push_back(it);
  }

  return relatives;
}

// Returns the entry whose key best serves the query: an exact match
// when present, otherwise the relative sharing the longest common prefix
// with the target key. Returns nullptr when no relative exists.
const QuickTuningDbEntry *findClosestEntry(StringRef arch, KernelType op,
                                           Type dataType, bool isAccel) {
  const auto relatives = getRelatives(arch, op, dataType, isAccel);
  if (relatives.empty())
    return nullptr;

  std::string target = makeKey(arch, op, dataType);
  auto it = lowerBoundByKey(relatives, target);
  if (it == relatives.end())
    return relatives.back();
  if (it == relatives.begin() || entryKey(*it) == target)
    return *it;

  auto prev = std::prev(it);
  StringRef itKey = entryKey(*it);
  StringRef prevKey = entryKey(*prev);
  auto mismatchNext =
      std::mismatch(target.begin(), target.end(), itKey.begin());
  auto mismatchPrev =
      std::mismatch(target.begin(), target.end(), prevKey.begin());

  if (mismatchNext.first < mismatchPrev.first)
    return *prev;
  // On ties, prefer the lexicographically larger candidate.
  return *it;
}

// Wraps findClosestEntry with arch normalization and a bf16->f16 retry.
// Returns the chosen entry, or nullptr.
const QuickTuningDbEntry *resolveEntry(StringRef arch, KernelType op,
                                       Type dataType, bool isAccel) {
  arch = normalizeArch(arch);
  const auto *entry = findClosestEntry(arch, op, dataType, isAccel);
  // f16 stands in for missing bf16 entries.
  if (!entry && dataType.isBF16())
    entry = findClosestEntry(arch, op, Float16Type::get(dataType.getContext()),
                             isAccel);
  return entry;
}

} // namespace

SmallVector<StringRef>
mlir::rock::QuickTuningDb::lookup(StringRef arch, KernelType op, Type dataType,
                                  bool isAccel,
                                  std::optional<uint64_t> problemKeyHash) {
  arch = normalizeArch(arch);

  if (dataType.getIntOrFloatBitWidth() == 4 && dataType.isFloat() &&
      op == KernelType::Gemm && !lookupArchInfo(arch).hasScaledGemm)
    llvm::report_fatal_error(Twine("fp4 gemm is not supported on ") + arch);

  auto key = makeKey(arch, op, dataType);

  const auto *entry = resolveEntry(arch, op, dataType, isAccel);
  if (!entry)
    llvm::report_fatal_error(
        Twine("Quick-tuning parameters not found for key ") + key);

  LLVM_DEBUG(llvm::dbgs() << "Quick-tuning lookup for key " << key
                          << " resolved to " << entryKey(*entry) << "\n");

  SmallVector<StringRef> result;
  if (problemKeyHash)
    result = lookupProblem(*entry, *problemKeyHash);
  if (result.empty())
    result.assign(entry->setCover, entry->setCover + entry->setCoverSize);

  result.truncate(std::min<size_t>(result.size(), getMaxListLength()));
  return result;
}

StringRef mlir::rock::QuickTuningDb::resolveKey(StringRef arch, KernelType op,
                                                Type dataType, bool isAccel) {
  const auto *entry = resolveEntry(arch, op, dataType, isAccel);
  return entry ? entryKey(*entry) : StringRef();
}

bool mlir::rock::QuickTuningDb::isSortedByKey() {
  return std::is_sorted(
      std::begin(kQuickTuningDb), std::end(kQuickTuningDb),
      [](const auto &a, const auto &b) { return entryKey(a) < entryKey(b); });
}

bool mlir::rock::QuickTuningDb::problemMapsAreSortedByHash() {
  for (const auto &entry : kQuickTuningDb) {
    if (entry.problemMap == nullptr)
      continue;
    if (!std::is_sorted(entry.problemMap,
                        entry.problemMap + entry.problemMapSize,
                        [](const ProblemRef &a, const ProblemRef &b) {
                          return a.hash < b.hash;
                        }))
      return false;
  }
  return true;
}
