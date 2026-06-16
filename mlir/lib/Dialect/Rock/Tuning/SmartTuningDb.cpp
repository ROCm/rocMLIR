//===- SmartTuningDb.cpp - Learned tuning-config ranking -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/SmartTuningDb.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cmath>

using namespace mlir;
using namespace mlir::rock;

namespace {

#define SMART_TUNING_DB_ARRAYS
#include "SmartTuningDb.inc"
#undef SMART_TUNING_DB_ARRAYS

const SmartTuningDb::Model kSmartTuningDb[] = {
#define SMART_TUNING_DB_ENTRIES
#include "SmartTuningDb.inc"
#undef SMART_TUNING_DB_ENTRIES
};

constexpr const char *kSeparator = "_";

// Extracts the bare "gfx<N>" identifier from a possibly-decorated arch string.
StringRef normalizeArch(StringRef arch) {
  auto gfxPos = arch.find("gfx");
  if (gfxPos == StringRef::npos)
    return StringRef();
  auto remaining = arch.substr(gfxPos);
  auto endPos =
      remaining.find_if_not([](char c) { return llvm::isAlnum(c); }, 3);
  return remaining.substr(0, endPos);
}

// Op suffix used in model keys. Mirrors the trainer's op names; ops without a
// learned model return "" (no match).
StringRef getOpString(KernelType op) {
  switch (op) {
  case KernelType::Conv:
  case KernelType::ConvBwdData:
  case KernelType::ConvBwdWeight:
    return "conv";
  case KernelType::Gemm:
    return "gemm";
  case KernelType::Attention:
    return "attention";
  case KernelType::GemmElementwiseGemm:
  case KernelType::ConvElementwiseGemm:
    return StringRef();
  }
  llvm_unreachable("Unknown KernelType");
}

double evalTree(const SmartTuningDb::TreeNode *nodes, unsigned root,
                ArrayRef<double> features) {
  const SmartTuningDb::TreeNode *node = &nodes[root];
  while (node->splitFeature >= 0) {
    bool goLeft = features[node->splitFeature] <= node->threshold;
    node = &nodes[goLeft ? node->left : node->right];
  }
  return node->leafValue;
}

} // namespace

double SmartTuningDb::stageMargin(const Stage &stage,
                                  ArrayRef<double> features) {
  if (stage.nodes == nullptr)
    return 0.0;
  double margin = stage.bias;
  for (unsigned i = 0; i < stage.numRoots; ++i)
    margin += evalTree(stage.nodes, stage.roots[i], features);
  return margin;
}

const SmartTuningDb::Model *SmartTuningDb::resolveModel(StringRef arch,
                                                        KernelType op) {
  StringRef normArch = normalizeArch(arch);
  StringRef opStr = getOpString(op);
  if (normArch.empty() || opStr.empty())
    return nullptr;

  std::string key = (Twine(normArch) + kSeparator + opStr).str();
  const Model *it = std::lower_bound(
      std::begin(kSmartTuningDb), std::end(kSmartTuningDb), StringRef(key),
      [](const Model &m, StringRef k) { return StringRef(m.key) < k; });
  if (it != std::end(kSmartTuningDb) && StringRef(it->key) == key)
    return it;
  return nullptr;
}

SmallVector<unsigned>
SmartTuningDb::rankConfigs(const Model &model,
                           ArrayRef<ArrayRef<double>> featureRows,
                           unsigned maxK, double applicThreshold) {
  unsigned n = featureRows.size();
  // Applicability is a hard tier boundary in margin space: prob >= threshold
  // <=> margin >= logit(threshold).
  double marginThreshold = std::log(applicThreshold / (1.0 - applicThreshold));
  bool hasApplic = model.applicability.nodes != nullptr;
  bool hasOptimal = model.optimality.nodes != nullptr;

  SmallVector<double> applicMargin(n), optimalMargin(n);
  SmallVector<unsigned> tierA, tierB;
  for (unsigned i = 0; i < n; ++i) {
    applicMargin[i] =
        hasApplic ? stageMargin(model.applicability, featureRows[i]) : 0.0;
    optimalMargin[i] =
        hasOptimal ? stageMargin(model.optimality, featureRows[i]) : 0.0;
    bool applicable = !hasApplic || applicMargin[i] >= marginThreshold;
    (applicable ? tierA : tierB).push_back(i);
  }

  // Tier A: predicted-applicable, ordered by optimality (tie-break by
  // applicability) when an optimality stage exists, else by applicability.
  llvm::stable_sort(tierA, [&](unsigned a, unsigned b) {
    if (hasOptimal && optimalMargin[a] != optimalMargin[b])
      return optimalMargin[a] > optimalMargin[b];
    return applicMargin[a] > applicMargin[b];
  });
  // Tier B: the rest, ordered by applicability.
  llvm::stable_sort(tierB, [&](unsigned a, unsigned b) {
    return applicMargin[a] > applicMargin[b];
  });

  SmallVector<unsigned> ranked;
  ranked.reserve(std::min<size_t>(n, maxK));
  for (unsigned i : tierA) {
    if (ranked.size() >= maxK)
      return ranked;
    ranked.push_back(i);
  }
  for (unsigned i : tierB) {
    if (ranked.size() >= maxK)
      return ranked;
    ranked.push_back(i);
  }
  return ranked;
}

bool SmartTuningDb::isSortedByKey() {
  return std::is_sorted(std::begin(kSmartTuningDb), std::end(kSmartTuningDb),
                        [](const Model &a, const Model &b) {
                          return StringRef(a.key) < StringRef(b.key);
                        });
}
