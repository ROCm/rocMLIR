//===- SmartTuningDb.h - Learned tuning-config ranking -------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Public API of the smart-tuning model database: a static table of learned
// per-(arch, op) gradient-boosted-tree models that rank candidate perfConfigs
// best-first. The models are embedded as plain tree *data* (mirroring
// QuickTuningDb) and walked by a generic evaluator here -- there is no link
// against LightGBM/Treelite.
//
// Each model has two stages, mirroring the offline trainer
// (mlir/utils/performance/analysis/tuning_eval):
//   * applicability -- will this config compile/run at all?
//   * optimality    -- given it runs, is it near the per-problem best?
// A config's feature vector is the same for both stages; see
// SmartTuningFeatures.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_SMART_TUNING_DB_H
#define MLIR_DIALECT_ROCK_SMART_TUNING_DB_H

#include "mlir/Dialect/Rock/IR/Rock.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace rock {
namespace SmartTuningDb {

// A flattened decision-tree node. Internal nodes have splitFeature >= 0 and go
// to nodes[left] iff features[splitFeature] <= threshold, else nodes[right].
// Leaves have splitFeature < 0 and contribute leafValue to the margin.
struct TreeNode {
  int splitFeature;
  double threshold;
  int left;
  int right;
  double leafValue;
};

// One boosted-tree stage. `roots[i]` indexes the root of tree i in `nodes`. The
// stage margin is `bias + sum_i evalTree(roots[i])`. An absent (unfitted) stage
// has nodes == nullptr and is treated as neutral.
struct Stage {
  const TreeNode *nodes;
  unsigned numNodes;
  const unsigned *roots;
  unsigned numRoots;
  double bias;
};

// One learned model for an (arch, op), keyed by "<arch>_<op>".
struct Model {
  const char *key;
  unsigned numFeatures;
  Stage applicability;
  Stage optimality;
};

// Resolves the model for (arch, op), or nullptr when none is embedded (the
// caller should then fall back to a non-learned tuning space). `arch` may be a
// decorated target string; only the bare gfx identifier is matched.
const Model *resolveModel(StringRef arch, KernelType op);

// Margin (pre-sigmoid score) of a stage for a feature vector. Neutral (0) for
// an absent stage.
double stageMargin(const Stage &stage, ArrayRef<double> features);

// Reorders candidate indices [0, featureRows.size()) best-first, mirroring the
// offline proposer: predicted-applicable configs first (ordered by predicted
// optimality), then the rest (ordered by predicted applicability). Truncated to
// `maxK`. `featureRows[i]` is candidate i's feature vector (length
// numFeatures). `applicThreshold` is the applicability probability above which
// a config is treated as applicable (the tier boundary, not a hard filter).
SmallVector<unsigned> rankConfigs(const Model &model,
                                  ArrayRef<ArrayRef<double>> featureRows,
                                  unsigned maxK, double applicThreshold = 0.5);

// -- Database invariants (always true; exposed for tests) --------------------

// True if the entries are sorted lexicographically by key (a precondition for
// the binary-search resolution in resolveModel).
bool isSortedByKey();

} // namespace SmartTuningDb
} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_SMART_TUNING_DB_H
