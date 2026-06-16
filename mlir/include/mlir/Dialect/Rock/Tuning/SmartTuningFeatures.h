//===- SmartTuningFeatures.h - Feature extraction for smart tuning -------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builds the numeric feature vector a SmartTuningDb model scores. This is the
// C++ mirror of mlir/utils/performance/analysis/tuning_eval/features.py: the
// feature names, order, and arithmetic must match the offline trainer exactly,
// or a model's predictions are meaningless. The unittest pins the order and the
// values against goldens captured from the Python pipeline (the same drift
// guard QuickTuningProblemKey.cpp uses for the problem-key hash).
//
// Extraction consumes a small serialized problem signature (not an MLIR op) so
// it stays decoupled and directly testable; the smart-tuning caller fills the
// signature from the op.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_SMART_TUNING_FEATURES_H
#define MLIR_DIALECT_ROCK_SMART_TUNING_FEATURES_H

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace rock {
namespace SmartTuningFeatures {

// Serialized GEMM problem signature (mirrors the Python ProblemSig columns the
// gemm feature path reads).
struct GemmSig {
  StringRef arch;  // target, e.g. "gfx942" (decorations tolerated)
  StringRef dtype; // "f32" / "f16" / "bf16" / "i8" / "fp8" / "fp4" ...
  int64_t numCU = 64;
  int64_t numChiplets = 1;
  bool transA = false;
  bool transB = false;
  int64_t g = 1;
  int64_t m = 0;
  int64_t k = 0;
  int64_t n = 0;
};

// Serialized convolution problem signature (mirrors the conv ProblemSig columns
// CONV_COLUMNS in quickTuningGen). `c`/`k` are the totals across groups and
// `direction` is "fwd"/"bwd"/"wrw", matching the .debug representation the
// model was trained on; the layout strings are the lowercase rocMLIR convention
// with spatial dims encoded as '0' (Y/H) and '1' (X/W), e.g. "gkc01", "ngc01".
struct ConvSig {
  StringRef arch;
  StringRef dtype;
  int64_t numCU = 64;
  int64_t numChiplets = 1;
  StringRef direction;
  StringRef filterLayout;
  StringRef inputLayout;
  StringRef outputLayout;
  int64_t n = 0, c = 0, h = 0, w = 0, k = 0, y = 0, x = 0;
  int64_t strideH = 1, strideW = 1;
  int64_t dilationH = 1, dilationW = 1;
  int64_t paddingH = 0, paddingW = 0;
};

// Serialized attention problem signature (mirrors ATTENTION_COLUMNS).
struct AttentionSig {
  StringRef arch;
  StringRef dtype;
  int64_t numCU = 64;
  int64_t numChiplets = 1;
  bool transQ = false, transK = false, transV = false, transO = false;
  bool causal = false, returnLSE = false;
  int64_t splitKV = 1;
  bool withAttnScale = false, withAttnBias = false;
  int64_t g = 1, numHeadsQ = 1, numHeadsKV = 1;
  int64_t seqLenQ = 0, seqLenK = 0, headDimQK = 0, headDimV = 0;
};

// The dtype string used by the feature pipeline for an element type. Matches
// the trainer's data-type keys (and QuickTuningDb's suffixes).
StringRef dtypeString(Type elementType);

// Canonical per-op feature order, matching features.py feature_record and the
// committed <arch>_<op>_features.txt. Pinned; the unittest guards drift.
ArrayRef<StringRef> gemmFeatureNames();
ArrayRef<StringRef> convFeatureNames();
ArrayRef<StringRef> attentionFeatureNames();

// Appends the feature vector for one (problem, perfConfig) to `out`, in
// canonical order. The appended count equals the matching
// *FeatureNames().size().
void gemmFeatures(const GemmSig &sig, StringRef perfConfig,
                  SmallVectorImpl<double> &out);
void convFeatures(const ConvSig &sig, StringRef perfConfig,
                  SmallVectorImpl<double> &out);
void attentionFeatures(const AttentionSig &sig, StringRef perfConfig,
                       SmallVectorImpl<double> &out);

} // namespace SmartTuningFeatures
} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_SMART_TUNING_FEATURES_H
