//===- SmartTuningFeatures.cpp - Feature extraction for smart tuning -----===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// C++ mirror of tuning_eval/features.py (GEMM path). Every value here must
// match the Python pipeline the model was trained on; the unittest pins both
// the feature order and the arithmetic against goldens.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/SmartTuningFeatures.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"

#include "llvm/ADT/StringSwitch.h"

#include <algorithm>
#include <cmath>
#include <numeric>

using namespace mlir;
using namespace mlir::rock;

namespace {

// Bytes per element, keyed by the trainer's dtype strings. Default 2.0 mirrors
// features.py's _DTYPE_BYTES.get(dtype, 2.0).
double dtypeBytes(StringRef dtype) {
  return llvm::StringSwitch<double>(dtype)
      .Case("f32", 4.0)
      .Case("f16", 2.0)
      .Case("bf16", 2.0)
      .Case("i8", 1.0)
      .Case("fp8", 1.0)
      .Case("bf8", 1.0)
      .Case("fp4", 0.5)
      .Case("f4E2M1FN", 0.5)
      .Default(2.0);
}

// log2(value), or 0.0 for value <= 0 (features.py _lg / math.log2 guard).
double lg(double value) { return value > 0.0 ? std::log2(value) : 0.0; }

// ceil(a / b), or 0.0 for b <= 0 (features.py _ceil_div).
double ceilDiv(double a, double b) { return b > 0.0 ? std::ceil(a / b) : 0.0; }

// Parse one perfConfig param to int; -1 if non-numeric (features.py _safe_int).
int64_t paramInt(StringRef p) {
  int64_t v;
  if (p.trim().getAsInteger(10, v))
    return -1;
  return v;
}

// A parsed perfConfig: optional format tag, version, and the raw int params.
// Mirrors quickTuningGen.parse_perfconfig.
struct ParsedConfig {
  bool hasFormat = false;
  StringRef format;
  int version = 1;
  SmallVector<int64_t> params;
};

ParsedConfig parsePerfConfig(StringRef perfConfig) {
  SmallVector<StringRef> parts;
  perfConfig.split(parts, ':');
  ParsedConfig out;
  auto splitParams = [&](StringRef s) {
    SmallVector<StringRef> raw;
    s.split(raw, ',');
    for (StringRef r : raw)
      out.params.push_back(paramInt(r));
  };
  auto parseVersion = [](StringRef s) {
    int v;
    if (s.drop_front(1).getAsInteger(10, v))
      return 1;
    return v;
  };
  if (parts.size() == 3) {
    out.hasFormat = true;
    out.format = parts[0];
    out.version = parseVersion(parts[1]);
    splitParams(parts[2]);
  } else if (parts.size() == 2) {
    if (parts[0].starts_with("v")) {
      out.version = parseVersion(parts[0]);
      splitParams(parts[1]);
    } else {
      out.hasFormat = true;
      out.format = parts[0];
      out.version = 1;
      splitParams(parts[1]);
    }
  } else {
    splitParams(perfConfig);
  }
  return out;
}

// Index of the Split-K param for a parsed config, or -1 (quickTuningGen
// get_splitk_value). kPack sits immediately before it (get_kpack_value).
int splitKIndex(const ParsedConfig &c) {
  if (c.hasFormat && c.format == "attn")
    return c.version >= 3 ? 8 : (c.version >= 2 ? 7 : -1);
  return c.version >= 4 ? 7 : (c.version >= 2 ? 6 : -1);
}
int kpackIndex(const ParsedConfig &c) {
  int idx = splitKIndex(c);
  return idx < 0 ? -1 : idx - 1;
}

int64_t paramAt(const ParsedConfig &c, int idx) {
  if (idx >= 0 && idx < static_cast<int>(c.params.size()))
    return c.params[idx];
  return -1;
}

// Offsets of (mPerBlock, nPerBlock, kpackPerBlock, mPerWave) within a parsed
// perfConfig, per op. Mirrors features.py _CONFIG_FIELD_POS: gemm/conv list the
// tile fields first; attention interleaves the two GEMMs' tiles.
struct FieldPos {
  int mPerBlock, nPerBlock, kpackPerBlock, mPerWave;
};
constexpr FieldPos kGemmPos{0, 1, 2, 3};
constexpr FieldPos kConvPos{0, 1, 2, 3};
constexpr FieldPos kAttnPos{0, 2, 3, 4};

// Named + generic perfConfig features (features.py parse_config).
struct ConfigFeatures {
  double mPerBlock, nPerBlock, kpackPerBlock, mPerWave, kpack, splitK;
  double generic[14];
};

ConfigFeatures parseConfigFeatures(StringRef perfConfig, FieldPos pos) {
  ParsedConfig c = parsePerfConfig(perfConfig);
  ConfigFeatures f;
  f.mPerBlock = paramAt(c, pos.mPerBlock);
  f.nPerBlock = paramAt(c, pos.nPerBlock);
  f.kpackPerBlock = paramAt(c, pos.kpackPerBlock);
  f.mPerWave = paramAt(c, pos.mPerWave);

  int64_t kpack = paramAt(c, kpackIndex(c));
  f.kpack = kpack ? static_cast<double>(kpack) : 1.0;
  int skIdx = splitKIndex(c);
  bool hasSplitK = skIdx >= 0 && skIdx < static_cast<int>(c.params.size());
  f.splitK = hasSplitK ? static_cast<double>(c.params[skIdx]) : 1.0;

  for (int i = 0; i < 14; ++i)
    f.generic[i] = paramAt(c, i);
  return f;
}

// Python floor division (rounds toward -inf), matching the `//` operator the
// conv lowering math relies on.
int64_t pyFloorDiv(int64_t a, int64_t b) {
  int64_t q = a / b, r = a % b;
  if (r != 0 && ((r < 0) != (b < 0)))
    --q;
  return q;
}

// ceil(a / b) == -((-a) // b) for b > 0; 0 otherwise (features.py _idiv_ceil).
int64_t idivCeil(int64_t a, int64_t b) {
  return b > 0 ? -pyFloorDiv(-a, b) : 0;
}

// features.py _conv_out_spatial.
void convOutSpatial(int64_t h, int64_t w, int64_t y, int64_t x, int64_t sh,
                    int64_t sw, int64_t dh, int64_t dw, int64_t ph, int64_t pw,
                    int64_t &ho, int64_t &wo) {
  ho = sh > 0 ? pyFloorDiv(h + 2 * ph - dh * (y - 1) - 1, sh) + 1 : 0;
  wo = sw > 0 ? pyFloorDiv(w + 2 * pw - dw * (x - 1) - 1, sw) + 1 : 0;
  ho = std::max<int64_t>(ho, 0);
  wo = std::max<int64_t>(wo, 0);
}

// Implicit-GEMM (M, N, K, G) of a conv, direction-aware (features.py
// conv_implicit_gemm / conv_bwd_data_gemm). For backward-data the G slot
// carries the sub-GEMM (kernel) count so the grid feature reflects all kernels.
struct ImplicitGemm {
  double m, n, k, g;
};

ImplicitGemm convImplicitGemm(const SmartTuningFeatures::ConvSig &s) {
  int64_t n = s.n, c = s.c, k = s.k, h = s.h, w = s.w, y = s.y, x = s.x;
  int64_t sh = s.strideH, sw = s.strideW, dh = s.dilationH, dw = s.dilationW;
  int64_t ph = s.paddingH, pw = s.paddingW;
  int64_t g = 1; // conv corpora carry no G column; features.py defaults to 1.
  int64_t ho, wo;
  convOutSpatial(h, w, y, x, sh, sw, dh, dw, ph, pw, ho, wo);

  if (s.direction == "bwd") {
    int64_t filTildaY = sh > 0 ? sh / std::gcd(sh, dh) : 1;
    int64_t filTildaX = sw > 0 ? sw / std::gcd(sw, dw) : 1;
    int64_t outTildaY = ho + idivCeil(dh * (y - 1), sh);
    int64_t outTildaX = wo + idivCeil(dw * (x - 1), sw);
    int64_t leftY =
        sh > 0 ? pyFloorDiv(std::max<int64_t>(0, ph - dh * (filTildaY - 1)), sh)
               : 0;
    int64_t leftX =
        sw > 0 ? pyFloorDiv(std::max<int64_t>(0, pw - dw * (filTildaX - 1)), sw)
               : 0;
    int64_t rightY = std::min(outTildaY, idivCeil(ph + h - 1, sh) + 1);
    int64_t rightX = std::min(outTildaX, idivCeil(pw + w - 1, sw) + 1);
    int64_t sliceY = std::max<int64_t>(0, rightY - leftY);
    int64_t sliceX = std::max<int64_t>(0, rightX - leftX);

    int64_t nGemm = n * sliceY * sliceX;
    int64_t numKernels = 0, kRep = 0;
    for (int64_t kid = 0; kid < filTildaY * filTildaX; ++kid) {
      int64_t itildaY = kid / filTildaX;
      int64_t itildaX = kid % filTildaX;
      int64_t dotY = idivCeil(y - itildaY, filTildaY);
      int64_t dotX = idivCeil(x - itildaX, filTildaX);
      if (dotY > 0 && dotX > 0) {
        ++numKernels;
        if (kid == 0)
          kRep = k * dotY * dotX;
      }
    }
    double kGemm = kRep > 0 ? kRep : k;
    return {static_cast<double>(c), static_cast<double>(nGemm), kGemm,
            static_cast<double>(g * std::max<int64_t>(numKernels, 1))};
  }
  if (s.direction == "wrw")
    return {static_cast<double>(k), static_cast<double>(c * y * x),
            static_cast<double>(n * ho * wo), static_cast<double>(g)};
  // fwd
  return {static_cast<double>(k), static_cast<double>(n * ho * wo),
          static_cast<double>(c * y * x), static_cast<double>(g)};
}

// Layout dims n,g,k,c,h(0),w(1): position of each char in the layout string, or
// -1 (features.py _layout_positions). Appends six positions in this order.
void appendLayoutPositions(SmallVectorImpl<double> &out, StringRef layout) {
  static const char kDims[] = {'n', 'g', 'k', 'c', '0', '1'};
  for (char ch : kDims) {
    size_t pos = layout.find(ch);
    out.push_back(pos == StringRef::npos ? -1.0 : static_cast<double>(pos));
  }
}

// Appends the op-independent feature tail in feature_record order: dtype (3),
// topology (2), arch_hw (8), parse_config (20), interaction (13) = 46 values.
// `m`/`n`/`k`/`g` are the implicit-GEMM dims the interaction features use.
void appendTail(SmallVectorImpl<double> &out, double m, double n, double k,
                double g, StringRef dtype, int64_t numCU, int64_t numChiplets,
                StringRef arch, StringRef perfConfig, FieldPos pos) {
  const double dbytes = dtypeBytes(dtype);

  // -- dtype_features --
  out.push_back(dbytes * 8.0);
  out.push_back(dtype.starts_with("i") ? 0.0 : 1.0);
  out.push_back((dtype == "fp4" || dtype == "f4E2M1FN") ? 1.0 : 0.0);

  // -- topology --
  out.push_back(static_cast<double>(numCU ? numCU : 64));
  out.push_back(static_cast<double>(numChiplets ? numChiplets : 1));

  // -- arch_hw (sourced from the same DB the trainer used) --
  AmdArchInfo info = lookupArchInfo(arch);
  out.push_back(static_cast<double>(info.waveSize));
  out.push_back(
      bitEnumContainsAll(info.defaultFeatures, GemmFeatures::mfma) ? 1.0 : 0.0);
  out.push_back(
      bitEnumContainsAll(info.defaultFeatures, GemmFeatures::wmma) ? 1.0 : 0.0);
  double ldsPerWg = static_cast<double>(info.maxSharedMemPerWG);
  double ldsPerCu = static_cast<double>(info.totalSharedMemPerCU);
  out.push_back(ldsPerWg);
  out.push_back(ldsPerCu);
  out.push_back(static_cast<double>(info.totalVGPRPerEU));
  out.push_back(static_cast<double>(info.maxWavesPerEU));
  out.push_back(static_cast<double>(info.numEUPerCU));

  // -- parse_config --
  ConfigFeatures cfg = parseConfigFeatures(perfConfig, pos);
  out.push_back(cfg.mPerBlock);
  out.push_back(cfg.nPerBlock);
  out.push_back(cfg.kpackPerBlock);
  out.push_back(cfg.mPerWave);
  out.push_back(cfg.kpack);
  out.push_back(cfg.splitK);
  for (int i = 0; i < 14; ++i)
    out.push_back(cfg.generic[i]);

  // -- interaction_features --
  double mPerBlock = cfg.mPerBlock > 0.0 ? cfg.mPerBlock : 1.0;
  double nPerBlock = cfg.nPerBlock > 0.0 ? cfg.nPerBlock : 1.0;
  double kpackPerBlock = cfg.kpackPerBlock > 0.0 ? cfg.kpackPerBlock : 1.0;
  double kpack = cfg.kpack > 0.0 ? cfg.kpack : 1.0;
  double kPerBlock = kpackPerBlock * kpack;
  double splitK = cfg.splitK > 0.0 ? cfg.splitK : 1.0;

  double mTiles = ceilDiv(m, mPerBlock);
  double nTiles = ceilDiv(n, nPerBlock);
  double kTiles = ceilDiv(k, kPerBlock);
  double totalWg = g * mTiles * nTiles * splitK;
  double numCu = static_cast<double>(numCU ? numCU : 64);

  double workImbalance = 1.0;
  if (totalWg > 0.0 && numCu > 0.0) {
    double maxPerCu = std::ceil(totalWg / numCu);
    workImbalance = (maxPerCu * numCu) / totalWg;
  }

  double tileLdsBytes = (mPerBlock + nPerBlock) * kPerBlock * dbytes;
  double ldsFraction = 0.0, ldsBlocksPerCu = 0.0;
  if (tileLdsBytes > 0.0) {
    ldsFraction = ldsPerWg > 0.0 ? tileLdsBytes / ldsPerWg : 0.0;
    ldsBlocksPerCu = ldsPerCu > 0.0 ? std::floor(ldsPerCu / tileLdsBytes) : 0.0;
  }

  auto divisible = [](double dim, double tile) {
    int64_t t = static_cast<int64_t>(tile);
    return (t != 0 && static_cast<int64_t>(dim) % t == 0) ? 1.0 : 0.0;
  };

  out.push_back(mTiles);
  out.push_back(nTiles);
  out.push_back(kTiles);
  out.push_back(totalWg);
  out.push_back(numCu > 0.0 ? totalWg / numCu : 0.0);
  out.push_back(workImbalance);
  out.push_back(divisible(m, mPerBlock));
  out.push_back(divisible(n, nPerBlock));
  out.push_back(mTiles * mPerBlock - m);
  out.push_back(nTiles * nPerBlock - n);
  out.push_back(tileLdsBytes);
  out.push_back(ldsFraction);
  out.push_back(ldsBlocksPerCu);
}

// Shared tail feature names (features.py: dtype, topology, arch_hw,
// parse_config, interaction), identical across ops. Appends 46 names.
void appendTailNames(SmallVectorImpl<StringRef> &names) {
  static const StringRef kTail[] = {
      // dtype_features
      "dtype_bits", "dtype_is_float", "dtype_is_scaled",
      // topology
      "num_cu", "num_chiplets",
      // arch_hw
      "wave_size", "is_mfma", "is_wmma", "lds_bytes_per_wg", "lds_bytes_per_cu",
      "vgpr_per_eu", "waves_per_eu", "eu_per_cu",
      // parse_config (named + generic)
      "cfg_m_per_block", "cfg_n_per_block", "cfg_kpack_per_block",
      "cfg_m_per_wave", "cfg_kpack", "cfg_split_k", "cfg_p0", "cfg_p1",
      "cfg_p2", "cfg_p3", "cfg_p4", "cfg_p5", "cfg_p6", "cfg_p7", "cfg_p8",
      "cfg_p9", "cfg_p10", "cfg_p11", "cfg_p12", "cfg_p13",
      // interaction_features
      "m_tiles", "n_tiles", "k_tiles", "grid", "grid_per_cu", "work_imbalance",
      "m_divisible", "n_divisible", "m_pad_waste", "n_pad_waste",
      "tile_lds_bytes", "lds_fraction", "lds_blocks_per_cu"};
  names.append(std::begin(kTail), std::end(kTail));
}

} // namespace

StringRef SmartTuningFeatures::dtypeString(Type t) {
  if (t.isBF16())
    return "bf16";
  if (t.isFloat()) {
    switch (t.getIntOrFloatBitWidth()) {
    case 4:
      return "fp4";
    case 8:
      return "fp8";
    case 16:
      return "f16";
    case 32:
      return "f32";
    default:
      return "f32";
    }
  }
  if (t.isInteger() && t.getIntOrFloatBitWidth() == 8)
    return "i8";
  return "f32";
}

ArrayRef<StringRef> SmartTuningFeatures::gemmFeatureNames() {
  // Canonical order: must match features.py feature_record for gemm and the
  // committed <arch>_gemm_features.txt.
  static const SmallVector<StringRef> kNames = [] {
    SmallVector<StringRef> n = {
        // gemm_problem_features
        "trans_a",   "trans_b",   "g",     "m",         "n",
        "k",         "log_m",     "log_n", "log_k",     "log_g",
        "aspect_mn", "aspect_mk", "flops", "log_flops", "arith_intensity"};
    appendTailNames(n);
    return n;
  }();
  return kNames;
}

ArrayRef<StringRef> SmartTuningFeatures::convFeatureNames() {
  static const SmallVector<StringRef> kNames = [] {
    SmallVector<StringRef> n = {
        // conv_problem_features
        "is_fwd", "is_bwd", "is_wrw", "n", "c", "h", "w", "k", "y", "x",
        "log_n", "log_c", "log_h", "log_w", "log_k", "stride_h", "stride_w",
        "dil_h", "dil_w", "pad_h", "pad_w", "ho", "wo", "log_ho", "log_wo",
        "filter_area", "gemm_m", "gemm_n", "gemm_k", "log_gemm_m", "log_gemm_n",
        "log_gemm_k", "flops", "log_flops", "arith_intensity",
        // _layout_positions (fil, in, out)
        "fil_pos_n", "fil_pos_g", "fil_pos_k", "fil_pos_c", "fil_pos_h",
        "fil_pos_w", "in_pos_n", "in_pos_g", "in_pos_k", "in_pos_c", "in_pos_h",
        "in_pos_w", "out_pos_n", "out_pos_g", "out_pos_k", "out_pos_c",
        "out_pos_h", "out_pos_w"};
    appendTailNames(n);
    return n;
  }();
  return kNames;
}

ArrayRef<StringRef> SmartTuningFeatures::attentionFeatureNames() {
  static const SmallVector<StringRef> kNames = [] {
    SmallVector<StringRef> n = {
        // attention_problem_features
        "trans_q",        "trans_k",    "trans_v",     "trans_o",
        "causal",         "return_lse", "split_kv",    "with_attn_scale",
        "with_attn_bias", "g",          "num_heads_q", "num_heads_kv",
        "gqa_ratio",      "batch_q",    "seq_len_q",   "seq_len_k",
        "head_dim_qk",    "head_dim_v", "log_seq_q",   "log_seq_k",
        "log_head_qk",    "log_head_v", "log_batch_q", "seq_ratio",
        "is_square_seq",  "flops",      "log_flops",   "arith_intensity"};
    appendTailNames(n);
    return n;
  }();
  return kNames;
}

void SmartTuningFeatures::gemmFeatures(const GemmSig &sig, StringRef perfConfig,
                                       SmallVectorImpl<double> &out) {
  const double g = sig.g, m = sig.m, k = sig.k, n = sig.n;
  const double dbytes = dtypeBytes(sig.dtype);

  // -- gemm_problem_features --
  double flops = 2.0 * g * m * k * n;
  double bytesMoved = dbytes * (g * (m * k + k * n + m * n));
  double arithIntensity = bytesMoved > 0.0 ? flops / bytesMoved : 0.0;
  out.push_back(sig.transA ? 1.0 : 0.0);
  out.push_back(sig.transB ? 1.0 : 0.0);
  out.push_back(g);
  out.push_back(m);
  out.push_back(n);
  out.push_back(k);
  out.push_back(lg(m));
  out.push_back(lg(n));
  out.push_back(lg(k));
  out.push_back(lg(g));
  out.push_back(n > 0.0 ? m / n : 0.0);
  out.push_back(k > 0.0 ? m / k : 0.0);
  out.push_back(flops);
  out.push_back(lg(flops));
  out.push_back(arithIntensity);

  appendTail(out, m, n, k, g, sig.dtype, sig.numCU, sig.numChiplets, sig.arch,
             perfConfig, kGemmPos);
}

void SmartTuningFeatures::convFeatures(const ConvSig &sig, StringRef perfConfig,
                                       SmallVectorImpl<double> &out) {
  const double dbytes = dtypeBytes(sig.dtype);
  int64_t ho, wo;
  convOutSpatial(sig.h, sig.w, sig.y, sig.x, sig.strideH, sig.strideW,
                 sig.dilationH, sig.dilationW, sig.paddingH, sig.paddingW, ho,
                 wo);
  ImplicitGemm ig = convImplicitGemm(sig);

  double flops = 2.0 * ig.m * ig.n * ig.k;
  double bytesMoved = dbytes * (double(sig.n) * sig.c * sig.h * sig.w +
                                double(sig.k) * sig.c * sig.y * sig.x +
                                double(sig.n) * sig.k * ho * wo);
  double arithIntensity = bytesMoved > 0.0 ? flops / bytesMoved : 0.0;

  // -- conv_problem_features --
  out.push_back(sig.direction == "fwd" ? 1.0 : 0.0);
  out.push_back(sig.direction == "bwd" ? 1.0 : 0.0);
  out.push_back(sig.direction == "wrw" ? 1.0 : 0.0);
  out.push_back(static_cast<double>(sig.n));
  out.push_back(static_cast<double>(sig.c));
  out.push_back(static_cast<double>(sig.h));
  out.push_back(static_cast<double>(sig.w));
  out.push_back(static_cast<double>(sig.k));
  out.push_back(static_cast<double>(sig.y));
  out.push_back(static_cast<double>(sig.x));
  out.push_back(lg(sig.n));
  out.push_back(lg(sig.c));
  out.push_back(lg(sig.h));
  out.push_back(lg(sig.w));
  out.push_back(lg(sig.k));
  out.push_back(static_cast<double>(sig.strideH));
  out.push_back(static_cast<double>(sig.strideW));
  out.push_back(static_cast<double>(sig.dilationH));
  out.push_back(static_cast<double>(sig.dilationW));
  out.push_back(static_cast<double>(sig.paddingH));
  out.push_back(static_cast<double>(sig.paddingW));
  out.push_back(static_cast<double>(ho));
  out.push_back(static_cast<double>(wo));
  out.push_back(lg(ho));
  out.push_back(lg(wo));
  out.push_back(static_cast<double>(sig.y * sig.x));
  out.push_back(ig.m);
  out.push_back(ig.n);
  out.push_back(ig.k);
  out.push_back(lg(ig.m));
  out.push_back(lg(ig.n));
  out.push_back(lg(ig.k));
  out.push_back(flops);
  out.push_back(lg(flops));
  out.push_back(arithIntensity);

  // -- _layout_positions (filter, input, output) --
  appendLayoutPositions(out, sig.filterLayout);
  appendLayoutPositions(out, sig.inputLayout);
  appendLayoutPositions(out, sig.outputLayout);

  appendTail(out, ig.m, ig.n, ig.k, ig.g, sig.dtype, sig.numCU, sig.numChiplets,
             sig.arch, perfConfig, kConvPos);
}

void SmartTuningFeatures::attentionFeatures(const AttentionSig &sig,
                                            StringRef perfConfig,
                                            SmallVectorImpl<double> &out) {
  const double dbytes = dtypeBytes(sig.dtype);
  double sq = sig.seqLenQ, sk = sig.seqLenK, dqk = sig.headDimQK,
         dv = sig.headDimV;
  double batchQ = double(sig.g) * sig.numHeadsQ;
  double batchKV = double(sig.g) * sig.numHeadsKV;
  double mask = sig.causal ? 0.5 : 1.0;
  double flops = 2.0 * batchQ * mask * (sq * sk * dqk + sq * sk * dv);
  double bytesMoved = dbytes * (batchQ * sq * dqk + batchKV * sk * dqk +
                                batchKV * sk * dv + batchQ * sq * dv);
  double arithIntensity = bytesMoved > 0.0 ? flops / bytesMoved : 0.0;

  // -- attention_problem_features --
  out.push_back(sig.transQ ? 1.0 : 0.0);
  out.push_back(sig.transK ? 1.0 : 0.0);
  out.push_back(sig.transV ? 1.0 : 0.0);
  out.push_back(sig.transO ? 1.0 : 0.0);
  out.push_back(sig.causal ? 1.0 : 0.0);
  out.push_back(sig.returnLSE ? 1.0 : 0.0);
  out.push_back(static_cast<double>(sig.splitKV));
  out.push_back(sig.withAttnScale ? 1.0 : 0.0);
  out.push_back(sig.withAttnBias ? 1.0 : 0.0);
  out.push_back(static_cast<double>(sig.g));
  out.push_back(static_cast<double>(sig.numHeadsQ));
  out.push_back(static_cast<double>(sig.numHeadsKV));
  out.push_back(sig.numHeadsKV > 0 ? double(sig.numHeadsQ) / sig.numHeadsKV
                                   : 1.0);
  out.push_back(batchQ);
  out.push_back(sq);
  out.push_back(sk);
  out.push_back(dqk);
  out.push_back(dv);
  out.push_back(lg(sq));
  out.push_back(lg(sk));
  out.push_back(lg(dqk));
  out.push_back(lg(dv));
  out.push_back(lg(batchQ));
  out.push_back(sk > 0.0 ? sq / sk : 1.0);
  out.push_back(sq == sk ? 1.0 : 0.0);
  out.push_back(flops);
  out.push_back(lg(flops));
  out.push_back(arithIntensity);

  appendTail(out, sq, sk, dqk, batchQ, sig.dtype, sig.numCU, sig.numChiplets,
             sig.arch, perfConfig, kAttnPos);
}
