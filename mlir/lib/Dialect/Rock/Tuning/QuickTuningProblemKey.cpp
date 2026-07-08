//===- QuickTuningProblemKey.cpp - MLIR op -> tuning problem key ---------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements QuickTuningDb::computeProblemKeyHash. Each formatter mirrors
// quickTuningGen.py:make_problem_key byte-for-byte: fields joined by '_',
// booleans as 0/1, integers as plain decimal. Column lists are pinned to
// the *_COLUMNS in the generator; the unit/pytest goldens guard drift.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/QuickTuningDb.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

using namespace mlir;
using namespace mlir::rock;

namespace {

template <typename T>
void streamKeyField(llvm::raw_ostream &os, const T &v) {
  if constexpr (std::is_same_v<T, bool>)
    os << static_cast<int>(v);
  else
    os << v;
}

template <typename... Ts>
std::string joinKey(const Ts &...vs) {
  std::string out;
  llvm::raw_string_ostream os(out);
  size_t i = 0;
  ((os << (i++ ? "_" : ""), streamKeyField(os, vs)), ...);
  return out;
}

// Reverses the filter/input/output layout encoding: drops the
// 'g'/'gi'/'go' axis (the table column has no group dim) and remaps '0'
// and '1' to `zeroChar`/`oneChar`.
std::string layoutFromArrayAttr(ArrayAttr layoutAttr, char zeroChar,
                                char oneChar) {
  std::string result;
  result.reserve(layoutAttr.size());
  for (Attribute a : layoutAttr) {
    char c = cast<StringAttr>(a).getValue()[0];
    if (c == 'g')
      continue;
    if (c == '0')
      c = zeroChar;
    else if (c == '1')
      c = oneChar;
    result.push_back(c);
  }
  return result;
}

// GEMM_COLUMNS = [TransA, TransB, G, M, K, N]
std::string makeGemmKey(rock::GemmOp op) {
  GemmSize size = op.getGemmSize();
  return joinKey(op.getATransposed(), op.getBTransposed(), size.g, size.m,
                 size.k, size.n);
}

// CONV_COLUMNS = [Direction, FilterLayout, InputLayout, OutputLayout, N, C,
// H, W, K, Y, X, DilationH, DilationW, StrideH, StrideW, PaddingH,
// PaddingW]
//
// Returns an empty string when the op is not a 2D conv.
std::string makeConvKey(RockConvInterface convIF, KernelType opType) {
  StringRef direction;
  switch (opType) {
  case KernelType::Conv:
    direction = "fwd";
    break;
  case KernelType::ConvBwdData:
    direction = "bwd";
    break;
  case KernelType::ConvBwdWeight:
    direction = "wrw";
    break;
  default:
    return {};
  }

  Operation *op = convIF.getOperation();
  auto fLayout = op->getAttrOfType<ArrayAttr>("filter_layout");
  auto iLayout = op->getAttrOfType<ArrayAttr>("input_layout");
  auto oLayout = op->getAttrOfType<ArrayAttr>("output_layout");
  if (!fLayout || !iLayout || !oLayout)
    return {};

  // Build name -> shape-index maps for the canonical 2D layout.
  llvm::StringMap<unsigned> fMap, iMap;
  for (auto [i, a] : llvm::enumerate(fLayout))
    fMap[cast<StringAttr>(a).getValue()] = i;
  for (auto [i, a] : llvm::enumerate(iLayout))
    iMap[cast<StringAttr>(a).getValue()] = i;

  auto find = [](const llvm::StringMap<unsigned> &m,
                 StringRef k) -> std::optional<unsigned> {
    auto it = m.find(k);
    return it == m.end() ? std::nullopt : std::optional<unsigned>(it->second);
  };
  auto ni = find(iMap, "ni"), ci = find(iMap, "ci"), gi = find(iMap, "gi");
  auto hi = find(iMap, "0i"), wi = find(iMap, "1i");
  auto kI = find(fMap, "k"), gI = find(fMap, "g");
  auto y = find(fMap, "0"), x = find(fMap, "1");
  if (!ni || !ci || !gi || !hi || !wi || !kI || !gI || !y || !x)
    return {};

  auto pad = extractFromIntegerArrayAttr<int64_t>(convIF.getPadding());
  auto stride = extractFromIntegerArrayAttr<int64_t>(convIF.getStrides());
  auto dilate = extractFromIntegerArrayAttr<int64_t>(convIF.getDilations());
  if (pad.size() < 4 || stride.size() < 2 || dilate.size() < 2)
    return {};

  ArrayRef<int64_t> in =
      cast<ShapedType>(convIF.getInput().getType()).getShape();
  ArrayRef<int64_t> fil =
      cast<ShapedType>(convIF.getFilter().getType()).getShape();

  return joinKey(direction, layoutFromArrayAttr(fLayout, 'y', 'x'),
                 layoutFromArrayAttr(iLayout, 'h', 'w'),
                 layoutFromArrayAttr(oLayout, 'h', 'w'), in[*ni],
                 in[*ci] * in[*gi], in[*hi], in[*wi], fil[*kI] * fil[*gI],
                 fil[*y], fil[*x], dilate[0], dilate[1], stride[0], stride[1],
                 pad[0], pad[2]);
}

// ATTENTION_COLUMNS = [TransQ, TransK, TransV, TransO, Causal, ReturnLSE,
// SplitKV, WithAttnScale, WithAttnBias, G, SeqLenQ, SeqLenK, NumHeadsQ,
// NumHeadsKV, HeadDimQK, HeadDimV]
//
// WithAttnScale/Bias are inferred from preSoftmaxElemWiseInputs (scale, then
// bias; preceded by two quantization inputs for i8). Returns an empty string
// when shape metadata is missing.
std::string makeAttentionKey(rock::AttentionOp op) {
  auto qType = cast<ShapedType>(op.getQueries().getType());
  ArrayRef<int64_t> qShape = qType.getShape();
  ArrayRef<int64_t> kShape =
      cast<ShapedType>(op.getKeys().getType()).getShape();
  ArrayRef<int64_t> vShape =
      cast<ShapedType>(op.getValues().getType()).getShape();
  if (qShape.size() < 3 || kShape.size() < 3 || vShape.size() < 3)
    return {};
  int64_t numHeadsQ = op.getNumHeadsQ();
  if (numHeadsQ <= 0)
    return {};

  bool transQ = op.getQTransposed();
  bool transK = op.getKTransposed();
  bool transV = op.getVTransposed();

  unsigned quantPrefix = qType.getElementType().isInteger(8) ? 2u : 0u;
  unsigned numElemInputs = op.getPreSoftmaxElemWiseInputs().size();
  unsigned scaling =
      numElemInputs > quantPrefix ? numElemInputs - quantPrefix : 0u;

  return joinKey(transQ, transK, transV, op.getOTransposed(), op.getCausal(),
                 op.getLse() != nullptr, static_cast<int64_t>(op.getSplitKV()),
                 scaling >= 1, scaling >= 2, qShape[0] / numHeadsQ,
                 transQ ? qShape[2] : qShape[1], transK ? kShape[1] : kShape[2],
                 numHeadsQ, op.getNumHeadsKV(), transQ ? qShape[1] : qShape[2],
                 transV ? vShape[1] : vShape[2]);
}

// Returns the problem key string, or empty if `op` has no mapping.
std::string tryMakeProblemKey(Operation *op) {
  if (!op)
    return {};
  if (auto gemmOp = dyn_cast<rock::GemmOp>(op))
    return makeGemmKey(gemmOp);
  if (auto convIF = dyn_cast<RockConvInterface>(op))
    if (auto wrapper = dyn_cast<RockGemmWrapperInterface>(op))
      return makeConvKey(convIF, wrapper.getKernelType());
  if (auto attnOp = dyn_cast<rock::AttentionOp>(op))
    return makeAttentionKey(attnOp);
  return {};
}

} // namespace

FailureOr<uint64_t>
mlir::rock::QuickTuningDb::computeProblemKeyHash(Operation *op) {
  std::string key = tryMakeProblemKey(op);
  if (key.empty())
    return failure();
  return llvm::xxh3_64bits(key);
}
