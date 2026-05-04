# `migraphx.attention` and the preSoftmaxBody contract

This is a developer-facing companion to the canonical op definition in
[`MIGraphX.td`][td]. It collects the cross-cutting contracts that several
files (verifier, host decompose, GPU lowering, C API, `rocmlir-gen`) all
have to agree on. New contributors should read this before adding a new
elementwise op to the body or changing how either lowering path consumes
the op.

[td]: ../../include/mlir/Dialect/MIGraphX/IR/MIGraphX.td

## What `migraphx.attention` represents

`migraphx.attention` is the high-level MIGraphX representation of scaled
dot-product attention. The operation captures the algebra

```
result = softmax(preSoftmaxBody(Q * K^T, extras...)) * V
```

with optional features `kvcache`, `causal`, `prefix_offset`,
`sliding_window`, and `splitkv` (flash decoding). It is the producer-side
contract that MIGraphX uses when lowering its own attention nodes; the
two consumers in this repo are the host decompose
(`migraphx-transform`'s `AttentionDecompose`) and the GPU lowering
(`migraphx-attention-to-rock`).

See the operation description in [`MIGraphX.td`][td] for the full
operand/attribute list and the per-feature semantics. This document only
covers the contracts that span more than one file.

## preSoftmaxBody contract

The `preSoftmaxBody` region is a single-block region that performs the
elementwise fusion between the first GEMM (`Q*K^T`) and the softmax. Its
contract has four parts:

1. **Block argument layout.** The block's first argument is always the
   `Q*K^T` result. Block arguments `1..N` are the `preSoftmaxElemWiseInputs`
   in declaration order. Both `AttentionOp::verify` and
   `AttentionDecompose` rely on this order; the host decompose maps
   `block.getArgument(0) -> qk` and `block.getArgument(i+1) ->
   preSoftmaxElemWiseInputs[i]`.

2. **Allowed ops.** The body may only contain ops listed in
   [`AttentionUtils.h::isAllowedInPreSoftmaxBody`][utils] plus the
   terminator (`migraphx.yield`). Adding an op anywhere else (e.g.
   directly in the GPU body builder) will be rejected by the verifier.

3. **Yield value.** The body must terminate with `migraphx.yield <value>`
   where `<value>` has the same shape as the QK output. The element type
   of the yield is free: callers can use it to widen QK to the chosen
   `softmaxType`, mask out positions with `-inf`, etc. The empty-body
   form (`migraphx.yield` with no operand) is reserved for "no
   elementwise fusion needed", which requires zero
   `preSoftmaxElemWiseInputs` and float `Q`.

4. **Lock-step rule.** [`isAllowedInPreSoftmaxBody`][utils] and
   [`MIGraphXAttentionToRock::lowerMIGraphXElementwiseToScalar`][lower]
   are the single source of truth for what a body op means. The verifier
   asks only "is this op allowed?", but the GPU lowering asserts at
   runtime that every allowed op has a scalar lowering. Adding a new
   body op therefore requires touching both: the allowlist (so the
   verifier accepts it) and the dispatch table in
   `lowerMIGraphXElementwiseToScalar` (so the GPU lowering knows how to
   emit it). Tests live in
   `test/Conversion/MIGraphXAttentionToRock/attention-to-rock.mlir` and
   `test/Conversion/MIGraphXAttentionDecompose/attention-decompose.mlir`.

[utils]: ../../include/mlir/Dialect/MIGraphX/IR/AttentionUtils.h
[lower]: ../../lib/Conversion/MIGraphXAttentionToRock/MIGraphXAttentionToRock.cpp

### Quantized attention: the dequant-in-body rule

When `Q` and `K` are integer typed (`i8`), the first GEMM is lowered to
`migraphx.quant_dot`, whose output is `i32`. softmax is a float-only
operation, so something has to bridge `i32 -> float`. The body is the
only place where the user's quantization scale and zero point are
visible, so the verifier requires:

> Integer `Q` requires a non-empty `preSoftmaxBody` that dequantizes the
> `i32` QK output to a float type (e.g. with
> `migraphx.dequantizelinear`); `softmaxType` alone does not synthesize
> a scale.

`softmaxType` only chooses *which* float type the softmax runs in; it
does not invent a dequantize. A bare `migraphx.convert i32 -> f32`
would be a raw bit-width cast that feeds enormous accumulator values to
softmax and produces effectively-one-hot garbage, which is why the
empty-body case is rejected for integer `Q`.

Producers that want quantized attention must therefore emit at least one
body op that reads the `i32` QK and yields a float (typically a
`migraphx.dequantizelinear` followed by any masking / scaling). The
existing E2E tests in
`test/fusion/pr-e2e/migraphx-attention/mixr-attention-first-gemm-i8-*.mlir`
show the canonical shape.

## Lowering polarity

`migraphx.attention` has two consumers in the kernel pipeline, and they
use opposite `rock.kernel` polarity guards so they can run in the same
pipeline without stepping on each other:

| Function attribute | Pass that handles `migraphx.attention`     | Result                                       |
|--------------------|--------------------------------------------|----------------------------------------------|
| `rock.kernel`      | `migraphx-attention-to-rock` (GPU path)    | One `rock.attention` op (kernel generator).  |
| no `rock.kernel`   | `migraphx-transform` / `AttentionDecompose`| Decomposed to primitive `migraphx` ops.      |

The pipeline polarity test
`test/Conversion/MIGraphXAttentionToRock/attention-pipeline-polarity.mlir`
locks in this contract end-to-end. Anything that adds a third path (or
removes one of the two guards) must update both that test and the
matching guard in the other pass.

## perf_config forwarding

`rock.attention` accepts a `perf_config` string attribute that
`tuningRunner.py` (with `--operation attention`) and the kernel
generator both consume. `migraphx.attention` carries `perf_config` as a
discardable string attribute, and `MIGraphXAttentionToRock` copies it
straight onto the produced `rock.attention` op. This means high-level
tuning hints attached to a `migraphx.attention` reach the kernel
generator unchanged. The forwarding is verified by the
`attention_with_perf_config` lit test in `attention-to-rock.mlir`.

## Adding a new feature flag

`features` is a bit-flag enum (`MIXR_AttentionFeaturesAttr`) that
selects attention variants (causal / kvcache / sliding window / splitkv
/ prefix offset). Adding a new feature requires touching:

1. The enum definition in [`MIGraphX.td`][td].
2. The verifier's feature/operand/attribute pairing rules in
   `lib/Dialect/MIGraphX/IR/MIGraphX.cpp` (so missing operands /
   orphan operands / orphan attributes are rejected at op-construction
   time).
3. The C API constructor in
   `lib/CAPI/Dialect/MIGraphX.cpp` (so the same orphan-attribute /
   orphan-operand checks are also applied at the API boundary, which
   gives clearer diagnostics than the post-construction verifier).
4. Both lowerings: host (`MIGraphXTransform`'s `AttentionDecompose`)
   and GPU (`MIGraphXAttentionToRock`).
5. Positive and negative lit tests in
   `test/Dialect/MIGraphX/{ops.mlir,invalid.mlir}` plus a clone-verifier
   E2E test in `test/fusion/pr-e2e/migraphx-attention/`.

The verifier helper
`AttentionOp::verify` in `MIGraphX.cpp` already has the "feature ↔
attribute" and "feature ↔ operand" rejection patterns laid out as
explicit branches; new features should follow the same shape so the
diagnostics stay consistent.
