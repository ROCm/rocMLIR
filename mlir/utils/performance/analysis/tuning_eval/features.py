# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Feature extraction and labeling for GEMM/conv/attention tuning problems.

Consumes only serialized data -- a ProblemSig and a perfConfig string -- so
the feature pipeline stays decoupled from the compiler. The config parser is
op-aware: it mirrors the field order of the matching attribute in
RockDialect.cpp (AccelGemmParamsAttr for gemm/conv, GemmGemmParamsAttr for
attention) for the leading tile fields (mPerBlock, nPerBlock, kpackPerBlock,
mPerWave) and uses quickTuningGen.get_splitk_value for split-K; remaining
params are exposed as generic, fixed-width slots so version drift cannot break
the vector shape. Resource-pressure features (LDS footprint/occupancy) are
normalized by per-arch capacity so the signal transfers across targets.
"""

import math
from collections import OrderedDict
from functools import lru_cache
from typing import Dict

# amd_arch_db (performance/) and quickTuningGen (analysis/) are put on sys.path
# by the package __init__.
from amd_arch_db import GemmFeatures, has_feature, lookup_arch_info

import quickTuningGen

from .corpus import ProblemSig

# Default coverage threshold; matches quickTuningGen's set-cover default so the
# harness label and the shipped DB speak the same "good enough" language.
DEFAULT_THRESHOLD = 0.93

# Fixed width for the generic cfg_p* slots: every perfConfig's params are
# padded/truncated to this many entries so feature vectors stay fixed-size even
# as newer perfConfig versions add fields.
_MAX_CONFIG_PARAMS = 14

# Bytes per element by dtype suffix (as keyed by quickTuningGen / the DB).
_DTYPE_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "i8": 1.0,
    "fp8": 1.0,
    "bf8": 1.0,
    "fp4": 0.5,
    "f4E2M1FN": 0.5,
}

_DEFAULT_NUM_CU = 64

# Default LDS capacities (64 KiB) used only as standalone defaults for
# interaction_features() when it is called without an arch context;
# feature_record always passes the authoritative per-arch capacities from
# amd_arch_db. The per-workgroup and per-CU caps are distinct quantities and
# only coincide for these fallback defaults.
_DEFAULT_LDS_BYTES_PER_WG = 64.0 * 1024.0
_DEFAULT_LDS_BYTES_PER_CU = 64.0 * 1024.0


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _col_int(sig: ProblemSig, name: str, default: int = 0) -> int:
    """Read an integer problem column, returning ``default`` if it is absent
    (e.g. conv corpora have no explicit group 'G' column)."""
    if name not in sig.column_names:
        return default
    return _safe_int(sig.column(name), default)


@lru_cache(maxsize=None)
def arch_hw(arch: str) -> Dict[str, float]:
    """Physical arch features so the model can transfer rather than only shard.

    Sourced from amd_arch_db (the same database the compiler uses) so resource
    capacities and accelerator support are authoritative and per-arch;
    resource-pressure interactions are then normalized by these so the signal
    transfers across targets. Cached per arch; callers must treat the returned
    dict as read-only (feature_record only reads from it)."""
    info = lookup_arch_info(arch.lower())
    df = info.default_features
    # max_shared_mem_per_wg bounds one workgroup's LDS (feasibility); the per-CU
    # total bounds how many workgroups co-reside (occupancy).
    return {
        "wave_size": float(info.wave_size),
        "is_mfma": float(bool(has_feature(df, GemmFeatures.MFMA))),
        "is_wmma": float(bool(has_feature(df, GemmFeatures.WMMA))),
        "lds_bytes_per_wg": float(info.max_shared_mem_per_wg),
        "lds_bytes_per_cu": float(info.total_shared_mem_per_cu),
        "vgpr_per_eu": float(info.total_vgpr_per_eu),
        "waves_per_eu": float(info.max_waves_per_eu),
        "eu_per_cu": float(info.num_eu_per_cu),
    }


def dtype_features(dtype: str) -> Dict[str, float]:
    bits = _DTYPE_BYTES.get(dtype, 2.0) * 8.0
    is_float = 0.0 if dtype.startswith("i") else 1.0
    is_scaled = 1.0 if dtype in ("fp4", "f4E2M1FN") else 0.0
    return {"dtype_bits": bits, "dtype_is_float": is_float, "dtype_is_scaled": is_scaled}


# Offsets of (mPerBlock, nPerBlock, kpackPerBlock, mPerWave) within a
# perfConfig's params, per op. gemm/conv accel configs list these tile fields
# first; attention interleaves the two GEMMs' tiles (mPerBlockG0, mPerBlockG1,
# nPerBlockG0, ...), so its QK^T tile sits at different offsets. This must match
# the dialect's perfConfig field order -- a wrong mapping silently feeds the
# interaction features bogus tile sizes.
_CONFIG_FIELD_POS = {
    "gemm": {
        "m_per_block": 0,
        "n_per_block": 1,
        "kpack_per_block": 2,
        "m_per_wave": 3
    },
    "conv": {
        "m_per_block": 0,
        "n_per_block": 1,
        "kpack_per_block": 2,
        "m_per_wave": 3
    },
    "attention": {
        "m_per_block": 0,
        "n_per_block": 2,
        "kpack_per_block": 3,
        "m_per_wave": 4
    },
}


def parse_config(perf_config: str, op: str = "gemm") -> Dict[str, float]:
    """Extract named + generic perfConfig features, using the op's field layout.

    The named fields (mPerBlock/nPerBlock/kpackPerBlock/mPerWave) are mapped per
    op so the interaction features get the correct tile sizes; split-K and the
    generic ``cfg_p*`` slots are layout-independent.
    """
    _, _, params = quickTuningGen.parse_perfconfig(perf_config)
    ints = []
    for p in params:
        ints.append(_safe_int(p, -1))

    def at(idx: int) -> int:
        return ints[idx] if 0 <= idx < len(ints) else -1

    pos = _CONFIG_FIELD_POS.get(op, _CONFIG_FIELD_POS["gemm"])
    split_k = quickTuningGen.get_splitk_value(perf_config)
    kpack = quickTuningGen.get_kpack_value(perf_config)
    feats: Dict[str, float] = {
        "cfg_m_per_block": float(at(pos["m_per_block"])),
        "cfg_n_per_block": float(at(pos["n_per_block"])),
        "cfg_kpack_per_block": float(at(pos["kpack_per_block"])),
        "cfg_m_per_wave": float(at(pos["m_per_wave"])),
        "cfg_kpack": float(kpack if kpack else 1),
        "cfg_split_k": float(split_k if split_k is not None else 1),
    }
    for i in range(_MAX_CONFIG_PARAMS):
        feats[f"cfg_p{i}"] = float(at(i))
    return feats


def _ceil_div(a: float, b: float) -> float:
    if b <= 0:
        return 0.0
    return math.ceil(a / b)


def interaction_features(m: int,
                         n: int,
                         k: int,
                         g: int,
                         num_cu: int,
                         cfg: Dict[str, float],
                         dtype_bytes: float = 2.0,
                         lds_bytes_per_wg: float = _DEFAULT_LDS_BYTES_PER_WG,
                         lds_bytes_per_cu: float = _DEFAULT_LDS_BYTES_PER_CU) -> Dict[str, float]:
    """Problem x config features (the highest-leverage signal)."""
    m_per_block = cfg["cfg_m_per_block"] if cfg["cfg_m_per_block"] > 0 else 1.0
    n_per_block = cfg["cfg_n_per_block"] if cfg["cfg_n_per_block"] > 0 else 1.0
    kpack_per_block = cfg["cfg_kpack_per_block"] if cfg["cfg_kpack_per_block"] > 0 else 1.0
    kpack = cfg.get("cfg_kpack", 1.0) if cfg.get("cfg_kpack", 1.0) > 0 else 1.0
    # True K elements staged per main-loop iteration = kpackPerBlock * kpack.
    k_per_block = kpack_per_block * kpack
    split_k = cfg["cfg_split_k"] if cfg["cfg_split_k"] > 0 else 1.0

    m_tiles = _ceil_div(m, m_per_block)
    n_tiles = _ceil_div(n, n_per_block)
    k_tiles = _ceil_div(k, k_per_block)
    total_wg = g * m_tiles * n_tiles * split_k

    # Mirror of computeWorkImbalance in RockTuningImpl.cpp: how lopsided the
    # per-CU work distribution is (1.0 == perfectly balanced).
    if total_wg > 0 and num_cu > 0:
        max_per_cu = math.ceil(total_wg / num_cu)
        work_imbalance = (max_per_cu * num_cu) / total_wg
    else:
        work_imbalance = 1.0

    # Resource-pressure proxies, normalized by the arch's LDS capacity so the
    # signal transfers across targets. The A and B operand tiles are staged in
    # LDS; footprint = (mPerBlock + nPerBlock) * kTile * dtype_bytes, where
    # kTile = kpackPerBlock * kpack (true K elements per iteration). This omits
    # double-buffering and alignment padding, so it is a lower bound on the real
    # allocation. lds_fraction is vs the per-workgroup cap (a feasibility
    # signal); lds_blocks_per_cu is an occupancy ceiling -- how many workgroups
    # can co-reside before the per-CU LDS runs out.
    tile_lds_bytes = (m_per_block + n_per_block) * k_per_block * dtype_bytes
    if tile_lds_bytes > 0:
        lds_fraction = tile_lds_bytes / lds_bytes_per_wg if lds_bytes_per_wg > 0 else 0.0
        lds_blocks_per_cu = (math.floor(lds_bytes_per_cu /
                                        tile_lds_bytes) if lds_bytes_per_cu > 0 else 0.0)
    else:
        lds_fraction = 0.0
        lds_blocks_per_cu = 0.0

    return {
        "m_tiles": float(m_tiles),
        "n_tiles": float(n_tiles),
        "k_tiles": float(k_tiles),
        "grid": float(total_wg),
        "grid_per_cu": float(total_wg / num_cu) if num_cu > 0 else 0.0,
        "work_imbalance": float(work_imbalance),
        "m_divisible": float(m % int(m_per_block) == 0),
        "n_divisible": float(n % int(n_per_block) == 0),
        "m_pad_waste": float(m_tiles * m_per_block - m),
        "n_pad_waste": float(n_tiles * n_per_block - n),
        "tile_lds_bytes": float(tile_lds_bytes),
        "lds_fraction": float(lds_fraction),
        "lds_blocks_per_cu": float(lds_blocks_per_cu),
    }


def gemm_problem_features(sig: ProblemSig) -> Dict[str, float]:
    m = _safe_int(sig.column("M"))
    n = _safe_int(sig.column("N"))
    k = _safe_int(sig.column("K"))
    g = _safe_int(sig.column("G"), 1)
    trans_a = _safe_int(sig.column("TransA"))
    trans_b = _safe_int(sig.column("TransB"))

    dbytes = _DTYPE_BYTES.get(sig.dtype, 2.0)
    flops = 2.0 * g * m * k * n
    bytes_moved = dbytes * (g * (m * k + k * n + m * n))
    arith_intensity = flops / bytes_moved if bytes_moved > 0 else 0.0

    return {
        "trans_a": float(trans_a),
        "trans_b": float(trans_b),
        "g": float(g),
        "m": float(m),
        "n": float(n),
        "k": float(k),
        "log_m": math.log2(m) if m > 0 else 0.0,
        "log_n": math.log2(n) if n > 0 else 0.0,
        "log_k": math.log2(k) if k > 0 else 0.0,
        "log_g": math.log2(g) if g > 0 else 0.0,
        "aspect_mn": float(m / n) if n > 0 else 0.0,
        "aspect_mk": float(m / k) if k > 0 else 0.0,
        "flops": flops,
        "log_flops": math.log2(flops) if flops > 0 else 0.0,
        "arith_intensity": arith_intensity,
    }


def _lg(value: float) -> float:
    return math.log2(value) if value > 0 else 0.0


# Layout strings (e.g. "ngc01") encode dim order with n,g,k,c plus 0=H, 1=W.
# Position of each dim is a deterministic, fit-free encoding so two problems
# that differ only by layout get distinct features instead of colliding.
_LAYOUT_DIMS = (("n", "n"), ("g", "g"), ("k", "k"), ("c", "c"), ("h", "0"), ("w", "1"))


def _layout_positions(layout, prefix: str) -> Dict[str, float]:
    s = str(layout)
    return {
        f"{prefix}_pos_{name}": float(s.index(ch)) if ch in s else -1.0 for name, ch in _LAYOUT_DIMS
    }


def _conv_out_spatial(h, w, y, x, sh, sw, dh, dw, ph, pw):
    ho = (h + 2 * ph - dh * (y - 1) - 1) // sh + 1 if sh > 0 else 0
    wo = (w + 2 * pw - dw * (x - 1) - 1) // sw + 1 if sw > 0 else 0
    return max(ho, 0), max(wo, 0)


def _idiv_ceil(a: int, b: int) -> int:
    return -(-a // b) if b > 0 else 0


def conv_bwd_data_gemm(n, c, k, h, w, y, x, ho, wo, sh, sw, dh, dw, ph, pw):
    """Exact backward-data implicit-GEMM (M, N, K, num_kernels).

    Reimplements ConvBwdDataOp::getGemmSize: strided backward-data is split via
    the gcd(stride, dilation) "tilda/dot" decomposition into num_kernels
    sub-GEMMs, each (M=C, N=N*prod(tildaSlice), K=K*prod(dotSlice)). M and N are
    kernel-independent; K is taken at kernelId 0 (the largest slice, which bounds
    the tile counts). num_kernels scales the launch grid. Reduces exactly to the
    flat (C, N*H*W, K*Y*X) for unit stride/dilation."""
    fil_tilda_y = sh // math.gcd(sh, dh) if sh > 0 else 1
    fil_tilda_x = sw // math.gcd(sw, dw) if sw > 0 else 1
    out_tilda_y = ho + _idiv_ceil(dh * (y - 1), sh)
    out_tilda_x = wo + _idiv_ceil(dw * (x - 1), sw)
    left_y = max(0, ph - dh * (fil_tilda_y - 1)) // sh if sh > 0 else 0
    left_x = max(0, pw - dw * (fil_tilda_x - 1)) // sw if sw > 0 else 0
    right_y = min(out_tilda_y, _idiv_ceil(ph + h - 1, sh) + 1)
    right_x = min(out_tilda_x, _idiv_ceil(pw + w - 1, sw) + 1)
    slice_y = max(0, right_y - left_y)
    slice_x = max(0, right_x - left_x)

    m = c
    n_gemm = n * slice_y * slice_x
    num_kernels = 0
    k_rep = 0
    for kid in range(fil_tilda_y * fil_tilda_x):
        itilda_y = kid // fil_tilda_x
        itilda_x = kid % fil_tilda_x
        dot_y = _idiv_ceil(y - itilda_y, fil_tilda_y)
        dot_x = _idiv_ceil(x - itilda_x, fil_tilda_x)
        if dot_y > 0 and dot_x > 0:  # mirrors backwardDataKernelIds filtering
            num_kernels += 1
            if kid == 0:
                k_rep = k * dot_y * dot_x
    return m, n_gemm, (k_rep if k_rep > 0 else k), max(num_kernels, 1)


def conv_implicit_gemm(sig: ProblemSig):
    """Map a conv problem to its implicit-GEMM (M, N, K, G), direction-aware.

    This is the same lowering the codegen uses, so the tiling/imbalance
    interaction features carry over directly from the GEMM path. For
    backward-data the G slot carries the sub-GEMM (kernel) count so the grid
    feature reflects all launched kernels."""
    d = str(sig.column("Direction"))
    g = _col_int(sig, "G", 1)
    n, c, k = _safe_int(sig.column("N")), _safe_int(sig.column("C")), _safe_int(sig.column("K"))
    h, w = _safe_int(sig.column("H")), _safe_int(sig.column("W"))
    y, x = _safe_int(sig.column("Y")), _safe_int(sig.column("X"))
    sh, sw = _safe_int(sig.column("StrideH"), 1), _safe_int(sig.column("StrideW"), 1)
    dh, dw = _safe_int(sig.column("DilationH"), 1), _safe_int(sig.column("DilationW"), 1)
    ph, pw = _safe_int(sig.column("PaddingH")), _safe_int(sig.column("PaddingW"))
    ho, wo = _conv_out_spatial(h, w, y, x, sh, sw, dh, dw, ph, pw)
    if d == "bwd":  # gradient w.r.t. input
        m, n_gemm, k_gemm, num_kernels = conv_bwd_data_gemm(n, c, k, h, w, y, x, ho, wo, sh, sw, dh,
                                                            dw, ph, pw)
        return m, n_gemm, k_gemm, g * num_kernels
    if d == "wrw":  # gradient w.r.t. weights
        return k, c * y * x, n * ho * wo, g
    return k, n * ho * wo, c * y * x, g  # fwd


def conv_problem_features(sig: ProblemSig) -> Dict[str, float]:
    d = str(sig.column("Direction"))
    n, c, k = _safe_int(sig.column("N")), _safe_int(sig.column("C")), _safe_int(sig.column("K"))
    h, w = _safe_int(sig.column("H")), _safe_int(sig.column("W"))
    y, x = _safe_int(sig.column("Y")), _safe_int(sig.column("X"))
    sh, sw = _safe_int(sig.column("StrideH"), 1), _safe_int(sig.column("StrideW"), 1)
    dh, dw = _safe_int(sig.column("DilationH"), 1), _safe_int(sig.column("DilationW"), 1)
    ph, pw = _safe_int(sig.column("PaddingH")), _safe_int(sig.column("PaddingW"))
    ho, wo = _conv_out_spatial(h, w, y, x, sh, sw, dh, dw, ph, pw)
    gm, gn, gk, _ = conv_implicit_gemm(sig)

    dbytes = _DTYPE_BYTES.get(sig.dtype, 2.0)
    flops = 2.0 * gm * gn * gk
    bytes_moved = dbytes * (n * c * h * w + k * c * y * x + n * k * ho * wo)
    arith_intensity = flops / bytes_moved if bytes_moved > 0 else 0.0

    feats: Dict[str, float] = {
        "is_fwd": float(d == "fwd"),
        "is_bwd": float(d == "bwd"),
        "is_wrw": float(d == "wrw"),
        "n": float(n),
        "c": float(c),
        "h": float(h),
        "w": float(w),
        "k": float(k),
        "y": float(y),
        "x": float(x),
        "log_n": _lg(n),
        "log_c": _lg(c),
        "log_h": _lg(h),
        "log_w": _lg(w),
        "log_k": _lg(k),
        "stride_h": float(sh),
        "stride_w": float(sw),
        "dil_h": float(dh),
        "dil_w": float(dw),
        "pad_h": float(ph),
        "pad_w": float(pw),
        "ho": float(ho),
        "wo": float(wo),
        "log_ho": _lg(ho),
        "log_wo": _lg(wo),
        "filter_area": float(y * x),
        "gemm_m": float(gm),
        "gemm_n": float(gn),
        "gemm_k": float(gk),
        "log_gemm_m": _lg(gm),
        "log_gemm_n": _lg(gn),
        "log_gemm_k": _lg(gk),
        "flops": flops,
        "log_flops": _lg(flops),
        "arith_intensity": arith_intensity,
    }
    feats.update(_layout_positions(sig.column("FilterLayout"), "fil"))
    feats.update(_layout_positions(sig.column("InputLayout"), "in"))
    feats.update(_layout_positions(sig.column("OutputLayout"), "out"))
    return feats


def _flag(value) -> float:
    """Robustly coerce a truthy column (bool, 'True'/'False', 0/1) to 0.0/1.0."""
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in ("true", "1") else 0.0
    return 1.0 if bool(value) else 0.0


def attention_implicit_gemm(sig: ProblemSig):
    """Map attention to its dominant (first) GEMM -- the QK^T score matrix --
    for the tiling/imbalance interaction features.

    M = seqLenQ, N = seqLenK, K = headDimQK, G = batch * numHeadsQ. The second
    GEMM (scores @ V) tiles over the same seqLenQ rows, so the first GEMM is
    the representative grid for work-imbalance purposes."""
    g = _safe_int(sig.column("G"), 1)
    nhq = _safe_int(sig.column("NumHeadsQ"), 1)
    sq = _safe_int(sig.column("SeqLenQ"))
    sk = _safe_int(sig.column("SeqLenK"))
    dqk = _safe_int(sig.column("HeadDimQK"))
    return sq, sk, dqk, g * nhq


def attention_problem_features(sig: ProblemSig) -> Dict[str, float]:
    g = _safe_int(sig.column("G"), 1)
    nhq = _safe_int(sig.column("NumHeadsQ"), 1)
    nhkv = _safe_int(sig.column("NumHeadsKV"), 1)
    sq = _safe_int(sig.column("SeqLenQ"))
    sk = _safe_int(sig.column("SeqLenK"))
    dqk = _safe_int(sig.column("HeadDimQK"))
    dv = _safe_int(sig.column("HeadDimV"))
    causal = _flag(sig.column("Causal"))
    batch_q = g * nhq
    batch_kv = g * nhkv

    dbytes = _DTYPE_BYTES.get(sig.dtype, 2.0)
    # Two matmuls: QK^T (sq x sk x dqk) and P@V (sq x dv x sk). Causal masks
    # roughly half the score matrix.
    mask = 0.5 if causal else 1.0
    flops = 2.0 * batch_q * mask * (sq * sk * dqk + sq * sk * dv)
    bytes_moved = dbytes * (batch_q * sq * dqk + batch_kv * sk * dqk + batch_kv * sk * dv +
                            batch_q * sq * dv)
    arith_intensity = flops / bytes_moved if bytes_moved > 0 else 0.0

    return {
        "trans_q": _flag(sig.column("TransQ")),
        "trans_k": _flag(sig.column("TransK")),
        "trans_v": _flag(sig.column("TransV")),
        "trans_o": _flag(sig.column("TransO")),
        "causal": causal,
        "return_lse": _flag(sig.column("ReturnLSE")),
        "split_kv": float(_safe_int(sig.column("SplitKV"), 1)),
        "with_attn_scale": _flag(sig.column("WithAttnScale")),
        "with_attn_bias": _flag(sig.column("WithAttnBias")),
        "g": float(g),
        "num_heads_q": float(nhq),
        "num_heads_kv": float(nhkv),
        "gqa_ratio": float(nhq / nhkv) if nhkv > 0 else 1.0,
        "batch_q": float(batch_q),
        "seq_len_q": float(sq),
        "seq_len_k": float(sk),
        "head_dim_qk": float(dqk),
        "head_dim_v": float(dv),
        "log_seq_q": _lg(sq),
        "log_seq_k": _lg(sk),
        "log_head_qk": _lg(dqk),
        "log_head_v": _lg(dv),
        "log_batch_q": _lg(batch_q),
        "seq_ratio": float(sq / sk) if sk > 0 else 1.0,
        "is_square_seq": float(sq == sk),
        "flops": flops,
        "log_flops": _lg(flops),
        "arith_intensity": arith_intensity,
    }


def problem_features(sig: ProblemSig) -> Dict[str, float]:
    """Problem-only features, dispatched by op."""
    if sig.op == "conv":
        return conv_problem_features(sig)
    if sig.op == "attention":
        return attention_problem_features(sig)
    return gemm_problem_features(sig)


def implicit_mnkg(sig: ProblemSig):
    """(M, N, K, G) for the interaction features, dispatched by op."""
    if sig.op == "conv":
        return conv_implicit_gemm(sig)
    if sig.op == "attention":
        return attention_implicit_gemm(sig)
    return (_safe_int(sig.column("M")), _safe_int(sig.column("N")), _safe_int(sig.column("K")),
            _safe_int(sig.column("G"), 1))


# Problem-only features used as the nearest-neighbor distance metric, per op.
_DISTANCE_FEATURES = {
    "gemm": ("trans_a", "trans_b", "log_g", "log_m", "log_n", "log_k", "aspect_mn", "aspect_mk"),
    "conv": ("is_fwd", "is_bwd", "log_n", "log_c", "log_h", "log_w", "log_k", "y", "x", "stride_h",
             "stride_w", "log_gemm_m", "log_gemm_n", "log_gemm_k", "in_pos_c", "fil_pos_c"),
    "attention": ("causal", "log_seq_q", "log_seq_k", "log_head_qk", "log_head_v", "log_batch_q",
                  "gqa_ratio", "seq_ratio", "trans_q", "trans_k", "trans_v", "trans_o"),
}


def distance_features(op: str):
    return _DISTANCE_FEATURES.get(op, _DISTANCE_FEATURES["gemm"])


def feature_record(sig: ProblemSig, perf_config: str) -> "OrderedDict[str, float]":
    """Full feature dict for one (problem, config). Stable key order."""
    num_cu = sig.num_cu if sig.num_cu else _DEFAULT_NUM_CU

    rec: "OrderedDict[str, float]" = OrderedDict()
    rec.update(problem_features(sig))
    rec.update(dtype_features(sig.dtype))
    rec["num_cu"] = float(num_cu)
    # Chiplet (XCD) count: on MCM parts the grid is distributed across chiplets,
    # so it shifts the work-imbalance sweet spot. 1 for monolithic / unknown.
    rec["num_chiplets"] = float(sig.num_chiplets if sig.num_chiplets else 1)
    arch = arch_hw(sig.arch)
    rec.update(arch)
    cfg = parse_config(perf_config, sig.op)
    rec.update(cfg)
    m, n, k, g = implicit_mnkg(sig)
    dtype_bytes = _DTYPE_BYTES.get(sig.dtype, 2.0)
    rec.update(
        interaction_features(m, n, k, g, int(num_cu), cfg, dtype_bytes, arch["lds_bytes_per_wg"],
                             arch["lds_bytes_per_cu"]))
    return rec


def label(tflops: float, best: float, threshold: float = DEFAULT_THRESHOLD) -> int:
    """1 if the config is within ``threshold`` of the per-problem best."""
    if tflops is None or math.isnan(tflops) or best is None or math.isnan(best) or best <= 0:
        return 0
    return int(tflops >= best * threshold)
