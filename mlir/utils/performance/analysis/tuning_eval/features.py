# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Feature extraction via the compiler -- a single source of truth.

Features are computed by ``rocmlir-gen --emit-features``: the exact same C++
extractor the deployed model scorer uses (``SmartTuningFeatures.cpp``). This
module is a thin client -- it reconstructs the rocmlir-gen invocation for a
problem (reusing ``tuning_space``'s command builders), pipes the candidate
perfConfigs in on stdin, and parses the CSV the tool prints (a header row of
feature names, then one ``"perfConfig",v0,v1,...`` row per config). Training
and inference therefore share one feature implementation; there is no Python
reimplementation to keep in parity.

The feature *vector* (names, order, values) is defined entirely by the C++
side; callers read it back here and never compute features themselves.
"""

import csv
import io
import subprocess
from collections import OrderedDict
from typing import List, Optional

from .corpus import ProblemSig
from .tuning_space import problem_argv

# Default coverage threshold; matches quickTuningGen's set-cover default so the
# harness label and the shipped DB speak the same "good enough" language.
DEFAULT_THRESHOLD = 0.93

# Problem-only feature names used as the nearest-neighbor distance metric, per
# op. These are a subset of the names the C++ extractor emits; the values are
# read out of a full feature record (they do not depend on the perfConfig).
_DISTANCE_FEATURES = {
    "gemm": ("trans_a", "trans_b", "log_g", "log_m", "log_n", "log_k", "aspect_mn", "aspect_mk"),
    "conv": ("is_fwd", "is_bwd", "log_n", "log_c", "log_h", "log_w", "log_k", "y", "x", "stride_h",
             "stride_w", "log_gemm_m", "log_gemm_n", "log_gemm_k", "in_pos_c", "fil_pos_c"),
    "attention": ("causal", "log_seq_q", "log_seq_k", "log_head_qk", "log_head_v", "log_batch_q",
                  "gqa_ratio", "seq_ratio", "trans_q", "trans_k", "trans_v", "trans_o"),
}


def distance_features(op: str):
    return _DISTANCE_FEATURES.get(op, _DISTANCE_FEATURES["gemm"])


class FeatureExtractor:
    """Computes feature vectors by shelling out to ``rocmlir-gen --emit-features``.

    One subprocess serves a whole problem: all candidate perfConfigs go in on
    stdin and come back as CSV rows in the same order, so featurizing is batched
    per problem rather than per (problem, config).
    """

    def __init__(self, mlir_build_dir: Optional[str] = None, timeout: int = 120):
        import perfRunner
        if not mlir_build_dir:
            mlir_build_dir = perfRunner.find_mlir_build_dir()
        paths = perfRunner.create_paths(None, mlir_build_dir)
        if not paths.mlir_paths:
            raise RuntimeError(
                "rocmlir-gen build dir not found; pass mlir_build_dir to FeatureExtractor")
        self._gen = paths.mlir_paths.rocmlir_gen_path
        self._timeout = timeout

    def records(self, sig: ProblemSig, configs: List[str]) -> "List[OrderedDict[str, float]]":
        """Feature dicts for ``configs`` (input order preserved)."""
        if not configs:
            return []
        argv = [self._gen] + problem_argv(sig) + ["--emit-features", "-perf_config=-"]
        proc = subprocess.run(argv,
                              input="\n".join(configs) + "\n",
                              capture_output=True,
                              text=True,
                              timeout=self._timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"rocmlir-gen --emit-features failed for {sig.problem_key}: "
                               f"{proc.stderr.strip()}")
        return _parse_features_csv(proc.stdout)

    def record(self, sig: ProblemSig, perf_config: str) -> "OrderedDict[str, float]":
        recs = self.records(sig, [perf_config])
        if not recs:
            raise RuntimeError(f"rocmlir-gen --emit-features produced no row for {perf_config}")
        return recs[0]


def _parse_features_csv(stdout: str) -> "List[OrderedDict[str, float]]":
    """Parse the ``--emit-features`` CSV: header of names (after the leading
    perf_config column), then one numeric row per config."""
    rows = [r for r in csv.reader(io.StringIO(stdout)) if r]
    if not rows:
        raise RuntimeError("rocmlir-gen --emit-features produced no output")
    names = rows[0][1:]  # drop the perf_config column
    out: "List[OrderedDict[str, float]]" = []
    for row in rows[1:]:
        out.append(OrderedDict((name, float(v)) for name, v in zip(names, row[1:])))
    return out


# Process-wide extractor, configured once (optionally with an explicit build
# dir) and reused by the proposers. Lazily auto-discovers the build dir.
_EXTRACTOR: Optional[FeatureExtractor] = None


def configure_extractor(mlir_build_dir: Optional[str] = None) -> FeatureExtractor:
    """Set the process-wide extractor (e.g. to honor a --mlir-build-dir flag)."""
    global _EXTRACTOR
    _EXTRACTOR = FeatureExtractor(mlir_build_dir)
    return _EXTRACTOR


def get_extractor() -> FeatureExtractor:
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = FeatureExtractor()
    return _EXTRACTOR


def feature_records(sig: ProblemSig, configs: List[str]) -> "List[OrderedDict[str, float]]":
    """Feature dicts for several configs of one problem (batched, ordered)."""
    return get_extractor().records(sig, configs)


def feature_record(sig: ProblemSig, perf_config: str) -> "OrderedDict[str, float]":
    """Full feature dict for one (problem, config)."""
    return get_extractor().record(sig, perf_config)


def label(tflops: float, best: float, threshold: float = DEFAULT_THRESHOLD) -> int:
    """1 if the config is within ``threshold`` of the per-problem best."""
    import math
    if tflops is None or math.isnan(tflops) or best is None or math.isnan(best) or best <= 0:
        return 0
    return int(tflops >= best * threshold)
