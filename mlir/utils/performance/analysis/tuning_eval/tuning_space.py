# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Per-problem candidate pool from the compiler's own tuning-space enumerator.

Instead of reimplementing applicability (paramsProbablyValid / couldBePerformant)
in Python, this calls ``rocmlir-gen --emit-tuning-space=<kind>``, which prints the
applicable perfConfigs for a problem straight from ``createTunableParamSpace``
(no GPU, no execution). The serialization (``getPerfConfigStr``) matches what the
tuning driver records, so the strings line up with the oracle. Configs that are
statically applicable but fail at compile/run time simply show up as NaN in the
oracle (wasted budget) -- exactly the production behavior.
"""

import subprocess
from typing import Dict, List, Optional, Tuple

from .corpus import ProblemSig

# perfRunner (performance/) is put on sys.path by the package __init__.

# Output element type recorded alongside each GEMM dtype in the tier1 corpus.
_OUT_DTYPE = {
    "i8": "i32",
    "fp8": "f32",
    "bf8": "f32",
    "fp4": "f32",
    "f4E2M1FN": "f32",
}


def out_dtype_for(dtype: str) -> str:
    return _OUT_DTYPE.get(dtype, dtype)


def _bool_flag(value) -> str:
    """Coerce a truthy column (bool, 'True'/'False', 0/1) to rocmlir-gen's
    true/false spelling."""
    if isinstance(value, str):
        return "true" if value.strip().lower() in ("true", "1") else "false"
    return "true" if bool(value) else "false"


def _hw_argv(sig: ProblemSig) -> List[str]:
    """The -num_cu / -num_chiplets flags for a problem, so the constructed op's
    occupancy attributes match the tuned .debug row (and thus the features)."""
    argv: List[str] = []
    if sig.num_cu:
        argv += ["-num_cu", str(sig.num_cu)]
    if sig.num_chiplets:
        argv += ["-num_chiplets", str(sig.num_chiplets)]
    return argv


def problem_argv(sig: ProblemSig) -> List[str]:
    """rocmlir-gen flags describing a problem, sans binary and action flag.

    Shared by the tuning-space pool (``--emit-tuning-space``) and the feature
    extractor (``--emit-features``) so the two always describe the same op.
    """
    if sig.op == "conv":
        return _conv_argv(sig)
    if sig.op == "attention":
        return _attention_argv(sig)
    return _gemm_argv(sig)


def _gemm_argv(sig: ProblemSig) -> List[str]:
    ta = bool(int(sig.column("TransA")))
    tb = bool(int(sig.column("TransB")))
    return [
        "--arch", sig.arch, "-operation", "gemm", "-t", sig.dtype, "-out_datatype",
        out_dtype_for(sig.dtype), "-g",
        str(sig.column("G")), "-m",
        str(sig.column("M")), "-k",
        str(sig.column("K")), "-n",
        str(sig.column("N")), f"-transA={ta}", f"-transB={tb}"
    ] + _hw_argv(sig)


def _conv_argv(sig: ProblemSig) -> List[str]:
    operation = {
        "fwd": "conv",
        "bwd": "conv_bwd_data",
        "wrw": "conv_bwd_weight",
    }[str(sig.column("Direction"))]
    return [
        "--arch",
        sig.arch,
        "--operation",
        operation,
        "-t",
        sig.dtype,
        "--fil_layout",
        str(sig.column("FilterLayout")),
        "--in_layout",
        str(sig.column("InputLayout")),
        "--out_layout",
        str(sig.column("OutputLayout")),
        "--batchsize",
        str(sig.column("N")),
        "--in_channels",
        str(sig.column("C")),
        "--in_h",
        str(sig.column("H")),
        "--in_w",
        str(sig.column("W")),
        "--out_channels",
        str(sig.column("K")),
        "--fil_h",
        str(sig.column("Y")),
        "--fil_w",
        str(sig.column("X")),
        "--dilation_h",
        str(sig.column("DilationH")),
        "--dilation_w",
        str(sig.column("DilationW")),
        "--conv_stride_h",
        str(sig.column("StrideH")),
        "--conv_stride_w",
        str(sig.column("StrideW")),
        "--padding_h",
        str(sig.column("PaddingH")),
        "--padding_w",
        str(sig.column("PaddingW")),
        # The conv corpora are group-1 (the .debug schema carries no conv G
        # column); describe the op for a single group accordingly.
        "--groupsize",
        "1",
    ] + _hw_argv(sig)


def _attention_argv(sig: ProblemSig) -> List[str]:
    return [
        "--arch", sig.arch, "-operation", "attention", "-t", sig.dtype, "-g",
        str(sig.column("G")), "-seq_len_q",
        str(sig.column("SeqLenQ")), "-seq_len_k",
        str(sig.column("SeqLenK")), "-num_heads_q",
        str(sig.column("NumHeadsQ")), "-num_heads_kv",
        str(sig.column("NumHeadsKV")), "-head_dim_qk",
        str(sig.column("HeadDimQK")), "-head_dim_v",
        str(sig.column("HeadDimV")), f"-transQ={_bool_flag(sig.column('TransQ'))}",
        f"-transK={_bool_flag(sig.column('TransK'))}",
        f"-transV={_bool_flag(sig.column('TransV'))}",
        f"-transO={_bool_flag(sig.column('TransO'))}",
        f"-causal={_bool_flag(sig.column('Causal'))}",
        f"-return_lse={_bool_flag(sig.column('ReturnLSE'))}", f"-split_kv={sig.column('SplitKV')}",
        f"-with-attn-scale={_bool_flag(sig.column('WithAttnScale'))}",
        f"-with-attn-bias={_bool_flag(sig.column('WithAttnBias'))}"
    ] + _hw_argv(sig)


class EmitTuningSpacePool:
    """Callable ``ProblemSig -> [perfConfig]`` backed by rocmlir-gen.

    Results are cached per ``(arch, dtype, problem_key)``; the cache is
    train-independent, so a single instance can be shared across CV folds.
    """

    def __init__(self,
                 mlir_build_dir: Optional[str] = None,
                 kind: str = "full",
                 timeout: int = 120):
        import perfRunner
        self._paths = perfRunner.create_paths(None, mlir_build_dir)
        if not self._paths.mlir_paths:
            raise RuntimeError(
                "rocMLIR build dir not found; pass mlir_build_dir to EmitTuningSpacePool")
        self._gen = self._paths.mlir_paths.rocmlir_gen_path
        self._kind = kind
        self._timeout = timeout
        self._cache: Dict[Tuple[str, str, str], List[str]] = {}

    def __call__(self, sig: ProblemSig) -> List[str]:
        key = (sig.arch, sig.dtype, sig.problem_key)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        configs = self._enumerate(sig)
        self._cache[key] = configs
        return configs

    def command(self, sig: ProblemSig) -> List[str]:
        return [self._gen] + problem_argv(sig) + [f"--emit-tuning-space={self._kind}"]

    def _enumerate(self, sig: ProblemSig) -> List[str]:
        argv = self.command(sig)
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=self._timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"rocmlir-gen --emit-tuning-space failed for {sig.problem_key}: "
                               f"{proc.stderr.strip()}")
        return parse_tuning_space(proc.stdout)


def parse_tuning_space(stdout: str) -> List[str]:
    """One perfConfig per line; blank lines dropped, order preserved."""
    return [line.strip() for line in stdout.splitlines() if line.strip()]
