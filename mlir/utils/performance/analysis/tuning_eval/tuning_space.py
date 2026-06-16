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
        if sig.op == "conv":
            return self._conv_command(sig)
        if sig.op == "attention":
            return self._attention_command(sig)
        return self._gemm_command(sig)

    def _gemm_command(self, sig: ProblemSig) -> List[str]:
        ta = bool(int(sig.column("TransA")))
        tb = bool(int(sig.column("TransB")))
        argv = [
            self._gen, "--arch", sig.arch, "-operation", "gemm", "-t", sig.dtype, "-out_datatype",
            out_dtype_for(sig.dtype), "-g",
            str(sig.column("G")), "-m",
            str(sig.column("M")), "-k",
            str(sig.column("K")), "-n",
            str(sig.column("N")), f"-transA={ta}", f"-transB={tb}",
            f"--emit-tuning-space={self._kind}"
        ]
        if sig.num_cu:
            argv += ["--num_cu", str(sig.num_cu)]
        return argv

    def _conv_command(self, sig: ProblemSig) -> List[str]:
        operation = {
            "fwd": "conv",
            "bwd": "conv_bwd_data",
            "wrw": "conv_bwd_weight",
        }[str(sig.column("Direction"))]
        argv = [
            self._gen,
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
            # column); emit the applicable space for a single group accordingly.
            "--groupsize",
            "1",
            f"--emit-tuning-space={self._kind}"
        ]
        if sig.num_cu:
            argv += ["--num_cu", str(sig.num_cu)]
        return argv

    def _attention_command(self, sig: ProblemSig) -> List[str]:

        def flag(name: str) -> str:
            v = sig.column(name)
            if isinstance(v, str):
                return "true" if v.strip().lower() in ("true", "1") else "false"
            return "true" if bool(v) else "false"

        argv = [
            self._gen, "--arch", sig.arch, "-operation", "attention", "-t", sig.dtype, "-g",
            str(sig.column("G")), "-seq_len_q",
            str(sig.column("SeqLenQ")), "-seq_len_k",
            str(sig.column("SeqLenK")), "-num_heads_q",
            str(sig.column("NumHeadsQ")), "-num_heads_kv",
            str(sig.column("NumHeadsKV")), "-head_dim_qk",
            str(sig.column("HeadDimQK")), "-head_dim_v",
            str(sig.column("HeadDimV")), f"-transQ={flag('TransQ')}", f"-transK={flag('TransK')}",
            f"-transV={flag('TransV')}", f"-transO={flag('TransO')}", f"-causal={flag('Causal')}",
            f"-return_lse={flag('ReturnLSE')}", f"-split_kv={sig.column('SplitKV')}",
            f"-with-attn-scale={flag('WithAttnScale')}", f"-with-attn-bias={flag('WithAttnBias')}",
            f"--emit-tuning-space={self._kind}"
        ]
        if sig.num_cu:
            argv += ["--num_cu", str(sig.num_cu)]
        return argv

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
