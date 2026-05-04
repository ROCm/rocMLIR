# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Golden tests for problem-key formatting and hashing.

The same hashes are pinned in the C++ unit tests; drift fails both
suites.
"""
import sys
from pathlib import Path

import pytest

_test_dir = Path(__file__).resolve().parent
_analysis_dir = _test_dir.parent / "analysis"
if str(_analysis_dir) not in sys.path:
    sys.path.insert(0, str(_analysis_dir))

import quickTuningGen  # noqa: E402


class TestStringFormat:
    """make_problem_key joins values with '_' via Python's str()."""

    def test_gemm(self):
        # GEMM_COLUMNS = [TransA, TransB, G, M, K, N]; bools are 0/1 ints
        # by the time they reach make_problem_key.
        assert quickTuningGen.make_problem_key(
            (0, 0, 1, 1024, 1024, 1024)) == "0_0_1_1024_1024_1024"

    def test_gemm_transposed(self):
        assert quickTuningGen.make_problem_key((1, 1, 1, 64, 128, 256)) == "1_1_1_64_128_256"

    def test_conv(self):
        # CONV_COLUMNS = [Direction, FilterLayout, InputLayout, OutputLayout,
        # N, C, H, W, K, Y, X, DilationH, DilationW, StrideH, StrideW,
        # PaddingH, PaddingW]
        assert quickTuningGen.make_problem_key(
            ("fwd", "kcyx", "nchw", "nkhw", 1, 64, 14, 14, 128, 3, 3, 1, 1, 1, 1, 1,
             1)) == "fwd_kcyx_nchw_nkhw_1_64_14_14_128_3_3_1_1_1_1_1_1"

    def test_attention(self):
        # ATTENTION_COLUMNS = [TransQ, TransK, TransV, TransO, Causal,
        # ReturnLSE, SplitKV, WithAttnScale, WithAttnBias, G, SeqLenQ,
        # SeqLenK, NumHeadsQ, NumHeadsKV, HeadDimQK, HeadDimV]
        assert quickTuningGen.make_problem_key((0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 256, 256, 8, 8, 64,
                                                64)) == "0_0_0_0_0_0_1_0_0_1_256_256_8_8_64_64"


class TestLoadFilesBoolCast:
    """Bool columns must be coerced to int at ingest, not at key-format."""

    def test_bool_columns_become_int(self, tmp_path):
        import pandas as pd
        f = tmp_path / "in.tsv"
        cols = ['Chip', 'DataType', 'TransA', 'TransB', 'G', 'M', 'K', 'N', 'PerfConfig', 'TFlops']
        rows = [['gfx942', 'f32', True, False, 1, 1024, 1024, 1024, 'pc', 1.0]]
        pd.DataFrame(rows, columns=cols).to_csv(f, sep='\t', index=False)

        per_dtype = quickTuningGen.load_files([f], 'gemm', no_splitk=False)
        df = per_dtype['f32']
        assert df['TransA'].dtype.kind in ('i', 'u')
        assert df['TransB'].dtype.kind in ('i', 'u')


class TestHashGolden:

    @pytest.mark.parametrize(
        "key,expected",
        [
            # Sanity: classic xxh3_64("hello").
            ("hello", 0x9555E8555C62DCFD),
            # GEMM examples.
            ("0_0_1_1024_1024_1024", 0x8BB71834A431CBDD),
            ("1_1_1_64_128_256", 0xBFD6CCD314CA3040),
            # CONV example (fwd, NCHW-layout, 1x64x14x14 -> 128 ch, 3x3 filter).
            ("fwd_kcyx_nchw_nkhw_1_64_14_14_128_3_3_1_1_1_1_1_1", 0xA6F3626951158D16),
            # Attention example (no scale/bias, 1 head-group, 256x256, 8 heads).
            ("0_0_0_0_0_0_1_0_0_1_256_256_8_8_64_64", 0xC75F3355EE11CD53),
        ],
    )
    def test_known_hash(self, key, expected):
        assert quickTuningGen.hash_problem_key(key) == expected
