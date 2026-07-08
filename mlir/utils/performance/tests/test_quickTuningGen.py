# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Golden tests for problem-key formatting and hashing.

The hash goldens here are mirrored in QuickTuningDbTests.cpp; drift fails
both suites.
"""
import sys
from pathlib import Path

import pytest

_test_dir = Path(__file__).resolve().parent
_analysis_dir = _test_dir.parent / "analysis"
if str(_analysis_dir) not in sys.path:
    sys.path.insert(0, str(_analysis_dir))

import quickTuningGen  # noqa: E402


# make_problem_key joins values with '_' via Python's str(). Bool inputs are
# already coerced to int by load_files (see test_load_files_bool_cast).
@pytest.mark.parametrize(
    "fields,expected",
    [
        # GEMM_COLUMNS = [TransA, TransB, G, M, K, N]
        ((0, 0, 1, 1024, 1024, 1024), "0_0_1_1024_1024_1024"),
        ((1, 1, 1, 64, 128, 256), "1_1_1_64_128_256"),
        # CONV_COLUMNS = [Direction, FilterLayout, InputLayout, OutputLayout,
        #                 N, C, H, W, K, Y, X, DH, DW, SH, SW, PH, PW]
        (("fwd", "kcyx", "nchw", "nkhw", 1, 64, 14, 14, 128, 3, 3, 1, 1, 1, 1, 1, 1),
         "fwd_kcyx_nchw_nkhw_1_64_14_14_128_3_3_1_1_1_1_1_1"),
        # ATTENTION_COLUMNS = [TransQ, TransK, TransV, TransO, Causal,
        #                      ReturnLSE, SplitKV, WithAttnScale, WithAttnBias,
        #                      G, SeqLenQ, SeqLenK, NumHeadsQ, NumHeadsKV,
        #                      HeadDimQK, HeadDimV]
        ((0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 256, 256, 8, 8, 64, 64),
         "0_0_0_0_0_0_1_0_0_1_256_256_8_8_64_64"),
    ],
)
def test_make_problem_key(fields, expected):
    assert quickTuningGen.make_problem_key(fields) == expected


def test_load_files_bool_cast(tmp_path):
    """Bool columns must be coerced to int at ingest, not at key-format
    time -- str(True) == 'True' would silently corrupt every hash."""
    import pandas as pd
    f = tmp_path / "in.tsv"
    cols = ['Chip', 'DataType', 'TransA', 'TransB', 'G', 'M', 'K', 'N', 'PerfConfig', 'TFlops']
    rows = [['gfx942', 'f32', True, False, 1, 1024, 1024, 1024, 'pc', 1.0]]
    pd.DataFrame(rows, columns=cols).to_csv(f, sep='\t', index=False)

    df = quickTuningGen.load_files([f], 'gemm', no_splitk=False)['f32']
    assert df['TransA'].dtype.kind in ('i', 'u')
    assert df['TransB'].dtype.kind in ('i', 'u')


# hash_problem_key uses xxh3_64. These goldens are mirrored in
# QuickTuningDbTests.cpp; drift fails both suites.
@pytest.mark.parametrize(
    "key,expected",
    [
        ("0_0_1_1024_1024_1024", 0x8BB71834A431CBDD),
        ("1_1_1_64_128_256", 0xBFD6CCD314CA3040),
        ("fwd_kcyx_nchw_nkhw_1_64_14_14_128_3_3_1_1_1_1_1_1", 0xA6F3626951158D16),
        ("bwd_kcyx_nchw_nkhw_1_64_14_14_128_3_3_1_1_1_1_1_1", 0x61DCF2C43198890D),
        ("wrw_kcyx_nchw_nkhw_1_64_14_14_128_3_3_1_1_1_1_1_1", 0xDB5C922433670198),
        ("0_0_0_0_0_0_1_0_0_1_256_256_8_8_64_64", 0xC75F3355EE11CD53),
        ("0_0_0_0_0_0_1_1_0_1_256_256_8_8_64_64", 0x24A5C1C837083EAB),
    ],
)
def test_hash_problem_key(key, expected):
    assert quickTuningGen.hash_problem_key(key) == expected
