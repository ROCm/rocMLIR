#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pure-Python coverage for attention tuning DB compatibility.

The tuning DB key for attention has grown optional boolean flags over time
(``-with-attn-scale``, ``-with-attn-bias``, ``-transBias``). False-valued flags
are identity cases, so old DB rows that omit them should still be readable:
``read_tuning_db`` canonicalizes every stored key through
``AttentionConfiguration``, which re-adds the (default false) flags. True-valued
flags describe different generated kernels and must not be silently matched
against an old all-false row.

# RUN: %python %s %t
"""

from pathlib import Path
import sys
import types
import unittest

MLIR_DIR = Path(__file__).resolve().parents[2]
PERF_DIR = MLIR_DIR / "utils" / "performance"
TESTS_DIR = PERF_DIR / "tests"
sys.path.insert(0, str(PERF_DIR))
sys.path.insert(0, str(PERF_DIR / "analysis"))

# Inject mock 'hip'/'amd_arch_db' modules so perfRunner imports without ROCm.
exec(
    open(TESTS_DIR / "mock_hip.py").read(), {
        "__file__": str(TESTS_DIR / "mock_hip.py"),
        "__name__": "mock_hip"
    })


def stub_optional_pulp():
    """Let this parsing-only test import quickTuningGen without PuLP installed."""
    sys.modules.setdefault("pulp", types.SimpleNamespace())


stub_optional_pulp()

import perfRunner  # noqa: E402
from perfRunner import AttentionConfiguration, read_tuning_db  # noqa: E402
from quickTuningGen import get_target_columns, load_data  # noqa: E402

# Pin the attention dtype list so config construction never probes real hardware.
perfRunner.DATA_TYPES_ATTENTION = perfRunner.DATA_TYPES_ATTENTION_MFMA

ARCH = "gfx950:sramecc+:xnack-"
NUM_CU = 256
NUM_CHIPLETS = 8
PERFCONFIG = "attn:v4:32,256,32,1,1,4,16,1,1,0,0,-1,-1,-1,-1,-1,-1"
TMP_PREFIX = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/attention-tuning-db-compat"))


def make_config(extra_flags):
    """Build a canonical attention config with the requested optional flags."""
    key = ("-t f16 "
           "-transQ false -transK false -transV false -transO false "
           "-causal false -return_lse false -split_kv 1 -g 1 "
           "-seq_len_q 16 -seq_len_k 16 -num_heads_q 1 -num_heads_kv 1 "
           "-head_dim_qk 32 -head_dim_v 32 "
           f"{extra_flags}")
    return AttentionConfiguration.from_command_line(key.split(), ARCH, NUM_CU, NUM_CHIPLETS)


def write_tuning_db(path, config_field):
    """Write a one-row tuning DB using the given attention problem key."""
    path.write_text("# arch\tnumCUs\tnumChiplets\ttestVector\tperfConfig\tTFlops\n"
                    f"{ARCH}\t{NUM_CU}\t{NUM_CHIPLETS}\t{config_field}\t{PERFCONFIG}\t0.0\n")


def drop_flags(config_str, *flags):
    """Return a legacy key by removing selected false-valued flags."""
    for flag in flags:
        config_str = config_str.replace(flag, "")
    return config_str


class AttentionTuningDbCompatTest(unittest.TestCase):
    """Compatibility tests for attention tuning DB and debug TSV parsing."""

    def setUp(self):
        """Give each test its own temporary file prefix."""
        self.tmp_prefix = Path(f"{TMP_PREFIX}.{self._testMethodName}")

    def read_db_from_legacy_key(self, legacy_key):
        """Read a tuning DB written with a legacy key (canonicalized on read)."""
        path = Path(f"{self.tmp_prefix}.tsv")
        write_tuning_db(path, legacy_key)
        return read_tuning_db(str(path), AttentionConfiguration, NUM_CU, NUM_CHIPLETS)

    def db_key(self, config):
        return (ARCH, NUM_CU, NUM_CHIPLETS, config.to_command_line())

    def test_read_tuning_db_matches_legacy_all_false_attention_flags(self):
        """Old DB rows may omit all false-valued attention flags."""
        current_config = make_config(
            "-with-attn-scale false -with-attn-bias false -transBias false")
        legacy_key = drop_flags(current_config.to_command_line(), " -with-attn-scale false",
                                " -with-attn-bias false", " -transBias false")

        db = self.read_db_from_legacy_key(legacy_key)
        self.assertEqual(db.get(self.db_key(current_config)), PERFCONFIG)

    def test_read_tuning_db_matches_pre_transbias_scale_bias_key(self):
        """Scale+bias DB rows from before transBias still match non-transposed bias."""
        current_config = make_config("-with-attn-scale true -with-attn-bias true -transBias false")
        legacy_key = drop_flags(current_config.to_command_line(), " -transBias false")

        db = self.read_db_from_legacy_key(legacy_key)
        self.assertEqual(db.get(self.db_key(current_config)), PERFCONFIG)

    def test_read_tuning_db_keeps_true_scale_bias_distinct(self):
        """True scale/bias flags must not fall back to old all-false DB rows."""
        all_false_config = make_config(
            "-with-attn-scale false -with-attn-bias false -transBias false")
        legacy_all_false_key = drop_flags(all_false_config.to_command_line(),
                                          " -with-attn-scale false", " -with-attn-bias false",
                                          " -transBias false")

        scale_bias_config = make_config(
            "-with-attn-scale true -with-attn-bias true -transBias false")

        db = self.read_db_from_legacy_key(legacy_all_false_key)
        self.assertNotIn(self.db_key(scale_bias_config), db)

    def test_read_tuning_db_keeps_true_trans_bias_distinct(self):
        """Transposed bias must not fall back to an old all-false DB row."""
        all_false_config = make_config(
            "-with-attn-scale false -with-attn-bias false -transBias false")
        legacy_all_false_key = drop_flags(all_false_config.to_command_line(),
                                          " -with-attn-scale false", " -with-attn-bias false",
                                          " -transBias false")

        trans_bias_config = make_config(
            "-with-attn-scale false -with-attn-bias true -transBias true")

        self.assertIn("-transBias true", trans_bias_config.to_command_line())
        db = self.read_db_from_legacy_key(legacy_all_false_key)
        self.assertNotIn(self.db_key(trans_bias_config), db)

    def test_quick_tuning_gen_defaults_missing_trans_bias_column(self):
        """Legacy debug TSV rows without TransBias get a false default."""
        debug_path = Path(f"{self.tmp_prefix}.debug")
        debug_path.write_text(
            "DataType\tChip\tnumCU\tnumChiplets\tTransQ\tTransK\tTransV\tTransO\t"
            "Causal\tReturnLSE\tSplitKV\tWithAttnScale\tWithAttnBias\tG\tSeqLenQ\t"
            "SeqLenK\tNumHeadsQ\tNumHeadsKV\tHeadDimQK\tHeadDimV\tPerfConfig\tTFlops\n"
            f"f16\tgfx950\t{NUM_CU}\t{NUM_CHIPLETS}\tFalse\tFalse\tFalse\tFalse\t"
            f"False\tFalse\t1\tTrue\tTrue\t1\t16\t16\t1\t1\t32\t32\t{PERFCONFIG}\t1.0\n")

        df = load_data([str(debug_path)], no_splitk=False)
        self.assertIn("TransBias", df.columns)
        self.assertTrue(df["TransBias"].eq(False).all())

        grouped = df.groupby(get_target_columns("attention") + ["PerfConfig"],
                             as_index=False)["TFlops"].max()
        self.assertFalse(grouped.empty)

    def test_quick_tuning_gen_fills_trans_bias_nan_in_mixed_files(self):
        """Mixing legacy (no TransBias) and new (TransBias) TSVs must not drop
        the legacy rows: concat leaves NaN in the legacy rows, and TransBias is
        a groupby key, so an unfilled NaN would be silently dropped."""
        cols_no_tb = ("DataType\tChip\tnumCU\tnumChiplets\tTransQ\tTransK\tTransV\tTransO\t"
                      "Causal\tReturnLSE\tSplitKV\tWithAttnScale\tWithAttnBias\tG\tSeqLenQ\t"
                      "SeqLenK\tNumHeadsQ\tNumHeadsKV\tHeadDimQK\tHeadDimV\tPerfConfig\tTFlops\n")
        legacy_path = Path(f"{self.tmp_prefix}.legacy.debug")
        legacy_path.write_text(
            cols_no_tb + f"f16\tgfx950\t{NUM_CU}\t{NUM_CHIPLETS}\tFalse\tFalse\tFalse\tFalse\t"
            f"False\tFalse\t1\tTrue\tTrue\t1\t16\t16\t1\t1\t32\t32\t{PERFCONFIG}\t1.0\n")

        cols_tb = (
            "DataType\tChip\tnumCU\tnumChiplets\tTransQ\tTransK\tTransV\tTransO\t"
            "Causal\tReturnLSE\tSplitKV\tWithAttnScale\tWithAttnBias\tTransBias\tG\tSeqLenQ\t"
            "SeqLenK\tNumHeadsQ\tNumHeadsKV\tHeadDimQK\tHeadDimV\tPerfConfig\tTFlops\n")
        new_path = Path(f"{self.tmp_prefix}.new.debug")
        new_path.write_text(
            cols_tb + f"f16\tgfx950\t{NUM_CU}\t{NUM_CHIPLETS}\tFalse\tFalse\tFalse\tFalse\t"
            f"False\tFalse\t1\tTrue\tTrue\tTrue\t1\t16\t16\t1\t1\t32\t32\t{PERFCONFIG}\t2.0\n")

        df = load_data([str(legacy_path), str(new_path)], no_splitk=False)
        self.assertFalse(df["TransBias"].isna().any())

        grouped = df.groupby(get_target_columns("attention") + ["PerfConfig"],
                             as_index=False)["TFlops"].max()
        # Both problems survive: legacy TransBias=False and new TransBias=True.
        # Without the NaN fill, the legacy row would be dropped and only one
        # group would remain.
        self.assertEqual(len(grouped), 2)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
