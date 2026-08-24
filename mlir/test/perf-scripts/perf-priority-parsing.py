#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pure-Python coverage for -perf_priority in problem configs.

Config files may carry a -perf_priority flag recording how much of a model's
runtime a config was responsible for, so the tuner knows what to work on first.
It describes the config's importance rather than the problem itself, so every
parser has to accept it, ignore it, and keep it out of the tuning DB key.

# RUN: %python %s
"""

from pathlib import Path
import sys
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

import perfRunner  # noqa: E402
from perfRunner import AttentionConfiguration, ConvConfiguration  # noqa: E402
from perfRunner import ConvGemmConfiguration, GemmConfiguration  # noqa: E402
from perfRunner import GemmGemmConfiguration, drop_perf_priority  # noqa: E402

# Pin the attention dtype list so config construction never probes real hardware.
perfRunner.DATA_TYPES_ATTENTION = perfRunner.DATA_TYPES_ATTENTION_MFMA

ARCH = "gfx950:sramecc+:xnack-"
NUM_CU = 256
NUM_CHIPLETS = 8

CONV = ("conv -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 "
        "-k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1")
GEMM = "-t f32 -out_datatype f32 -transA false -transB false -g 1 -m 1024 -n 1024 -k 1024"
GEMM_GEMM = ("-t f16 -transA false -transB false -transC false -transO false "
             "-g 1 -m 1024 -n 1024 -k 64 -gemmO 64")
CONV_GEMM = ("-t f16 -f NCHW -I NCHW -transC false -transO false -n 1 -c 64 -H 56 -W 56 "
             "-k 64 -y 1 -x 1 -gemmO 64 -u 1 -v 1 -p 0 -q 0 -l 1 -j 1 -g 1")
ATTENTION = ("-t f32 -transQ false -transK false -transV false -transO false "
             "-causal false -return_lse false -split_kv 1 -num_heads_q 1 -num_heads_kv 1 "
             "-g 1 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 "
             "-with-attn-scale false -with-attn-bias false")

# The trailing field is how many tokens precede the flags: conv configs lead with a
# positional datatype, the rest are flags all the way down.
CONFIGS = [
    ("conv", ConvConfiguration, CONV, 1),
    ("gemm", GemmConfiguration, GEMM, 0),
    ("gemm_gemm", GemmGemmConfiguration, GEMM_GEMM, 0),
    ("conv_gemm", ConvGemmConfiguration, CONV_GEMM, 0),
    ("attention", AttentionConfiguration, ATTENTION, 0),
]

# Parsers that report an unrecognized flag. Conv is excluded because its getopt
# call silently discards flags outside the option string.
STRICT_CONFIGS = [entry for entry in CONFIGS if entry[0] != "conv"]


def parse(conf_class, config_str):
    """Build a configuration the way perfRunner and tuningRunner both do."""
    return conf_class.from_command_line(config_str.split(), ARCH, NUM_CU, NUM_CHIPLETS)


def with_leading_priority(config_str, lead, value):
    """Put -perf_priority ahead of the flags rather than after them."""
    tokens = config_str.split()
    return " ".join(tokens[:lead] + ["-perf_priority", str(value)] + tokens[lead:])


class PerfPriorityParsingTest(unittest.TestCase):
    """-perf_priority must be accepted everywhere and change nothing."""

    def test_trailing_perf_priority_is_ignored(self):
        """Appending the flag leaves the parsed problem untouched."""
        for name, conf_class, config_str, _lead in CONFIGS:
            with self.subTest(op=name):
                plain = parse(conf_class, config_str)
                prioritized = parse(conf_class, f"{config_str} -perf_priority 15")
                self.assertEqual(plain.to_command_line(), prioritized.to_command_line())

    def test_leading_perf_priority_is_ignored(self):
        """The flag is dropped wherever the generating script put it."""
        for name, conf_class, config_str, lead in CONFIGS:
            with self.subTest(op=name):
                plain = parse(conf_class, config_str)
                prioritized = parse(conf_class, with_leading_priority(config_str, lead, 15))
                self.assertEqual(plain.to_command_line(), prioritized.to_command_line())

    def test_perf_priority_stays_out_of_the_tuning_key(self):
        """A priority must not split one problem into two tuning DB entries."""
        for name, conf_class, config_str, _lead in CONFIGS:
            with self.subTest(op=name):
                config = parse(conf_class, f"{config_str} -perf_priority 15")
                self.assertNotIn("perf_priority", config.to_command_line())

    def test_differing_priorities_give_the_same_key(self):
        """The same problem tuned from two models must still be one problem."""
        for name, conf_class, config_str, _lead in CONFIGS:
            with self.subTest(op=name):
                low = parse(conf_class, f"{config_str} -perf_priority 1")
                high = parse(conf_class, f"{config_str} -perf_priority 99")
                self.assertEqual(low.to_command_line(), high.to_command_line())

    def test_conv_padding_survives_perf_priority(self):
        """getopt reads -perf_priority as -p with the value "erf_priority"."""
        config = parse(ConvConfiguration, f"{CONV} -perf_priority 15")
        self.assertEqual(config.padding_h, 0)
        self.assertEqual(config.padding_w, 0)

    def test_unknown_flags_are_still_rejected(self):
        """Ignoring one flag must not turn the parsers permissive."""
        for name, conf_class, config_str, _lead in STRICT_CONFIGS:
            with self.subTest(op=name):
                with self.assertRaises(ValueError):
                    parse(conf_class, f"{config_str} -perf_priorities 15")

    def test_drop_perf_priority_removes_flag_and_value(self):
        """The MIOpen benchmark path reuses this to clean the argv it forwards."""
        self.assertEqual(drop_perf_priority(['-g', '1', '-perf_priority', '15', '-m', '2']),
                         ['-g', '1', '-m', '2'])
        self.assertEqual(drop_perf_priority(['-g', '1', '-perf_priority', '15']), ['-g', '1'])
        self.assertEqual(drop_perf_priority(['-g', '1']), ['-g', '1'])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
