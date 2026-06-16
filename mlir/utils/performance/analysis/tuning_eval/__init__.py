# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tuning evaluation harness.

A standalone, plug-n-play framework for comparing perfConfig "proposers"
(set cover, nearest-problem, learned model, random) against the absolute-best
TFlops recorded in exhaustive tuning corpora, under a fixed compile-time
budget. The harness is intentionally decoupled from the compiler: proposers
communicate via plain problem signatures and perfConfig strings, never MLIR ops.

See ``__main__.py`` (``python -m tuning_eval``) for the evaluation CLI and
``train.py`` for fitting + compiling a model to C.
"""

import sys
from pathlib import Path

# The submodules import flat helper modules that live in the parent
# directories: quickTuningGen in analysis/, and amd_arch_db / perfRunner /
# tuningRunner in performance/. Put both on sys.path once, here, so importing
# the package makes them available without each submodule repeating the dance.
_ANALYSIS_DIR = Path(__file__).resolve().parent.parent
_PERF_DIR = _ANALYSIS_DIR.parent
for _p in (_ANALYSIS_DIR, _PERF_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
