# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Expand standardized tier1 config files into the set of expected problems.

The tier1 config files (``configs/tier1-<op>-configs``) are the source of truth
for which problems *should* have been tuned. tuningRunner expands each line
(filling in missing dtype/trans axes) and canonicalizes it before tuning, so the
.debug data the harness consumes is exactly that expanded set -- minus any
problem whose tuning failed or was skipped.

Rather than re-implement that expansion (and risk drifting from the data
generator), this module reuses tuningRunner/perfRunner directly:

  * ``tuningRunner.load_configs`` -- the same expansion+canonicalization
    tuningRunner itself runs;
  * ``<ConfigClass>.from_command_line`` + ``table_entry`` -- the same objects
    that emit the .debug rows, so the derived problem key is identical to the
    one ``corpus.py`` builds via ``quickTuningGen``.

The result lets the harness report which expected problems are missing from a
loaded corpus (i.e. a data-collection gap, like the large-head-dim attention
problems whose exhaustive sweep never completed).
"""

from typing import Callable, Dict, Set, Tuple

import pandas as pd

# quickTuningGen (analysis/) and tuningRunner / perfRunner / perfCommonUtils
# (performance/) are put on sys.path by the package __init__.
import quickTuningGen

# (dtype, problem_key) -- matches the corpus's (table_key[2], problem_key).
ExpectedProblem = Tuple[str, str]


def expected_problems(op: str, configs_file: str, arch: str, num_cu: int,
                      num_chiplets: int) -> Set[ExpectedProblem]:
    """Expand ``configs_file`` for ``op`` into the set of (dtype, problem_key).

    Reuses tuningRunner's loader and the perfRunner config classes so the keys
    line up bit-for-bit with what ``corpus.py`` derives from the .debug rows."""
    import tuningRunner
    from perfCommonUtils import Operation

    op_type = Operation.from_name(op)
    conf_class = tuningRunner.get_config_class(op_type)
    canonical = tuningRunner.load_configs(op_type, configs_file, arch, num_cu, num_chiplets)
    target_cols = quickTuningGen.get_target_columns(op)

    rows = []
    for test_vector in canonical:
        conf = conf_class.from_command_line(test_vector.split(), arch, num_cu, num_chiplets)
        rows.append(conf.table_entry(float('nan')))
    if not rows:
        return set()

    # Reproduce corpus.py's normalization exactly: bool columns -> int so the
    # joined problem key matches (e.g. TransA True -> "1", not "True").
    df = pd.DataFrame(rows)
    bool_cols = df.select_dtypes(include='bool').columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)

    out: Set[ExpectedProblem] = set()
    for row in df.itertuples(index=False):
        row_d = row._asdict()
        key = quickTuningGen.make_problem_key(tuple(row_d[c] for c in target_cols))
        out.add((str(row_d['DataType']), key))
    return out


def corpus_problems(corpus, op: str, arch: str) -> Set[ExpectedProblem]:
    """The (dtype, problem_key) pairs present in ``corpus`` for one (arch, op)."""
    have: Set[ExpectedProblem] = set()
    for key in corpus.keys():
        if key[0] != arch or key[1] != op:
            continue
        for pk in corpus.problem_keys(key):
            have.add((key[2], pk))
    return have


def expected_by_arch(corpus, op: str, configs_file: str,
                     log: Callable[..., None]) -> Dict[str, Set[ExpectedProblem]]:
    """Expand the config spec *per corpus arch* and report coverage per arch.

    tuningRunner.load_configs is arch/topology-aware (it expands with each
    arch's num_cu/num_chiplets), so the expected problem set is computed once
    per arch present in the corpus rather than once globally. Returns
    ``arch -> {(dtype, problem_key)}`` so the caller can restrict each shard to
    its own arch's expected set (a shape expected for one arch is never used to
    keep/drop another arch's problems)."""
    arches = sorted({k[0] for k in corpus.keys() if k[1] == op})
    out: Dict[str, Set[ExpectedProblem]] = {}
    for arch in arches:
        num_cu, num_chiplets = corpus.arch_meta.get(arch, (0, 0))
        expected = expected_problems(op, configs_file, arch, num_cu or 0, num_chiplets or 0)
        out[arch] = expected
        have = corpus_problems(corpus, op, arch)
        missing = sorted(expected - have)
        extra = sorted(have - expected)
        log(f"config spec [{arch}]: {len(expected)} expected from {configs_file}; "
            f"corpus has {len(have)} ({len(expected & have)} matched, "
            f"{len(missing)} missing, {len(extra)} extra not in spec)")
        if missing:
            by_dtype: Dict[str, list] = {}
            for dtype, pk in missing:
                by_dtype.setdefault(dtype, []).append(pk)
            for dtype in sorted(by_dtype):
                log(f"  MISSING from data [{arch}/{dtype}] ({len(by_dtype[dtype])}):")
                for pk in by_dtype[dtype]:
                    log(f"    {pk}")
    return out
