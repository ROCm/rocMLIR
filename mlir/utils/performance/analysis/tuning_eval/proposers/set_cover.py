# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Set-cover proposer: today's quick-tuning behavior, the bar to beat.

Fits the ILP set cover on the *training* problems (reusing
quickTuningGen.solve_set_cover) and, for an unseen test problem, returns the
global cover ordered by how many training problems each config covers --
exactly the unknown-problem fallback the shipped DB uses.

The harness fits the proposer on a single arch's data at a time (the
orchestration groups by arch), so a fitted instance only ever covers one arch.

The per-dtype covers go straight through quickTuningGen.solve_set_cover. The
cross-dtype fallback (used for the held-out-dtype split: an unseen dtype of the
arch) cannot, because solve_set_cover groups problems by the op's target columns
alone -- so the same shape under two dtypes would collapse into one group and its
TFlops would be compared across dtypes. The fallback instead computes coverage
per (dtype-shard, problem) (each relative to its *own* best) and solves one ILP
over that arch's dtypes, so throughput is never compared across dtypes.
"""

import math
from typing import Dict, List, Tuple

import pandas as pd
import pulp

from ..corpus import Corpus, ProblemSig, TableKey
from ..features import DEFAULT_THRESHOLD
from .base import ConfigProposer

# quickTuningGen is put on sys.path by the package __init__.
import quickTuningGen

# A coverage element is one concrete (shard, problem); its key is sortable for
# determinism. ``CoverElem = (table_key, problem_key)``.
CoverElem = Tuple[TableKey, str]


class SetCoverProposer(ConfigProposer):
    name = "set_cover"

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        super().__init__()
        self._threshold = threshold
        # Per (arch,op,dtype): the ordered cover, and the problem-hash -> cover
        # index map (the same structure QuickTuningDb::lookup consumes).
        self._covers: Dict[TableKey, List[str]] = {}
        self._problem_maps: Dict[TableKey, Dict[int, List[int]]] = {}
        # Cross-dtype fallback (held-out dtype) over the fitted arch's dtypes.
        self._fallback_cover: List[str] = []
        self._fallback_problem_map: Dict[int, List[int]] = {}

    def fit(self, train: Corpus) -> None:
        self._covers = {}
        self._problem_maps = {}
        for key in train.keys():
            df = self._build_frame(train, key)
            if df.empty:
                continue
            _, op, _ = key
            cover, problem_map = self._solve(df, op)
            self._covers[key] = cover
            self._problem_maps[key] = problem_map
        # Cross-dtype fallback for the held-out-dtype split, solved over the
        # arch's dtypes while keeping per-(dtype-shard, problem) identity (see
        # module docstring).
        self._fallback_cover, self._fallback_problem_map = self._solve_fallback(train)

    def propose(self, sig: ProblemSig, budget: int) -> List[str]:
        cover, _ = self.cover_and_map(sig.table_key)
        return cover[:budget]

    def cover_and_map(self, key: TableKey) -> Tuple[List[str], Dict[int, List[int]]]:
        """Ordered cover and problem-hash -> cover-index map for a key, falling
        back to the cross-dtype cover when the exact (dtype) key is unseen."""
        cover = self._covers.get(key)
        if cover is not None:
            return cover, self._problem_maps.get(key, {})
        return self._fallback_cover, self._fallback_problem_map

    def _solve(self, df: pd.DataFrame, op: str) -> Tuple[List[str], Dict[int, List[int]]]:
        return quickTuningGen.solve_set_cover(df, op, self._threshold)

    def _solve_fallback(self, train: Corpus) -> Tuple[List[str], Dict[int, List[int]]]:
        """Solve one set cover over every (dtype) shard of the fitted arch,
        treating each (dtype-shard, problem) as an independent element covered by
        the configs within ``threshold`` of *that element's own* best. No TFlops
        are compared across dtypes."""
        coverage: Dict[CoverElem, List[str]] = {}
        for key in train.keys():
            for problem_key, cfgs in train.tables.get(key, {}).items():
                valid = {c: t for c, t in cfgs.items() if t is not None and not math.isnan(t)}
                if not valid:
                    continue
                best = max(valid.values())
                good = [c for c, t in valid.items() if t >= best * self._threshold]
                if good:
                    coverage[(key, problem_key)] = good
        return _cover_from_coverage(coverage)

    @staticmethod
    def _build_frame(train: Corpus, key: TableKey) -> pd.DataFrame:
        _, op, _ = key
        target_cols = quickTuningGen.get_target_columns(op)
        rows: List[Tuple] = []
        for problem_key, cfgs in train.tables.get(key, {}).items():
            col_tuple = train.problem_cols[key][problem_key]
            for perf_config, tflops in cfgs.items():
                if tflops is None or math.isnan(tflops):
                    continue
                rows.append((*col_tuple, perf_config, tflops))
        return pd.DataFrame(rows, columns=target_cols + ['PerfConfig', 'TFlops'])


def _cover_from_coverage(
        coverage: Dict[CoverElem, List[str]]) -> Tuple[List[str], Dict[int, List[int]]]:
    """Minimal set cover over pre-computed per-element coverage sets.

    Each element (a concrete (shard, problem)) already lists the configs that
    are good enough *for it*, so this only solves the ILP -- it never compares
    TFlops across elements. Returns the cover ordered by how many elements each
    config covers, and a problem-key-hash -> cover-index map (hashes merged when
    a shape recurs across shards), matching SetCoverProposer's per-key output."""
    if not coverage:
        return [], {}
    elements = sorted(coverage.keys())
    configs = sorted({c for good in coverage.values() for c in good})
    cfg_idx = {c: j for j, c in enumerate(configs)}
    elem_cfg_idxs = [[cfg_idx[c] for c in coverage[el]] for el in elements]

    prob = pulp.LpProblem("FallbackSetCover", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", range(len(configs)), cat="Binary")
    prob += pulp.lpSum(x.values())
    for idxs in elem_cfg_idxs:
        prob += pulp.lpSum(x[j] for j in idxs) >= 1
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if status != pulp.LpStatusOptimal:
        raise RuntimeError(f"fallback set cover failed: {pulp.LpStatus.get(status, 'Unknown')}")

    selected = [j for j in range(len(configs)) if x[j].varValue == 1]
    counts = {j: sum(1 for idxs in elem_cfg_idxs if j in idxs) for j in selected}
    selected.sort(key=lambda j: counts[j], reverse=True)
    cover = [configs[j] for j in selected]
    cover_pos = {j: i for i, j in enumerate(selected)}

    problem_map: Dict[int, List[int]] = {}
    for el, idxs in zip(elements, elem_cfg_idxs):
        h = quickTuningGen.hash_problem_key(el[1])
        covering = [cover_pos[j] for j in idxs if j in cover_pos]
        problem_map[h] = sorted(set(problem_map.get(h, []) + covering))
    return cover, problem_map
