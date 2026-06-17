# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Nearest-known proposer: the deployed per-problem lookup, relaxed.

Mirrors how the shipped quick-tuning DB serves a problem: hash the problem and,
on a hit, return exactly the cover entries that ``problem_map`` records for it
(`QuickTuningDb::lookup`). Since an unseen problem is always a hash miss, we
relax to the *nearest known* training problem (in standardized problem space)
and reuse its ``problem_map`` entry, topping up from the global cover order.

Everything routes through the set cover + problem_map produced by
``quickTuningGen.solve_set_cover`` (via SetCoverProposer), so this baseline only
ever proposes deployable cover entries -- no raw-oracle peeking.
"""

import math
from typing import Dict, List, Optional, Tuple

from ..corpus import Corpus, ProblemSig, TableKey
from ..features import distance_features, feature_record
from .base import ConfigProposer
from .set_cover import SetCoverProposer

# quickTuningGen is put on sys.path by the package __init__.
import quickTuningGen


class NearestKnownProposer(ConfigProposer):
    name = "nearest"

    def __init__(self, set_cover: Optional[SetCoverProposer] = None, k_neighbors: int = 1):
        super().__init__()
        # An externally supplied SetCoverProposer is assumed already fitted on
        # the same train corpus (lets the driver avoid solving the ILP twice).
        self._sc = set_cover
        self._owns_sc = set_cover is None
        self._k = k_neighbors
        # table_key -> list[(standardized_vec, problem_hash)]
        self._index: Dict[TableKey, List[Tuple[List[float], int]]] = {}
        self._stats: Dict[TableKey, Tuple[List[float], List[float]]] = {}

    def fit(self, train: Corpus) -> None:
        if self._owns_sc:
            self._sc = SetCoverProposer()
            self._sc.fit(train)
        self._index = {}
        self._stats = {}
        for key in train.keys():
            raw_vecs: List[List[float]] = []
            hashes: List[int] = []
            for problem_key in train.problem_keys(key):
                sig = train.sig(key, problem_key)
                raw_vecs.append(self._raw_vec(sig, self._any_config(train, key, problem_key)))
                hashes.append(quickTuningGen.hash_problem_key(problem_key))
            if not raw_vecs:
                continue
            mean, std = self._compute_stats(raw_vecs)
            self._stats[key] = (mean, std)
            self._index[key] = [
                (self._standardize(v, mean, std), h) for v, h in zip(raw_vecs, hashes)
            ]

    def propose(self, sig: ProblemSig, budget: int) -> List[str]:
        cover, problem_map = self._sc.cover_and_map(sig.table_key)
        if not cover:
            return []
        h = quickTuningGen.hash_problem_key(sig.problem_key)
        idxs = problem_map.get(h)
        if idxs is None:
            idxs = self._nearest_idxs(sig, problem_map, cover[0])

        result = [cover[i] for i in idxs if i < len(cover)]
        if len(result) < budget:
            seen = set(result)
            for cfg in cover:  # top up in global coverage order
                if cfg not in seen:
                    result.append(cfg)
                    seen.add(cfg)
                if len(result) >= budget:
                    break
        return result[:budget]

    # ---- helpers ------------------------------------------------------

    def _nearest_idxs(self, sig: ProblemSig, problem_map: Dict[int, List[int]],
                      config: str) -> List[int]:
        index = self._index.get(sig.table_key)
        if not index:
            return []
        mean, std = self._stats[sig.table_key]
        query = self._standardize(self._raw_vec(sig, config), mean, std)
        neighbors = sorted(index, key=lambda row: self._dist(query, row[0]))
        idxs: List[int] = []
        seen = set()
        for _, h in neighbors[:max(self._k, 1)]:
            for i in problem_map.get(h, []):
                if i not in seen:
                    seen.add(i)
                    idxs.append(i)
        return idxs

    @staticmethod
    def _any_config(train: Corpus, key: TableKey, problem_key: str) -> str:
        """A config to featurize the problem with. The distance features are
        problem-only, so any recorded config of the problem yields the same
        values; we just need one the compiler will accept."""
        return next(iter(train.measured(key, problem_key)))

    def _raw_vec(self, sig: ProblemSig, config: str) -> List[float]:
        rec = feature_record(sig, config)
        return [rec[name] for name in distance_features(sig.op)]

    @staticmethod
    def _compute_stats(vecs: List[List[float]]) -> Tuple[List[float], List[float]]:
        n = len(vecs)
        dim = len(vecs[0])
        mean = [sum(v[i] for v in vecs) / n for i in range(dim)]
        std = []
        for i in range(dim):
            var = sum((v[i] - mean[i])**2 for v in vecs) / n
            std.append(math.sqrt(var) if var > 0 else 1.0)
        return mean, std

    @staticmethod
    def _standardize(vec: List[float], mean: List[float], std: List[float]) -> List[float]:
        return [(vec[i] - mean[i]) / std[i] for i in range(len(vec))]

    @staticmethod
    def _dist(a: List[float], b: List[float]) -> float:
        return sum((a[i] - b[i])**2 for i in range(len(a)))
