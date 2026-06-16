# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Metrics: regret@k, coverage@k, proposal coverage, and the eval loop.

regret@k is measured against the absolute best recorded for the problem, so
the set cover and a model are compared on the same "how close to optimal in
k picks" axis. A proposed config that has no valid measurement counts as a
wasted-budget pick (contributes 0 achieved TFlops).

Scoring is done directly against the recorded corpus oracle: a proposed config
with no recorded (or a failed/NaN) measurement scores NaN, i.e. wasted budget.
"""

import math
from typing import Dict, List

from .corpus import Corpus, ProblemSig
from .features import DEFAULT_THRESHOLD
from .proposers.base import ConfigProposer


def _valid(value: float) -> bool:
    return value is not None and not math.isnan(value)


def regret_at_k(ordered: List[str], achieved: Dict[str, float], best: float, k: int) -> float:
    """1 - (best achieved within top-k) / (absolute best). NaN if no ref."""
    if not _valid(best) or best <= 0:
        return float('nan')
    top = ordered[:k]
    vals = [achieved.get(c, float('nan')) for c in top]
    valid = [v for v in vals if _valid(v)]
    achieved_best = max(valid) if valid else 0.0
    return max(0.0, min(1.0, 1.0 - achieved_best / best))


def coverage_at_k(ordered: List[str],
                  achieved: Dict[str, float],
                  best: float,
                  k: int,
                  threshold: float = DEFAULT_THRESHOLD) -> float:
    """1.0 if some top-k config is within ``threshold`` of best, else 0.0."""
    if not _valid(best) or best <= 0:
        return float('nan')
    for c in ordered[:k]:
        v = achieved.get(c, float('nan'))
        if _valid(v) and v >= best * threshold:
            return 1.0
    return 0.0


def proposal_coverage(ordered: List[str], achieved: Dict[str, float]) -> float:
    """Fraction of proposed configs that have a valid measurement."""
    if not ordered:
        return float('nan')
    valid = sum(1 for c in ordered if _valid(achieved.get(c, float('nan'))))
    return valid / len(ordered)


def evaluate(proposer: ConfigProposer,
             corpus: Corpus,
             test_sigs: List[ProblemSig],
             budgets: List[int],
             threshold: float = DEFAULT_THRESHOLD) -> List[Dict]:
    """Score a fitted proposer over test problems at each budget.

    Returns one row per (problem, budget). Proposes once at the max budget and
    evaluates prefixes, so ordering is consistent across budgets.
    """
    max_budget = max(budgets)
    rows: List[Dict] = []
    for sig in test_sigs:
        ordered = proposer.propose(sig, max_budget)
        recorded = corpus.measured(sig.table_key, sig.problem_key)
        achieved = {cfg: recorded.get(cfg, float('nan')) for cfg in ordered}
        best = corpus.best(sig.table_key, sig.problem_key)
        for k in budgets:
            rows.append({
                "proposer": proposer.name,
                "arch": sig.arch,
                "op": sig.op,
                "dtype": sig.dtype,
                "problem_key": sig.problem_key,
                "budget": k,
                "regret": regret_at_k(ordered, achieved, best, k),
                "coverage": coverage_at_k(ordered, achieved, best, k, threshold),
                "proposal_coverage": proposal_coverage(ordered[:k], achieved),
                "best": best,
            })
    return rows
