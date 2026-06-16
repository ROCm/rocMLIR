# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Model proposer: an offline-trained two-stage LightGBM classifier.

Two stages, both gradient-boosted tree classifiers over the same
(problem, config) features:

  1. *Applicability* -- predict whether the pair produces a valid measurement
     (the codegen accepts it and it runs). Trained on every recorded pair,
     using NaN measurements as the negative class.
  2. *Optimality* -- predict whether the pair is "within threshold of the
     per-problem best" (the same label the set cover uses). Trained only on
     the applicable pairs, so it ranks runnable configs rather than learning
     to avoid the inapplicable ones a second time.

At propose time we tier: predicted-applicable configs first (ordered by
predicted optimality), then the rest (ordered by predicted applicability) as a
fallback. This kills the "proposed only inapplicable configs" tail while
keeping the optimality ranking that drives average-case regret down.

The model is arch-agnostic: it fits one model on whatever corpus it is given.
The orchestration (``tuning_eval.train`` / ``python -m tuning_eval``) groups the
corpus by arch and fits one model per arch from a single-arch corpus -- so data
is never mixed across architectures, mirroring quickTuningGen's per-(arch, op)
grouping. Training is always from scratch on the full data passed to ``fit``.
The fitted boosters are exposed via ``stage_boosters`` so the deployment path
can compile them to C; see ``export.py``.
"""

import math
import os
import random
from collections import defaultdict, deque
from typing import Callable, List, Optional, Tuple

import pandas as pd

from ..corpus import Corpus, ProblemSig
from ..features import DEFAULT_THRESHOLD, feature_record, label
from .base import ConfigProposer, PoolProvider

# A training item before featurization:
#   (sig, config, applicable, optimal)
# ``optimal`` is only meaningful when ``applicable`` is 1.
_Item = Tuple[ProblemSig, str, int, int]

# Stage names, also used as the namespace for exported C bundles.
APPLICABILITY = "applicability"
OPTIMALITY = "optimality"


def _valid(tflops) -> bool:
    return tflops is not None and not math.isnan(tflops) and tflops > 0


def _has(sig: ProblemSig, name: str) -> bool:
    return name in sig.column_names


def _group_key(sig: ProblemSig) -> str:
    """Coarse problem group for balanced subsampling under the training cap.

    CV showed balancing helps only for conv *direction* (bwd/wrw are tiny
    minorities drowned out by fwd). Splitting further by dtype or gemm/attn
    structural axes over-samples rare groups and regresses the majority
    (conv fwd f32 -14pp; gemm uniformly worse). So only conv names direction;
    every other op shares a single queue (a no-op for balancing)."""
    if sig.op == "conv" and _has(sig, "Direction"):
        return "conv:" + str(sig.column("Direction"))
    return sig.op


class ModelProposer(ConfigProposer):
    name = "model"

    def __init__(self,
                 threshold: float = DEFAULT_THRESHOLD,
                 n_estimators: int = 100,
                 seed: int = 0,
                 pool_provider: Optional[PoolProvider] = None,
                 max_train_pairs: Optional[int] = None,
                 applic_threshold: float = 0.5,
                 group_subsample: bool = True,
                 balanced_class_weight: bool = True,
                 max_depth: Optional[int] = None,
                 learning_rate: float = 0.1):
        super().__init__(pool_provider)
        self._threshold = threshold
        self._n_estimators = n_estimators
        self._seed = seed
        # LightGBM (gradient-boosted trees) backend. max_depth caps tree depth
        # (None = -1, no limit); learning_rate is the boosting step size.
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        # Two independent training knobs (decoupled on purpose):
        #  - group_subsample: when over the cap, draw each stratum round-robin
        #    across _group_key groups so rare problem groups (conv bwd/wrw)
        #    survive instead of being washed out. No-op for single-group ops.
        #  - balanced_class_weight: weight classes inversely to frequency so the
        #    rare positive label (applicable / near-best) is not drowned by the
        #    common negative. Orthogonal to subsampling.
        self._group_subsample = group_subsample
        self._balanced_class_weight = balanced_class_weight
        # Cap on (problem, config) training rows for large corpora. The
        # subsample is stratified to retain rare positives when possible.
        self._max_train_pairs = max_train_pairs
        # Probability above which a config is treated as applicable at propose
        # time (tier boundary, not a hard filter).
        self._applic_threshold = applic_threshold
        self._clf_applic = None
        self._clf_optimal = None
        self._feature_names: Optional[List[str]] = None

    def is_fitted(self) -> bool:
        """True once ``fit`` has produced at least one classifier."""
        return self._feature_names is not None and (self._clf_applic is not None or
                                                    self._clf_optimal is not None)

    def stage_boosters(self) -> List[Tuple[str, object]]:
        """(stage_name, LightGBM Booster) for each fitted stage, for export."""
        out = []
        if self._clf_applic is not None:
            out.append((APPLICABILITY, self._clf_applic.booster_))
        if self._clf_optimal is not None:
            out.append((OPTIMALITY, self._clf_optimal.booster_))
        return out

    def fit(self, train: Corpus) -> None:
        """Fit both stages on ``train`` from scratch.

        ``train`` is expected to be a single arch's data (the orchestration
        groups by arch); the model itself is arch-agnostic and simply fits on
        whatever it is given."""
        self._clf_applic = None
        self._clf_optimal = None
        self._feature_names = None

        # Build training items: one (sig, config, applicable, optimal) per
        # recorded (problem, config) pair. Stage 1 trains on all of them;
        # stage 2 trains only on the applicable ones.
        items: List[_Item] = []
        for key in train.keys():
            for problem_key in train.problem_keys(key):
                sig = train.sig(key, problem_key)
                best = train.best(key, problem_key)
                for perf_config, tflops in train.measured(key, problem_key).items():
                    applicable = int(_valid(tflops))
                    optimal = label(tflops, best, self._threshold) if applicable else 0
                    items.append((sig, perf_config, applicable, optimal))

        if not items:
            return

        # Subsample *before* featurizing -- on the exhaustive corpus this is the
        # difference between featurizing millions of pairs and a few hundred
        # thousand.
        items = self._subsample(items)

        rows: List[List[float]] = []
        applic_labels: List[int] = []
        optimal_labels: List[int] = []
        for sig, perf_config, applicable, optimal in items:
            rec = feature_record(sig, perf_config)
            if self._feature_names is None:
                self._feature_names = list(rec.keys())
            rows.append([rec[name] for name in self._feature_names])
            applic_labels.append(applicable)
            optimal_labels.append(optimal)

        # Stage 1: applicability over every row.
        self._clf_applic = self._fit_clf(rows, applic_labels)

        # Stage 2: optimality over only the applicable rows ("given it runs, is
        # it near-best?").
        applic_idx = [i for i, a in enumerate(applic_labels) if a == 1]
        opt_rows = [rows[i] for i in applic_idx]
        opt_labels = [optimal_labels[i] for i in applic_idx]
        self._clf_optimal = self._fit_clf(opt_rows, opt_labels)

    def _fit_clf(self, rows: List[List[float]], labels: List[int]):
        # Single-class (or empty) training data -> no usable classifier.
        if not rows or len(set(labels)) < 2:
            return None
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth if self._max_depth else -1,
            learning_rate=self._learning_rate,
            class_weight="balanced" if self._balanced_class_weight else None,
            random_state=self._seed,
            # Reproducible across runs/threads (so the eval and the exported C
            # are stable for a fixed seed).
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
            # Bounded, concrete thread count. lightgbm's n_jobs=-1 sentinel and
            # very high thread counts stall on machines with many cores (OpenMP
            # oversubscription on small training sets); a modest cap is fast
            # everywhere and, with deterministic=True, leaves the result thread
            # -count independent.
            n_jobs=min(os.cpu_count() or 1, 16))
        # Fit on a named DataFrame so the booster (and the exported C) carry the
        # real feature names, and predict-time inputs (also named) line up --
        # avoiding sklearn's nameless-array feature-name warning.
        clf.fit(self._frame(rows), labels)
        return clf

    def _frame(self, rows: List[List[float]]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=self._feature_names, dtype=float)

    def _balanced_take(self, idxs: List[int], n: int, group_of: Callable[[int], str]) -> List[int]:
        """Take ``n`` indices spread across problem groups via round-robin, so a
        rare group (e.g. bwd conv) is fully retained instead of being sampled
        away by a dominant group. Deterministic."""
        if len(idxs) <= n:
            return idxs
        buckets = defaultdict(list)
        for i in idxs:
            buckets[group_of(i)].append(i)
        rng = random.Random(self._seed)
        queues = []
        for g in sorted(buckets):
            b = buckets[g]
            rng.shuffle(b)
            queues.append(deque(b))
        out: List[int] = []
        while len(out) < n and any(queues):
            for q in queues:
                if q:
                    out.append(q.popleft())
                    if len(out) >= n:
                        break
        return out

    def _subsample(self, items: List[_Item]) -> List[_Item]:
        """Three-way stratified cap: keep the rare optimal positives, plus a
        mix of applicable-but-suboptimal and inapplicable rows so both
        classifiers see two classes. Group-balanced within each stratum so rare
        problem groups (e.g. bwd conv) survive the cap. Deterministic."""
        cap = self._max_train_pairs
        if cap is None or len(items) <= cap:
            return items
        opt = [i for i, it in enumerate(items) if it[2] == 1 and it[3] == 1]
        valid_only = [i for i, it in enumerate(items) if it[2] == 1 and it[3] == 0]
        invalid = [i for i, it in enumerate(items) if it[2] == 0]
        rng = random.Random(self._seed)
        group_of = lambda i: _group_key(items[i][0])  # noqa: E731

        def take(idxs: List[int], n: int) -> List[int]:
            if self._group_subsample:
                return self._balanced_take(idxs, n, group_of)
            return idxs if len(idxs) <= n else rng.sample(idxs, n)

        third = cap // 3
        keep_opt = take(opt, third)
        rem = cap - len(keep_opt)
        keep_valid = take(valid_only, rem // 2)
        keep_inval = take(invalid, rem - len(keep_valid))
        keep = keep_opt + keep_valid + keep_inval
        rng.shuffle(keep)
        return [items[i] for i in keep]

    def propose(self, sig: ProblemSig, budget: int) -> List[str]:
        pool = self._candidate_pool(sig)
        if not pool:
            return []
        if not self.is_fitted():
            return list(pool)[:budget]
        clf_applic, clf_optimal = self._clf_applic, self._clf_optimal

        vecs = []
        for cfg in pool:
            rec = feature_record(sig, cfg)
            vecs.append([rec[name] for name in self._feature_names])
        n = len(pool)

        p_applic = self._positive_proba(clf_applic, vecs) if clf_applic else [1.0] * n

        # Stage 1 partitions the pool; stage 2 (optimality) only runs on the
        # predicted-applicable tier.
        tier_a_idx = [i for i in range(n) if p_applic[i] >= self._applic_threshold]
        tier_b_idx = [i for i in range(n) if p_applic[i] < self._applic_threshold]

        if clf_optimal and tier_a_idx:
            p_opt = self._positive_proba(clf_optimal, [vecs[i] for i in tier_a_idx])
            tier_a = sorted(zip(tier_a_idx, p_opt),
                            key=lambda ip: (ip[1], p_applic[ip[0]]),
                            reverse=True)
        else:
            tier_a = sorted(((i, p_applic[i]) for i in tier_a_idx),
                            key=lambda ip: ip[1],
                            reverse=True)
        tier_b = sorted(((i, p_applic[i]) for i in tier_b_idx), key=lambda ip: ip[1], reverse=True)

        ranked = [pool[i] for i, _ in tier_a] + [pool[i] for i, _ in tier_b]
        return ranked[:budget]

    def _positive_proba(self, clf, vecs: List[List[float]]) -> List[float]:
        classes = list(clf.classes_)
        proba = clf.predict_proba(self._frame(vecs))
        if 1 in classes:
            pos = classes.index(1)
            return [row[pos] for row in proba]
        # Positive class never observed; treat all as equally unpromising.
        return [0.0 for _ in vecs]
