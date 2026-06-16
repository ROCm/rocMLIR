# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Random proposer: the sanity floor every other method must beat."""

import random
from typing import List, Optional

from ..corpus import Corpus, ProblemSig
from .base import ConfigProposer, PoolProvider

# quickTuningGen is put on sys.path by the package __init__.
import quickTuningGen


class RandomProposer(ConfigProposer):
    name = "random"

    def __init__(self, seed: int = 0, pool_provider: Optional[PoolProvider] = None):
        super().__init__(pool_provider)
        self._seed = seed

    def fit(self, train: Corpus) -> None:
        # Random ignores the training data; the pool comes from the provider.
        pass

    def propose(self, sig: ProblemSig, budget: int) -> List[str]:
        pool = list(self._candidate_pool(sig))
        # Deterministic per-problem ordering so reports are reproducible across
        # processes. Python's str/tuple __hash__ is salted per interpreter, so
        # seed the RNG from the stable xxh3_64 problem-key hash instead.
        token = f"{self._seed}|{sig.arch}|{sig.op}|{sig.dtype}|{sig.problem_key}"
        rng = random.Random(quickTuningGen.hash_problem_key(token))
        rng.shuffle(pool)
        return pool[:budget]
