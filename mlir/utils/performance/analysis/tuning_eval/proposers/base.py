# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""ConfigProposer: the black-box interface every method conforms to."""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from ..corpus import Corpus, ProblemSig

# A pool provider maps a problem to its candidate perfConfig list. The only
# source of truth is the per-problem applicable tuning space from
# EmitTuningSpacePool (rocmlir-gen --emit-tuning-space); proposers that rank a
# pool (random, model) require one.
PoolProvider = Callable[[ProblemSig], List[str]]


class ConfigProposer(ABC):
    """Given a problem and a budget, emit an ordered list of perfConfigs.

    Subclasses must set ``name`` and implement ``fit`` and ``propose``.
    Proposers may fit on training problems (set cover, model, nearest) or
    ignore the training data (random). They source candidates from their own
    self-contained state -- never by reaching into the compiler directly; the
    only compiler dependency is the injectable ``pool_provider``.
    """

    name: str = "base"

    def __init__(self, pool_provider: Optional[PoolProvider] = None):
        self._pool_provider = pool_provider

    @abstractmethod
    def fit(self, train: Corpus) -> None:
        """Train/index on the training corpus. May be a no-op."""

    @abstractmethod
    def propose(self, sig: ProblemSig, budget: int) -> List[str]:
        """Return up to ``budget`` perfConfigs, best-first."""

    def _candidate_pool(self, sig: ProblemSig) -> List[str]:
        """Per-problem candidate pool from the injected ``pool_provider`` (the
        applicable tuning space from rocmlir-gen --emit-tuning-space)."""
        if self._pool_provider is None:
            raise RuntimeError(f"{self.name}: a pool provider is required "
                               "(rocmlir-gen --emit-tuning-space)")
        return self._pool_provider(sig)
