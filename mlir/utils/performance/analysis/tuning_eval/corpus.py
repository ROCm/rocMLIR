# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Corpus loading: turn tuningRunner .debug files into an evaluation oracle.

Reuses quickTuningGen's loaders/column definitions so the harness and the
quick-tuning DB generator never drift. The oracle maps

    (arch, op, dtype) -> problem_key -> {perfConfig: best_tflops}

where ``best_tflops`` is the max recorded TFlops for that (problem, config)
pair (NaN means every measurement failed/timed out). The candidate pool a
proposer ranks comes from the compiler (rocmlir-gen --emit-tuning-space), not
from this corpus; the corpus is purely the measurement oracle.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

# quickTuningGen / amd_arch_db are put on sys.path by the package __init__.
import quickTuningGen

# A table key uniquely identifies a (arch, op, dtype) shard of the corpus.
TableKey = Tuple[str, str, str]


@dataclass(frozen=True)
class ProblemSig:
    """Full, serializable description of a single tuning problem.

    Carries everything a proposer needs to decide its own scope (per-dtype
    vs dtype-as-feature, per-arch vs arch-as-feature) without the harness
    dictating it.
    """
    arch: str
    op: str
    dtype: str
    problem_key: str
    column_names: Tuple[str, ...]
    columns: Tuple[object, ...]
    num_cu: Optional[int] = None
    num_chiplets: Optional[int] = None

    @property
    def table_key(self) -> TableKey:
        return (self.arch, self.op, self.dtype)

    def column(self, name: str):
        """Look up a raw problem column by name (e.g. 'M')."""
        return self.columns[self.column_names.index(name)]


@dataclass
class Corpus:
    """In-memory oracle built from one or more .debug files."""

    # (arch, op, dtype) -> problem_key -> {perfConfig: best_tflops}
    tables: Dict[TableKey, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    # (arch, op, dtype) -> problem_key -> raw column tuple
    problem_cols: Dict[TableKey, Dict[str, Tuple[object, ...]]] = field(default_factory=dict)
    # arch -> (num_cu, num_chiplets); best-effort, may be (None, None)
    arch_meta: Dict[str, Tuple[Optional[int], Optional[int]]] = field(default_factory=dict)

    # ---- construction -------------------------------------------------

    @classmethod
    def from_debug_files(cls, files: List[str], no_splitk: bool = False) -> "Corpus":
        """Build a corpus from tuningRunner .debug files.

        Grouping, dtype splitting and (problem, config) -> best aggregation
        all reuse quickTuningGen so the keys match the shipped DB exactly.
        """
        corpus = cls()
        groups = quickTuningGen.group_files(files)
        for (arch, op), file_list in groups.items():
            corpus._absorb_arch_meta(arch, file_list)
            dtype_data = quickTuningGen.load_files(file_list, op, no_splitk)
            target_cols = quickTuningGen.get_target_columns(op)
            for dtype, df in dtype_data.items():
                corpus._absorb_table((arch, op, dtype), op, target_cols, df)
        return corpus

    def _absorb_arch_meta(self, arch: str, files: List[str]) -> None:
        if arch in self.arch_meta:
            return
        num_cu: Optional[int] = None
        num_chiplets: Optional[int] = None
        try:
            head = pd.read_csv(files[0], sep='\t', nrows=1)
            for col in ('numCU', 'numCUs', 'num_cu'):
                if col in head.columns:
                    num_cu = int(head[col].iloc[0])
                    break
            for col in ('numChiplets', 'num_chiplets'):
                if col in head.columns:
                    num_chiplets = int(head[col].iloc[0])
                    break
        except (ValueError, KeyError, IndexError, pd.errors.ParserError):
            pass
        self.arch_meta[arch] = (num_cu, num_chiplets)

    def _absorb_table(self, key: TableKey, op: str, target_cols: List[str],
                      df: pd.DataFrame) -> None:
        # quickTuningGen.load_files already aggregates each (problem, config)
        # pair to its max TFlops, so a single assignment per pair is the best
        # recorded value (no max() needed here).
        table = self.tables.setdefault(key, {})
        cols = self.problem_cols.setdefault(key, {})
        for row in df.itertuples(index=False):
            row_d = row._asdict()
            col_tuple = tuple(row_d[c] for c in target_cols)
            problem_key = quickTuningGen.make_problem_key(col_tuple)
            perf_config = row_d['PerfConfig']
            tflops = float(row_d['TFlops'])
            table.setdefault(problem_key, {})[perf_config] = tflops
            cols.setdefault(problem_key, col_tuple)

    # ---- queries ------------------------------------------------------

    def keys(self) -> List[TableKey]:
        return sorted(self.tables.keys())

    def problem_keys(self, key: TableKey) -> List[str]:
        return sorted(self.tables.get(key, {}).keys())

    def measured(self, key: TableKey, problem_key: str) -> Dict[str, float]:
        """Recorded {perfConfig: tflops} for a problem (NaN = failed)."""
        return self.tables.get(key, {}).get(problem_key, {})

    def best(self, key: TableKey, problem_key: str) -> float:
        """Absolute-best TFlops recorded for a problem (NaN if none valid)."""
        vals = [
            t for t in self.measured(key, problem_key).values()
            if t is not None and not math.isnan(t)
        ]
        return max(vals) if vals else float('nan')

    def vocabulary(self, key: TableKey) -> List[str]:
        """All perfConfigs seen for any problem under this table key."""
        seen = set()
        for cfgs in self.tables.get(key, {}).values():
            seen.update(cfgs.keys())
        return sorted(seen)

    def sig(self, key: TableKey, problem_key: str) -> ProblemSig:
        arch, op, dtype = key
        target_cols = quickTuningGen.get_target_columns(op)
        num_cu, num_chiplets = self.arch_meta.get(arch, (None, None))
        return ProblemSig(arch=arch,
                          op=op,
                          dtype=dtype,
                          problem_key=problem_key,
                          column_names=tuple(target_cols),
                          columns=self.problem_cols[key][problem_key],
                          num_cu=num_cu,
                          num_chiplets=num_chiplets)

    def sigs(self, key: Optional[TableKey] = None) -> List[ProblemSig]:
        """All problem signatures, optionally restricted to one table key."""
        target_keys = [key] if key is not None else self.keys()
        out: List[ProblemSig] = []
        for k in target_keys:
            for pk in self.problem_keys(k):
                out.append(self.sig(k, pk))
        return out

    # ---- derivation ---------------------------------------------------

    def subset(self, keep) -> "Corpus":
        """Return a new corpus keeping only problems where keep(sig) is True."""
        out = Corpus(arch_meta=dict(self.arch_meta))
        for key, problems in self.tables.items():
            for problem_key, cfgs in problems.items():
                if not keep(self.sig(key, problem_key)):
                    continue
                out.tables.setdefault(key, {})[problem_key] = dict(cfgs)
                out.problem_cols.setdefault(key, {})[problem_key] = \
                    self.problem_cols[key][problem_key]
        return out

    def arches(self) -> List[str]:
        """Architectures present in the corpus, sorted."""
        return sorted({key[0] for key in self.tables})

    def by_arch(self) -> List[Tuple[str, "Corpus"]]:
        """Split into one independent sub-corpus per arch, sorted by arch.

        The harness processes each arch separately (one model / cover scope per
        arch), so this is the top-level grouping -- equivalent to invoking the
        tool once per arch, mirroring quickTuningGen's per-(arch, op) grouping.
        """
        return [(arch, self.subset(lambda sig, a=arch: sig.arch == a)) for arch in self.arches()]
