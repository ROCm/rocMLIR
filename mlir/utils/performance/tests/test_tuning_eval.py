# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for the tuning evaluation harness.

Asserts invariants (not absolute performance) on a tiny committed GEMM
fixture: the oracle has zero regret, regrets are bounded, results are
deterministic, set-cover/model beat the random floor, and the model trains,
proposes, and compiles to C end-to-end. No GPU is required, but a built
rocmlir-gen is: features come from ``rocmlir-gen --emit-features`` (the single
source of truth the deployed scorer also uses). The C-export parity test
self-skips when no C compiler / codegen toolchain is available.

The candidate pool is the per-problem applicable tuning space (rocmlir-gen
--emit-tuning-space) in production; here the pool-ranking proposers (random,
model) are fed a fake provider built from the fixture's recorded configs.
"""

import math
import os
import sys
from pathlib import Path
from typing import List

import pytest

_test_dir = Path(__file__).resolve().parent
_analysis_dir = _test_dir.parent / "analysis"
if str(_analysis_dir) not in sys.path:
    sys.path.insert(0, str(_analysis_dir))

# Inject the mock 'hip' module so perfRunner (imported by the feature extractor)
# loads without ROCm. Must run first.
exec(
    open(_test_dir / "mock_hip.py").read(), {
        "__file__": str(_test_dir / "mock_hip.py"),
        "sys": sys
    })

from tuning_eval import export  # noqa: E402
from tuning_eval import features  # noqa: E402

# Features are computed by rocmlir-gen; point the extractor at the build
# (ROCMLIR_BUILD_DIR overrides the auto-discovered location).
features.configure_extractor(os.environ.get("ROCMLIR_BUILD_DIR"))
from tuning_eval.corpus import Corpus  # noqa: E402
from tuning_eval.metrics import evaluate  # noqa: E402
from tuning_eval.proposers import ModelProposer  # noqa: E402
from tuning_eval.proposers import NearestKnownProposer  # noqa: E402
from tuning_eval.proposers import RandomProposer  # noqa: E402
from tuning_eval.proposers import SetCoverProposer  # noqa: E402
from tuning_eval.proposers.base import ConfigProposer  # noqa: E402
from tuning_eval.splits import held_out_dtype  # noqa: E402
from tuning_eval.splits import kfold_problems  # noqa: E402
from tuning_eval.tuning_space import EmitTuningSpacePool  # noqa: E402
from tuning_eval.tuning_space import out_dtype_for  # noqa: E402
from tuning_eval.tuning_space import parse_tuning_space  # noqa: E402

_FIXTURE = str(_test_dir / "data" / "tuning_eval_gemm_sample.debug")
_BUDGETS = [1, 2, 4]
_BEST_CONFIG = "v2:128,128,8,32,16,4,1,1,0"  # dominant best for every fixture problem


class _OracleProposer(ConfigProposer):
    """Upper bound: orders a problem's configs by recorded TFlops."""

    name = "oracle"

    def __init__(self, corpus: Corpus):
        self._corpus = corpus

    def fit(self, train: Corpus) -> None:
        pass

    def propose(self, sig, budget: int) -> List[str]:
        measured = self._corpus.measured(sig.table_key, sig.problem_key)
        ranked = sorted(((c, t) for c, t in measured.items() if not math.isnan(t)),
                        key=lambda ct: ct[1],
                        reverse=True)
        return [c for c, _ in ranked[:budget]]


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return Corpus.from_debug_files([_FIXTURE])


@pytest.fixture(scope="module")
def split(corpus):
    # In-distribution split: the first fold of a grouped 2-fold partition
    # (~50/50 by problem shape), which is what the harness now uses.
    return next(iter(kfold_problems(corpus, k=2, seed=0)))


def _corpus_pool(corpus: Corpus):
    """A pool provider that stands in for rocmlir-gen --emit-tuning-space by
    returning the configs the corpus recorded for a problem (compiler-free)."""

    def pool(sig):
        return sorted(corpus.measured(sig.table_key, sig.problem_key).keys())

    return pool


@pytest.fixture(scope="module")
def pool(corpus):
    return _corpus_pool(corpus)


def _mean_regret(rows, proposer_name, budget):
    vals = [r["regret"] for r in rows if r["proposer"] == proposer_name and r["budget"] == budget]
    return sum(vals) / len(vals)


def test_corpus_loads_expected_keys(corpus):
    keys = corpus.keys()
    assert ("gfx942", "gemm", "f32") in keys
    assert ("gfx942", "gemm", "f16") in keys
    vocab = corpus.vocabulary(("gfx942", "gemm", "f32"))
    assert _BEST_CONFIG in vocab
    assert len(vocab) == 4


def test_best_is_dominant_config(corpus):
    for key in corpus.keys():
        for pk in corpus.problem_keys(key):
            measured = corpus.measured(key, pk)
            best_cfg = max(measured, key=lambda c: measured[c])
            assert best_cfg == _BEST_CONFIG
            assert corpus.best(key, pk) == measured[_BEST_CONFIG]


def test_split_is_nonempty_and_disjoint(corpus, split):
    train, test_sigs = split
    assert test_sigs, "expected a non-empty test set"
    train_keys = {(s.table_key, s.problem_key) for s in train.sigs()}
    test_keys = {(s.table_key, s.problem_key) for s in test_sigs}
    assert train_keys, "expected a non-empty train set"
    assert train_keys.isdisjoint(test_keys)


def test_oracle_has_zero_regret(corpus, split):
    _, test_sigs = split
    rows = evaluate(_OracleProposer(corpus), corpus, test_sigs, _BUDGETS)
    for r in rows:
        assert r["regret"] == pytest.approx(0.0)


def test_regrets_are_bounded(corpus, split, pool):
    train, test_sigs = split
    for proposer in (RandomProposer(seed=0, pool_provider=pool), SetCoverProposer(),
                     NearestKnownProposer(), ModelProposer(seed=0, pool_provider=pool)):
        proposer.fit(train)
        rows = evaluate(proposer, corpus, test_sigs, _BUDGETS)
        for r in rows:
            assert not math.isnan(r["regret"])
            assert 0.0 <= r["regret"] <= 1.0


def test_set_cover_and_model_beat_random(corpus, split, pool):
    train, test_sigs = split

    rnd = RandomProposer(seed=0, pool_provider=pool)
    rnd.fit(train)
    rnd_rows = evaluate(rnd, corpus, test_sigs, _BUDGETS)

    for proposer in (SetCoverProposer(), ModelProposer(seed=0, pool_provider=pool)):
        proposer.fit(train)
        rows = evaluate(proposer, corpus, test_sigs, _BUDGETS)
        assert _mean_regret(rows, proposer.name, 1) <= _mean_regret(rnd_rows, "random", 1)


def test_set_cover_selects_dominant_config(corpus, split):
    train, test_sigs = split
    proposer = SetCoverProposer()
    proposer.fit(train)
    # Only the dominant config clears the 0.93 threshold, so the cover is {c2}.
    assert proposer.propose(test_sigs[0], 4) == [_BEST_CONFIG]


def test_model_is_deterministic(corpus, split, pool):
    pytest.importorskip("lightgbm")
    train, test_sigs = split

    def run():
        m = ModelProposer(seed=0, pool_provider=pool)
        m.fit(train)
        return [r["regret"] for r in evaluate(m, corpus, test_sigs, _BUDGETS)]

    assert run() == run()


def test_model_fits_and_proposes(corpus, split, pool):
    pytest.importorskip("lightgbm")
    train, test_sigs = split
    m = ModelProposer(seed=0, max_depth=4, n_estimators=50, pool_provider=pool)
    m.fit(train)
    out = m.propose(test_sigs[0], 4)
    assert out and len(out) <= 4
    # deterministic for a fixed seed
    m2 = ModelProposer(seed=0, max_depth=4, n_estimators=50, pool_provider=pool)
    m2.fit(train)
    assert m2.propose(test_sigs[0], 4) == out


def test_proposal_coverage_full_for_emit_pool(corpus, split, pool):
    pytest.importorskip("lightgbm")
    train, test_sigs = split
    proposer = ModelProposer(seed=0, pool_provider=pool)
    proposer.fit(train)
    rows = evaluate(proposer, corpus, test_sigs, _BUDGETS)
    for r in rows:
        assert r["proposal_coverage"] == pytest.approx(1.0)


def test_nearest_proposes_only_cover_entries(corpus, split):
    train, test_sigs = split
    proposer = NearestKnownProposer()
    proposer.fit(train)
    cover, _ = proposer._sc.cover_and_map(test_sigs[0].table_key)
    out = proposer.propose(test_sigs[0], 4)
    assert out, "nearest should fall back to a known problem's cover entries"
    assert set(out).issubset(set(cover))
    # The fixture's only good config is the dominant one, so it must surface.
    assert out[0] == _BEST_CONFIG


def test_nearest_exact_hash_hit_on_train_problem(corpus, split):
    train, _ = split
    proposer = NearestKnownProposer()
    proposer.fit(train)
    train_sig = train.sigs()[0]
    assert proposer.propose(train_sig, 4) == [_BEST_CONFIG]


def _write_debug_with_inapplicable(path) -> str:
    """Many GEMM problems, four configs each; one config (cfg_d, the only one
    with kpack=8) is always NaN (inapplicable) and would otherwise look
    attractive. The grid is large enough that LightGBM (default
    min_child_samples) can split out cfg_d's rows."""
    cfg_a = "v2:128,128,8,32,16,4,1,1,0"  # best, applicable
    cfg_b = "v2:256,256,8,32,16,4,1,1,0"  # mid, applicable
    cfg_c = "v2:64,64,8,16,16,4,1,1,0"  # low, applicable
    cfg_d = "v2:128,256,8,32,16,8,1,1,0"  # always inapplicable (kpack=8)
    header = ("Chip\tnumCUs\tnumChiplets\tDataType\tTransA\tTransB\tG\tM\tK\tN"
              "\tPerfConfig\tTFlops")
    lines = [header]
    shapes = [(m, k, n)
              for m in (256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 2048)
              for k in (256, 512, 768)
              for n in (512, 1024)]
    for m, k, n in shapes:
        base = (m + k + n) / 64.0
        for cfg, tf in [(cfg_a, base * 1.25), (cfg_b, base * 1.0), (cfg_c, base * 0.8),
                        (cfg_d, "NaN")]:
            lines.append(f"gfx942\t64\t1\tf32\t0\t0\t1\t{m}\t{k}\t{n}\t{cfg}\t{tf}")
    p = path / "inapplicable.debug"
    p.write_text("\n".join(lines) + "\n")
    return str(p), cfg_a, cfg_d


def test_two_stage_model_deprioritizes_inapplicable(tmp_path):
    pytest.importorskip("lightgbm")
    debug, cfg_a, cfg_d = _write_debug_with_inapplicable(tmp_path)
    corpus = Corpus.from_debug_files([debug])
    # The emit pool would include cfg_d; here we mirror that from the corpus so
    # the test confirms the model still deprioritizes the inapplicable config.
    mdl = ModelProposer(seed=0, pool_provider=_corpus_pool(corpus))
    mdl.fit(corpus)
    # Stage 1 must have trained (both applicable and inapplicable rows present).
    assert mdl._clf_applic is not None
    sig = corpus.sigs()[0]
    # Three applicable configs exist, so a budget-3 proposal must not waste a
    # slot on the always-inapplicable config.
    out = mdl.propose(sig, 3)
    assert cfg_d not in out
    assert out[0] == cfg_a


def test_pool_provider_overrides_candidate_pool(corpus, split):
    pytest.importorskip("lightgbm")
    train, test_sigs = split
    forced = ["v2:64,64,8,16,16,4,1,1,0"]

    rnd = RandomProposer(seed=0, pool_provider=lambda sig: list(forced))
    rnd.fit(train)
    assert rnd.propose(test_sigs[0], 4) == forced

    mdl = ModelProposer(seed=0, pool_provider=lambda sig: list(forced))
    mdl.fit(train)
    assert set(mdl.propose(test_sigs[0], 4)).issubset(set(forced))


def test_kfold_partitions_all_problems_disjointly(corpus):
    folds = list(kfold_problems(corpus, k=3, seed=0))
    assert len(folds) == 3
    all_problems = {(s.table_key, s.problem_key) for s in corpus.sigs()}
    union_test = set()
    for train, test_sigs in folds:
        test_keys = {(s.table_key, s.problem_key) for s in test_sigs}
        train_keys = {(s.table_key, s.problem_key) for s in train.sigs()}
        assert test_keys.isdisjoint(train_keys)
        union_test |= test_keys
    assert union_test == all_problems


def test_kfold_is_deterministic(corpus):

    def fingerprint():
        return [
            sorted((s.table_key, s.problem_key)
                   for s in test)
            for _, test in kfold_problems(corpus, k=3, seed=0)
        ]

    assert fingerprint() == fingerprint()


def test_held_out_dtype_splits_on_dtype(corpus):
    train, test_sigs = held_out_dtype(corpus, "f16")
    assert test_sigs and all(s.dtype == "f16" for s in test_sigs)
    assert all(s.dtype != "f16" for s in train.sigs())


def _write_cross_dtype_debug(path) -> str:
    """One GEMM shape under three dtypes; each dtype's best config differs and
    the dtypes have very different TFlops magnitudes. The merge bug (comparing
    TFlops across dtypes) would drop the lower-magnitude dtype's best config
    from the fallback cover; the per-shard fix keeps both."""
    header = ("Chip\tnumCUs\tnumChiplets\tDataType\tTransA\tTransB\tG\tM\tK\tN"
              "\tPerfConfig\tTFlops")
    cfg_x = "v2:128,128,8,32,16,4,1,1,0"
    cfg_y = "v2:256,256,8,32,16,4,1,1,0"
    rows = [
        # f32 (low magnitude): cfg_x is best.
        ("f32", cfg_x, 100.0),
        ("f32", cfg_y, 50.0),
        # i8 (high magnitude): cfg_y is best.
        ("i8", cfg_x, 200.0),
        ("i8", cfg_y, 400.0),
        # f16 only exists so it can be held out as the test shard.
        ("f16", cfg_x, 60.0),
        ("f16", cfg_y, 30.0),
    ]
    lines = [header]
    for dtype, cfg, tf in rows:
        lines.append(f"gfx942\t64\t1\t{dtype}\t0\t0\t1\t1024\t1024\t1024\t{cfg}\t{tf}")
    p = path / "cross_dtype.debug"
    p.write_text("\n".join(lines) + "\n")
    return str(p), cfg_x, cfg_y


def test_fallback_cover_does_not_merge_across_dtypes(tmp_path):
    debug, cfg_x, cfg_y = _write_cross_dtype_debug(tmp_path)
    corpus = Corpus.from_debug_files([debug])
    train, _ = held_out_dtype(corpus, "f16")  # train = f32 + i8
    sc = SetCoverProposer()
    sc.fit(train)
    cover = sc._fallback_cover
    # Each train shard's own best must be covered; if TFlops were merged across
    # dtypes, i8's 400 would mask f32's 100 and cfg_x would be missing.
    assert cfg_x in cover
    assert cfg_y in cover


def _write_multi_arch_debug(path) -> List[str]:
    """The same GEMM problems under two archs, with arch-dependent TFlops (one
    file per arch, since quickTuningGen treats a .debug file as a single arch)."""
    header = ("Chip\tnumCUs\tnumChiplets\tDataType\tTransA\tTransB\tG\tM\tK\tN"
              "\tPerfConfig\tTFlops")
    cfg_a = "v2:128,128,8,32,16,4,1,1,0"
    cfg_b = "v2:256,256,8,32,16,4,1,1,0"
    cfg_c = "v2:64,64,8,16,16,4,1,1,0"
    cfg_d = "v2:128,256,8,32,16,8,1,1,0"  # always inapplicable
    shapes = [(m, k, n)
              for m in (256, 512, 768, 1024, 1536, 2048)
              for k in (256, 512)
              for n in (512, 1024)]
    files = []
    for arch, scale in (("gfx942", 1.0), ("gfx90a", 0.7)):
        lines = [header]
        for m, k, n in shapes:
            base = (m + k + n) / 64.0 * scale
            for cfg, tf in [(cfg_a, base * 1.25), (cfg_b, base * 1.0), (cfg_c, base * 0.8),
                            (cfg_d, "NaN")]:
                lines.append(f"{arch}\t64\t1\tf32\t0\t0\t1\t{m}\t{k}\t{n}\t{cfg}\t{tf}")
        p = path / f"multi_arch_{arch}.debug"
        p.write_text("\n".join(lines) + "\n")
        files.append(str(p))
    return files


def test_by_arch_partitions_corpus_independently(tmp_path):
    corpus = Corpus.from_debug_files(_write_multi_arch_debug(tmp_path))
    assert corpus.arches() == ["gfx90a", "gfx942"]

    groups = corpus.by_arch()
    assert [arch for arch, _ in groups] == ["gfx90a", "gfx942"]
    # Each sub-corpus is single-arch and holds only that arch's problems --
    # the harness processes them independently (one model / cover per arch).
    for arch, sub in groups:
        assert {k[0] for k in sub.keys()} == {arch}
        assert all(s.arch == arch for s in sub.sigs())
    total = sum(len(sub.problem_keys(k)) for _, sub in groups for k in sub.keys())
    assert total == sum(len(corpus.problem_keys(k)) for k in corpus.keys())


def test_model_fits_per_arch_subcorpus(tmp_path):
    pytest.importorskip("lightgbm")
    corpus = Corpus.from_debug_files(_write_multi_arch_debug(tmp_path))
    # Fit one model per arch on its own sub-corpus (what train.py does); each
    # only ever sees its own arch, so data is never mixed across archs.
    for arch, sub in corpus.by_arch():
        m = ModelProposer(seed=0, pool_provider=_corpus_pool(corpus))
        m.fit(sub)
        assert m.is_fitted()
        sig = sub.sigs()[0]
        assert m.propose(sig, 3)  # ranks its own arch's pool


def _bare_pool(kind="exhaustive"):
    """An EmitTuningSpacePool with command-building state only (no build dir),
    so command() argv can be checked without invoking rocmlir-gen."""
    p = object.__new__(EmitTuningSpacePool)
    p._gen = "rocmlir-gen"
    p._kind = kind
    return p


def _gemm_sig():
    from tuning_eval.corpus import ProblemSig
    cols = ("TransA", "TransB", "G", "M", "K", "N")
    return ProblemSig(arch="gfx942",
                      op="gemm",
                      dtype="i8",
                      problem_key="g",
                      column_names=cols,
                      columns=(0, 0, 1, 1024, 1024, 1024),
                      num_cu=304)


def test_gemm_command_argv_uses_out_dtype():
    argv = _bare_pool().command(_gemm_sig())
    assert argv[:4] == ["rocmlir-gen", "--arch", "gfx942", "-operation"]
    # i8 GEMM accumulates into i32; the emitted space must reflect that.
    assert "-out_datatype" in argv and argv[argv.index("-out_datatype") + 1] == "i32"
    assert "--emit-tuning-space=exhaustive" in argv
    assert "-num_cu" in argv and argv[argv.index("-num_cu") + 1] == "304"


def test_attention_command_flag_spellings():
    sig = _attention_sig()
    argv = _bare_pool().command(sig)
    # rocmlir-gen spells these with dashes; the rest use underscores. A drift in
    # either would silently break enumeration, so pin the exact spellings.
    assert "-with-attn-scale=false" in argv
    assert "-with-attn-bias=false" in argv
    assert any(a.startswith("-return_lse=") for a in argv)
    assert any(a.startswith("-num_heads_q") or a == "-num_heads_q" for a in argv)


def test_parse_tuning_space_and_out_dtype():
    out = parse_tuning_space("v2:1\n\n  v2:2  \nv2:3\n")
    assert out == ["v2:1", "v2:2", "v2:3"]
    assert out_dtype_for("i8") == "i32"
    assert out_dtype_for("f16") == "f16"


def test_export_model_emits_inc_and_features(tmp_path, split, pool):
    pytest.importorskip("lightgbm")
    train, _ = split
    m = ModelProposer(seed=0, n_estimators=20, pool_provider=pool)
    m.fit(train)
    written = export.export_model(m, tmp_path / "models", "gfx942", "gemm")

    inc = tmp_path / "models" / "Gfx942Gemm.inc"
    feats = tmp_path / "models" / "gfx942_gemm_features.txt"
    assert set(written) == {inc, feats}
    assert inc.exists() and feats.exists()

    text = inc.read_text()
    # Two-phase include guards + the (arch, op) entry key, mirroring QuickTuningDb.
    assert "#ifdef SMART_TUNING_DB_ARRAYS" in text
    assert "#ifdef SMART_TUNING_DB_ENTRIES" in text
    assert '"gfx942_gemm"' in text
    assert "SmartTuningDb::TreeNode" in text
    # The feature count baked into the entry matches the contract sidecar.
    n_features = len(feats.read_text().splitlines())
    assert f'"gfx942_gemm", {n_features},' in text


def test_inc_trees_reproduce_lightgbm(split, pool):
    """The flattened node arrays the .inc embeds must reproduce LightGBM's raw
    score exactly -- this pins the semantics the C++ evaluator mirrors."""
    pytest.importorskip("lightgbm")
    import numpy as np
    train, _ = split
    m = ModelProposer(seed=0, n_estimators=20, pool_provider=pool)
    m.fit(train)
    stages = m.stage_boosters()
    assert stages, "expected at least one fitted stage"
    n_features = len(m._feature_names)
    rng = np.random.RandomState(1)
    x = rng.rand(48, n_features).astype(float)
    for _, booster in stages:
        nodes, roots, bias = export._stage_arrays(booster, n_features)
        mine = np.array(
            [bias + sum(export._eval(nodes, r, list(x[i])) for r in roots) for i in range(len(x))])
        ref = booster.predict(x, raw_score=True)
        assert float(np.max(np.abs(mine - ref))) < 1e-6


def test_label_uses_threshold():
    assert features.label(120.0, 120.0) == 1
    assert features.label(112.0, 120.0) == 1  # 112 >= 0.93 * 120
    assert features.label(80.0, 120.0) == 0
    assert features.label(float("nan"), 120.0) == 0


def test_feature_record_has_stable_named_fields(corpus):
    sig = corpus.sigs(("gfx942", "gemm", "f32"))[0]
    rec = features.feature_record(sig, _BEST_CONFIG)
    assert rec["cfg_m_per_block"] == 128.0
    assert rec["cfg_n_per_block"] == 128.0
    assert "work_imbalance" in rec
    assert rec["is_mfma"] == 1.0


def _conv_sig():
    from tuning_eval.corpus import ProblemSig
    cols = ("Direction", "FilterLayout", "InputLayout", "OutputLayout", "N", "C", "H", "W", "K",
            "Y", "X", "DilationH", "DilationW", "StrideH", "StrideW", "PaddingH", "PaddingW")
    vals = ("fwd", "gkc01", "ngc01", "ngk01", 1, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1)
    return ProblemSig(arch="gfx942",
                      op="conv",
                      dtype="f16",
                      problem_key="conv_test",
                      column_names=cols,
                      columns=vals,
                      num_cu=304)


def test_conv_implicit_gemm_mapping():
    # fwd: out 28x28 (3x3, pad1, stride1) -> M=K=128, N=1*28*28=784, K=C*Y*X=1152.
    # These problem features come straight from rocmlir-gen --emit-features.
    rec = features.feature_record(_conv_sig(), _BEST_CONFIG)
    assert rec["is_fwd"] == 1.0 and rec["is_bwd"] == 0.0
    assert (rec["ho"], rec["wo"]) == (28.0, 28.0)
    assert rec["gemm_m"] == 128.0
    assert rec["gemm_n"] == 784.0
    assert rec["gemm_k"] == 1152.0
    assert rec["in_pos_c"] == 2.0  # 'ngc01' -> channel at index 2


def test_config_parser_extracts_kpack():
    # gemm v2: kpack at idx5 (one before split-K at idx6).
    assert features.feature_record(_gemm_sig(), "v2:128,128,8,32,16,4,1,1,0")["cfg_kpack"] == 4.0
    # attn v3: kpack at idx7.
    rec = features.feature_record(_attention_sig(), "attn:v3:16,16,32,4,16,16,16,8,1,1,2,0,1")
    assert rec["cfg_kpack"] == 8.0


def test_conv_feature_record_and_distance_features():
    sig = _conv_sig()
    rec = features.feature_record(sig, _BEST_CONFIG)
    # config + interaction features are shared with the gemm path
    assert "cfg_m_per_block" in rec and "work_imbalance" in rec
    for name in features.distance_features("conv"):
        assert name in rec


def test_feature_record_includes_num_chiplets():
    from tuning_eval.corpus import ProblemSig
    sig = ProblemSig(arch="gfx942",
                     op="conv",
                     dtype="f16",
                     problem_key="conv_test",
                     column_names=_conv_sig().column_names,
                     columns=_conv_sig().columns,
                     num_cu=304,
                     num_chiplets=8)
    assert features.feature_record(sig, _BEST_CONFIG)["num_chiplets"] == 8.0
    # When the problem carries no chiplet count, rocmlir-gen falls back to the
    # arch default (so the op is well-formed); just assert it is a sane value.
    assert features.feature_record(_conv_sig(), _BEST_CONFIG)["num_chiplets"] >= 1.0


def test_group_key_splits_conv_direction_only():
    from dataclasses import replace

    from tuning_eval.proposers.model import _group_key

    assert _group_key(_conv_sig()) == "conv:fwd"
    bwd_cols = ("bwd",) + _conv_sig().columns[1:]
    assert _group_key(replace(_conv_sig(), columns=bwd_cols)) == "conv:bwd"
    # dtype does not split groups (CV: dtype+dir balancing regressed fwd f32).
    assert _group_key(_conv_sig()) == _group_key(replace(_conv_sig(), dtype="i8"))
    # gemm/attention share one queue -- balancing is a no-op for them.
    assert _group_key(_attention_sig()) == "attention"
    assert _group_key(_attention_sig(causal="True")) == "attention"


def _attention_sig(causal="False"):
    from tuning_eval.corpus import ProblemSig
    cols = ("TransQ", "TransK", "TransV", "TransO", "Causal", "ReturnLSE", "SplitKV",
            "WithAttnScale", "WithAttnBias", "G", "SeqLenQ", "SeqLenK", "NumHeadsQ", "NumHeadsKV",
            "HeadDimQK", "HeadDimV")
    vals = ("False", "False", "False", "False", causal, "False", 1, "False", "False", 2, 1024, 1024,
            16, 4, 64, 64)
    return ProblemSig(arch="gfx942",
                      op="attention",
                      dtype="f16",
                      problem_key="attn_test",
                      column_names=cols,
                      columns=vals,
                      num_cu=304)


_ATTN_CONFIG = "attn:v3:16,16,32,4,16,16,16,4,1,1,2,0,1"


def test_attention_implicit_gemm_and_features():
    rec = features.feature_record(_attention_sig(), _ATTN_CONFIG)
    assert rec["gqa_ratio"] == 4.0  # 16 q heads / 4 kv heads
    assert rec["is_square_seq"] == 1.0
    assert rec["causal"] == 0.0


def test_attention_causal_halves_flops():
    base = features.feature_record(_attention_sig(causal="False"), _ATTN_CONFIG)["flops"]
    causal = features.feature_record(_attention_sig(causal="True"), _ATTN_CONFIG)["flops"]
    assert causal == pytest.approx(base * 0.5)


def test_attention_feature_record_and_distance_features():
    sig = _attention_sig()
    rec = features.feature_record(sig, "attn:v3:16,16,32,4,16,16,16,4,1,1,2,0,1")
    assert "cfg_m_per_block" in rec and "work_imbalance" in rec
    for name in features.distance_features("attention"):
        assert name in rec


def test_arch_resource_features_present_and_normalized(corpus):
    # gfx942 is wave64, mfma (not wmma), with 64 KiB LDS -- the C++ extractor
    # sources these from amd_arch_db, the same DB the compiler uses.
    sig = corpus.sigs(("gfx942", "gemm", "f32"))[0]
    rec = features.feature_record(sig, "v2:128,128,8,32,16,4,1,1,0")
    assert rec["wave_size"] == 64.0
    assert rec["is_mfma"] == 1.0 and rec["is_wmma"] == 0.0
    assert rec["lds_bytes_per_cu"] == 64.0 * 1024.0
    for name in ("lds_bytes_per_wg", "vgpr_per_eu", "waves_per_eu", "eu_per_cu", "tile_lds_bytes",
                 "lds_fraction", "lds_blocks_per_cu"):
        assert name in rec
    # A larger tile must use strictly more LDS and admit fewer co-resident WGs.
    big = features.feature_record(sig, "v2:256,256,8,32,16,4,1,1,0")
    small = features.feature_record(sig, "v2:64,64,8,32,16,4,1,1,0")
    assert big["tile_lds_bytes"] > small["tile_lds_bytes"]
    assert big["lds_blocks_per_cu"] <= small["lds_blocks_per_cu"]
    assert 0.0 < small["lds_fraction"] < 1.0


def test_config_features_are_op_aware():
    # attn:v3 layout is mPerBlockG0, mPerBlockG1, nPerBlockG0, kpackPerBlock,
    # mPerWave, ... so the QK^T tile must read 0,2,3,4 -- not the gemm 0,1,2,3.
    attn = features.feature_record(_attention_sig(), "attn:v3:16,32,64,8,128,16,16,4,1,1,2,0,1")
    assert attn["cfg_m_per_block"] == 16.0  # mPerBlockG0 (p0)
    assert attn["cfg_n_per_block"] == 64.0  # nPerBlockG0 (p2), not mPerBlockG1
    assert attn["cfg_kpack_per_block"] == 8.0  # kpackPerBlock (p3)
    assert attn["cfg_m_per_wave"] == 128.0  # mPerWave (p4)
    # The gemm layout reads the same string positionally (0,1,2,3).
    gemm = features.feature_record(_gemm_sig(), "v2:128,128,8,32,16,4,1,1,0")
    assert gemm["cfg_m_per_block"] == 128.0
    assert gemm["cfg_n_per_block"] == 128.0
    assert gemm["cfg_kpack_per_block"] == 8.0
    assert gemm["cfg_m_per_wave"] == 32.0
