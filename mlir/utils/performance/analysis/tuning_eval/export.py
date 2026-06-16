# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compile a fitted model's decision trees to an embeddable C++ data table.

The deployment artifact is a ``.inc`` of plain tree-ensemble *data* (flattened
node arrays + per-stage tree roots), mirroring ``QuickTuningDb``'s per-key
``.inc`` files: the build glues them into one table that a single generic
evaluator (``SmartTuningDb.cpp``) walks. There is no link against
LightGBM/Treelite at deploy time, and -- unlike a per-model code generator --
many ``(arch, op)`` models embed into one library with no symbol collisions.

A LightGBM binary classifier predicts ``sigmoid(margin)`` where
``margin = bias + sum over trees of the reached leaf value``; each numeric split
goes left iff ``feature[i] <= threshold``. We flatten that structure directly
from ``booster.dump_model()`` (no Treelite needed) and verify the reconstructed
margin matches LightGBM's raw score before emitting.

One model per ``(arch, op)`` (the orchestration fits one arch at a time); the
artifacts are namespaced by arch. Written under ``out_dir``:

    <Arch><Op>.inc                 the embeddable tree table (two-phase include)
    <arch>_<op>_features.txt       input contract: feature order, one per line
"""

from pathlib import Path
from typing import List, Tuple

# Number of (feature-vector -> margin) probes used to validate that the
# flattened trees reproduce LightGBM's raw score, and to recover the bias.
_BIAS_PROBES = 64
_BIAS_TOL = 1e-6

# A flattened tree node: leaves carry ``leaf_value`` (split_feature == -1);
# internal nodes carry the split and child indices into the same node array.
_Node = Tuple[int, float, int, int, float]  # (splitFeature, threshold, left, right, leafValue)


def features_file(out_dir: Path, arch: str, op: str) -> Path:
    return out_dir / f"{arch}_{op}_features.txt"


def _to_pascal(key: str) -> str:
    """'gfx942_gemm' -> 'Gfx942Gemm' (matches QuickTuningDb's naming)."""
    return "".join(part.capitalize() for part in key.split("_"))


def _flatten_tree(tree: dict, nodes: List[_Node]) -> int:
    """Append ``tree``'s nodes to ``nodes`` (post-order); return its root index."""
    if "leaf_value" in tree:
        idx = len(nodes)
        nodes.append((-1, 0.0, -1, -1, float(tree["leaf_value"])))
        return idx
    decision = tree.get("decision_type", "<=")
    if decision != "<=":
        # We only emit numeric "<=" splits; categorical splits would need a
        # different evaluator and never occur for our continuous features.
        raise ValueError(f"unsupported split decision_type: {decision!r}")
    idx = len(nodes)
    nodes.append((-1, 0.0, -1, -1, 0.0))  # placeholder, patched below
    left = _flatten_tree(tree["left_child"], nodes)
    right = _flatten_tree(tree["right_child"], nodes)
    nodes[idx] = (int(tree["split_feature"]), float(tree["threshold"]), left, right, 0.0)
    return idx


def _eval(nodes: List[_Node], root: int, x: List[float]) -> float:
    """Walk one tree: go left iff feature[split] <= threshold (LightGBM)."""
    split_feature, threshold, left, right, leaf = nodes[root]
    while split_feature >= 0:  # internal node
        nxt = left if x[split_feature] <= threshold else right
        split_feature, threshold, left, right, leaf = nodes[nxt]
    return leaf


def _stage_arrays(booster, num_features: int) -> Tuple[List[_Node], List[int], float]:
    """Flatten a stage's boosted trees and recover its bias.

    Returns (nodes, roots, bias) such that, for any feature vector x,
    ``bias + sum(eval(nodes, root, x) for root in roots)`` equals LightGBM's
    raw-score margin. Raises if the reconstruction disagrees."""
    import numpy as np

    dumped = booster.dump_model()
    nodes: List[_Node] = []
    roots: List[int] = []
    for tree in dumped["tree_info"]:
        roots.append(_flatten_tree(tree["tree_structure"], nodes))

    rng = np.random.RandomState(0)
    probes = rng.rand(_BIAS_PROBES, num_features).astype(float)
    raw = booster.predict(probes, raw_score=True)
    mine = np.array(
        [sum(_eval(nodes, r, list(probes[i])) for r in roots) for i in range(len(probes))])
    diffs = raw - mine
    bias = float(np.mean(diffs))
    if float(np.max(np.abs(diffs - bias))) > _BIAS_TOL:
        raise RuntimeError("flattened trees do not reproduce LightGBM's raw score "
                           "(non-constant bias); the dump format may have changed")
    return nodes, roots, bias


def _format_nodes(name: str, nodes: List[_Node]) -> List[str]:
    lines = [f"static const ::mlir::rock::SmartTuningDb::TreeNode {name}[] = {{"]
    for split_feature, threshold, left, right, leaf in nodes:
        lines.append(f"  {{{split_feature}, {threshold!r}, {left}, {right}, {leaf!r}}},")
    lines.append("};")
    return lines


def _format_roots(name: str, roots: List[int]) -> List[str]:
    lines = [f"static const unsigned {name}[] = {{"]
    for i in range(0, len(roots), 16):
        lines.append("  " + ", ".join(str(r) for r in roots[i:i + 16]) + ",")
    lines.append("};")
    return lines


def _format_inc(key: str, num_features: int, stages: dict) -> str:
    """Render the two-phase ``.inc`` for one ``(arch, op)`` model.

    ``stages`` maps stage name -> (nodes, roots, bias); a missing stage (not
    fitted) is emitted as a null/empty stage the evaluator treats as neutral.
    """
    pascal = _to_pascal(key)
    lines = [f"// {pascal}.inc -- auto-generated by tuning_eval.train"]

    lines.append("#ifdef SMART_TUNING_DB_ARRAYS")
    for stage in ("applicability", "optimality"):
        sp = _to_pascal(stage)
        data = stages.get(stage)
        if data is None:
            continue
        nodes, roots, _ = data
        lines += _format_nodes(f"kNodes{pascal}{sp}", nodes)
        lines += _format_roots(f"kRoots{pascal}{sp}", roots)
    lines.append("#endif")

    lines.append("#ifdef SMART_TUNING_DB_ENTRIES")
    lines.append(f'{{"{key}", {num_features},')
    for stage in ("applicability", "optimality"):
        sp = _to_pascal(stage)
        data = stages.get(stage)
        if data is None:
            lines.append("  {nullptr, 0, nullptr, 0, 0.0},")
            continue
        nodes, roots, bias = data
        lines.append(f"  {{kNodes{pascal}{sp}, {len(nodes)}, "
                     f"kRoots{pascal}{sp}, {len(roots)}, {bias!r}}},")
    lines.append("},")
    lines.append("#endif")
    return "\n".join(lines) + "\n"


def export_model(model, out_dir, arch: str, op: str) -> List[Path]:
    """Emit the embeddable tree table for one ``(arch, op)`` model plus the
    feature-order contract sidecar. Returns the paths written."""
    if not model.is_fitted():
        raise RuntimeError("export_model called before a successful fit")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    num_features = len(model._feature_names)
    stages = {
        stage: _stage_arrays(booster, num_features) for stage, booster in model.stage_boosters()
    }

    key = f"{arch}_{op}"
    inc_path = out / f"{_to_pascal(key)}.inc"
    inc_path.write_text(_format_inc(key, num_features, stages))

    feats = features_file(out, arch, op)
    feats.write_text("\n".join(model._feature_names) + "\n")
    return [inc_path, feats]
