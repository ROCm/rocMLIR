#!/usr/bin/env python3
"""Add migraphx-linalg,highlevel RUN lines to eligible E2E test files."""

import os
import re
import sys

E2E_DIRS = [
    "mlir/test/fusion/pr-e2e",
    "mlir/test/fusion/e2e",
    "mlir/test/fusion/nightly-misc-e2e",
    "mlir/test/fusion/resnet50-e2e",
]

NOT_IMPLEMENTED_OPS = [
    "migraphx.quant_convolution",
    "migraphx.softmax",
    "migraphx.reduce_mean",
    "migraphx.reduce_max",
]

def find_mlir_files(root, dirs):
    files = []
    for d in dirs:
        full = os.path.join(root, d)
        if not os.path.isdir(full):
            continue
        for dirpath, _, filenames in os.walk(full):
            for f in filenames:
                if f.endswith(".mlir"):
                    files.append(os.path.join(dirpath, f))
    return sorted(files)

def should_skip(filepath, content):
    if "migraphx-linalg" in content:
        return "already has migraphx-linalg"
    for op in NOT_IMPLEMENTED_OPS:
        if op in content:
            return f"uses not-implemented op: {op}"
    if "!migraphx." not in content:
        return "not in migraphx dialect"
    return None

def get_run_lines(content):
    """Extract (line_index, line_text) for all RUN lines."""
    results = []
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^//\s*RUN:", line):
            full_line = line
            while full_line.rstrip().endswith("\\") and i + 1 < len(lines):
                i += 1
                full_line += "\n" + lines[i]
            results.append((i - full_line.count("\n"), full_line))
        i += 1
    return results

def pick_run_line_to_duplicate(run_lines):
    """Pick the best RUN line to duplicate: prefer ones that actually run the test
    (contain xmir-runner or mlir-runner)."""
    runner_lines = [(idx, l) for idx, l in run_lines
                    if "xmir-runner" in l or "mlir-runner" in l]
    if runner_lines:
        return runner_lines[-1]
    return run_lines[-1]

def transform_first_rocmlir_driver(run_line):
    """Replace migraphx,highlevel with migraphx-linalg,highlevel in the first rocmlir-driver invocation only."""
    # Split the pipeline into segments separated by |
    # We need to find the first rocmlir-driver invocation and change its pipeline flags
    segments = run_line.split("|")
    transformed = False
    new_segments = []
    for seg in segments:
        if not transformed and "rocmlir-driver" in seg:
            # Check if this segment has kernel-pipeline or host-pipeline with migraphx,highlevel
            if re.search(r'-kernel-pipeline[= ]migraphx,highlevel', seg) or \
               re.search(r'-host-pipeline[= ]migraphx,highlevel', seg):
                seg = re.sub(
                    r'(-kernel-pipeline[= ])migraphx,highlevel',
                    r'\1migraphx-linalg,highlevel',
                    seg
                )
                seg = re.sub(
                    r'(-host-pipeline[= ])migraphx,highlevel',
                    r'\1migraphx-linalg,highlevel',
                    seg
                )
                transformed = True
        new_segments.append(seg)
    if not transformed:
        return None
    return "|".join(new_segments)

def process_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    reason = should_skip(filepath, content)
    if reason:
        return False, reason

    run_lines = get_run_lines(content)
    if not run_lines:
        return False, "no RUN lines found"

    idx, chosen_line = pick_run_line_to_duplicate(run_lines)
    new_line = transform_first_rocmlir_driver(chosen_line)
    if new_line is None:
        return False, "no migraphx,highlevel pipeline found in RUN line"
    if new_line == chosen_line:
        return False, "transform produced no change"

    # Insert the new RUN line after the last RUN line
    last_run_idx = run_lines[-1][0]
    last_run_text = run_lines[-1][1]
    # Count how many physical lines the last RUN line spans
    last_run_span = last_run_text.count("\n") + 1

    lines = content.split("\n")
    insert_after = last_run_idx + last_run_span - 1
    lines.insert(insert_after + 1, new_line)

    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    return True, "added migraphx-linalg RUN line"

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    files = find_mlir_files(root, E2E_DIRS)
    modified = 0
    skipped = 0
    for filepath in files:
        ok, msg = process_file(filepath)
        rel = os.path.relpath(filepath, root)
        if ok:
            print(f"  MODIFIED: {rel}")
            modified += 1
        else:
            skipped += 1
            # Only print interesting skip reasons (not dialect/already-has)
            if "no migraphx,highlevel" in msg or "no RUN" in msg or "no change" in msg:
                print(f"  SKIPPED ({msg}): {rel}")
    print(f"\nDone. Modified: {modified}, Skipped: {skipped}")

if __name__ == "__main__":
    main()
