#!/usr/bin/env python3
"""Regression tests for Jenkinsfile failure classification patterns."""

import re
import unittest
from pathlib import Path


def _extract_function_source(file_text: str, signature: str) -> str:
    start = file_text.find(signature)
    if start < 0:
        raise ValueError(f"Could not find function signature: {signature}")

    brace_start = file_text.find("{", start)
    if brace_start < 0:
        raise ValueError(f"Could not find opening brace for: {signature}")

    depth = 0
    for idx in range(brace_start, len(file_text)):
        char = file_text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return file_text[start : idx + 1]

    raise ValueError(f"Could not find closing brace for: {signature}")


class TestClassifyBuildFailurePatterns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jenkinsfile_path = Path(__file__).resolve().parents[2] / "Jenkinsfile"
        jenkinsfile_text = jenkinsfile_path.read_text(encoding="utf-8")
        cls.classify_source = _extract_function_source(
            jenkinsfile_text,
            "Map<String,String> classifyBuildFailure(String logText)",
        )

    def test_migraphx_failure_scenario_is_classified(self):
        self.assertIn(
            "if (!reason && logText.contains('Configuring incomplete, errors occurred!'))",
            self.classify_source,
        )
        self.assertIn(
            "reason = 'MIGraphX: CMake configuration failed (check CMakeError.log / CMakeOutput.log)'",
            self.classify_source,
        )

    def test_migraphx_stages_are_in_stage_lookup(self):
        stage_list_match = re.search(
            r"def stageNames = \[(.*?)\]",
            self.classify_source,
            re.DOTALL,
        )
        self.assertIsNotNone(stage_list_match, "Could not locate stageNames list")

        stages = re.findall(r"'([^']+)'", stage_list_match.group(1))
        self.assertIn("MIGraphX", stages)
        self.assertIn("Build and Verify MIGraphX with MLIR", stages)

    def test_migraphx_scenario_precedes_fallback_reason(self):
        migraphx_idx = self.classify_source.find("Configuring incomplete, errors occurred!")
        fallback_idx = self.classify_source.find(
            "Could not match a known error pattern. See build log for details."
        )

        self.assertNotEqual(migraphx_idx, -1, "Missing MIGraphX failure scenario")
        self.assertNotEqual(fallback_idx, -1, "Missing fallback reason")
        self.assertLess(
            migraphx_idx,
            fallback_idx,
            "MIGraphX scenario should be checked before fallback reason is assigned",
        )


if __name__ == "__main__":
    unittest.main()
