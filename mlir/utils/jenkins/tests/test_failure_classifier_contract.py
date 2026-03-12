import pathlib
import unittest


JENKINSFILE_PATH = pathlib.Path(__file__).resolve().parents[1] / "Jenkinsfile"


def _classifier_block() -> str:
    content = JENKINSFILE_PATH.read_text(encoding="utf-8")
    start_marker = "Map<String,String> classifyBuildFailure(String logText) {"
    end_marker = '\n}\n\n// Parse "Aborted by USERNAME" from console log'

    start = content.find(start_marker)
    if start < 0:
        raise AssertionError("classifyBuildFailure function not found in Jenkinsfile")

    end = content.find(end_marker, start)
    if end < 0:
        raise AssertionError("Could not find end of classifyBuildFailure function")

    return content[start:end]


class TestFailureClassifierContract(unittest.TestCase):
    def test_migraphx_cmake_classification_is_stage_guarded(self) -> None:
        block = _classifier_block()
        self.assertIn(
            "if (!reason && migraphxStagePos >= 0 && cmakeConfigErrorPos > migraphxStagePos)",
            block,
        )
        self.assertNotIn(
            "logText.contains('Configuring incomplete, errors occurred!')",
            block,
        )

    def test_migraphx_reason_is_assigned_inside_guarded_branch(self) -> None:
        block = _classifier_block()
        guard_pos = block.find(
            "if (!reason && migraphxStagePos >= 0 && cmakeConfigErrorPos > migraphxStagePos) {"
        )
        reason_pos = block.find(
            "reason = 'MIGraphX: CMake configuration failed (check CMakeError.log / CMakeOutput.log)'"
        )
        fallback_pos = block.find(
            "if (!reason) reason = 'Could not match a known error pattern. See build log for details.'"
        )

        self.assertGreaterEqual(guard_pos, 0, "MIGraphX guarded branch missing")
        self.assertGreater(reason_pos, guard_pos, "MIGraphX reason assignment moved outside guard")
        self.assertGreater(
            fallback_pos,
            reason_pos,
            "Fallback reason must remain after MIGraphX-specific classification",
        )

    def test_stage_names_list_includes_migraphx_stages(self) -> None:
        block = _classifier_block()
        self.assertIn("'MIGraphX'", block)
        self.assertIn("'Build and Verify MIGraphX with MLIR'", block)


if __name__ == "__main__":
    unittest.main()
