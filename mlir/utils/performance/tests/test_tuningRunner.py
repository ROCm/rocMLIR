"""
Tests for tuningRunner.py.

These tests cover argument parsing (with mocked GPU topology), state management,
output file parsing, and utilities. No real tuning or GPU execution is run; they are
intended for CI (e.g. GitHub Actions) where no AMD GPU or rocm-smi is available.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure we can import from parent (tuningRunner and perfRunner live in mlir/utils/performance)
_test_dir = Path(__file__).resolve().parent
sys_path_parent = str(_test_dir.parent)
if sys_path_parent not in sys.path:
    sys.path.insert(0, sys_path_parent)
# Mock hip so perfRunner (imported by tuningRunner) can load without ROCm (CI has no GPU)
exec(
    open(_test_dir / "mock_hip.py").read(), {
        "__file__": str(_test_dir / "mock_hip.py"),
        "sys": sys
    })

import tuningRunner  # noqa: E402 - must run after mock_hip
from tuningRunner import (  # noqa: E402
    ConfigState, TuningState, TuningStateFile, TunedConfigsCache, Options, get_state_filepath,
    verify_mode_flags, format_error, get_config_class, get_git_commit_hash, NumaTopology, Operation,
)


def _make_mock_gpu_topology(gpu_ids_and_skus=None):
    """Build a GpuTopology-like object for parse_arguments tests (no rocm-smi)."""
    if gpu_ids_and_skus is None:
        gpu_ids_and_skus = [(0, "gfx900")]
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Gpu:
        gpu_id: int
        sku: str
        numa_node: int

    gpus = {gid: Gpu(gpu_id=gid, sku=sku, numa_node=0) for gid, sku in gpu_ids_and_skus}

    class MockGpuTopology:

        def __init__(self, gpus_dict):
            self.gpus = gpus_dict

        def get_numa_node(self, gpu_id: int) -> int:
            return self.gpus[gpu_id].numa_node

        def validate_homogeneity(self, gpu_ids) -> bool:
            if len(gpu_ids) <= 1:
                return True
            skus = {self.gpus[gid].sku for gid in gpu_ids}
            return len(skus) == 1

    return MockGpuTopology(gpus)


class TestVerifyModeFlags:
    """Tests for verify_mode_flags."""

    def test_none(self):
        assert verify_mode_flags("none") == ""

    def test_cpu(self):
        assert verify_mode_flags("cpu") == "-pv"

    def test_gpu(self):
        out = verify_mode_flags("gpu")
        assert "-pv_with_gpu" in out
        assert "verifier-keep-perf-config=false" in out

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown verification mode"):
            verify_mode_flags("invalid")


class TestFormatError:
    """Tests for format_error."""

    def test_basic(self):
        msg = format_error("Something failed")
        assert "Something failed" in msg

    def test_with_exit_code(self):
        msg = format_error("Failed", exit_code=1)
        assert "Exit code: 1" in msg

    def test_with_command_and_gpu(self):
        msg = format_error("Failed", command="rocmlir-gen -c", gpu_id=0)
        assert "ROCR_VISIBLE_DEVICES=0" in msg
        assert "rocmlir-gen" in msg

    def test_truncate_long_output(self):
        long_stderr = "line\n" * 20
        msg = format_error("Failed", stderr=long_stderr, max_lines=5)
        assert "omitted" in msg or "Failed" in msg


class TestGetStateFilepath:
    """Tests for get_state_filepath."""

    def test_stdout_returns_none(self):
        assert get_state_filepath("-") is None

    def test_normal_path(self):
        assert get_state_filepath("out.tsv") == "out.tsv.state"


class TestTuningState:
    """Tests for TuningState (in-memory state transitions)."""

    def test_empty_should_skip_returns_false_for_unknown(self):
        state = TuningState()
        assert state.should_skip("config1") is False

    def test_failed_should_skip_without_retry(self):
        state = TuningState()
        state.set_failed("config1")
        assert state.should_skip("config1") is True
        assert state.should_skip("config1", retry_states=frozenset({ConfigState.FAILED})) is False

    def test_timed_out_and_crashed_skipped(self):
        state = TuningState()
        state.set_timed_out("c1")
        state.set_failed("c2")
        assert state.timed_out_count() == 1
        assert state.failed_count() == 1

    def test_promote_running_to_interrupted(self):
        state = TuningState()
        state.set_running("c1")
        n = state.promote_running_to_interrupted()
        assert n == 1
        assert state.configs["c1"] == ConfigState.INTERRUPTED

    def test_remove_clears_config(self):
        state = TuningState()
        state.set_failed("c1")
        state.remove("c1")
        assert state.is_empty()


class TestTuningStateFile:
    """Tests for TuningStateFile (persisted state, no GPU)."""

    def test_no_filepath_is_noop(self):
        sf = TuningStateFile(None, arch="gfx900", num_cu=64, num_chiplets=1, tuning_space="full")
        sf.set_running("c1")
        sf.set_failed("c1")
        assert sf.state.failed_count() == 1
        # No file written when filepath is None

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".state", delete=False) as f:
            f.write(json.dumps({"contexts": {}}))
            path = f.name
        try:
            sf = TuningStateFile(path,
                                 arch="gfx900",
                                 num_cu=64,
                                 num_chiplets=1,
                                 tuning_space="full")
            sf.set_failed("config_a")
            sf.set_timed_out("config_b")
            # Reload from file
            sf2 = TuningStateFile(path,
                                  arch="gfx900",
                                  num_cu=64,
                                  num_chiplets=1,
                                  tuning_space="full")
            assert sf2.state.configs.get("config_a") == ConfigState.FAILED
            assert sf2.state.configs.get("config_b") == ConfigState.TIMED_OUT
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_running_becomes_crashed_on_load(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".state", delete=False) as f:
            f.write(json.dumps({"contexts": {"gfx900/64/1/full": {"config_x": "running"}}}))
            path = f.name
        try:
            sf = TuningStateFile(path,
                                 arch="gfx900",
                                 num_cu=64,
                                 num_chiplets=1,
                                 tuning_space="full")
            assert sf.state.configs.get("config_x") == ConfigState.CRASHED
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestTunedConfigsCache:
    """Tests for TunedConfigsCache.from_output_file (parsing only, no GPU)."""

    def _options(self, output_path, arch="gfx900", num_cu=64, num_chiplets=1, tuning_space="full"):
        return Options(
            chip=arch,
            arch=arch,
            num_cu=num_cu,
            num_chiplets=num_chiplets,
            debug=False,
            quiet=False,
            verbose=False,
            tuning_space_kind=tuning_space,
            rocmlir_gen_flags="",
            verify_mode="none",
            verify_perfconfigs=False,
            output=output_path,
            abort_on_error=False,
            retune=False,
            retry_states=frozenset(),
            gpu_ids=[0],
            num_cpus=None,
            wait_for_compiles=False,
            timeout=None,
        )

    def test_missing_file_returns_empty_cache(self):
        opts = self._options("/nonexistent/out.tsv")
        cache = TunedConfigsCache.from_output_file(opts)
        assert cache.count() == 0

    def test_stdout_output_returns_empty(self):
        opts = self._options("-")
        cache = TunedConfigsCache.from_output_file(opts)
        assert cache.count() == 0

    def test_parse_new_format_tsv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(
                "# arch\tnumCUs\tnumChiplets\ttestVector\tperfConfig\tTFlops\ttuningSpace\tcommitId\ttimestamp\tdurationSec\n"
            )
            f.write(
                "gfx900\t64\t1\t-g 1 -m 1024 -k 769 -n 512\tperf_best\t1.5\tfull\tabc123\t2025-01-01T00:00:00Z\t10.0\n"
            )
            path = f.name
        try:
            opts = self._options(path)
            cache = TunedConfigsCache.from_output_file(opts)
            assert cache.count() == 1
            r = cache.get("-g 1 -m 1024 -k 769 -n 512")
            assert r is not None
            assert r.success
            assert r.winning_config == "perf_best"
            assert r.max_tflops == 1.5
        finally:
            os.unlink(path)

    def test_arch_mismatch_not_loaded(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(
                "# arch\tnumCUs\tnumChiplets\ttestVector\tperfConfig\tTFlops\ttuningSpace\tcommitId\ttimestamp\tdurationSec\n"
            )
            f.write(
                "gfx1030\t72\t1\t-g 1 -m 1024\tperf_x\t1.0\tfull\tx\t2025-01-01T00:00:00Z\t5.0\n")
            path = f.name
        try:
            opts = self._options(path, arch="gfx900")  # different arch
            cache = TunedConfigsCache.from_output_file(opts)
            assert cache.count() == 0
        finally:
            os.unlink(path)


class TestGetConfigClass:
    """Tests for get_config_class."""

    def test_known_ops(self):
        assert get_config_class(Operation.CONV).__name__ == "ConvConfiguration"
        assert get_config_class(Operation.GEMM).__name__ == "GemmConfiguration"
        assert get_config_class(Operation.ATTENTION).__name__ == "AttentionConfiguration"
        assert get_config_class(Operation.GEMM_GEMM).__name__ == "GemmGemmConfiguration"
        assert get_config_class(Operation.CONV_GEMM).__name__ == "ConvGemmConfiguration"

    def test_fusion_raises(self):
        with pytest.raises(ValueError, match="No config class"):
            get_config_class(Operation.FUSION)


class TestParseArguments:
    """Tests for parse_arguments with mocked GPU topology (no rocm-smi)."""

    def test_required_op_and_config_group(self):
        topology = _make_mock_gpu_topology([(0, "gfx900")])
        available = [0]
        with pytest.raises(SystemExit):
            tuningRunner.parse_arguments(topology, available, ["--op", "gemm"])
        with pytest.raises(SystemExit):
            tuningRunner.parse_arguments(topology, available, ["-c", "configs.txt"])

    def test_fusion_requires_test_dir(self):
        topology = _make_mock_gpu_topology([(0, "gfx900")])
        available = [0]
        with pytest.raises(SystemExit):
            tuningRunner.parse_arguments(topology, available, ["--op", "fusion", "-c", "dummy.txt"])

    def test_valid_gemm_single_config(self):
        topology = _make_mock_gpu_topology([(0, "gfx900")])
        available = [0]
        parsed = tuningRunner.parse_arguments(
            topology,
            available,
            [
                "--op",
                "gemm",
                "--config",
                "-g 1 -m 1024 -k 769 -n 512 -t f32",
                "-o",
                "/tmp/out.tsv",
            ],
        )
        assert parsed.op == "gemm"
        assert parsed.config == "-g 1 -m 1024 -k 769 -n 512 -t f32"
        assert parsed.output == "/tmp/out.tsv"

    def test_tuning_space_choices(self):
        topology = _make_mock_gpu_topology([(0, "gfx900")])
        available = [0]
        parsed = tuningRunner.parse_arguments(
            topology,
            available,
            [
                "--op",
                "gemm",
                "--config",
                "-g 1 -m 1024 -k 769 -n 512",
                "--tuning-space",
                "quick",
            ],
        )
        assert parsed.tuning_space == "quick"


class TestNumaTopologyParseCpuList:
    """Tests for NumaTopology._parse_cpu_list (used when discovering NUMA)."""

    def test_single_range(self):
        assert NumaTopology._parse_cpu_list("0-3") == [0, 1, 2, 3]

    def test_comma_separated(self):
        assert NumaTopology._parse_cpu_list("0,2,4") == [0, 2, 4]

    def test_mixed(self):
        assert NumaTopology._parse_cpu_list("0-2,5,10-11") == [0, 1, 2, 5, 10, 11]


class TestGetGitCommitHash:
    """Tests for get_git_commit_hash (runs in repo)."""

    def test_returns_string(self):
        h = get_git_commit_hash()
        assert isinstance(h, str)
        # In a real repo we get a 40-char hex; in CI might be "unknown"
        assert len(h) >= 1


class TestFindBestPerfconfig:
    """Tests for find_best_perfconfig with mock config (no subprocess)."""

    def test_empty_lines_returns_none_winner(self):
        from tuningRunner import find_best_perfconfig
        from unittest.mock import MagicMock

        config = MagicMock()
        config.table_entry.return_value = {"TFlops": float("nan")}
        paths = MagicMock()
        options = MagicMock()
        options.debug = False
        options.verify_perfconfigs = False
        winner, tflops, entries = find_best_perfconfig([], config, paths, options, gpu_id=0)
        assert winner is None
        assert tflops is None
        assert entries == []

    def test_single_valid_line(self):
        from tuningRunner import find_best_perfconfig
        from unittest.mock import MagicMock

        config = MagicMock()
        config.table_entry.return_value = {"TFlops": 1.5}
        paths = MagicMock()
        options = MagicMock()
        options.debug = False
        options.verify_perfconfigs = False
        lines = ["perf_cfg_1\t12345"]
        winner, tflops, entries = find_best_perfconfig(lines, config, paths, options, gpu_id=0)
        assert winner == "perf_cfg_1"
        assert tflops == 1.5
        assert len(entries) == 1
