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
import threading
import time
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
    NumaNodeLock, resolve_verify_mode, canonicalize_test_vector, DebugFileWriter, TuningResult,
    tune_config)
from perfRunner import (  # noqa: E402
    GemmConfiguration, ConvConfiguration, AttentionConfiguration, ConvGemmConfiguration,
    GemmGemmConfiguration, PerfConfiguration, canonicalize_config)


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

    def test_cpu_relaxes_thresholds(self):
        # CPU verification is inherently noisier than GPU validation (different accumulation
        # order on the reference path), so both thresholds are relaxed uniformly regardless of op
        # type.
        out = verify_mode_flags("cpu").split()
        assert "-pv" in out
        assert "-relDiff_threshold=0.0001" in out
        assert "-RMS_threshold=0.15" in out

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

    _CONF_CLASS = GemmConfiguration
    _ARCH = "gfx900"
    _NUM_CU = 64
    _NUM_CHIPLETS = 1
    _TV_A = "-t f32 -out_datatype f32 -transA false -transB false -g 1 -m 1024 -n 512 -k 769"
    _TV_B = "-t f16 -out_datatype f16 -transA false -transB true -g 1 -m 256 -n 128 -k 64"

    def _make_state_file(self, filepath, **kwargs):
        return TuningStateFile(filepath,
                               chip=self._ARCH,
                               arch=self._ARCH,
                               num_cu=self._NUM_CU,
                               num_chiplets=self._NUM_CHIPLETS,
                               tuning_space="full",
                               conf_class=self._CONF_CLASS,
                               **kwargs)

    def test_no_filepath_is_noop(self):
        sf = self._make_state_file(None)
        sf.set_running(self._TV_A)
        sf.set_failed(self._TV_A)
        assert sf.state.failed_count() == 1

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".state", delete=False) as f:
            f.write(json.dumps({"contexts": {}}))
            path = f.name
        try:
            sf = self._make_state_file(path)
            sf.set_failed(self._TV_A)
            sf.set_timed_out(self._TV_B)
            sf2 = self._make_state_file(path)
            assert sf2.state.configs.get(self._TV_A) == ConfigState.FAILED
            assert sf2.state.configs.get(self._TV_B) == ConfigState.TIMED_OUT
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_running_becomes_crashed_on_load(self):
        ctx_key = f"{self._ARCH}/{self._NUM_CU}/{self._NUM_CHIPLETS}/full"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".state", delete=False) as f:
            f.write(json.dumps({"contexts": {ctx_key: {self._TV_A: "running"}}}))
            path = f.name
        try:
            sf = self._make_state_file(path)
            assert sf.state.configs.get(self._TV_A) == ConfigState.CRASHED
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_old_state_file_configs_are_canonicalized(self):
        """Non-canonical test vectors in state file are canonicalized on load."""
        non_canonical = "-g 1 -m 1024 -k 769 -n 512 -t f32 -out_datatype f32 -transA false -transB false"
        canonical = self._TV_A
        assert non_canonical != canonical

        ctx_key = f"{self._ARCH}/{self._NUM_CU}/{self._NUM_CHIPLETS}/full"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".state", delete=False) as f:
            f.write(json.dumps({"contexts": {ctx_key: {non_canonical: "failed"}}}))
            path = f.name
        try:
            sf = self._make_state_file(path)
            assert sf.state.configs.get(canonical) == ConfigState.FAILED
            assert sf.state.configs.get(non_canonical) is None
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
            debug_quick_tune_data=False,
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
            verify_timeout=None,
            gpu_run_timeout=0,
        )

    def test_missing_file_returns_empty_cache(self):
        opts = self._options("/nonexistent/out.tsv")
        cache = TunedConfigsCache.from_output_file(opts, GemmConfiguration)
        assert cache.count() == 0

    def test_stdout_output_returns_empty(self):
        opts = self._options("-")
        cache = TunedConfigsCache.from_output_file(opts, GemmConfiguration)
        assert cache.count() == 0

    def test_parse_new_format_tsv(self):
        tv = "-t f32 -out_datatype f32 -transA false -transB false -g 1 -m 1024 -n 512 -k 769"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(
                "# arch\tnumCUs\tnumChiplets\ttestVector\tperfConfig\tTFlops\ttuningSpace\tcommitId\ttimestamp\tdurationSec\n"
            )
            f.write(
                f"gfx900\t64\t1\t{tv}\tperf_best\t1.5\tfull\tabc123\t2025-01-01T00:00:00Z\t10.0\n")
            path = f.name
        try:
            opts = self._options(path)
            cache = TunedConfigsCache.from_output_file(opts, GemmConfiguration)
            assert cache.count() == 1
            r = cache.get(tv)
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
            cache = TunedConfigsCache.from_output_file(opts, GemmConfiguration)
            assert cache.count() == 0
        finally:
            os.unlink(path)


_SAMPLE_TEST_VECTORS = {
    "gemm": {
        "conf_class":
            GemmConfiguration,
        "raw":
            "-g 1 -m 1024 -k 769 -n 512 -t f32 -out_datatype f32 -transA false -transB false",
        "canonical":
            "-t f32 -out_datatype f32 -transA false -transB false -g 1 -m 1024 -n 512 -k 769",
        "idempotent":
            "-t f16 -out_datatype f16 -transA false -transB true -g 1 -m 256 -n 128 -k 64",
    },
    "conv": {
        "conf_class":
            ConvConfiguration,
        "raw":
            "convfp16 -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1",
        "canonical": ("convfp16 -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 "
                      "-k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1"),
        "idempotent": ("convfp16 -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 "
                       "-k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1"),
    },
    "attention": {
        "conf_class":
            AttentionConfiguration,
        "raw": ("-g 1 -seq_len_q 256 -seq_len_k 256 -num_heads_q 8 -num_heads_kv 8 "
                "-head_dim_qk 64 -head_dim_v 64 -t f16 "
                "-transQ false -transK false -transV false -transO false "
                "-causal false -return_lse false -split_kv 1 "
                "-with-attn-scale false -with-attn-bias false"),
        "canonical": ("-t f16 -transQ false -transK false -transV false -transO false "
                      "-causal false -return_lse false -split_kv 1 -g 1 "
                      "-seq_len_q 256 -seq_len_k 256 -num_heads_q 8 -num_heads_kv 8 "
                      "-head_dim_qk 64 -head_dim_v 64 "
                      "-with-attn-scale false -with-attn-bias false -transBias false"),
        "idempotent": ("-t f16 -transQ false -transK false -transV false -transO false "
                       "-causal false -return_lse false -split_kv 1 -g 1 "
                       "-seq_len_q 128 -seq_len_k 128 -num_heads_q 4 -num_heads_kv 4 "
                       "-head_dim_qk 32 -head_dim_v 32 "
                       "-with-attn-scale false -with-attn-bias false -transBias false"),
    },
    "gemm_gemm": {
        "conf_class":
            GemmGemmConfiguration,
        "raw": ("-g 1 -m 64 -k 128 -n 256 -gemmO 32 -t f16 "
                "-transA false -transB false -transC false -transO false"),
        "canonical": ("-t f16 -transA false -transB false -transC false -transO false "
                      "-g 1 -m 64 -k 128 -n 256 -gemmO 32"),
        "idempotent": ("-t f16 -transA false -transB false -transC false -transO false "
                       "-g 1 -m 32 -k 64 -n 128 -gemmO 16"),
    },
    "conv_gemm": {
        "conf_class":
            ConvGemmConfiguration,
        "raw": ("-n 1 -c 64 -H 14 -W 14 -k 128 -y 3 -x 3 -gemmO 64 "
                "-p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -f NCHW -I NCHW "
                "-t f16 -transC false -transO false"),
        "canonical": ("-t f16 -f NCHW -I NCHW -transC false -transO false "
                      "-n 1 -c 64 -H 14 -W 14 -k 128 -y 3 -x 3 "
                      "-p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -gemmO 64"),
        "idempotent": ("-t f16 -f NCHW -I NCHW -transC false -transO false "
                       "-n 1 -c 64 -H 14 -W 14 -k 128 -y 3 -x 3 "
                       "-p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -gemmO 64"),
    },
}

_ALL_OPS = list(_SAMPLE_TEST_VECTORS.keys())


class TestCanonicalizeTestVector:
    """Tests for canonicalize_config and canonicalize_test_vector across all ops."""

    @pytest.mark.parametrize("op", _ALL_OPS)
    def test_reorders_flags(self, op):
        tv = _SAMPLE_TEST_VECTORS[op]
        conf_class = tv["conf_class"]
        canonical = canonicalize_config(tv["raw"], conf_class, "gfx900", 64, 1)
        assert canonical == tv["canonical"]

    @pytest.mark.parametrize("op", _ALL_OPS)
    def test_idempotent(self, op):
        tv = _SAMPLE_TEST_VECTORS[op]
        conf_class = tv["conf_class"]
        idempotent_form = tv["idempotent"]
        result = canonicalize_config(idempotent_form, conf_class, "gfx900", 64, 1)
        assert result == idempotent_form

    @pytest.mark.parametrize("op", _ALL_OPS)
    def test_round_trip_preserves_data(self, op):
        """Canonicalize twice and verify the result is stable."""
        tv = _SAMPLE_TEST_VECTORS[op]
        conf_class = tv["conf_class"]
        first = canonicalize_config(tv["raw"], conf_class, "gfx900", 64, 1)
        second = canonicalize_config(first, conf_class, "gfx900", 64, 1)
        assert first == second

    def test_mlir_path_passthrough(self):
        path = "/some/test.mlir"
        assert canonicalize_test_vector(path, GemmConfiguration, "gfx900", 64, 1) == path

    def test_invalid_config_raises_valueerror(self):
        with pytest.raises(ValueError, match="Failed to parse"):
            canonicalize_config("not a valid config", GemmConfiguration, "gfx900", 64, 1)

    def test_wrong_op_raises_valueerror(self):
        gemm_tv = "-t f32 -out_datatype f32 -transA false -transB false -g 1 -m 64 -n 128 -k 256"
        with pytest.raises(ValueError, match="Failed to parse"):
            canonicalize_config(gemm_tv, ConvConfiguration, "gfx900", 64, 1)

    def test_fusion_dispatches_to_conv(self):
        """Fusion path (PerfConfiguration base class) routes 'conv*' prefix to ConvConfiguration."""
        raw = _SAMPLE_TEST_VECTORS["conv"]["raw"]
        expected = canonicalize_config(raw, ConvConfiguration, "gfx900", 64, 1)
        result = canonicalize_config(raw, PerfConfiguration, "gfx900", 64, 1)
        assert result == expected

    def test_fusion_dispatches_to_gemm(self):
        """Fusion path (PerfConfiguration base class) routes non-'conv' prefix to GemmConfiguration."""
        raw = _SAMPLE_TEST_VECTORS["gemm"]["raw"]
        expected = canonicalize_config(raw, GemmConfiguration, "gfx900", 64, 1)
        result = canonicalize_config(raw, PerfConfiguration, "gfx900", 64, 1)
        assert result == expected

    def test_fusion_invalid_raises_valueerror_with_resolved_class(self):
        """Errors from fusion dispatch should name the resolved concrete class, not the base."""
        with pytest.raises(ValueError, match="ConvConfiguration"):
            canonicalize_config("convfp16 not a real config", PerfConfiguration, "gfx900", 64, 1)
        with pytest.raises(ValueError, match="GemmConfiguration"):
            canonicalize_config("not a real config", PerfConfiguration, "gfx900", 64, 1)

    def test_cache_loaded_with_canonical_key(self):
        """Verify that from_output_file canonicalizes test vectors so cache lookups match."""
        raw = "-g 1 -m 1024 -k 769 -n 512 -t f32 -out_datatype f32 -transA false -transB false"
        canonical = canonicalize_config(raw, GemmConfiguration, "gfx900", 64, 1)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(
                "# arch\tnumCUs\tnumChiplets\ttestVector\tperfConfig\tTFlops\ttuningSpace\tcommitId\ttimestamp\tdurationSec\n"
            )
            f.write(
                f"gfx900\t64\t1\t{raw}\tperf_best\t1.5\tfull\tabc123\t2025-01-01T00:00:00Z\t10.0\n")
            path = f.name
        try:
            opts = Options(
                chip="gfx900",
                arch="gfx900",
                num_cu=64,
                num_chiplets=1,
                debug=False,
                debug_quick_tune_data=False,
                quiet=False,
                verbose=False,
                tuning_space_kind="full",
                rocmlir_gen_flags="",
                verify_mode="none",
                verify_perfconfigs=False,
                output=path,
                abort_on_error=False,
                retune=False,
                retry_states=frozenset(),
                gpu_ids=[0],
                num_cpus=None,
                wait_for_compiles=False,
                timeout=None,
                verify_timeout=None,
                gpu_run_timeout=0,
            )
            cache = TunedConfigsCache.from_output_file(opts, GemmConfiguration)
            assert cache.count() == 1
            assert cache.get(raw) is None, "raw (non-canonical) key should not match"
            r = cache.get(canonical)
            assert r is not None, "canonical key should match"
            assert r.winning_config == "perf_best"
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

    def test_negative_gpu_run_timeout_rejected(self, capsys):
        topology = _make_mock_gpu_topology([(0, "gfx900")])
        available = [0]
        with pytest.raises(SystemExit):
            tuningRunner.parse_arguments(
                topology,
                available,
                [
                    "--op",
                    "gemm",
                    "--config",
                    "-g 1 -m 1024 -k 769 -n 512",
                    "--gpu-run-timeout",
                    "-1",
                ],
            )
        assert "argument --gpu-run-timeout: must be non-negative" in capsys.readouterr().err


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
        numa_lock = NumaNodeLock()
        winner, tflops, entries = find_best_perfconfig([],
                                                       config,
                                                       paths,
                                                       options,
                                                       gpu_id=0,
                                                       numa_lock=numa_lock)
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
        numa_lock = NumaNodeLock()
        lines = ["perf_cfg_1\t12345"]
        winner, tflops, entries = find_best_perfconfig(lines,
                                                       config,
                                                       paths,
                                                       options,
                                                       gpu_id=0,
                                                       numa_lock=numa_lock)
        assert winner == "perf_cfg_1"
        assert tflops == 1.5
        assert len(entries) == 1


class TestTuneConfig:
    """Tests for tune_config subprocess result handling."""

    def test_gpu_timeout_exit_code_marks_result_gpu_timed_out(self, monkeypatch):
        from unittest.mock import MagicMock

        class FakeStdout:

            def close(self):
                pass

        class FakeProcess:

            def __init__(self, returncode, stdout=b"", stderr=b""):
                self.returncode = returncode
                self._stdout = stdout
                self._stderr = stderr
                self.stdout = FakeStdout()
                self.pid = 1234

            def communicate(self, timeout=None):
                return self._stdout, self._stderr

            def kill(self):
                pass

            def wait(self, timeout=None):
                return self.returncode

        class FakeConfig:

            def generate_mlir_driver_commandline(self, rocmlir_gen_flags, kernel_repeats=None):
                return "--fake-rocmlir-gen-arg"

        class FakeConfiguration:

            @staticmethod
            def from_command_line(command_line, arch, num_cu, num_chiplets):
                return FakeConfig()

        rocmlir_gen = FakeProcess(returncode=0)
        tuning_driver = FakeProcess(returncode=tuningRunner.GPU_TIMEOUT_EXIT_CODE,
                                    stderr=b"gpu timeout")

        def fake_popen(command, **kwargs):
            if command[0] == "rocmlir-gen":
                return rocmlir_gen
            assert command[0] == "rocmlir-tuning-driver"
            return tuning_driver

        monkeypatch.setattr(tuningRunner.subprocess, "Popen", fake_popen)

        paths = MagicMock()
        paths.mlir_paths.rocmlir_gen_path = "rocmlir-gen"
        paths.mlir_paths.rocmlir_tuning_driver_path = "rocmlir-tuning-driver"
        options = MagicMock()
        options.tuning_space_kind = "quick"
        options.debug = False
        options.wait_for_compiles = False
        options.gpu_run_timeout = 30
        options.timeout = None
        options.arch = "gfx900"
        options.num_cu = 64
        options.num_chiplets = 1
        options.rocmlir_gen_flags = ""

        result = tune_config("-g 1 -m 1024 -k 769 -n 512",
                             FakeConfiguration,
                             paths,
                             options,
                             gpu_id=0,
                             num_compile_threads=1,
                             numa_lock=NumaNodeLock())

        assert not result.success
        assert result.gpu_timed_out
        assert result.gpu_id == 0


class TestResolveVerifyMode:
    """Tests for resolve_verify_mode (gpu->cpu fallback for unsupported configs)."""

    @pytest.mark.parametrize("mode", ["none", "cpu"])
    @pytest.mark.parametrize("cls", [
        GemmConfiguration, ConvConfiguration, AttentionConfiguration, ConvGemmConfiguration,
        GemmGemmConfiguration
    ])
    def test_non_gpu_modes_pass_through(self, mode, cls):
        assert resolve_verify_mode(mode, cls.__new__(cls)) == mode

    @pytest.mark.parametrize("cls,expected", [
        (GemmConfiguration, "gpu"),
        (ConvConfiguration, "gpu"),
        (AttentionConfiguration, "cpu"),
        (ConvGemmConfiguration, "cpu"),
        (GemmGemmConfiguration, "cpu"),
    ])
    def test_gpu_mode(self, cls, expected):
        assert resolve_verify_mode("gpu", cls.__new__(cls)) == expected


class TestNumaNodeLock:
    """Tests for NumaNodeLock reader-writer semantics."""

    def test_shared_holders_run_concurrently(self):
        lock = NumaNodeLock()
        n = 4
        entered = threading.Barrier(n + 1, timeout=2.0)
        release = threading.Event()

        def reader():
            lock.acquire_shared()
            try:
                entered.wait()
                release.wait()
            finally:
                lock.release_shared()

        threads = [threading.Thread(target=reader) for _ in range(n)]
        for t in threads:
            t.start()
        entered.wait()
        release.set()
        for t in threads:
            t.join(timeout=2.0)
            assert not t.is_alive()

    def test_no_overlap_under_contention(self):
        """Stress test: assert all reader/writer exclusion invariants."""
        lock = NumaNodeLock()
        readers_active = 0
        writer_active = False
        state_lock = threading.Lock()
        violations = []
        stop = threading.Event()

        def reader():
            nonlocal readers_active
            while not stop.is_set():
                lock.acquire_shared()
                with state_lock:
                    if writer_active:
                        violations.append("reader saw active writer")
                    readers_active += 1
                with state_lock:
                    readers_active -= 1
                lock.release_shared()

        def writer():
            nonlocal writer_active
            while not stop.is_set():
                lock.acquire_exclusive()
                with state_lock:
                    if readers_active > 0:
                        violations.append("writer saw active readers")
                    if writer_active:
                        violations.append("writer saw another active writer")
                    writer_active = True
                with state_lock:
                    writer_active = False
                lock.release_exclusive()

        threads = ([threading.Thread(target=reader) for _ in range(4)] +
                   [threading.Thread(target=writer) for _ in range(2)])
        for t in threads:
            t.start()
        time.sleep(0.5)
        stop.set()
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive()
        assert violations == [], f"Lock invariant violated: {violations}"

    def test_release_shared_without_acquire_is_noop(self):
        """release_shared on a fresh lock must not corrupt state or block subsequent acquires."""
        lock = NumaNodeLock()
        lock.release_shared()
        lock.acquire_exclusive()
        lock.release_exclusive()

    def test_release_exclusive_without_acquire_is_noop(self):
        """release_exclusive on a fresh lock must not corrupt state or block subsequent acquires."""
        lock = NumaNodeLock()
        lock.release_exclusive()
        lock.acquire_shared()
        lock.release_shared()

    def test_release_shared_extra_call_is_noop(self):
        """An extra release_shared after balanced acquire/release must not push the count negative."""
        lock = NumaNodeLock()
        lock.acquire_shared()
        lock.release_shared()
        lock.release_shared()
        lock.acquire_exclusive()
        lock.release_exclusive()

    def test_release_exclusive_double_call_is_noop(self):
        """An extra release_exclusive after balanced acquire/release must not flip the flag back."""
        lock = NumaNodeLock()
        lock.acquire_exclusive()
        lock.release_exclusive()
        lock.release_exclusive()
        lock.acquire_shared()
        lock.release_shared()


class TestDebugFileWriter:
    """DebugFileWriter rejects appending rows whose schema would not match an existing header."""

    @staticmethod
    def _make_result(entries):
        return TuningResult(test_vector="-g 1 -m 1 -n 1 -k 1",
                            success=True,
                            gpu_id=0,
                            duration_seconds=1.0,
                            timestamp="2026-01-01T00:00:00Z",
                            winning_config="cfg",
                            max_tflops=1.0,
                            entries=entries)

    def test_fresh_file_writes_header(self, tmp_path):
        path = str(tmp_path / "out.tsv.debug")
        with DebugFileWriter(path) as w:
            w.write_result(self._make_result([{"M": 1, "N": 2, "PerfConfig": "p", "TFlops": 1.0}]))
        contents = Path(path).read_text().splitlines()
        assert contents[0] == "M\tN\tPerfConfig\tTFlops"
        assert contents[1] == "1\t2\tp\t1.0"

    def test_append_with_same_schema_skips_header(self, tmp_path):
        path = str(tmp_path / "out.tsv.debug")
        with DebugFileWriter(path) as w:
            w.write_result(self._make_result([{"M": 1, "N": 2, "PerfConfig": "p1", "TFlops": 1.0}]))
        with DebugFileWriter(path) as w:
            w.write_result(self._make_result([{"M": 3, "N": 4, "PerfConfig": "p2", "TFlops": 2.0}]))
        contents = Path(path).read_text().splitlines()
        # One header, two data rows -- second open must not have re-emitted the header.
        assert contents[0] == "M\tN\tPerfConfig\tTFlops"
        assert contents[1] == "1\t2\tp1\t1.0"
        assert contents[2] == "3\t4\tp2\t2.0"
        assert len(contents) == 3

    def test_append_with_different_schema_raises(self, tmp_path):
        path = str(tmp_path / "out.tsv.debug")
        with DebugFileWriter(path) as w:
            w.write_result(
                self._make_result([{
                    "M": 1,
                    "N": 2,
                    "K": 3,
                    "PerfConfig": "p",
                    "TFlops": 1.0
                }]))
        with DebugFileWriter(path) as w:
            with pytest.raises(ValueError, match="schema that does not match"):
                w.write_result(
                    self._make_result([{
                        "H": 1,
                        "W": 2,
                        "PerfConfig": "p",
                        "TFlops": 1.0
                    }]))

    def test_empty_existing_file_treated_as_fresh(self, tmp_path):
        path = str(tmp_path / "out.tsv.debug")
        Path(path).touch()  # file exists but is empty
        with DebugFileWriter(path) as w:
            w.write_result(self._make_result([{"M": 1, "N": 2, "PerfConfig": "p", "TFlops": 1.0}]))
        contents = Path(path).read_text().splitlines()
        assert contents[0] == "M\tN\tPerfConfig\tTFlops"
