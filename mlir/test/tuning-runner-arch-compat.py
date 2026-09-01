# RUN: %python %s

import importlib.util
import math
import pathlib
import re
import sys
import tempfile
import types
from enum import Enum


def _install_import_stubs():
    """Stub optional runtime dependencies so this test is hermetic."""
    if "numpy" not in sys.modules:
        numpy_stub = types.ModuleType("numpy")
        numpy_stub.isfinite = math.isfinite
        sys.modules["numpy"] = numpy_stub

    if "pandas" not in sys.modules:
        pandas_stub = types.ModuleType("pandas")

        class _DataFrame:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def to_csv(self, *args, **kwargs):
                del args, kwargs

        pandas_stub.DataFrame = _DataFrame
        sys.modules["pandas"] = pandas_stub

    if "tqdm" not in sys.modules:
        tqdm_stub = types.ModuleType("tqdm")

        def _tqdm(*args, **kwargs):
            del args, kwargs

            class _DummyBar:
                def update(self, *_args, **_kwargs):
                    pass

                def set_postfix_str(self, *_args, **_kwargs):
                    pass

                def close(self):
                    pass

            return _DummyBar()

        _tqdm.write = lambda *_args, **_kwargs: None
        tqdm_stub.tqdm = _tqdm
        sys.modules["tqdm"] = tqdm_stub

    if "perfCommonUtils" not in sys.modules:
        perf_common_utils_stub = types.ModuleType("perfCommonUtils")

        class _Operation(Enum):
            CONV = "conv"
            GEMM = "gemm"
            ATTENTION = "attention"
            GEMM_GEMM = "gemm_gemm"
            CONV_GEMM = "conv_gemm"
            FUSION = "fusion"

            @classmethod
            def from_name(cls, name: str):
                for op in cls:
                    if op.value == name:
                        return op
                raise ValueError(f"unknown operation: {name}")

        perf_common_utils_stub.CORRECT_RESULT_RE = re.compile(r".*")
        perf_common_utils_stub.Operation = _Operation
        sys.modules["perfCommonUtils"] = perf_common_utils_stub

    if "perfRunner" not in sys.modules:
        perf_runner_stub = types.ModuleType("perfRunner")
        for name in [
                "AttentionConfiguration",
                "ConvConfiguration",
                "ConvGemmConfiguration",
                "GemmConfiguration",
                "GemmGemmConfiguration",
                "Paths",
                "PerfConfiguration",
        ]:
            setattr(perf_runner_stub, name, type(name, (), {}))
        sys.modules["perfRunner"] = perf_runner_stub


def _load_tuning_runner_module():
    _install_import_stubs()
    module_path = pathlib.Path(__file__).resolve().parents[1] / "utils" / "performance" / "tuningRunner.py"
    spec = importlib.util.spec_from_file_location("tuningRunner", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_options(tuning_runner, output_path: str):
    return tuning_runner.Options(debug=False,
                                 tuning_space_kind="quick",
                                 quiet=True,
                                 verbose=False,
                                 chip="gfx942",
                                 arch="gfx942:sramecc+:xnack-",
                                 num_cu=120,
                                 num_chiplets=2,
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
                                 timeout=None)


def _test_output_writer_uses_arch(tuning_runner):
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = pathlib.Path(tmpdir) / "tuning.tsv"
        options = _make_options(tuning_runner, str(output_path))
        tuning_runner.get_git_commit_hash = lambda: "deadbeef"

        result = tuning_runner.TuningResult(test_vector="gemm_test",
                                            success=True,
                                            duration_seconds=1.4,
                                            timestamp="2026-03-13T16:00:00Z",
                                            winning_config="mock_config",
                                            max_tflops=123.0)

        with tuning_runner.OutputFileWriter(str(output_path), options) as writer:
            writer.write_result(result)

        lines = output_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        first_data_field = lines[1].split("\t")[0]
        assert first_data_field == options.arch
        assert first_data_field != options.chip


def _test_cache_accepts_arch_and_legacy_chip(tuning_runner):
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = pathlib.Path(tmpdir) / "cache.tsv"
        options = _make_options(tuning_runner, str(output_path))
        tuning_runner.get_git_commit_hash = lambda: "deadbeef"

        header = "# " + "\t".join(tuning_runner.OUTPUT_HEADER_COLUMNS)
        rows = [
            "\t".join([
                options.chip, "120", "2", "legacy_chip_vector", "cfg_a", "11.2", "quick",
                "deadbeef", "2026-03-13T16:00:00Z", "1.0"
            ]),
            "\t".join([
                options.arch, "120", "2", "full_arch_vector", "cfg_b", "12.2", "quick",
                "deadbeef", "2026-03-13T16:01:00Z", "1.1"
            ]),
            "\t".join([
                "gfx900", "120", "2", "wrong_arch_vector", "cfg_c", "13.2", "quick", "deadbeef",
                "2026-03-13T16:02:00Z", "1.2"
            ]),
        ]
        output_path.write_text("\n".join([header] + rows) + "\n", encoding="utf-8")

        cache = tuning_runner.TunedConfigsCache.from_output_file(options)
        assert cache.count() == 2
        assert cache.contains("legacy_chip_vector")
        assert cache.contains("full_arch_vector")
        assert not cache.contains("wrong_arch_vector")


def _test_cache_accepts_old_header_tuning_space_format(tuning_runner):
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = pathlib.Path(tmpdir) / "old-format.tsv"
        options = _make_options(tuning_runner, str(output_path))
        tuning_runner.get_git_commit_hash = lambda: "deadbeef"

        old_header = "# arch\tnumCUs\tnumChiplets\ttestVector\tperfConfig (quick)\tTFlops"
        old_data = "\t".join([options.arch, "120", "2", "old_format_vector", "cfg_old", "9.1"])
        output_path.write_text(f"{old_header}\n{old_data}\n", encoding="utf-8")

        cache = tuning_runner.TunedConfigsCache.from_output_file(options)
        assert cache.count() == 1
        assert cache.contains("old_format_vector")


def main():
    tuning_runner = _load_tuning_runner_module()
    _test_output_writer_uses_arch(tuning_runner)
    _test_cache_accepts_arch_and_legacy_chip(tuning_runner)
    _test_cache_accepts_old_header_tuning_space_format(tuning_runner)


if __name__ == "__main__":
    main()
