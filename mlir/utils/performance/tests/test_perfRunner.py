"""
Tests for perfRunner.py.

These tests cover parsing, tuning DB format, layout helpers, and pure logic that does not
require a real GPU or ROCm. They run in CI (e.g. GitHub Actions) where no AMD GPU is available.
"""
import math
import os
import sys
import tempfile
from pathlib import Path

# Ensure we can import from parent (perfRunner lives in mlir/utils/performance)
_test_dir = Path(__file__).resolve().parent
sys_path_parent = str(_test_dir.parent)
if sys_path_parent not in sys.path:
    sys.path.insert(0, sys_path_parent)
# Mock hip so perfRunner can be imported without ROCm (CI has no GPU)
exec(open(_test_dir / "mock_hip.py").read(), {"__file__": str(_test_dir / "mock_hip.py"), "sys": sys})

import perfRunner  # noqa: E402 - must run after mock_hip


class TestParseTuningDbLine:
    """Tests for parse_tuning_db_line (legacy, v2, v3 formats)."""

    def test_legacy_three_entries(self):
        out = perfRunner.parse_tuning_db_line(["gfx900", "config1", "perf1"], 120, 1)
        assert out == ("gfx900", 120, 1, "config1", "perf1")

    def test_v2_four_entries(self):
        out = perfRunner.parse_tuning_db_line(
            ["gfx900", "120", "config1", "perf1"], fallback_num_chiplets=1
        )
        assert out == ("gfx900", 120, 1, "config1", "perf1")

    def test_v3_five_entries(self):
        out = perfRunner.parse_tuning_db_line(
            ["gfx900", "120", "2", "config1", "perf1", "1.5"]
        )
        assert out == ("gfx900", 120, 2, "config1", "perf1")

    def test_v3_extra_columns(self):
        out = perfRunner.parse_tuning_db_line(
            ["gfx90x", "304", "8", "gemm -m 1024", "perf_x", "2.0", "extra"]
        )
        assert out == ("gfx90x", 304, 8, "gemm -m 1024", "perf_x")

    def test_invalid_returns_none(self):
        assert perfRunner.parse_tuning_db_line([]) is None
        assert perfRunner.parse_tuning_db_line(["a"]) is None
        assert perfRunner.parse_tuning_db_line(["a", "b"]) is None


class TestReadTuningDb:
    """Tests for read_tuning_db."""

    def test_read_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("")
            path = f.name
        try:
            db = perfRunner.read_tuning_db(path)
            assert db == {}
        finally:
            os.unlink(path)

    def test_read_with_header_and_comments(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("# arch\tconfig\tperfconfig\n")
            f.write("gfx900\t-g 1 -m 1024\tperf_1\n")
            f.write("\n")
            f.write("gfx900\t-g 1 -m 2048\tperf_2\n")
            path = f.name
        try:
            db = perfRunner.read_tuning_db(path, fallback_num_cu=120, fallback_num_chiplets=1)
            assert len(db) == 2
            assert db[("gfx900", 120, 1, "-g 1 -m 1024")] == "perf_1"
            assert db[("gfx900", 120, 1, "-g 1 -m 2048")] == "perf_2"
        finally:
            os.unlink(path)

    def test_read_nonexistent_returns_none(self):
        db = perfRunner.read_tuning_db("/nonexistent/path.tsv")
        assert db is None


class TestGetNumChiplets:
    """Tests for get_num_chiplets (pure logic, no GPU)."""

    def test_gfx942_304(self):
        assert perfRunner.get_num_chiplets("gfx942", 304) == 8

    def test_gfx942_80(self):
        assert perfRunner.get_num_chiplets("gfx942", 80) == 4

    def test_gfx950(self):
        assert perfRunner.get_num_chiplets("gfx950", 228) == 8

    def test_default_one(self):
        assert perfRunner.get_num_chiplets("gfx900", 64) == 1
        assert perfRunner.get_num_chiplets("gfx1030", 72) == 1


class TestParseDataTypes:
    """Tests for parse_data_types (gemm data types)."""

    def test_empty_returns_defaults(self):
        dtypes, out_map = perfRunner.parse_data_types(None)
        assert "f32" in dtypes
        assert out_map.get("f32") == "f32"

    def test_single_type(self):
        dtypes, out_map = perfRunner.parse_data_types(["f16"])
        assert dtypes == ["f16"]
        assert out_map["f16"] == "f16"

    def test_i8_maps_to_i32(self):
        dtypes, out_map = perfRunner.parse_data_types(["i8"])
        assert "i8" in dtypes
        assert out_map["i8"] == "i32"

    def test_fp8_maps_to_f32(self):
        dtypes, out_map = perfRunner.parse_data_types(["fp8"])
        assert out_map["fp8"] == "f32"

    def test_pair_notation(self):
        dtypes, out_map = perfRunner.parse_data_types(["fp8_fp8"])
        assert "fp8" in dtypes
        assert out_map["fp8"] == "fp8"


class TestLayoutHelpers:
    """Tests for input/output/filter layout conversion."""

    def test_input_layouts(self):
        assert perfRunner.input_layouts("NCHW") == "nchw"

    def test_output_layouts(self):
        # OUTPUT_LAYOUT_MAP: C -> k, so NCHW -> nkhw
        assert perfRunner.output_layouts("NCHW") == "nkhw"

    def test_filter_layouts(self):
        # FILTER_LAYOUT_MAP: H->y, W->x, so NCHW -> kcyx
        assert perfRunner.filter_layouts("NCHW") == "kcyx"

    def test_inverse_roundtrip(self):
        layout = "NHWC"
        assert perfRunner.inverse_input_layouts(perfRunner.input_layouts(layout)) == layout
        assert perfRunner.inverse_output_layouts(perfRunner.output_layouts(layout)) == layout
        assert perfRunner.inverse_filter_layouts(perfRunner.filter_layouts(layout)) == layout


class TestGetNanoseconds:
    """Tests for get_nanoseconds (reads CSV from rocprof)."""

    def test_missing_file_returns_nan(self):
        ns = perfRunner.get_nanoseconds("/nonexistent/path.csv")
        assert math.isnan(ns)

    def test_valid_csv(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            f.write("KernelName,AverageNs,SomeOther\n")
            f.write("kern1,1000,0\n")
            f.write("kern2,2000,0\n")
            path = f.name
        try:
            ns = perfRunner.get_nanoseconds(path)
            assert ns == 3000
        finally:
            os.unlink(path)


class TestGetProfilerOutputPath:
    """Tests for get_profiler_output_path (arch-dependent path)."""

    def test_gfx950_returns_base(self):
        assert perfRunner.get_profiler_output_path("gfx950", "results.csv") == "results.csv"

    def test_other_arch_returns_pmc_subdir(self):
        p = perfRunner.get_profiler_output_path("gfx900", "results.csv")
        assert p == os.path.join("pmc_1", "results.csv")


class TestGetMetricArgsForRocprof:
    """Tests for get_metric_args_for_rocprof."""

    def test_gfx950_no_metrics(self):
        args = perfRunner.get_metric_args_for_rocprof("gfx950")
        assert args == []

    def test_other_arch_uses_metrics_file(self):
        args = perfRunner.get_metric_args_for_rocprof("gfx900")
        assert "-i" in args
        assert any("rocmlir_metrics" in str(x) for x in args)


class TestGetMiliseconds:
    """Tests for get_miliseconds (kernel time parsing)."""

    def test_match(self):
        out = perfRunner.get_miliseconds(b"some output\nkernel time: 1.234\n")
        assert out == 1.234

    def test_no_match_returns_nan(self):
        out = perfRunner.get_miliseconds(b"no kernel time here")
        assert math.isnan(out)
