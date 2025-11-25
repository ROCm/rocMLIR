import numpy as np

import tuningRunner


def test_verify_mode_flags():
    assert tuningRunner.verify_mode_flags("none") == ""
    assert tuningRunner.verify_mode_flags("cpu").strip() == "-pv"
    assert "--verifier-keep-perf-config=false" in tuningRunner.verify_mode_flags("gpu")
    try:
        tuningRunner.verify_mode_flags("unknown")
        assert False, "verify_mode_flags should raise for unknown modes"
    except ValueError:
        pass


def test_get_winning_config_prefers_fastest(monkeypatch):
    class DummyConfig:
        def __init__(self):
            self.perfconfigs = []

        def set_perfconfig(self, perfconfig):
            self.perfconfigs.append(perfconfig)

        def table_entry(self, nanoseconds):
            score = np.nan
            if not np.isnan(nanoseconds):
                score = 1000.0 / nanoseconds
            return {"TFlops": score}

    options = tuningRunner.Options(
        debug=False,
        tuning_space_kind="full",
        quiet=True,
        arch="gfx1200",
        num_cu=10,
        rocmlir_gen_flags="",
        verify_mode="none",
        verify_perfconfigs=False,
        tflops=False,
        compact_print=True,
    )

    dummy_config = DummyConfig()
    all_data = []
    winner, max_tflops = tuningRunner.get_winning_config(
        [b"fast\t5", b"slow\t10", b"skip_me\tN/A"],
        "vector",
        dummy_config,
        all_data,
        paths=None,
        options=options,
    )

    assert winner == "fast"
    assert max_tflops == 200.0
    assert len(all_data) == 3
    assert dummy_config.perfconfigs[0] == "fast"
    assert dummy_config.perfconfigs[1] == "slow"
