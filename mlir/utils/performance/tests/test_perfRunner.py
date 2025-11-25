import perfRunner


def test_layout_round_trip():
    assert perfRunner.input_layouts("NCHW") == "nchw"
    assert perfRunner.output_layouts("NCHW") == "nkhw"
    assert perfRunner.filter_layouts("NCHW") == "kcyx"
    assert perfRunner.inverse_input_layouts("nchw") == "NCHW"
    assert perfRunner.inverse_output_layouts("nkhw") == "NCHW"
    assert perfRunner.inverse_filter_layouts("kcyx") == "NCHW"


def test_get_conv_configurations_uses_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(perfRunner, "DIRECTIONS", ["-F 1"])
    monkeypatch.setattr(perfRunner, "DATA_TYPES", ["conv"])
    monkeypatch.setattr(perfRunner, "LAYOUTS", ["NHWC"])
    monkeypatch.setattr(perfRunner, "get_chip", lambda: "gfx1200")

    config_file = tmp_path / "conv.txt"
    config_file.write_text(
        "-n 1 -c 2 -H 3 -W 4 -k 5 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -g 1\n"
    )

    configs = perfRunner.get_conv_configurations(str(config_file))

    assert configs == [
        "conv -F 1 -f NHWC -I NHWC -O NHWC "
        "-n 1 -c 2 -H 3 -W 4 -k 5 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -g 1"
    ]


def test_conv_configuration_command_generation():
    cfg = perfRunner.ConvConfiguration(
        "f32",
        "fwd",
        "NCHW",
        "NCHW",
        "NCHW",
        n=1,
        c=2,
        hi=3,
        wi=4,
        k=5,
        y=1,
        x=1,
        conv_stride_h=1,
        conv_stride_w=1,
        padding_h=0,
        padding_w=0,
        dilation_h=1,
        dilation_w=1,
        group=1,
        arch="gfx1200",
        num_cu=10,
    )

    command = cfg.generate_mlir_driver_commandline("", kernel_repeats=5)
    assert "--operation conv" in command
    assert "--fil_layout kcyx" in command
    assert "--in_layout nchw" in command
    assert "--out_layout nkhw" in command
    assert "--kernel-repeats 5" in command
    assert "--perf_config=" in command

    round_trip = cfg.to_command_line()
    assert "conv " in round_trip
    assert "-F 1" in round_trip
    assert "-f NCHW" in round_trip
    assert "-I NCHW" in round_trip
    assert "-O NCHW" in round_trip


def test_gemm_configuration_command_generation():
    cfg = perfRunner.GemmConfiguration(
        dtype="f32",
        out_dtype="f32",
        g=1,
        m=64,
        k=32,
        n=16,
        trans_a=False,
        trans_b=True,
        scaled_gemm=True,
        scale_a_dtype="f32",
        scale_b_dtype="f8E8M0FNU",
        trans_scale_a=True,
        trans_scale_b=False,
        arch="gfx1200",
        num_cu=8,
        perf_config="best",
    )

    command = cfg.generate_mlir_driver_commandline("", kernel_repeats=7)
    assert "-operation gemm" in command
    assert "-scaledGemm" in command
    assert "-scale_a_dtype f32" in command
    assert "-scale_b_dtype f8E8M0FNU" in command
    assert "-transScaleA=True" in command
    assert "-transScaleB" not in command  # only emitted when True
    assert "--kernel-repeats 7" in command
    assert "--perf_config=best" in command

    round_trip = cfg.to_command_line()
    assert "-t f32" in round_trip
    assert "-out_datatype f32" in round_trip
    assert "-transA false" in round_trip
    assert "-transB true" in round_trip
    assert "-scaledGemm" in round_trip
    assert "-scale_a_dtype f32" in round_trip
    assert "-scale_b_dtype f8E8M0FNU" in round_trip
    assert "-transScaleA true" in round_trip
    assert "-transScaleB" not in round_trip


def test_attention_configuration_command_generation(monkeypatch):
    monkeypatch.setattr(perfRunner, "DATA_TYPES_ATTENTION", ["f16"])

    cfg = perfRunner.AttentionConfiguration(
        dtype="f16",
        g=1,
        seq_len_q=128,
        seq_len_k=128,
        num_heads_q=4,
        num_heads_kv=4,
        head_dim_qk=64,
        head_dim_v=64,
        with_attn_scale=True,
        with_attn_bias=False,
        trans_q=False,
        trans_k=True,
        trans_v=False,
        trans_o=True,
        causal=True,
        return_lse=False,
        split_kv=1,
        arch="gfx1200",
        num_cu=12,
        perf_config="fast",
    )

    command = cfg.generate_mlir_driver_commandline("", kernel_repeats=3)
    assert "-operation attention" in command
    assert "-t f16" in command
    assert "-with-attn-scale=True" in command
    assert "-with-attn-bias=False" in command
    assert "-transQ=False" in command
    assert "-transK=True" in command
    assert "-transO=True" in command
    assert "-causal=True" in command
    assert "-return_lse=False" in command
    assert "-split_kv=1" in command
    assert "--kernel-repeats 3" in command
    assert "--perf_config=fast" in command

    round_trip = cfg.to_command_line()
    assert "-t f16" in round_trip
    assert "-transQ false" in round_trip
    assert "-transK true" in round_trip
    assert "-transO true" in round_trip
    assert "-causal true" in round_trip
    assert "-return_lse false" in round_trip
    assert "-split_kv 1" in round_trip
    assert "-with-attn-scale true" in round_trip
    assert "-with-attn-bias false" in round_trip


def test_conv_configuration_from_to_command_line_round_trip(monkeypatch):
    monkeypatch.setattr(perfRunner, "get_chip", lambda: "gfx1200")
    args = [
        "conv",
        "-F",
        "1",
        "-f",
        "NHWC",
        "-I",
        "NHWC",
        "-O",
        "NHWC",
        "-n",
        "2",
        "-c",
        "4",
        "-H",
        "8",
        "-W",
        "8",
        "-k",
        "16",
        "-y",
        "3",
        "-x",
        "3",
        "-u",
        "1",
        "-v",
        "1",
        "-p",
        "1",
        "-q",
        "1",
        "-l",
        "1",
        "-j",
        "1",
        "-g",
        "1",
    ]
    cfg = perfRunner.ConvConfiguration.from_command_line(args, arch="gfx1200", num_cu=6)
    round_trip = cfg.to_command_line()
    assert "-n 2" in round_trip
    assert "-c 4" in round_trip
    assert "-F 1" in round_trip
    assert cfg.arch == "gfx1200"
    assert cfg.num_cu == 6


def test_gemm_configuration_from_to_command_line_round_trip():
    args = [
        "-t",
        "f16",
        "-out_datatype",
        "f16",
        "-g",
        "1",
        "-m",
        "64",
        "-k",
        "32",
        "-n",
        "16",
        "-transA",
        "false",
        "-transB",
        "true",
        "-scaledGemm",
        "-scale_a_dtype",
        "f32",
        "-scale_b_dtype",
        "f8E8M0FNU",
        "-perf_config",
        "cfg",
    ]
    cfg = perfRunner.GemmConfiguration.from_command_line(args, arch="gfx1200", num_cu=4)
    round_trip = cfg.to_command_line()
    assert "-t f16" in round_trip
    assert "-out_datatype f16" in round_trip
    assert "-g 1" in round_trip
    assert "-transA false" in round_trip
    assert "-transB true" in round_trip
    assert "-scale_a_dtype f32" in round_trip
    assert "-scale_b_dtype f8E8M0FNU" in round_trip
    assert cfg.perfconfig == "cfg"
    assert cfg.arch == "gfx1200"
    assert cfg.num_cu == 4


def test_attention_configuration_from_to_command_line_round_trip(monkeypatch):
    monkeypatch.setattr(perfRunner, "DATA_TYPES_ATTENTION", ["f16"])
    args = [
        "-t",
        "f16",
        "-g",
        "2",
        "-seq_len_q",
        "128",
        "-seq_len_k",
        "128",
        "-num_heads_q",
        "4",
        "-num_heads_kv",
        "4",
        "-head_dim_qk",
        "64",
        "-head_dim_v",
        "64",
        "-with-attn-scale",
        "true",
        "-with-attn-bias",
        "false",
        "-transQ",
        "false",
        "-transK",
        "true",
        "-transV",
        "false",
        "-transO",
        "true",
        "-causal",
        "true",
        "-return_lse",
        "false",
        "-split_kv",
        "1",
        "-perf_config",
        "pc",
    ]
    cfg = perfRunner.AttentionConfiguration.from_command_line(args, arch="gfx1200", num_cu=10)
    round_trip = cfg.to_command_line()
    assert "-t f16" in round_trip
    assert "-g 2" in round_trip
    assert "-seq_len_q 128" in round_trip
    assert "-num_heads_q 4" in round_trip
    assert "-with-attn-scale true" in round_trip
    assert "-with-attn-bias false" in round_trip
    assert "-transK true" in round_trip
    assert "-causal true" in round_trip
    assert cfg.perfconfig == "pc"
    assert cfg.arch == "gfx1200"
    assert cfg.num_cu == 10


def test_get_gemm_configurations_scaled(monkeypatch, tmp_path):
    monkeypatch.setattr(perfRunner, "get_chip", lambda: "gfx1200")
    config_file = tmp_path / "gemm.txt"
    config_file.write_text("-m 128 -k 64 -n 256 -scaledGemm\n")

    configs = perfRunner.get_gemm_configurations(
        str(config_file), datatypes=["f16"], scale_types=["f32", "f8E8M0FNU"]
    )

    assert len(configs) == 16  # 1 dtype * 2 transA * 2 transB * 2 scale_a * 2 scale_b
    assert any(
        "-scale_a_dtype f32" in entry and "-scale_b_dtype f8E8M0FNU" in entry
        for entry in configs
    )
    assert all("-t f16" in entry for entry in configs)
