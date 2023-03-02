import getopt
import sys

# Useful tabled conversions
MLIR_FILTER_LAYOUTS = {"NCHW": "kcyx", "NHWC": "kyxc"}
MLIR_OUTPUT_LAYOUTS = {"NCHW": "nkhw", "NHWC": "nhwk"}
MLIR_OPS = {"F": "conv2d", "W": "conv2d_bwd_weight", "B": "conv2d_bwd_data"}
MLIR_DATA_TYPE = {"FP16": "f16", "FP32": "f32", "INT8INT8INT32": "i8"}


# Convert a key,value pair from the tuning DB to a "--option value" parameter for rocmlir-gen
def cvtTuningDBToRocmlirgen(key, val):
    if key == "layout":
        parsedName = f"--in_layout {val.lower()} --out_layout {MLIR_OUTPUT_LAYOUTS[val]} --fil_layout {MLIR_FILTER_LAYOUTS[val]} "
    elif key in [
        "in_w",
        "in_h",
        "in_channels",
        "batchsize",
        "fil_h",
        "fil_w",
        "out_channels",
        "dilation_h",
        "dilation_w",
        "conv_stride_h",
        "conv_stride_w",
    ]:
        parsedName = f"--{key} {val}"
    elif key == "pad_w":
        parsedName = f"--padding_w {val}"
    elif key == "pad_h":
        parsedName = f"--padding_h {val}"
    elif key == "data_type":
        parsedName = f"-t {MLIR_DATA_TYPE[val]}"
    elif key == "direction":
        parsedName = f"--operation {MLIR_OPS[val]}"
    elif key == "params":
        parsedName = f"--perf_config={val}"
    else:
        return None

    return parsedName


# Parse an MIOpenConfig into a valid set of DB fields
def cvtMIOpenToTuningDB(
    config, overrideDataType=None, overrideDirection=None, overrideLayout=None
):
    argv = config.split(" ")
    MIOpenConfig = {}

    # determine dataType from argv[0]
    if overrideDataType:
        MIOpenConfig["data_type"] = overrideDataType
    elif argv[0] == "convfp16":
        MIOpenConfig["data_type"] = "FP16"
    elif argv[0] == "convint8":
        MIOpenConfig["data_type"] = "INT8INT8INT32"
    elif argv[0] == "conv":
        MIOpenConfig["data_type"] = "FP32"

    layout = None
    direction = None
    try:
        opts, _ = getopt.getopt(argv, "F:f:I:O:n:c:H:W:k:y:x:p:q:l:j:u:v:g:m:t:")
    except getopt.GetOptError:
        print("getopt error")
        sys.exit(1)

    for opt, arg in opts:
        if opt == "-F":
            if int(arg) == 1:
                direction = "F"
            elif int(arg) == 2:
                direction = "B"
            elif int(arg) == 4:
                direction = "W"
        elif opt == "-f":
            if layout is not None and layout != arg:
                raise ValueError("Mixed layouts")
            layout = arg
        elif opt == "-I":
            if layout is not None and layout != arg:
                raise ValueError("Mixed layouts")
            layout = arg
        elif opt == "-O":
            if layout is not None and layout != arg:
                raise ValueError("Mixed layouts")
            layout = arg
        elif opt == "-n":
            MIOpenConfig["batchsize"] = arg
        elif opt == "-c":
            MIOpenConfig["in_channels"] = arg
        elif opt == "-H":
            MIOpenConfig["in_h"] = arg
        elif opt == "-W":
            MIOpenConfig["in_w"] = arg
        elif opt == "-k":
            MIOpenConfig["out_channels"] = arg
        elif opt == "-y":
            MIOpenConfig["fil_h"] = arg
        elif opt == "-x":
            MIOpenConfig["fil_w"] = arg
        elif opt == "-u":
            MIOpenConfig["conv_stride_h"] = arg
        elif opt == "-v":
            MIOpenConfig["conv_stride_w"] = arg
        elif opt == "-p":
            MIOpenConfig["pad_h"] = arg
        elif opt == "-q":
            MIOpenConfig["pad_w"] = arg
        elif opt == "-l":
            MIOpenConfig["dilation_h"] = arg
        elif opt == "-j":
            MIOpenConfig["dilation_w"] = arg
        elif opt == "-g":
            MIOpenConfig["group_count"] = arg
        else:
            continue

    if overrideLayout:
        MIOpenConfig["layout"] = overrideLayout
    elif layout is not None:
        MIOpenConfig["layout"] = layout

    if overrideDirection:
        MIOpenConfig["direction"] = overrideDirection
    elif direction is not None:
        MIOpenConfig["direction"] = direction

    return MIOpenConfig
