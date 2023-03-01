import argparse
import getopt
import sqlite3
import sys

# Useful tabled conversions
MLIR_FILTER_LAYOUTS = {"NCHW": "kcyx", "NHWC": "kyxc"}
MLIR_OUTPUT_LAYOUTS = {"NCHW": "nkhw", "NHWC": "nhwk"}
MLIR_OPS = {"F": "conv2d", "W": "conv2d_bwd_weight", "B": "conv2d_bwd_data"}
MLIR_DATA_TYPE = {"FP16": "f16", "FP32": "f32", "INT8INT8INT32": "i8"}


# Convert a key,value pair from the tuning DB to a "--option value" parameter for rocmlir-gen
def convertToRocmgenParam(key, val):
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
def parseMIOpenConfig(
    config, arch, overrideDataType=None, overrideDirection=None, overrideLayout=None
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


# Select all the tables from the DB
def selectAllTables(cursor):
    sql_query_get_all_tables = """SELECT name FROM sqlite_master WHERE type='table';"""
    cursor.execute(sql_query_get_all_tables)
    tables = cursor.fetchall()
    return tables


# Select the perfConfig matching the convolution config
def selectPerfConfig(cursor, config):
    query = []
    for key, val in config.items():
        query.append(f'({key} = "{val}")')

    whereClause = ""
    if len(query):
        whereClause = f" WHERE ({' AND '.join(query)})"
    sql_query_select_perf_conifg = f"SELECT * FROM perf_db INNER JOIN config ON perf_db.config = config.id {whereClause} "
    cursor.execute(sql_query_select_perf_conifg)
    pc = cursor.fetchall()
    keys = [description[0] for description in cursor.description]
    return keys, pc


# Given a configuration generate the rocmlir-gen command
def getRocMlirCommand(cursor, config, arch):
    keys, perfConfigs = selectPerfConfig(cursor, config)
    rocmlirCommands = []
    for perfConfig in perfConfigs:
        rocmlirCommand = [f"rocmlir-gen -ph --arch {arch}"]
        for key, val in zip(keys, perfConfig):
            opt = convertToRocmgenParam(key, val)
            if opt:
                rocmlirCommand.append(opt)
        rocmlirCommands.append(rocmlirCommand)
    return rocmlirCommands


# Try to determine the arch
def getArch(parsed_args):
    arch = None
    if "gfx90a" in parsed_args.tuning_db:
        arch = "gfx90a"
    elif "gfx908" in parsed_args.tuning_db:
        arch = "gfx908"
    elif "gfx1030" in parsed_args.tuning_db:
        arch = "gfx1030"

    if parsed_args.arch and arch is None:
        arch = parsed_args.arch
    elif parsed_args.arch and arch != parsed_args.arch:
        raise ValueError(
            f"Architecture mismatch: --arch={parsed_args.arch} but got {arch} from the path"
        )

    if arch is None:
        raise ValueError("Cannot determine the arch the database was built for.")

    return arch


def main(args=None):
    parser = argparse.ArgumentParser(
        prog="MIOpen performance DB parser",
        description="This scripts parse the MIOpen performance DB  and generates rocmlir-gen calls",
    )

    parser.add_argument(
        "--miopen_config",
        type=str,
        default="",
        help="The specific config to test, if you want to test one",
    )

    parser.add_argument(
        "--miopen_configs_file",
        type=str,
        default=None,
        help="File of configurations to parse",
    )

    parser.add_argument("--tuning-db", type=str, help="Path to the tuning db")
    parser.add_argument("--direction", type=str, help="Conv direction (F|W|B)")
    parser.add_argument("--layout", type=str, help="Conv layout (NHWC|NCHW)")
    parser.add_argument(
        "--data_type", type=str, help="Conv data type (FP32|FP16|INT8INT8INT32)"
    )

    parser.add_argument(
        "--print-tables",
        action="store_true",
        help="Print the tables in the tuning database",
    )
    parser.add_argument(
        "--arch", type=str, help="Architecture we tuned for", default=None
    )
    parsed_args = parser.parse_args(args)

    con = sqlite3.connect(parsed_args.tuning_db)
    cursor = con.cursor()

    # Print the tables of the perf DB
    if parsed_args.print_tables:
        print("Table list:")
        for idx, t in enumerate(selectAllTables(cursor)):
            print(f"{idx}] {t[0]}")
        return
    arch = getArch(parsed_args)

    parsedConfigs = []
    rocmlirCommands = []
    if parsed_args.miopen_configs_file:
        with open(parsed_args.miopen_configs_file) as configs:
            for config in configs:
                parsedConfigs.append(
                    parseMIOpenConfig(
                        config,
                        arch,
                        parsed_args.data_type,
                        parsed_args.direction,
                        parsed_args.layout,
                    )
                )
    else:
        parsedConfigs.append(
            parseMIOpenConfig(
                parsed_args.miopen_config,
                arch,
                parsed_args.data_type,
                parsed_args.direction,
                parsed_args.layout,
            )
        )

    for parsedConfig in parsedConfigs:
        rocmlirCommands.extend(getRocMlirCommand(cursor, parsedConfig, arch))

    for cmd in rocmlirCommands:
        print(" ".join(cmd))


if __name__ == "__main__":
    sys.exit(main())
