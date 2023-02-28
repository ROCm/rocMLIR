import argparse
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
    else:
        return None

    return parsedName


def selectAllTables(cursor):
    sql_query_get_all_tables = """SELECT name FROM sqlite_master WHERE type='table';"""
    cursor.execute(sql_query_get_all_tables)
    tables = cursor.fetchall()
    return tables


def selectAllConfigs(cursor):
    sql_query_print_config = """SELECT * FROM config"""
    cursor.execute(sql_query_print_config)
    configs = cursor.fetchall()
    keys = [description[0] for description in cursor.description]
    return configs, keys


def selectPerfConfig(cursor, perf_config_id):
    sql_query_select_perf_conifg = (
        f"""SELECT * FROM perf_db WHERE config={perf_config_id}"""
    )
    cursor.execute(sql_query_select_perf_conifg)
    pc = cursor.fetchall()
    return pc[0]


def getRocMlirCommand(cursor, config, keys, arch):
    configId = config[0]
    perfConfig = selectPerfConfig(cursor, configId)
    rocmlirCommand = [f"rocmlir-gen -ph --arch {arch}"]
    for key, val in zip(keys, config):
        opt = convertToRocmgenParam(key, val)
        if opt:
            rocmlirCommand.append(opt)
    rocmlirCommand.append(f"--perf_config {perfConfig[3]}")


def main(args=None):
    parser = argparse.ArgumentParser(
        prog="MIOpen performance DB parser",
        description="This scripts parse the MIOpen performance DB  and generates rocmlir-gen calls",
    )

    parser.add_argument("--tuning-db", type=str, help="Path to the tuning db")
    parser.add_argument(
        "--print-tables",
        action="store_true",
        help="Print the tables in the tuning database",
    )
    parser.add_argument("--arch", type=str, help="Architecture we tuned for")
    parsed_args = parser.parse_args(args)

    con = sqlite3.connect(parsed_args.tuning_db)
    cursor = con.cursor()

    # Print the tables of the perf DB
    if parsed_args.print_tables:
        print("Table list:")
        for idx, t in enumerate(selectAllTables(cursor)):
            print(f"{idx}] {t[0]}")
        return

    # Get all the configurations
    configs, keys = selectAllConfigs(cursor)

    # Convert all the configurations to rocmlir-gen commands
    for config in configs:
        rocmlirCommand = getRocMlirCommand(config, keys, parsed_args.arch)
        print(" ".join(rocmlirCommand))


if __name__ == "__main__":
    sys.exit(main())
