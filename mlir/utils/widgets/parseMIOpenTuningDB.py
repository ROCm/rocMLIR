#!/usr/bin/env python3

import argparse
import sqlite3
import sys
import cvtUtils


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
            opt = cvtUtils.cvtTuningDBToRocmlirgen(key, val)
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
                    cvtUtils.cvtMIOpenToTuningDB(
                        config,
                        parsed_args.data_type,
                        parsed_args.direction,
                        parsed_args.layout,
                    )
                )
    else:
        parsedConfigs.append(
            cvtUtils.cvtMIOpenToTuningDB(
                parsed_args.miopen_config,
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
