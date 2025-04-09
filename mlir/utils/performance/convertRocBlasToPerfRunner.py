#!/usr/bin/env python3

import argparse
import sys
import subprocess


def stringify(config):
    return ' '.join(key + " " + val for key, val in config.items())


def convertToPerfRunner(rocblasIns):
    perfRunnerIns = {}

    # Default values (from rocblas-bench)
    perfRunnerIns["-transA"] = "true"
    perfRunnerIns["-transB"] = "true"
    perfRunnerIns["-g"] = "1"
    perfRunnerIns["-m"] = "128"
    perfRunnerIns["-n"] = "128"
    perfRunnerIns["-k"] = "128"

    # Convert the values to perfRunner values
    for ii in range(1, len(rocblasIns), 2):
        if rocblasIns[ii] == "-m":
            perfRunnerIns["-m"] = rocblasIns[ii + 1]
        elif rocblasIns[ii] == "-k":
            perfRunnerIns["-k"] = rocblasIns[ii + 1]
        elif rocblasIns[ii] == "-n":
            perfRunnerIns["-n"] = rocblasIns[ii + 1]
        elif "_type" in rocblasIns[ii]:
            t = rocblasIns[ii + 1][0:3]
            if "-t" in perfRunnerIns and perfRunnerIns["-t"] != t:
                raise (ValueError("Mixed Layouts"))
            perfRunnerIns["-t"] = t
        elif rocblasIns[ii] == "--batch_count":
            perfRunnerIns["-g"] = rocblasIns[ii + 1]
        elif rocblasIns[ii] == "--transposeA" and rocblasIns[ii + 1] == "N":
            perfRunnerIns["-transA"] = "false"
        elif rocblasIns[ii] == "--transposeB" and rocblasIns[ii + 1] == "N":
            perfRunnerIns["-transB"] = "false"

    return stringify(perfRunnerIns)


def main():
    parser = argparse.ArgumentParser(
        prog="rocBLAS converter",
        description="converts rocblas-bench parameter to perfRunner parameters",
        allow_abbrev=False,
    )

    parser.add_argument("-c",
                        "--config-file",
                        type=str,
                        help="Config file to convert")

    parser.add_argument("-o", "--output-file", type=str, help="New configfile")

    parsed_args = parser.parse_args()

    fin = open(parsed_args.config_file, 'r')
    configs = []

    # Convert the input file line by line
    for l in fin:
        rocblasInputs = l.split(' ')
        if l.startswith("#"):
            continue
        if l.isspace():
            continue
        configs.append(convertToPerfRunner(rocblasInputs))

    # Save the result into the output file
    fout = open(parsed_args.output_file, 'w')
    cmdLine = subprocess.list2cmdline(sys.argv[0:])

    print("# This file has been generated with the following command:",
          file=fout)
    print(f"# {cmdLine}\n", file=fout)

    for config in configs:
        print(config, file=fout)


if __name__ == "__main__":
    sys.exit(main())
