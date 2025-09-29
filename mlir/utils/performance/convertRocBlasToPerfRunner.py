#!/usr/bin/env python3

import argparse
import sys
import subprocess


def stringify(config):
    return ' '.join(key + " " + val for key, val in config.items())


def convert_to_perfrunner(rocblas_ins):
    perfrunner_ins = {}

    # Default values (from rocblas-bench)
    perfrunner_ins["-transA"] = "true"
    perfrunner_ins["-transB"] = "true"
    perfrunner_ins["-g"] = "1"
    perfrunner_ins["-m"] = "128"
    perfrunner_ins["-n"] = "128"
    perfrunner_ins["-k"] = "128"

    # Convert the values to perfRunner values
    for ii in range(1, len(rocblas_ins), 2):
        if rocblas_ins[ii] == "-m":
            perfrunner_ins["-m"] = rocblas_ins[ii + 1]
        elif rocblas_ins[ii] == "-k":
            perfrunner_ins["-k"] = rocblas_ins[ii + 1]
        elif rocblas_ins[ii] == "-n":
            perfrunner_ins["-n"] = rocblas_ins[ii + 1]
        elif "_type" in rocblas_ins[ii]:
            t = rocblas_ins[ii + 1][0:3]
            if "-t" in perfrunner_ins and perfrunner_ins["-t"] != t:
                raise (ValueError("Mixed Layouts"))
            perfrunner_ins["-t"] = t
        elif rocblas_ins[ii] == "--batch_count":
            perfrunner_ins["-g"] = rocblas_ins[ii + 1]
        elif rocblas_ins[ii] == "--transposeA" and rocblas_ins[ii + 1] == "N":
            perfrunner_ins["-transA"] = "false"
        elif rocblas_ins[ii] == "--transposeB" and rocblas_ins[ii + 1] == "N":
            perfrunner_ins["-transB"] = "false"

    return stringify(perfrunner_ins)


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
    for line in fin:
        rocblas_inputs = line.split(' ')
        if line.startswith("#"):
            continue
        if line.isspace():
            continue
        configs.append(convert_to_perfrunner(rocblas_inputs))

    # Save the result into the output file
    fout = open(parsed_args.output_file, 'w')
    cmdline = subprocess.list2cmdline(sys.argv[0:])

    print("# This file has been generated with the following command:",
          file=fout)
    print(f"# {cmdline}\n", file=fout)

    for config in configs:
        print(config, file=fout)


if __name__ == "__main__":
    sys.exit(main())
