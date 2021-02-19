#!/usr/bin/python3.7

import csv
import getopt
import os
import subprocess
import sys

# global variables.
mlirBuildDir = './bin'
mlirMIOpenDriver = 'mlir-miopen-driver'
mlirROCmRunner = 'mlir-rocm-runner'
rocprof = '/opt/rocm/bin/rocprof'
MIOpenDriver = '/opt/rocm/miopen/bin/MIOpenDriver'
benchmarkingResultFileName = 'results.stats.csv'
roundDigits = 2

# Test vectors.
globalTestVector = '''
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 128 -H 58 -W 58 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 30 -W 30 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 16 -W 16 -k 512 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 128 -H 58 -W 58 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 30 -W 30 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 16 -W 16 -k 512 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -f kyxc -I nhwc -O nhwk -n 256 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 128 -H 58 -W 58 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 256 -H 30 -W 30 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 512 -H 16 -W 16 -k 512 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 1 -n 256 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 128 -H 58 -W 58 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 256 -H 30 -W 30 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 512 -H 16 -W 16 -k 512 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 1 -n 256 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 128 -H 58 -W 58 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 30 -W 30 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 16 -W 16 -k 512 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 128 -H 58 -W 58 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 30 -W 30 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 16 -W 16 -k 512 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -f kyxc -I nhwc -O nhwk -n 256 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
# trigger errors in the next config.
#convfp16 -F 2 -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
# trigger errors in the next configs.
#convfp16 -F 2 -n 256 -c 128 -H 58 -W 58 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 256 -H 30 -W 30 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
# trigger errors in the next config.
#convfp16 -F 2 -n 256 -c 512 -H 16 -W 16 -k 512 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 2 -n 256 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 128 -H 58 -W 58 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 256 -H 30 -W 30 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 512 -H 16 -W 16 -k 512 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 2 -n 256 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 128 -H 58 -W 58 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 30 -W 30 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 16 -W 16 -k 512 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 128 -H 58 -W 58 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 30 -W 30 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 16 -W 16 -k 512 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -f kyxc -I nhwc -O nhwk -n 256 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 128 -H 58 -W 58 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 256 -H 30 -W 30 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 512 -H 16 -W 16 -k 512 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
convfp16 -F 4 -n 256 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 128 -H 58 -W 58 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 256 -H 30 -W 30 -k 256 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 512 -H 16 -W 16 -k 512 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 512 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
conv -F 4 -n 256 -c 3 -H 230 -W 230 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
'''

# utility functions.
def getNanoSeconds(fileName):
    with open(fileName, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter = ',')

        row = next(reader)
        result = row['AverageNs']
        csv_file.close()
        return result

# convolution configurations.
class ConvConfiguration:
    def computeTFlops(self, ns):
        if ns == 0:
            return 0.0
        return round((2.0 * self.n * self.c * self.k * self.ho * self.wo * self.y * self.x) / (float(ns) * 1e-9) / 1e12, roundDigits)

    @classmethod
    def generateCSVHeader(cls):
        result = ''
        # print operation.
        result = result + 'Direction' + ','
        # print data type.
        result = result + 'DataType' + ','
        # print XDLOPS
        result = result + 'XDLOPS' + ','
        # print filter layout.
        result = result + 'FilterLayout' + ','
        # print input layout.
        result = result + 'InputLayout' + ','
        # print output layout.
        result = result + 'OutputLayout' + ','
        # print N
        result = result + 'N' + ','
        # print C
        result = result + 'C' + ','
        # print Hi
        result = result + 'H' + ','
        # print Wi
        result = result + 'W' + ','
        # print K
        result = result + 'K' + ','
        # print Y
        result = result + 'Y' + ','
        # print X
        result = result + 'X' + ','
        # print dilation height
        result = result + 'DilationH' + ','
        # print dilation width
        result = result + 'DilationW' + ','
        # print stride height
        result = result + 'StrideH' + ','
        # print stride width
        result = result + 'StrideW' + ','
        # print padding width
        result = result + 'PaddingH' + ','
        # print padding height
        result = result + 'PaddingW' + ','

        # benchmarking fields

        # print TFlops
        result = result + 'TFlops'
        return result

    def generateCSVContent(self, nanoSeconds):
        result = ''
        # print operation.
        result = result + self.direction + ','
        # print data type.
        result = result + self.dataType + ','
        # print XDLOPS
        result = result + str(self.xdlops) + ','
        # print filter layout.
        result = result + self.filterLayout + ','
        # print input layout.
        result = result + self.inputLayout + ','
        # print output layout.
        result = result + self.outputLayout + ','
        # print N
        result = result + str(self.n) + ','
        # print C
        result = result + str(self.c) + ','
        # print Hi
        result = result + str(self.hi) + ','
        # print Wi
        result = result + str(self.wi) + ','
        # print K
        result = result + str(self.k) + ','
        # print Y
        result = result + str(self.y) + ','
        # print X
        result = result + str(self.x) + ','
        # print dilation height
        result = result + str(self.dilationH) + ','
        # print dilation width
        result = result + str(self.dilationW) + ','
        # print stride height
        result = result + str(self.convStrideH) + ','
        # print stride width
        result = result + str(self.convStrideW) + ','
        # print padding width
        result = result + str(self.paddingH) + ','
        # print padding height
        result = result + str(self.paddingW) + ','

        # benchmarking fields

        # print TFlops
        result = result + str(self.computeTFlops(nanoSeconds))
        return result

    def generateMlirDriverCommandLine(self):
        result = ''
        # set operation.
        if self.direction == 'fwd':
            result = result + '--operation conv2d'
        elif self.direction == 'bwd':
            result = result + '--operation conv2d_bwd_data'
        elif self.direction == 'wrw':
            result = result + '--operation conv2d_bwd_weight'
        # set data type.
        result = result + ' -t ' + self.dataType
        # set XDLOPS
        if self.xdlops == True:
            result = result + ' -x2'
        # set filter layout.
        result = result + ' --fil_layout ' + self.filterLayout
        # set input layout.
        result = result + ' --in_layout ' + self.inputLayout
        # set output layout.
        result = result + ' --out_layout ' + self.outputLayout
        # set N
        result = result + ' --batchsize ' + str(self.n)
        # set C
        result = result + ' --in_channels ' + str(self.c)
        # set Hi
        result = result + ' --in_h ' + str(self.hi)
        # set Wi
        result = result + ' --in_w ' + str(self.wi)
        # set K
        result = result + ' --out_channels ' + str(self.k)
        # set Y
        result = result + ' --fil_w ' + str(self.y)
        # set X
        result = result + ' --fil_h ' + str(self.x)
        # set dilation height
        result = result + ' --dilation_h ' + str(self.dilationH)
        # set dilation width
        result = result + ' --dilation_w ' + str(self.dilationW)
        # set stride height
        result = result + ' --conv_stride_h ' + str(self.convStrideH)
        # set stride width
        result = result + ' --conv_stride_w ' + str(self.convStrideW)
        # set padding width
        result = result + ' --padding_h ' + str(self.paddingH)
        # set padding height
        result = result + ' --padding_w ' + str(self.paddingW)
        return result

    def __init__(self, argv):
        # setup default values.
        self.dataType = 'f32'
        self.xdlops = True
        self.direction = 'fwd' # fwd, bwd, wrw
        self.filterLayout = 'kcyx'
        self.inputLayout = 'nchw'
        self.outputLayout = 'nkhw'
        self.n = 128
        self.c = 1024
        self.hi = 14
        self.wi = 14
        self.k = 1024
        self.y = 1
        self.x = 1
        self.convStrideH = 1
        self.convStrideW = 1
        self.paddingH = 0
        self.paddingW = 0
        self.dilationH = 1
        self.dilationW = 1
        self.group = 1

        # determine dataType from argv[1]
        if argv[0] == 'conv':
            self.dataType = 'f32'
        elif argv[0] == 'convfp16':
            self.dataType = 'f16'
        elif argv[0] == 'convbfp16':
            self.dataType = 'bf16'

        try:
            # TBD:
            # implement -m ?
            # implement -t ?
            opts, args = getopt.getopt(argv[1:], "hX:F:f:I:O:n:c:H:W:k:y:x:p:q:l:j:u:v:g:m:t:")
        except getopt.GetOptError:
            print('getopt error')
            sys.exit(-1)

        for opt, arg in opts:
            if opt == '-X':
                # -X
                self.xdlops = (int(arg) != 0)
            elif opt == '-F':
                # -F
                # 1 fwd only
                # 2 bwd only
                # 4 wrw only
                # TBD:
                # 0 fwd+bwd+wrw
                # 3 fwd+bwd
                # 5 fwd+wrw
                # 6 bwd+wrw
                if int(arg) == 1:
                    self.direction = 'fwd'
                elif int(arg) == 2:
                    self.direction = 'bwd'
                elif int(arg) == 4:
                    self.direction = 'wrw'
            elif opt == '-f':
                # -f
                self.filterLayout = arg
            elif opt == '-I':
                # -I
                self.inputLayout = arg
            elif opt == '-O':
                # -O
                self.outputLayout = arg
            elif opt == "-n":
                # -n
                self.n = int(arg)
            elif opt == '-c':
                # -c
                self.c = int(arg)
            elif opt == '-H':
                # -H
                self.hi = int(arg)
            elif opt == '-W':
                # -W
                self.wi = int(arg)
            elif opt == '-k':
                # -k
                self.k = int(arg)
            elif opt == '-y':
                # -y
                self.y = int(arg)
            elif opt == '-x':
                # -x
                self.x = int(arg)
            elif opt == '-u':
                # -u
                self.convStrideH = int(arg)
            elif opt == '-v':
                # -v
                self.convStrideW = int(arg)
            elif opt == '-p':
                # -p
                self.paddingH = int(arg)
            elif opt == '-q':
                # -q
                self.paddingW = int(arg)
            elif opt == '-l':
                # -l
                self.dilationH = int(arg)
            elif opt == '-j':
                # -j
                self.dilationW = int(arg)
            elif opt == '-g':
                # -g
                self.group = int(arg)
            else:
                continue

        # Ho and Wo are computed.
        self.ho = (self.hi + self.paddingH * 2 - self.y) / self.convStrideH + 1
        self.wo = (self.wi + self.paddingW * 2 - self.x) / self.convStrideW + 1

def runConfigWithMLIR(commandLine):
    config = ConvConfiguration(commandLine)
    commandLineOptions = config.generateMlirDriverCommandLine()
    mlirMIOpenDriverCommand = mlirBuildDir + os.sep + mlirMIOpenDriver + ' -ph -c ' + commandLineOptions
    profilerCommand = rocprof + ' --hip-trace ' + mlirBuildDir + os.sep + mlirROCmRunner + ' --shared-libs=./lib/librocm-runtime-wrappers.so,./lib/libmlir_runner_utils.so --entry-point-result=void'
    
    # invoke mlir-miopen-driver.
    p1 = subprocess.Popen(mlirMIOpenDriverCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    # pipe to rocprof + mlir-rocm-runner.
    p2 = subprocess.Popen(profilerCommand.split(), stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p1.stdout.close() # Allow p1 to receive a SIGPIPE if p2 exits.
    # get output.
    p2.communicate()

def runConfigWithMIOpenDriver(commandLine):
    MIOpenDriverCommand = MIOpenDriver + ' ' + ' '.join(commandLine) + ' -V 0'
    profilerCommand = rocprof + ' --hip-trace ' + MIOpenDriverCommand

    # invoke rocprof + MIOpenDriver.
    p1 = subprocess.Popen(profilerCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    # get output.
    p1.communicate()
    
def output(string, outputFile):
    print(string)
    if outputFile != None:
        outputFile.write(string)
        outputFile.write('\n')

# Benchmarking function.
def benchmarkMLIR(commandLine, outputFile):
    config = ConvConfiguration(commandLine)
    runConfigWithMLIR(commandLine)
    # get nanoseconds from rocprof output.
    nanoSeconds = getNanoSeconds(benchmarkingResultFileName)
    output(config.generateCSVContent(nanoSeconds), outputFile)

def benchmarkMIOpen(commandLine, outputFile):
    config = ConvConfiguration(commandLine)
    if config.inputLayout == 'nchw':
        runConfigWithMIOpenDriver(commandLine)
        # get nanoseconds from rocprof output.
        nanoSeconds = getNanoSeconds(benchmarkingResultFileName)
    else:
        # skip the test for non-supported layouts.
        # MIOpenDriver currently only support NCHW.
        nanoSeconds = 0
    output(config.generateCSVContent(nanoSeconds), outputFile)

# Main function.
if __name__ == '__main__':
    if len(sys.argv) > 1:
        outputFile = None
        if sys.argv[1] == '-o':
            outputFile = open(sys.argv[2], 'w')
            sys.argv.pop(1)
            sys.argv.pop(1)

        output(ConvConfiguration.generateCSVHeader(), outputFile)
        if sys.argv[1] == '-b':
            # CSV batch benchmarking mode with MLIR.
            for testVector in globalTestVector.split(sep='\n'):
                if len(testVector) > 0 and testVector[0] != '#':
                    benchmarkMLIR(testVector.split(sep=' '), outputFile)
        elif sys.argv[1] == '-bmiopen':
            # CSV batch benchmarking mode with MIOpenDriver.
            for testVector in globalTestVector.split(sep='\n'):
                if len(testVector) > 0 and testVector[0] != '#':
                    benchmarkMIOpen(testVector.split(sep=' '), outputFile)
        elif sys.argv[1] == '-miopen':
            benchmarkMIOpen(sys.argv[2:], outputFile)
        else:
            benchmarkMLIR(sys.argv[1:], outputFile)

        if outputFile != None:
            outputFile.close()
