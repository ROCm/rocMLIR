#!/usr/bin/env python3
"""Script to sweep the parameters of the MIOpen driver for bugs

Note: This requires Python 3.7 or newer, use pyenv or the like to install it temporarily

Usage:
$ ninja mlir-miopen-driver mlir-rocm-runner ci-performance-scripts
$ cp ../mlir/utils/miopenParameterSweeps.py bin
$ stdbuf --output=L python3 ./bin/miopenParameterSweeps.py 2>&1 | stdbuf --output=L tee [output-file-of-choice]"""


import asyncio
import itertools
import re
import os
import sys
from typing import Sequence, Optional

import MIOpenDriver
from MIOpenDriver import ConvConfiguration

CORRECT_RESULT_RE = re.compile(r"data\s*=\s*\[1\]")

async def testConfig(config: ConvConfiguration) -> bool:
    """Runs the given configuration under mlir-miopen-driver without benchmarking.
    Returns whether the configuration ran successfully."""
    commandLineOptions = config.generateMlirDriverCommandLine()
    mlirMIOpenDriverCommand = '-pv_with_gpu -c ' + commandLineOptions
    compiler = await asyncio.create_subprocess_exec(os.path.join(MIOpenDriver.MLIR_BIN_DIR, MIOpenDriver.MLIR_MIOPEN_DRIVER),
        *mlirMIOpenDriverCommand.split(), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.DEVNULL)
    program, errors = await compiler.communicate()

    if compiler.returncode != 0:
        print(f"""Compiler did not complete succesfully for config {config!r}
Command line: {mlirMIOpenDriverCommand}
Output = {program.decode('utf-8')}
Errors = {errors.decode('utf-8')}
Return code = {compiler.returncode}""", file=sys.stderr)
        return False

    runner = await asyncio.create_subprocess_exec(os.path.join(MIOpenDriver.MLIR_BIN_DIR, MIOpenDriver.MLIR_ROCM_RUNNER), *MIOpenDriver.MLIR_ROCM_RUNNER_ARGS,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.PIPE)
    output, errors = await runner.communicate(input=program)
    runner.stdin.close()
    await runner.stdin.wait_closed()
    output = output.decode('utf-8')

    if runner.returncode != 0:
        print(f"""Runner execution failed for config {config!r}
Output = {output}
Errors = {errors.decode('utf-8')}
Return code = {runner.returncode}""", file=sys.stderr)
        return False
    if not CORRECT_RESULT_RE.search(output):
        print(f"""Convolution returned intorrect result for config {config!r}
Output = {output}
Errors = {errors.decode('utf-8')}""", file=sys.stderr)
        return False
    return True

def outputDim(inLen: int, filLen: int, padLen: int, strideLen: int, dilationLen: int) -> int:
    return (inLen + (2 * padLen) - ((filLen - 1) * dilationLen) - 1) // strideLen + 1

def shouldSucceed(config: ConvConfiguration):
    return outputDim(config.hi, config.y, config.paddingH, config.convStrideH, config.dilationH) > 0\
        and outputDim(config.wi, config.x, config.paddingW, config.convStrideW, config.dilationW) > 0\
        and not config.direction == "bwd"

def grouper(iterable, n):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

async def dropGoodConfig(config: ConvConfiguration) -> Optional[ConvConfiguration]:
    """Test the given `params`, returning the corresponding `config` on failure and `None` on success"""
    if not await testConfig(config):
        return config
    return None

async def sweepParameters() -> bool:
    PARAM_ITERATORS = [
        {"f16", "f32"},
        {"fwd", "bwd", "wrw"},
        {"NCHW", "NHWC"},
        # Dimensions (n, c, hi, wi, k, y, x)
        range(1, 9),
        range(1, 9),
        range(1, 9),
        range(1, 9),
        range(1, 9),
        range(1, 9),
        range(1, 9),
        # Stride
        range(1, 4),
        range(1, 4),
        # Padding
        range(0, 5),
        range(0, 5),
        # Dilation
        range(1, 4),
        range(1, 4),
        # Group
        {1},
        # xdlops
        {False, True}
    ]

    CONCURRENT_TESTS = 200
    REPORTING_INTERVAL = 10
    failingConfigs = []
    configs = (c for c in (ConvConfiguration(*p) for p in itertools.product(*PARAM_ITERATORS))
        if shouldSucceed(c))
    for i, configs in enumerate(grouper(configs, CONCURRENT_TESTS)):
        if i % REPORTING_INTERVAL == 0:
            print(f"{i * CONCURRENT_TESTS} tests complete")
        configsFuture = asyncio.gather(*(dropGoodConfig(c) for c in configs))
        try:
            configsResults = await configsFuture
        except Exception as e:
            configsFuture.cancel()
            raise e
        for result in configsResults:
            if result is not None:
                failingConfigs.append(result)
        del configsResults
        del configsFuture

    nFailed = len(failingConfigs)
    if nFailed > 0:
        print("Summary of failures:")
        for config in failingConfigs:
            print(repr(config))
    print(f"# of failures {nFailed}")
    return nFailed == 0

if __name__ == '__main__':
    asyncio.set_child_watcher(asyncio.FastChildWatcher())
    ret = asyncio.run(sweepParameters(), debug=True)
    sys.exit(int(not ret))
