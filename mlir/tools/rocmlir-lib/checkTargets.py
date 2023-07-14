# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os.path
import sys

# Ignore known targets not required for the librockCompiler library
ignored_targets = [
    'LLVMCFGuard',
    'LLVMFileCheck',
    'MLIRAMXToLLVMIRTranslation',
    'MLIRArmNeonToLLVMIRTranslation',
    'MLIRArmSMEToLLVMIRTranslation',
    'MLIRArmSVEToLLVMIRTranslation',
    'MLIRCAPIGPU',
    'MLIRDebug',
    'MLIREmitCDialect',
    'MLIRFromLLVMIRTranslationRegistration',
    'MLIRIRDL',
    'MLIRLLVMIRToLLVMTranslation',
    'MLIRLspServerLib',
    'MLIRLspServerSupportLib',
    'MLIRNVVMToLLVMIRTranslation',
    'MLIRObservers',
    'MLIROpenACCDialect',
    'MLIROpenACCToLLVMIRTranslation',
    'MLIROpenMPDialect',
    'MLIROpenMPToLLVMIRTranslation',
    'MLIROptLib',
    'MLIRPluginsLib',
    'MLIRSPIRVBinaryUtils',
    'MLIRSPIRVDeserialization',
    'MLIRSPIRVDialect',
    'MLIRSPIRVSerialization',
    'MLIRSPIRVTranslateRegistration',
    'MLIRTargetCpp',
    'MLIRTargetLLVMIRImport',
    'MLIRToLLVMIRTranslationRegistration',
    'MLIRX86VectorToLLVMIRTranslation',
    'llvm_gtest',
    'llvm_gtest_main'
]

def fopen(filename):
    try:
        file = open(filename)
    except FileNotFoundError:
        print(f'{scriptName}: file not found: {filename}')
        sys.exit(1)
    except OSError:
        print(f'{scriptName}: error opening file: {filename}')
        sys.exit(1)

    return file


if __name__ == '__main__':
    scriptName = os.path.basename(sys.argv[0])
    if len(sys.argv) != 5:
        print(f'{scriptName}: invalid number of parameters, required 4!')
        sys.exit(1)

    with fopen(sys.argv[4]) as f:
        libraries = [line.rstrip() for line in f]

    libraries = list(set(libraries) - set(ignored_targets))

    with fopen(sys.argv[3]) as f:
        targets = [line.rstrip() for line in f]

    targets.sort()

    missing_targets = list(set(libraries) - set(targets))
    removed_targets = list(set(targets) - set(libraries))

    if missing_targets:
        missing_targets.sort()
        print(f'** {scriptName}: error: `{sys.argv[2]}` of {sys.argv[1]} '
              f'is missing elements - ADD them or UPDATE ignored targets in this script:\n        ' +
              ' '.join(missing_targets))

    if removed_targets:
        removed_targets.sort()
        print(f'** {scriptName}: error: `{sys.argv[2]}` of {sys.argv[1]} '
              f'has invalid elements - REMOVE them:\n        ' + ' '.join(removed_targets))

    if missing_targets or removed_targets:
        sys.exit(1)
