# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'RocMLIR'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.td', '.mlir', '.toy', '.ll', '.tc', '.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlir_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))
config.substitutions.append(('%prefix_mlir', config.prefix_mlir))
config.substitutions.append(("%mlir_src_root", config.mlir_src_root))
config.substitutions.append(('%random_data', config.random_data))
config.substitutions.append(('%rocmlir_gen_flags', config.rocmlir_gen_flags))
config.substitutions.append(('%arch', config.arch))
config.substitutions.append(('%pv', config.populate_validation))

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

##############
# FIXME: adding a path to the environment isn't appearing to work as
#  expected, so below is a tmp workaround that inlines
#  use_default_substitutions() and subs in the path for FileCheck.
#llvm_config.with_environment('PATH', config.lit_tools_dir, append_path=True)
#llvm_config.use_default_substitutions()
##############
config.filecheck_executable = os.path.join(config.lit_tools_dir, 'FileCheck')
config.not_executable = os.path.join(config.lit_tools_dir, 'not')
config.count_executable = os.path.join(config.lit_tools_dir, 'count')
tool_patterns = [
    ToolSubst('FileCheck', config.filecheck_executable, unresolved='fatal'),
    ToolSubst('not', config.not_executable, unresolved='fatal'),
    ToolSubst('count', config.count_executable, unresolved='fatal'),]
config.substitutions.append(('%python', '"%s"' % (sys.executable)))
llvm_config.add_tool_substitutions(
   tool_patterns, [config.llvm_tools_dir])
##############

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt',
                   'lit.cfg.py', 'lit.site.cfg.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlir_obj_root, 'mlir', 'test')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.mlir_rock_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.lit_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [config.mlir_rock_tools_dir, config.mlir_tools_dir, config.llvm_tools_dir]
tools = [
    'rocmlir-opt',
    'rocmlir-translate'
]

# The following tools are optional
tools.extend([
    ToolSubst('%PYTHON', config.python_executable, unresolved='ignore'),
    ToolSubst('%linalg_test_lib_dir', config.linalg_test_lib_dir, unresolved='ignore'),
    ToolSubst('%mlir_runner_utils_dir', config.mlir_runner_utils_dir, unresolved='ignore'),
    ToolSubst('%conv_validation_wrapper_library_dir', config.conv_validation_wrapper_library_dir, unresolved='fatal'),
])

llvm_config.add_tool_substitutions(tools, tool_dirs)


# FileCheck -enable-var-scope is enabled by default in MLIR test
# This option avoids to accidentally reuse variable across -LABEL match,
# it can be explicitly opted-in by prefixing the variable name with $
config.environment['FILECHECK_OPTS'] = "-enable-var-scope --allow-unused-prefixes=false"


# LLVM can be configured with an empty default triple
# by passing ` -DLLVM_DEFAULT_TARGET_TRIPLE="" `.
# This is how LLVM filters tests that require the host target
# to be available for JIT tests.
if config.target_triple:
    config.available_features.add('default_triple')

# Add the python path for both the source and binary tree.
# Note that presently, the python sources come from the source tree and the
# binaries come from the build tree. This should be unified to the build tree
# by copying/linking sources to build.
if config.enable_bindings_python:
    llvm_config.with_environment('PYTHONPATH', [
        # TODO: Don't reference the llvm_obj_root here: the invariant is that
        # the python/ must be at the same level of the lib directory
        # where libMLIR.so is installed. This is presently not optimal from a
        # project separation perspective and a discussion on how to better
        # segment MLIR libraries needs to happen. See also
        # lib/Bindings/Python/CMakeLists.txt for where this is set up.
        os.path.join(config.llvm_obj_root, 'python'),
    ], append_path=True)
