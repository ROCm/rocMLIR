#!/usr/bin/env python3
""" A script to perform static tests for the mlir project.

This script runs clang-format and clang-tidy on the changes before a user 
merges them to the master branch.

The code was extracted from https://github.com/google/llvm-premerge-checks.

Example usage:
~/rocMLIR#  ln -s build/compile_commands.json compile_commands.json
~/rocMLIR#  python3 ./mlir/utils/jenkins/static-checks/premerge-checks.py
"""


# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys
import re
import subprocess
from typing import Tuple, Optional
import pathspec
import unidiff
import argparse
import multiprocessing
import git


def get_diff(base_commit) -> Tuple[bool, str]:
  diff_run = subprocess.run(
    f'/opt/rocm/llvm/bin/git-clang-format --binary /opt/rocm/llvm/bin/clang-format --diff {base_commit}', 
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
  )
  diff = diff_run.stdout.decode()
  print(diff)
  return diff

def check_external_file(filename: str) -> bool:
  regex = f'^external/'
  return re.search(regex, filename)

def run_clang_format(base_commit, ignore_config, ignore_external_files: bool = False):
  """Apply clang-format and return if no issues were found.
  Extracted from https://github.com/google/llvm-premerge-checks/blob/master/scripts/clang_format_report.py"""

  patch = get_diff(base_commit)
  patches = unidiff.PatchSet(patch)
  ignore_lines = []

  if ignore_config is not None and os.path.exists(ignore_config):
    ignore_lines = open(ignore_config, 'r').readlines()
  ignore = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, ignore_lines)
  patched_file: unidiff.PatchedFile
  success = True
  for patched_file in patches:
    # drop diff prefix
    patched_file_src = patched_file.source_file[2:]
    patched_file_tgt = patched_file.target_file[2:]
    if ignore.match_file(patched_file_src) or ignore.match_file(patched_file_tgt):
      continue
    if ignore_external_files:
      if check_external_file(patched_file_src) or check_external_file(patched_file_tgt):
        continue
    hunk: unidiff.Hunk
    for hunk in patched_file:
      success = False

  if not success:
    print(f'Please format your changes with clang-format by running `git-clang-format {base_commit}` or applying patch.')
    return False
  return True

def remove_ignored(diff_lines, ignore_patterns_lines):
  ignore = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, ignore_patterns_lines)
  good = True
  result = []
  for line in diff_lines:
    match = re.search(r'^diff --git (.*) (.*)$', line)
    if match:
      good = not (ignore.match_file(match.group(1)) and ignore.match_file(match.group(2)))
    if good:
      result.append(line)
  return result


def check_third_party_file(filename: str) -> bool:
  # We assume third party files to have absolute paths
  if filename[0] != '/':
    return False
  repo = git.Repo('.', search_parent_directories=True)
  regex = f'^{repo.working_tree_dir}'
  # Any file that does not belong to the repo, will be considered third party.
  return not re.search(regex, filename)


def run_clang_tidy(base_commit, ignore_config, ignore_external_files: bool = False):
  """Apply clang-tidy and return if no issues were found.
  Extracted from https://github.com/google/llvm-premerge-checks/blob/master/scripts/clang_tidy_report.py"""

  r = subprocess.run(f'git diff -U0 --no-prefix {base_commit}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  diff = r.stdout.decode("utf-8", "ignore")
  if ignore_config is not None and os.path.exists(ignore_config):
    ignore = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern,
                                          open(ignore_config, 'r').readlines())
    diff = remove_ignored(diff.splitlines(keepends=True), open(ignore_config, 'r'))
  else:
    ignore = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, [])
  cpu_count = multiprocessing.cpu_count()
  p = subprocess.Popen(['./external/llvm-project/clang-tools-extra/clang-tidy/tool/clang-tidy-diff.py', '-p0', '-quiet', '-j', str(cpu_count), '-extra-arg=-std=c++17'],
                       stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
  a = ''.join(diff)
  out = p.communicate(input=a.encode())[0].decode()
  # Typical finding looks like:
  # [cwd/]clang/include/clang/AST/DeclCXX.h:3058:20: error: ... [clang-diagnostic-error]
  pattern = '^([^:]*):(\\d+):(\\d+): (.*): (.*)'
  errors_count = 0
  warn_count = 0
  for line in out.splitlines(keepends=False):
    line = line.strip()
    line = line.replace(os.getcwd() + os.sep, '')

    if len(line) == 0 or line == 'No relevant changes found.':
      continue
    match = re.search(pattern, line)
    if match:
      file_name = match.group(1)
      line_pos = match.group(2)
      char_pos = match.group(3)
      severity = match.group(4)
      if severity in ['warning', 'error']:
        if ignore.match_file(file_name):
          print('{} is ignored by pattern and no comment will be added'.format(file_name))
          continue
        if check_third_party_file(file_name):
          print(f'{file_name} is ignored as its a third-party file and no comment will be added')
          continue
        if ignore_external_files:
          if check_external_file(file_name):
            print(f'{file_name} is ignored as its a external (upstream) file and no comment will be added')
            continue
        if severity == 'warning':
          warn_count += 1
        if severity == 'error':
          print('clang-tidy found error:',line)
          errors_count += 1

  if errors_count + warn_count != 0:
    print('clang-tidy found {} errors and {} warnings.'.format(errors_count, warn_count))

  if errors_count != 0:
    return False

  return True


if __name__ == '__main__':
  args = sys.argv[1:]
  parser = argparse.ArgumentParser(
        prog="rocMLIR premerge checker",
        description="A helper script to invoke linters",
        allow_abbrev=False,
  )
  parser.add_argument(
        "-b", "--base-commit",
        type=str,
        default='origin/develop',
        help="The base commit to lint againts"
  )
  parser.add_argument('--ignore-external', action='store_true', default=False)
  parsed_args = parser.parse_args(args)
  print(f"Running linters against base commit : {parsed_args.base_commit}")
  if not (
    run_clang_format(parsed_args.base_commit, './mlir/utils/jenkins/static-checks/clang-format.ignore', parsed_args.ignore_external) and
    run_clang_tidy(parsed_args.base_commit, './mlir/utils/jenkins/static-checks/clang-tidy.ignore', parsed_args.ignore_external)
  ):
    exit(1)
