from .directories import Directories
from .model import Group, build_groups
from .report import make_report_page, process_migraphx_output
from collections import OrderedDict
from datetime import datetime
import argparse
import subprocess
import shlex
import yaml
import os
import sys
import shutil
import re


def check_model(models):
  try:
    for model in models:
      if not os.path.exists(model['path']):
        raise RuntimeError(f'could not find: `{model["path"]}`')
  except KeyError as err:
    print(f'yaml config error: {err}')
    raise err


def collect_tuning_config(group, config, dirs):
  os.chdir(dirs.migraphx)

  for model in group.models: 
    tunung_config_file = f'{model.name}{model.type}.cfg'
    env = os.environ.copy()
    env['MIGRAPHX_ENABLE_MLIR'] = '1'
    env['MIGRAPHX_MLIR_TUNING_CFG'] = tunung_config_file

    migraphx_exe = os.path.join(dirs.migraphx, 'bin', 'migraphx-driver')
    args = [migraphx_exe, 'compile']
    if model.type:
      args.append(model.type)
    args.extend(['--onnx', model.path])
    if not model.is_static():
      args.append(model.params)

    cmd = ' '.join(args)
    print(f'executing: {cmd}')
    try:
      result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True, env=env)
    except subprocess.CalledProcessError as err:
      print(f'failed during the execution:')
      print(result.stderr)
      print(err)

    gemm_file = f'{tunung_config_file}.gemm'
    if os.path.exists(gemm_file):
      dest = os.path.join(dirs.tuning_config_dir, gemm_file)
      shutil.move(gemm_file, dest)

    conv_file = f'{tunung_config_file}.conv'
    if os.path.exists(conv_file):
      dest = os.path.join(dirs.tuning_config_dir, conv_file)
      shutil.move(conv_file, dest)

  print(f'tuning configs are saved to: {dirs.tuning_config_dir}')
  os.chdir(dirs.current_workdir)


def remove_duplicates_from_files(files):
  context = []
  for filepath in files:
    with open(filepath, 'r') as file:
      context.extend(file.readlines())

  return list(set(context))


def join_tuning_config(config, dirs):
  print(f'processing all tuning configs from: {dirs.tuning_config_dir}')
  conv_files = []; gemm_files = []
  for (dirpath, _, filenames) in os.walk(dirs.tuning_config_dir):
    for filename in filenames:
      db_file = os.path.join(dirpath, filename)
      suffix = db_file[-4:]
      if suffix == 'conv':
        conv_files.append(db_file)
      if suffix == 'gemm':
        gemm_files.append(db_file)

  conv_file, gemm_file = dirs.get_tuning_config_files(config)
  conv_configs = remove_duplicates_from_files(conv_files)
  with open(conv_file, 'w') as file:
    for line in conv_configs:
      file.write(line)

  gemm_configs = remove_duplicates_from_files(gemm_files)
  with open(gemm_file, 'w') as file:
    for line in gemm_configs:
      file.write(line)

  print(f'tuning configs are written to:')
  print(f'\t{conv_file}')
  print(f'\t{gemm_file}')


def run_tunner(config, dirs, verbose):
  os.chdir(dirs.rocmlir)
  conv_file, gemm_file = dirs.get_tuning_config_files(config)
  tuning_db = dirs.get_tuning_db_path(config)

  if os.path.exists(tuning_db):
    os.remove(tuning_db)

  def run_process(cmd):
    start_time = datetime.now()
    last_time = start_time
    process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE)
    print('| Time since start | Time in between | Tuned configs |')
    print('| ---------------- | --------------- | ------------- |')
    entire_output = []
    while process.poll() is None:
      output = process.stdout.readline().decode("utf-8")
      entire_output.append(output)
      output = output.strip()
      if output:
        header = 'Tuned :'
        if output.startswith(header):
          current_time = datetime.now()
          time_since_beginning = current_time - start_time
          time_between_tunes = current_time - last_time
          print(f'| {time_since_beginning} | {time_between_tunes} | {output[len(header):]} |')
          last_time = current_time
        elif verbose:
          print(output)
    return process.poll(), entire_output

  cmd = f'./bin/tuningRunner.py --op=gemm --configs_file=\"{gemm_file}\" --output=\"{tuning_db}\" --verify-mode=none'
  print(f'execute: {cmd}')
  _, captured_output = run_process(cmd)
  with open(f'{dirs.tuner_output_file}.gemm', 'w') as file:
    for line in captured_output:
      file.write(line)

  cmd = f'./bin/tuningRunner.py --op=conv --configs_file=\"{conv_file}\" --output=\"{tuning_db}\" --verify-mode=none'
  print(f'execute: {cmd}')
  _, captured_output = run_process(cmd)
  with open(f'{dirs.tuner_output_file}.conv', 'w') as file:
    for line in captured_output:
      file.write(line)

  os.chdir(dirs.current_workdir)


def adjust_tuning_db(config, dirs):
  tuning_db = dirs.get_tuning_db_path(config)
  if not os.path.exists(tuning_db):
    print(f'cannot open tuning db: {tuning_db}')
    sys.exit(-1)

  tuning_db_backup = f'{tuning_db}.bkp'
  if not os.path.exists(tuning_db_backup):
    shutil.copyfile(tuning_db, tuning_db_backup)
  print(f'\tbackup db file: {tuning_db_backup}')

  hipinfo = os.path.join(dirs.hiputils, 'hipinfo')
  if not os.path.exists(hipinfo):
    print(f'cannot find hipinfo at: {hipinfo}')
    sys.exit(-1)

  result = subprocess.run(f'{hipinfo}', shell=True, capture_output=True, text=True)
  match = re.search(r'numCU:\s+(\d+)', result.stdout)
  if match and match.group(1) is not None:
    num_cu = match.group(1)
  else:
    print(f'failed to get/process output from: {hipinfo}')
    sys.exit(-1)

  with open(tuning_db_backup, 'r') as file:
    content = file.readlines()

  with open(tuning_db, 'w') as file:
    for line in content:
      adjusted_line = re.sub(r'\t\d+\t', f'\t{num_cu}\t', line)
      file.write(adjusted_line)

  print(f'\ttuning db adjusted. See: {tuning_db}')


def evaluate_performance(group, config, dirs):
  os.chdir(dirs.migraphx)
  tuning_db = dirs.get_tuning_db_path(config)
  if not os.path.exists(tuning_db):
    print(f'cannot find tuning db: {tuning_db}')
    sys.exit(-1)

  test_envs = []
  env_copy = os.environ.copy()
  env_copy['MIGRAPHX_ENABLE_MLIR'] = '1'
  env_copy['MIGRAPHX_MLIR_TUNING_DB'] = f'\"{tuning_db}\"'
  test_envs.append(('mlir_on_tb_on', env_copy))

  env_copy = os.environ.copy()
  env_copy['MIGRAPHX_ENABLE_MLIR'] = '1'
  test_envs.append(('mlir_on_tb_off', env_copy))

  env_copy = os.environ.copy()
  env_copy['MIGRAPHX_ENABLE_MLIR'] = '0'
  test_envs.append(('mlir_off', env_copy))

  migraphx_exe = os.path.join(dirs.migraphx, 'bin', 'migraphx-driver')  
  for model in group.models:
    args = [migraphx_exe, 'perf']
    if model.type:
      args.append(model.type)
    args.extend(['--onnx', model.path])
    if not model.is_static():
      args.append(model.params)
    cmd = ' '.join(args)

    config_name = model.gen_config_result_dir_name()
    config_dir = os.path.join(dirs.results_dirs, config_name)
    os.makedirs(config_dir, exist_ok=True)  

    print(f'executing: {cmd}')
    for (env_name, curr_env) in test_envs:
      info = ''
      keys = ['MIGRAPHX_MLIR_TUNING_DB', 'MIGRAPHX_ENABLE_MLIR']
      for key in keys:
        if key in curr_env.keys():
          info += f'{key}={curr_env[key]} '

      print(f'\t{env_name}: {info}')
      result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=curr_env)
      output_file = os.path.join(config_dir, f'{env_name}.out')
      with open(output_file, 'w') as file:
        file.write(result.stdout)
      err_file = os.path.join(config_dir, f'{env_name}.err')
      with open(err_file, 'w') as file:
        file.write(result.stderr)

      if env_name == 'mlir_on_tb_on' and result.stderr:
        print('-' * 80)
        print(f'{result.stderr}')
        print('-' * 80)

  print(f'results are written: {dirs.results_dirs}')
  os.chdir(dirs.current_workdir)


def collect_output_data(group, config, dirs):
  data = OrderedDict()
  for model in group.models:
    config_name = model.gen_config_result_dir_name()
    config_dir = os.path.join(dirs.results_dirs, config_name)
    configs = ['mlir_on_tb_on', 'mlir_on_tb_off', 'mlir_off']

    data[model.type] = OrderedDict()
    entry = data[model.type]
    for config in configs:
      entry[config] = OrderedDict()
      filename = os.path.join(config_dir, f'{config}.out')
      if os.path.exists(filename):
        statistics, summary = process_migraphx_output(filename)
        entry[config]['statistics'] = statistics
        entry[config]['summary'] = summary
      else:
        print(f'cannot open: {filename}')
  return data


def main():
  parser = argparse.ArgumentParser(prog='migraphx runner for rocMLIR')
  parser.add_argument('-c','--config')
  parser.add_argument('-a','--action', choices=['show', 'clean-workdir', 'collect', 'join', 'tune', 'adjust', 'perf', 'report'])
  parser.add_argument('-v', '--verbose', action='store_true')
  args = parser.parse_args()

  try:
    with open(args.config, "r") as stream:
      config = yaml.safe_load(stream)
  except Exception as err:
    print('Fail to open config yaml-file')
    print(str(err))
    sys.exit(-1)

  dirs = Directories(config)
  groups = build_groups(config['models'])

  if args.action == 'show':
    for group in groups:
      print(group)

  if args.action == 'clean-workdir':
    if os.path.isdir(dirs.workdir):
      shutil.rmtree(dirs.workdir)

  if args.action == 'collect':
    for group in groups:
      collect_tuning_config(group, config, dirs)

  if args.action == 'join':
    join_tuning_config(config, dirs)

  if args.action == 'tune':
    run_tunner(config, dirs, args.verbose)

  if args.action == 'adjust':
    for group in groups:
      adjust_tuning_db(config, dirs)

  if args.action == 'perf':
    for group in groups:
      evaluate_performance(group, config, dirs)

  if args.action == 'report':
    for group in groups:
      data = collect_output_data(group, config, dirs)
      make_report_page(group, data)


if __name__ == '__main__':
  main()
