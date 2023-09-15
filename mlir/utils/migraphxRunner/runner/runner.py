import argparse
import subprocess
import yaml
import os
import sys
import shutil


class Directories:
  def __init__(self, config):
    try:
      self.current_workdir = os.getcwd()
      self.migraphx = os.path.expanduser(config['migraphx_path'])
      self.rocmlir = os.path.expanduser(config['rocmlir_path'])
      self.hiputils = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'hiputils'))
      self.workdir = os.path.expanduser(config['workdir_path'])
      self.tuning_config_dir = os.path.join(self.workdir, 'tuning_configs')
      self.__check_paths()
    except KeyError as err:
      print(f'yaml config error: {err}')
      sys.exit(-1)

  def __check_paths(self):
    if not os.path.isdir(self.migraphx):
      raise RuntimeError('could not find `migraphx` dir')
    if not os.path.isdir(self.rocmlir):
      raise RuntimeError('could not find `rocmlir` dir')
    if not os.path.isdir(self.hiputils):
      raise RuntimeError('could not find `hiputils` dir')
    if not os.path.isdir(self.workdir):
      os.makedirs(self.workdir, exist_ok=False)
    if not os.path.isdir(self.tuning_config_dir):
      os.makedirs(self.tuning_config_dir, exist_ok=False)
  
  def __str__(self):
    return f'current_workdir: {self.current_workdir}' \
           f'\nmigraphx: {self.migraphx}' \
           f'\nrocmlir: {self.rocmlir}' \
           f'\nhiputils: {self.hiputils}' \
           f'\nworkdir: {self.workdir}'


def check_model(models):
  try:
    for model in models:
      if not os.path.exists(model['path']):
        raise RuntimeError(f'could not find: `{model["path"]}`')
  except KeyError as err:
    print(f'yaml config error: {err}')
    raise err


def collect_tuning_config(model, config, dirs):
  os.chdir(dirs.migraphx)

  for test_type in model['types']: 
    tunung_config_file = f'{model["name"]}{test_type}.cfg'
    env = os.environ.copy()
    env['MIGRAPHX_ENABLE_MLIR'] = '1'
    env['MIGRAPHX_MLIR_TUNING_CFG'] = tunung_config_file

    migraphx_exe = os.path.join(dirs.migraphx, 'bin', 'migraphx-driver')
    args = [migraphx_exe, 'compile']
    if test_type:
      args.append(test_type)
    args.extend(['--onnx', model['path']])
    if 'params' in model:
      args.append(model['params'])

    cmd = ' '.join(args)
    print(f'executing: {cmd}')
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)

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


def read_files(files):
  context = []
  for filepath in files:
    filepath = files[0]
    with open(filepath, 'r') as file:
      context.extend(file.readlines())

  return list(set(context))


def join_tuning_config(config, dirs):
  print(f'processing all tuning configs from: {dirs.tuning_config_dir}')
  conv_files = []; gemm_files = []
  for (dirpath, dirnames, filenames) in os.walk(dirs.tuning_config_dir):
    for filename in filenames:
      db_file = os.path.join(dirpath, filename)
      suffix = db_file[-4:]
      if suffix == 'conv':
        conv_files.append(db_file)
      if suffix == 'gemm':
        gemm_files.append(db_file)

  conv_configs = read_files(conv_files)
  output = os.path.join(dirs.tuning_config_dir, 'cfg')
  with open(f'{output}.conv', 'w') as file:
    for line in conv_configs:
      file.write(line)

  gemm_configs = read_files(gemm_files)
  output = os.path.join(dirs.tuning_config_dir, config['tunung_config_name'])
  with open(f'{output}.gemm', 'w') as file:
    for line in gemm_configs:
      file.write(line)

  print(f'tuning configs are written to: {output}.(conv|gemm)')

def main():
  parser = argparse.ArgumentParser(prog='migraphx runner for rocMLIR')
  parser.add_argument('-c','--config')
  parser.add_argument('-a','--action', choices=['collect', 'join', 'run'])
  args = parser.parse_args()

  try:
    with open(args.config, "r") as stream:
      config = yaml.safe_load(stream)
  except Exception as err:
    print('Fail to open config yaml-file')
    print(str(err))
    sys.exit(-1)

  dirs = Directories(config)
  check_model(config['models'])
  model = config['models'][0]

  if args.action == 'collect':
    for model in models:
      collect_tuning_config(model, config, dirs)

  if args.action == 'join':
    join_tuning_config(config, dirs)

if __name__ == '__main__':
  main()
