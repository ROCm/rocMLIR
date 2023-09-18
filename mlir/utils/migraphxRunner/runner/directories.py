import os
import sys


class Directories:
  def __init__(self, config):
    try:
      self.current_workdir = os.getcwd()
      self.migraphx = os.path.expanduser(config['migraphx_path'])
      self.rocmlir = os.path.expanduser(config['rocmlir_path'])
      self.hiputils = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'hiputils'))
      self.workdir = os.path.expanduser(config['workdir_path'])
      self.tuning_config_dir = os.path.join(self.workdir, 'tuning_configs')
      self.results_dirs = os.path.join(self.workdir, 'results')
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
    os.makedirs(self.workdir, exist_ok=True)
    os.makedirs(self.tuning_config_dir, exist_ok=True)
    os.makedirs(self.results_dirs, exist_ok=True)
  
  def __str__(self):
    return f'current_workdir: {self.current_workdir}' \
           f'\nmigraphx: {self.migraphx}' \
           f'\nrocmlir: {self.rocmlir}' \
           f'\nhiputils: {self.hiputils}' \
           f'\nworkdir: {self.workdir}'

  def get_tuning_config_files(self, config):
    output = os.path.join(self.tuning_config_dir, config['tuning_config_name'])
    return f'{output}.conv', f'{output}.gemm'

  def get_tuning_db_path(self, config):
    return os.path.join(self.tuning_config_dir, config['tuning_db_name'])

  @classmethod
  def get_config_result_dir(cls, model_name, type, params):
    name = f'{model_name}{type}'
    if params:
      if type(params) == list:
        text = ' '.join(params)
        sha = hashlib.sha256(text.encode('UTF-8'))
        name += sha.hexdigest()[:8]
    return name