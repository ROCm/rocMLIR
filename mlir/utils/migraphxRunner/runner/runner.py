import argparse
import yaml
import os
import sys


class Directories:
  def __init__(self, config):
    try:
      self.current_workdir = os.path.dirname(__file__)
      self.migraphx = os.path.expanduser(config['migraphx_path'])
      self.rocmlir = os.path.expanduser(config['rocmlir_path'])
      self.hiputils = os.path.abspath(os.path.join(self.current_workdir, os.pardir, "hiputils"))
      self.workdir = os.path.expanduser(config['workdir_path'])
      self.__check_paths()
    except KeyError as err:
      print(f'yaml config error: {err}')
      sys.exit(-1)
    except Exception as err:
      print(str(err))
      sys.exit(-1)

  def __check_paths(self):
    if not os.path.isdir(self.migraphx):
      raise RuntimeError('could not find `migraphx` dir')
    if not os.path.isdir(self.rocmlir):
      raise RuntimeError('could not find `rocmlir` dir')
    if not os.path.isdir(self.hiputils):
      raise RuntimeError('could not find `hiputils` dir')
    if not os.path.isdir(self.workdir):
      os.makedirs(config['workdir_path'], exist_ok=False)
  
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
  except Exception as err:
    print(str(err))
    sys.exit(-1)


def main():
  parser = argparse.ArgumentParser(prog='migraphx runner for rocMLIR')
  parser.add_argument('-c','--config')
  args = parser.parse_args()

  try:
    with open(args.config, "r") as stream:
      config = yaml.safe_load(stream)
  except Exception as err:
    print('Fail to open config yaml-file')
    print(str(err))
    sys.exit(-1)

  dirs = Directories(config)
  print(dirs)
  check_model(config['models'])
  model = config['models'][0]


if __name__ == '__main__':
  main()
