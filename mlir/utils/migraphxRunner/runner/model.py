from abc import ABC, abstractmethod
import hashlib


def build_groups(descriptions):
  groups = []
  for desc in descriptions:
    if 'params' in desc.keys():
      groups.extend(Group.create_dynamic(desc))
    else:
      groups.append(Group.create_static(desc))
  return groups


class Group:
  def __init__(self, name, models):
    self.name = name
    self.models = models

  def __str__(self):
    text = f'group: {self.name}\n'
    for model in self.models:
      text += '\t' + str(model) + '\n'
    return text

  @classmethod
  def create_static(cls, description):
    name = description['name']
    path = description['path']
    types = description['types']

    models = []
    for type_test in types:
      model = Static(name, path, type_test)
      models.append(model)

    return Group(name, models)

  @classmethod
  def create_dynamic(cls, description):
    name = description['name']
    path = description['path']
    types = description['types']

    params = description['params']
    if 'batch_size' in description.keys():
      batch_sizes = description['batch_size']
    else:
      batch_sizes = [None]

    groups = []
    for batch_size in batch_sizes:
      models = []
      if batch_size:
        group_name = f'{name}-b{batch_size}'
        concrete_params = params.replace('<batch_size>', str(batch_size))
      else:
        group_name = name
        concrete_params = params
      for type_test in types:
        model = Dynamic(name, path, type_test, concrete_params, batch_size)
        models.append(model)
      groups.append(Group(group_name, models))
    return groups


class Model:
  def __init__(self, name, path, test_type):
    self.name = name
    self.path = path
    self.type = test_type

  @abstractmethod
  def is_static(self):
    pass

  @abstractmethod
  def get_full_name(self):
    pass


class Static(Model):
  def __init__(self, name, path, test_type):
    super(Static, self).__init__(name, path, test_type)

  def is_static(self):
    return True

  def get_full_name(self):
    return f'{self.name}{self.type}'

  def __str__(self):
    text = f'model: {self.name}'
    text += f' | kind: static'
    if self.type:
      text += f' | data type: {self.type}'
    else:
      text += ' | data type: default'
    return text


class Dynamic(Model):
  def __init__(self, name, path, test_type, params, batch_size=None):
    super(Dynamic, self).__init__(name, path, test_type)
    self.params = params
    self.batch_size = batch_size  

  def is_static(self):
    return False

  def get_full_name(self):
    full_name = f'{self.name}{self.type}'
    if self.batch_size:
      full_name += f'-b{self.batch_size}'
    text = ' '.join(self.params)
    sha = hashlib.sha256(text.encode('UTF-8'))
    full_name += '-' + sha.hexdigest()[:8]
    return full_name

  def __str__(self):
    text = f'model: {self.name}'
    text += f' | kind: dynamic'
    if self.type:
      text += f' | data type: {self.type}'
    else:
      text += ' | data type: default'

    if self.batch_size:
      text += f' | batch size: {self.batch_size}'

    text += f' | params: {self.params}'

    return text
