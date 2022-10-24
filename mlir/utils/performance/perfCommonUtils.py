import enum

class Operation(enum.IntEnum):
  CONV = 1
  GEMM = 2

  @staticmethod
  def fromName(name: str) -> 'self':
    name = name.lower()
    if name == 'conv':
      return Operation.CONV
    elif name == 'gemm':
      return Operation.GEMM
    else:
      raise ValueError(f"Unknown operation type {name}")
