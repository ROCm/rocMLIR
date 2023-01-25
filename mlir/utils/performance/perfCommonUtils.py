import enum
import re

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

CORRECT_RESULT_RE = re.compile('\[1\s*1\s*1\]')
