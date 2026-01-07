import enum
import re


class Operation(enum.IntEnum):
    CONV = 1
    GEMM = 2
    FUSION = 3
    ATTENTION = 4
    GEMM_GEMM = 5
    CONV_GEMM = 6

    @staticmethod
    def from_name(name: str) -> "Operation":
        name = name.lower()
        if name == 'conv':
            return Operation.CONV
        elif name == 'gemm':
            return Operation.GEMM
        elif name == 'attention':
            return Operation.ATTENTION
        elif name == 'gemm_gemm':
            return Operation.GEMM_GEMM
        elif name == 'conv_gemm':
            return Operation.CONV_GEMM
        elif name == 'fusion':
            return Operation.FUSION
        else:
            raise ValueError(f"Unknown operation type {name}")


CORRECT_RESULT_RE = re.compile('\[1\s*1\s*1\]')


class GEMMLibrary(enum.IntEnum):
    CK = 1
    HIPBLASLT = 2

    @staticmethod
    def from_name(name: str) -> "GEMMLibrary":
        name = name.lower()
        if name == 'ck':
            return GEMMLibrary.CK
        elif name == 'hipblaslt':
            return GEMMLibrary.HIPBLASLT
        else:
            raise ValueError(f"Unknown library {name}")
