# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Config proposers: the black boxes under test."""

from .base import ConfigProposer, PoolProvider
from .model import ModelProposer
from .nearest import NearestKnownProposer
from .random import RandomProposer
from .set_cover import SetCoverProposer

__all__ = [
    "ConfigProposer",
    "PoolProvider",
    "RandomProposer",
    "SetCoverProposer",
    "NearestKnownProposer",
    "ModelProposer",
]
