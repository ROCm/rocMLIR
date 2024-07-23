#!/usr/bin/env python3

# AMDDeviceLibsIncGen.py - embed device library bitcode as string constants
#
# Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
# Exceptions. See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (c) 2022 Advanced Micro Devices INc.
#
# Despite being in upstream directories, this is rocMLIR specific code that
# lets us embed the AMD device libraries as string constants

import sys
from pathlib import Path
from typing import List

def as_signed(byte: int) -> int:
  """Return the input byte as a signed value, that is, map [128, 255] to
  [-128, -1]."""
  if byte >= 128:
    return byte - 256
  return byte

def generate(outputPath: Path, rocmPath: Path, libs: List[str]) -> None:
  bcPath = rocmPath / "amdgcn" / "bitcode"
  with outputPath.open("w") as out:
    for lib in libs:
      with (bcPath / (lib + ".bc")).open("rb") as libFile:
        bcBytes = libFile.read()
      bcLen = len(bcBytes)
      print(f"static constexpr size_t {lib}_size = {bcLen};", file=out)
      print("""#if defined __GNUC__
__attribute__((aligned (4096)))
#elif defined _MSC_VER
__declspec(align(4096))
#endif""", file=out)
      print(f"static constexpr char {lib}_bytes[{lib}_size + 1] = {{", file=out)
      for i, byte in enumerate(bcBytes):
        print(f"{as_signed(byte):+4},", file=out, end=("\n" if i % 8 == 0 else " "))
      # Terminating null pointer needed for
      print("0x00};", file=out)
    print("static constexpr std::initializer_list<std::pair<llvm::StringRef, llvm::StringRef>> allLibList = {", file=out)
    for lib in libs:
      print(f"{{\"{lib}.bc\", llvm::StringRef({lib}_bytes, {lib}_size)}},", file=out)
    print("};", file=out)
    print("""static const llvm::StringMap<llvm::StringRef>& getDeviceLibraries() {
static const llvm::StringMap<llvm::StringRef> allLibs(allLibList);
return allLibs;
}""", file=out)

if __name__ == '__main__':
  generate(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:])
