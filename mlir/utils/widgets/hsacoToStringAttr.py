#!/usr/bin/env python3
import argparse
import os

# https://github.com/llvm/llvm-project/blob/dc37dc824aabbbe3d029519f43f0b348dcad7027/llvm/include/llvm/ADT/StringExtras.h#L125-L128
def isPrint(C: str) -> bool:
    UC = ord(C)
    return 0x20 <= UC <= 0x7E

# https://github.com/llvm/llvm-project/blob/dc37dc824aabbbe3d029519f43f0b348dcad7027/llvm/lib/Support/StringExtras.cpp#L62-L71
def printEscapedString(data, out):
    for val in data:
        c = chr(val)
        if c == '\\':
            out.write('\\' + c)
        elif isPrint(c) and c != '"':
            out.write(c)
        else:
            out.write('\\' + hex(ord(c) >> 4)[2:] + hex(ord(c) & 0x0F)[2:])
    print("mlir type: array<" + str(len(data)) + ", i8>")

def genAttrFromHsaco(args):
    with open(args.i, 'rb') as f:
        # Read the entire contents of the file into a bytes object
        data = f.read()

    with open(args.o, 'w') as out:
        printEscapedString(data, out)

def add_args():
    parser = argparse.ArgumentParser(
        description="Convert hsaco elf to rocMLIR serialized text.")

    parser.add_argument("-i", help="Input hsaco kernel file", required=True)
    parser.add_argument("-o", help="Output kernel text file", default=None)

    args = parser.parse_args()
    return args

def main(args):
    if (args.o == None):
        args.o = args.i.rsplit('.', maxsplit=1)[0] + ".attr"
    print("Converting from " + args.i + " to " + args.o)
    genAttrFromHsaco(args)

if __name__ == "__main__":
    arguments = add_args()
    main(arguments)
