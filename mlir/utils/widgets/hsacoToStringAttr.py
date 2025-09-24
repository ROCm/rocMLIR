#!/usr/bin/env python3
import argparse


# https://github.com/llvm/llvm-project/blob/dc37dc824aabbbe3d029519f43f0b348dcad7027/llvm/include/llvm/ADT/StringExtras.h#L125-L128
def is_print(c: str) -> bool:
    uc = ord(c)
    return 0x20 <= uc <= 0x7E


# https://github.com/llvm/llvm-project/blob/dc37dc824aabbbe3d029519f43f0b348dcad7027/llvm/lib/Support/StringExtras.cpp#L62-L71
def print_escaped_string(data, out):
    for val in data:
        c = chr(val)
        if c == '\\':
            out.write('\\' + c)
        elif is_print(c) and c != '"':
            out.write(c)
        else:
            out.write('\\' + hex(ord(c) >> 4)[2:] + hex(ord(c) & 0x0F)[2:])
    print("mlir type: array<" + str(len(data)) + ", i8>")


# This convert generate hsaco as attribute from args.i to args.o
def gen_attr_from_hsaco(args):
    with open(args.i, 'rb') as f:
        # Read the entire contents of the file into a bytes object
        data = f.read()

    with open(args.o, 'w') as out:
        print_escaped_string(data, out)


def add_args():
    parser = argparse.ArgumentParser(
        description="Convert hsaco elf to rocMLIR serialized text.")

    parser.add_argument("-i", help="Input hsaco kernel file", required=True)
    parser.add_argument("-o", help="Output kernel text file", default=None)

    args = parser.parse_args()
    return args


def main(args):
    if args.o is None:
        args.o = args.i.rsplit('.', maxsplit=1)[0] + ".attr"
    gen_attr_from_hsaco(args)


if __name__ == "__main__":
    arguments = add_args()
    main(arguments)
