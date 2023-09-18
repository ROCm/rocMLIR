#!/usr/bin/env python3

import os
import argparse
import shutil
import glob
import itertools
import tomli

RANDTYPE = {'f32' : 'float',
            'f16' : 'float',
            'bf16' : 'float',
            'i32' : 'int',
            'i8' : 'int'}

def generate_option_list(table, key1, key2):
    options_list=[]
    for item in table[key1]:
        options=[]
        for value in item[key2]:
            options.append(value)
        options_list.append(options)
    combinations=[]
    for opt in itertools.product(*options_list):
        combinations.append(opt);
    return combinations

def generate_op_variants_test(indir, outdir, type, file, opspec):
    opname,op = opspec
    with open(f"{indir}/{file}.e2e.template") as f:
        template = f.read()
    outfile = f"{outdir}/{file}-{opname}-{type}.e2e.mlir"
    with open(outfile, 'w') as f:
        # The instruction operand is in the middle for clamp, so we can't simply
        # substitute.  With a regexp replace we could capture the actual operand
        # but for now we'll just "know" what we have.
        op = op.format(operand='(%0)')
        f.write(template.format(op=op, type=type,
                                randtype=RANDTYPE[type],
                                randkind='fixed' if opname != 'rsqrt' else '1',
                                # So far clone-verification only works with f32.  Also, tanh
                                # fails it because math.tanh is converted by gpu-to-rocdl pass
                                # which isn't run on the cloned function.
                                disablep='-DISABLE' if type != 'f32' or opname == 'tanh' else ''))

def generate_type_only_test(indir, outdir, type, file):
    with open(f"{indir}/{file}.e2e.template") as f:
        template = f.read()
    outfile = f"{outdir}/{file}-{type}.e2e.mlir"
    with open(outfile, 'w') as f:
        f.write(template.format(type=type,
                                # So far clone-verification only works with f32.
                                disablep='-DISABLE' if type != 'f32' else ''))

def toml_loop(toml, indir, outdir):
    for suite in toml['suite']:
        combinations = generate_option_list(suite, 'axis', 'values')
        for test in combinations:
            if suite['kind'] == "op-variants":
                generate_op_variants_test(indir, outdir, *test)
            elif suite['kind'] == "type-only":
                generate_type_only_test(indir, outdir, *test)
            else:
                raise Exception("unknown test suite")


parser = argparse.ArgumentParser()
parser.add_argument('indir', default=os.getcwd())
parser.add_argument('outdir', default=os.getcwd())
args = parser.parse_args()

def run():
    indir = args.indir
    outdir = args.outdir

#     # shutil.copytree isn't recursively copying in some circumstances.
#     for dir,*_ in os.walk(indir):
#         if dir == indir:
#             dir = '.'
#         elif dir.startswith(indir):
#             dir = dir[len(indir):].lstrip('/')
#         outsubdir = os.path.join(outdir, dir)
#         os.makedirs(outsubdir, exist_ok=True)
#         for file in glob.glob(os.path.join(indir, '*.mlir')):
#             shutil.copy(file, outsubdir)

    def ignore_not_mlir(dir, files):
        return [f for f in files if not f.endswith('.mlir')
                                    and not os.path.isdir(os.path.join(dir, f))]
    shutil.copytree(indir, outdir, ignore=ignore_not_mlir, dirs_exist_ok=True)
    shutil.copy(os.path.join(indir, "lit.local.cfg"), outdir)

    with open(os.path.join(indir, "tests.toml"), 'rb') as f:
        toml = tomli.load(f)
    toml_loop(toml, indir, outdir)


if __name__ == '__main__':
    run()
    print("DONE!")
