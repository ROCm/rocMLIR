#!/usr/bin/env python3
# coding: utf-8

"""Loop through the E2E tests in this repository and collect information about them.

Expects to be run from mlir/utils/ (where it lives), though editing ROCK_ROOT
can change that.

Output is a file in pickle form containing the extracted data

Expects a newer Python than 3.6.9 - I used 3.10.4"""

import asyncio
from pathlib import Path
import sys
import re
import pickle
from typing import Dict, List, Union, Optional, Tuple, Iterable

ROCK_ROOT = Path('../..')
ROCK_BIN = ROCK_ROOT / 'build' / 'bin'
ROCK_GEN = ROCK_BIN / 'rocmlir-gen'
ROCMLIR_DRIVER = ROCK_BIN / 'rocmlir-driver'
E2E_TEST_ROOT = ROCK_ROOT / 'mlir' / 'test' / 'rocmlir-driver'

CONV_CONFIG_RE = re.compile(r'rocmlir-gen\s*([^|]+)\s*\|')
UNNEEDED_PARAMS_RE = re.compile(r'%pv|%random_data|%feature|-x2|-pv|-pv_with_gpu|-ph')
def clean_configs_from(data: str) -> List[str]:
    ret = []
    for match in CONV_CONFIG_RE.finditer(data):
        ret.append(UNNEEDED_PARAMS_RE.sub("", match[1]).strip())
    return ret

class OptPassException(Exception):
    pass

async def run_generator(config: str, xdlops: bool) -> str:
    args = config.split()
    if xdlops:
        args.append("-x2")
    proc = await asyncio.create_subprocess_exec(ROCK_GEN, *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    (stdout, stderr) = await proc.communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")
    if proc.returncode != 0:
        print(stdout)
        print(stderr)
        raise ValueError(f"Convolution generator failed for config '{config}', xdlops={xdlops}")
    return stdout

CONV_FUNC_FINDER = re.compile(r'func.*rock\.conv.*return\s*\}', re.MULTILINE | re.DOTALL)
def isolate_real_kernel(rock_gen_out: str) -> str:
    """Strip out utility kernels and return [one example of] the 'real' convolution.

    If there is only one kernel call, return it.

    Otherwise, return the kernel with GEMM ID 1, since in both wrw and bwd, this is the kernel
    that isn't responsible for zeroing or other such activities."""
    matches = CONV_FUNC_FINDER.findall(rock_gen_out)
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        return matches[1]
    else:
        raise ValueError("No convolution kernels in rocmlir-gen outputs")

async def get_kernel(config: str, xdlops: bool) -> str:
    return isolate_real_kernel(await run_generator(config, xdlops))

async def run_passes(kernel: str, passes: List[str]) -> Tuple[str, str]:
    proc = await asyncio.create_subprocess_exec(ROCMLIR_DRIVER, *passes,
                            stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate(kernel.encode("utf-8"))
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")
    if proc.returncode != 0:
        raise OptPassException({"stdout": stdout, "stderr": stderr})
    return (stdout, stderr)

# sample_config = '-batchsize=64 -in_channels=64 -in_h=56 -in_w=56 -out_channels=256 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 --operation conv2d -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -t f32'
# sample_kernel = await get_kernel(sample_config, True)
# sample_kernel_affix_tuning, sample_kernel_affix_tuning_debug =\
#     await run_passes(sample_kernel, ["-rock-affix-params", "-debug-only=rock-tuning-parameter"])
# print(sample_kernel_affix_tuning)
# print(sample_kernel_affix_tuning_debug)

# sample_kernel_gridwise, sample_kernel_gridwise_debug =\
#     await run_passes(sample_kernel_affix_tuning,
#                ["-rock-lowering", "-rock-lowering-step2",
#                 "-debug-only=rock-gridwise-to-blockwise"])
# print(sample_kernel_gridwise)
# print(sample_kernel_gridwise_debug)
# sample_kernel_stdlib, sample_kernel_stdlib_debug =\
#     await run_passes(sample_kernel_gridwise, ["-rock-lowering-step3", "-rock-lowering-step4",
#                                        "-debug-only=rock-blockwise-to-threadwise,rock-threadwise-to-stdlib"])
# print(sample_kernel_stdlib_debug)

def get_array_attribute(kernel: str, name: str) -> str:
    location = kernel.find(name)
    if location is None:
        raise ValueError(f"Can't find array attribute {name}")
    name_end = location + len(name)
    array_start = name_end + 4
    if kernel[name_end:array_start] != " = [":
        raise ValueError(f"Attribute {name} isn't an array")
    array_end = kernel.find(']', array_start)
    if array_end is None:
        raise ValueError(f"Couldn't find ] while parsing attribute {name}")
    return kernel[array_start:array_end]

def get_scalar_attr(kernel: str, name: str) -> str:
    location = kernel.find(name)
    if location is None:
        raise ValueError(f"Couldn't find attribute {name}")
    name_end = location + len(name)
    value_start = name_end + 3
    if kernel[name_end:value_start] != " = ":
        raise ValueError(f"Couldn't identify value of attribute {name}")
    value_end = kernel.find(',', value_start)
    if value_end is None:
        value_end = kernel.find('}', value_start)
    return kernel[value_start:value_end]

TYPED_INT_MATCH_RE = re.compile(r'(-?\d+)(?:\s*:\s*\w+)?,?')
def get_int_array_attr(kernel: str, name: str) -> List[int]:
    return [int(match[1]) for match
            in TYPED_INT_MATCH_RE.finditer(get_array_attribute(kernel, name))]
STRING_MATCH_RE = re.compile(r'"([^"]+)"')
def get_str_array_attr(kernel: str, name: str) -> List[str]:
    return [match[1] for match
               in STRING_MATCH_RE.finditer(get_array_attribute(kernel, name))]
def get_int_attr(kernel: str, name: str) -> int:
    return int(TYPED_INT_MATCH_RE.match(get_scalar_attr(kernel, name))[1])
def get_str_attr(kernel: str, name: str) -> str:
    return STRING_MATCH_RE.match(get_scalar_attr(kernel, name))[1]

DEBUG_KEY_RES: Dict[str, re.Pattern] = dict()
def get_debug(debug: str, key: str) -> str:
    global DEBUG_KEY_RES
    if key not in DEBUG_KEY_RES:
        DEBUG_KEY_RES[key] = re.compile(r'^' + key + r'\s*:\s*(.*)$', re.MULTILINE)
    return DEBUG_KEY_RES[key].search(debug)[1]
def get_debug_int(debug: str, key: str) -> int:
    return int(get_debug(debug, key))

VECTOR_COMPONENTS_RE = re.compile('^vector<(\d+)x(\w+)>$')
def parse_type(typ: str) -> Tuple[int, str]:
    vector_match = VECTOR_COMPONENTS_RE.match(typ)
    if vector_match:
        return (int(vector_match[1]), vector_match[2])
    return (1, typ)

GEMM_PADDING_RE = re.compile('#rock.padding_info<extraM = (\d+), extraK = (\d+), extraN = (\d+)>')
def get_padding_info(kernel: str) -> Tuple[int, int, int]:
    match = GEMM_PADDING_RE.search(kernel)
    return int(match[1]), int(match[2]), int(match[3])

def normalize_layout(layout_attr: List[str]) -> str:
    return ''.join(l[0] for l in layout_attr)

REDUNDANT_SIZES = frozenset({"no", "ci", "ko", "gi", "go"})
RENAMES = {"ni": "n"}
MEMREFS_RE = re.compile(r'rock.conv.* : ((?:memref<\w+>,?\s*)+)$', re.MULTILINE)
MEMREF_RE = re.compile(r'memref<(\w+)>')
def parse_sizes(kernel: str) -> List[List[int]]:
    types = MEMREFS_RE.search(kernel)
    ret = []
    for match in MEMREF_RE.finditer(types[1]):
        args = match[1]
        ret.append([int(v) for v in args.split('x')[:-1]])
    return ret

def get_op(kernel: str) -> str:
    if "rock.conv_bwd_data" in kernel:
        return "bwd"
    if "rock.conv_bwd_weight" in kernel:
        return "wrw"
    if "rock.conv" in kernel:
        return "fwd"
    raise ValueError("Kernel is not a convolution")

KernelDataT = Dict[str, Union[str, int]]
async def get_properties(config: str, xdlops: bool, filename: Optional[str] = None) -> Optional[KernelDataT]:
    kernel = await get_kernel(config, xdlops)
    try:
        tuning, tuning_dbg =\
            await run_passes(kernel, ["-rock-affix-params", "-debug-only=rock-tuning-parameter"])
        conv, conv_dbg = await run_passes(tuning, ["-rock-lowering"])
        grid, grid_dbg =\
            await run_passes(conv, ["-rock-lowering-step2", "-debug-only=rock-gridwise-to-blockwise"])
        block, block_dbg =\
            await run_passes(grid, ["-rock-lowering-step3", "-debug-only=rock-blockwise-to-threadwise"])
        thread, thread_dbg =\
            await run_passes(block, ["-rock-lowering-step4", "-debug-only=rock-threadwise-to-stdlib"])
    except OptPassException:
        print(f"Inapplicable config '{config}' xdlops={xdlops}")
        return None
    ret = {"Config": config}
    if filename is not None:
        ret["Filename"] = filename

    ret["Op"] = get_op(tuning)

    filter_layout = get_str_array_attr(tuning, "filter_layout")
    input_layout = get_str_array_attr(tuning, "input_layout")
    output_layout = get_str_array_attr(tuning, "output_layout")

    ret["Filter layout"] = normalize_layout(filter_layout)
    ret["Input layout"] = normalize_layout(input_layout)
    ret["Output layout"] = normalize_layout(output_layout)

    all_sizes = parse_sizes(tuning)
    for names, sizes in zip((filter_layout, input_layout, output_layout), all_sizes):
        for name, size in zip(names, sizes):
            if name in REDUNDANT_SIZES:
                continue
            ret[RENAMES.get(name, name)] = size

    ret["Pad h left"], ret["Pad h right"], ret["Pad w left"], ret["Pad w right"] =\
        get_int_array_attr(tuning, "padding")
    ret["Stride h"], ret["Stride w"] = get_int_array_attr(tuning, "strides")
    ret["Dilation h"], ret["Dilation w"] = get_int_array_attr(tuning, "dilations")

    ret["Pad M"], ret["Pad K"], ret["Pad N"] = get_padding_info(conv)

    ret["Block size"] = get_int_attr(tuning, "block_size")
    ret["Grid size"] = get_int_attr(tuning, "grid_size")

    ret["M"] = get_debug_int(grid_dbg, "M")
    ret["K"] = get_debug_int(grid_dbg, "K")
    ret["N"] = get_debug_int(grid_dbg, "N")
    ret["M per block"] = get_debug_int(grid_dbg, "MPerBlock")
    ret["K per block"] = get_debug_int(grid_dbg, "KPerBlock")
    ret["N per block"] = get_debug_int(grid_dbg, "NPerBlock")
    ret["KPack"] = get_debug_int(grid_dbg, "KPack")
    if xdlops:
        ret["M per wave"] = get_debug_int(grid_dbg, "mPerWave")
        ret["N per wave"] = get_debug_int(grid_dbg, "nPerWave")
    else:
        ret["M per thread"] = get_debug_int(grid_dbg, "MPerThread")
        ret["N per thread"] = get_debug_int(grid_dbg, "NPerThread")
    ret["Load A vector dim"] = get_debug_int(grid_dbg, "Corrected blockwise vector dim A")
    ret["Load B vector dim"] = get_debug_int(grid_dbg, "Corrected blockwise vector dim B")
    ret["Load A vector len"], ret["Load type"] =\
        parse_type(get_debug(grid_dbg, "Load type A"))
    ret["Load B vector len"] = parse_type(get_debug(grid_dbg, "Load type B"))[0]
    ret["Load A intermediate len"] = parse_type(get_debug(grid_dbg, "Intermediate type A"))[0]
    ret["Load B intermediate len"] = parse_type(get_debug(grid_dbg, "Intermediate type B"))[0]
    ret["LDS store A len"] = parse_type(get_debug(grid_dbg, "Store type A"))[0]
    ret["LDS store B len"] = parse_type(get_debug(grid_dbg, "Store type B"))[0]
    if xdlops:
        ret["Store C vector dim"] = get_debug_int(block_dbg, "Threadwise copy vector dimension")
        ret["Store C vector len"] = get_debug_int(block_dbg, "Data per copy")
        ret["Xdlop"] = get_str_attr(thread, "instr")
    return ret

async def parse_configs(path: Path, xdlops: bool,
        excludes: Iterable[str] = []) -> List[KernelDataT]:
    args = []
    for file in path.glob('**/*.mlir'):
        if any((exclude in str(file)) for exclude in excludes):
            continue
        with open(file, 'r', encoding='utf-8') as f:
            args.extend((c, file) for c in clean_configs_from(f.read()))
    mb_data = await asyncio.gather(*(get_properties(a[0], xdlops, a[1])
         for a in args))
    data = [d for d in mb_data if d is not None]
    return data

async def get_data() -> Tuple[List[KernelDataT], List[KernelDataT]]:
    raw_xdlops_data = await parse_configs(E2E_TEST_ROOT / 'auto_e2e',
            True, ["nonxdlops"]) + \
        await parse_configs(E2E_TEST_ROOT / 'e2e_for_pr', True)
    raw_non_xdlops_data = await parse_configs(E2E_TEST_ROOT / 'auto_e2e',
            False, ["PaddingXDLOPS"]) + \
        await parse_configs(E2E_TEST_ROOT / 'e2e_for_pr', False)
    return raw_xdlops_data, raw_non_xdlops_data

async def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} out_file", file=sys.stderr)
        sys.exit(1)
    if not ROCMLIR_DRIVER.exists():
        print(f"Cannot find rocmlir-driver at {ROCMLIR_DRIVER}", file=sys.stderr)
        sys.exit(1)
    filename = sys.argv[1]
    with open(filename, 'wb') as f:
        data = await get_data()
        pickle.dump(data, f)

if __name__ == '__main__':
    asyncio.run(main())
