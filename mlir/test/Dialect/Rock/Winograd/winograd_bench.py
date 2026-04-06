#!/usr/bin/env python3
"""Benchmark Winograd assembly kernels on GPU using HIP runtime.

Assembles a Winograd kernel, loads it via HIP, packs arguments, launches,
and measures execution time. Compares against GEMM baseline from perfRunner.py.

Usage:
  python3 winograd_bench.py --arch gfx942 --dtype f16 \
    --n 1 --c 64 --h 56 --w 56 --k 64 --r 3 --s 3
"""

import argparse
import ctypes
import os
import subprocess
import sys
import struct
import time
import numpy as np

def find_tool(name, extra_paths=None):
    """Find a tool by name on PATH or in extra_paths."""
    import shutil
    path = shutil.which(name)
    if path:
        return path
    for d in (extra_paths or []):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None

def assemble_winograd_kernel(kernel_file, arch, kernel_dir):
    """Assemble a .s file to .hsaco using clang + lld."""
    clang = find_tool("clang", ["/opt/rocm/llvm/bin"])
    lld = find_tool("ld.lld", ["/opt/rocm/llvm/bin"])
    if not clang or not lld:
        raise RuntimeError("clang or ld.lld not found")

    src = os.path.join(kernel_dir, kernel_file)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Kernel source not found: {src}")

    obj = f"/tmp/wino_bench_{os.getpid()}.o"
    hsaco = f"/tmp/wino_bench_{os.getpid()}.hsaco"

    subprocess.check_call([
        clang, "-x", "assembler",
        "-target", "amdgcn-amd-amdhsa",
        f"-mcpu={arch}",
        f"-I{kernel_dir}",
        "-c", src, "-o", obj
    ], stderr=subprocess.PIPE)

    subprocess.check_call([lld, "-shared", obj, "-o", hsaco],
                          stderr=subprocess.PIPE)

    with open(hsaco, "rb") as f:
        binary = f.read()

    os.unlink(obj)
    os.unlink(hsaco)
    return binary

def run_winograd_benchmark(args):
    """Run the Winograd kernel on GPU and measure time."""
    try:
        from hip import hip, hiprtc
    except ImportError:
        print("ERROR: hip-python not installed. Install with: pip install hip-python")
        sys.exit(1)

    def hip_check(result):
        if isinstance(result, tuple):
            err = result[0]
            if err != 0:
                raise RuntimeError(f"HIP error: {err}")
            return result[1] if len(result) > 1 else None
        if result != 0:
            raise RuntimeError(f"HIP error: {result}")
        return None

    # Determine kernel file based on args
    dtype = args.dtype
    if dtype == "f16":
        kernel_file = "Conv_Winograd_Rage_v4_9_0_fp16_fp32acc_f2x3_stride1.s"
        elem_size = 2
        np_dtype = np.float16
    elif dtype == "f32":
        kernel_file = "Conv_Winograd_Rage_v4_9_0_fp32_fp32acc_f2x3_stride1.s"
        elem_size = 4
        np_dtype = np.float32
    elif dtype == "bf16":
        kernel_file = "Conv_Winograd_Rage_v4_9_0_bf16_fp32acc_f2x3_stride1.s"
        elem_size = 2
        np_dtype = np.float16  # numpy doesn't have bf16, use f16 for sizes
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    kernel_dir = args.kernel_dir
    arch = args.arch

    print(f"Assembling {kernel_file} for {arch}...")
    hsaco_binary = assemble_winograd_kernel(kernel_file, arch, kernel_dir)
    print(f"  HSACO: {len(hsaco_binary)} bytes")

    # Compute dimensions
    N, C, H, W = args.n, args.c, args.h, args.w
    K, R, S = args.k, args.r, args.s
    pad_h, pad_w = R // 2, S // 2
    out_h = H - R + 1 + 2 * pad_h
    out_w = W - S + 1 + 2 * pad_w
    group = 1
    n_groups = args.n_groups

    # Allocate GPU buffers
    input_size = N * C * H * W * elem_size
    filter_size = K * C * R * S * elem_size
    output_size = N * K * out_h * out_w * elem_size

    input_dev = hip_check(hip.hipMalloc(input_size))
    filter_dev = hip_check(hip.hipMalloc(filter_size))
    output_dev = hip_check(hip.hipMalloc(output_size))

    # Initialize with random data
    input_host = np.random.randn(N * C * H * W).astype(np_dtype)
    filter_host = np.random.randn(K * C * R * S).astype(np_dtype)

    hip_check(hip.hipMemcpyHtoD(input_dev, input_host.ctypes.data, input_size))
    hip_check(hip.hipMemcpyHtoD(filter_dev, filter_host.ctypes.data, filter_size))
    hip_check(hip.hipMemset(output_dev, 0, output_size))

    # Load HSACO module
    module = hip_check(hip.hipModuleLoadData(hsaco_binary))

    # Find kernel function - need to know the exact name
    # Rage v4_9 kernel names follow the pattern:
    # miopenSp3AsmConvRage_v4_9_0_gfx9_<dtype>_f2x3_stride1
    if arch.startswith("gfx94") or arch.startswith("gfx9"):
        arch_tag = "gfx9"
    elif arch.startswith("gfx12"):
        arch_tag = "gfx12"
    else:
        arch_tag = "gfx9"

    dtype_map = {"f16": "fp16_fp32acc", "f32": "fp32_fp32acc", "bf16": "bf16_fp32acc"}
    kernel_name = f"miopenSp3AsmConvRage_v4_9_0_{arch_tag}_{dtype_map[dtype]}_f2x3_stride1"

    print(f"Loading kernel: {kernel_name}")
    kernel = hip_check(hip.hipModuleGetFunction(module, kernel_name.encode()))

    # Compute strides (NCHW layout, in elements)
    d_W = 1
    d_H = W
    d_C = H * W
    d_N = C * H * W
    d_G = C * H * W
    f_S = 1
    f_R = S
    f_C = R * S
    f_K = C * R * S
    f_G = K * C * R * S
    o_W = 1
    o_H = out_w
    o_K = out_h * out_w
    o_N = K * out_h * out_w
    o_G = K * out_h * out_w

    # V2 ABI flags for forward conv
    F_NKCHR_STRIDES = 1 << 9
    F_TENSOR_OFFSETS = 1 << 13
    F_USE_ACTIVATION_MODE = 1 << 14
    F_USE_EXTENDED_FLAGS_64 = 1 << 15
    flags64 = F_NKCHR_STRIDES | F_TENSOR_OFFSETS | F_USE_ACTIVATION_MODE | F_USE_EXTENDED_FLAGS_64

    # Pack V2 ABI kernel arguments (232 bytes)
    # The kernel argument buffer is packed as raw bytes
    arg_buf = bytearray(232)
    struct.pack_into("<I", arg_buf, 0, N)       # N
    struct.pack_into("<I", arg_buf, 4, C)       # C
    struct.pack_into("<I", arg_buf, 8, H)       # H
    struct.pack_into("<I", arg_buf, 12, W)      # W
    struct.pack_into("<I", arg_buf, 16, K)      # K
    struct.pack_into("<I", arg_buf, 20, n_groups) # n_groups
    struct.pack_into("<Q", arg_buf, 24, flags64)  # flags64
    # Get raw device pointers from NDBuffer objects
    input_ptr = input_dev._ptr if hasattr(input_dev, '_ptr') else int(input_dev)
    filter_ptr = filter_dev._ptr if hasattr(filter_dev, '_ptr') else int(filter_dev)
    output_ptr = output_dev._ptr if hasattr(output_dev, '_ptr') else int(output_dev)
    struct.pack_into("<Q", arg_buf, 32, input_ptr)   # data_addr
    struct.pack_into("<Q", arg_buf, 40, filter_ptr)   # filter_addr
    struct.pack_into("<Q", arg_buf, 48, output_ptr)   # output_addr
    struct.pack_into("<Q", arg_buf, 56, 0)      # reserved
    struct.pack_into("<I", arg_buf, 64, R)      # R
    struct.pack_into("<I", arg_buf, 68, S)      # S
    struct.pack_into("<i", arg_buf, 72, pad_h)  # pad_h
    struct.pack_into("<i", arg_buf, 76, pad_w)  # pad_w
    struct.pack_into("<I", arg_buf, 80, out_h)  # out_h
    struct.pack_into("<I", arg_buf, 84, out_w)  # out_w
    # bias_addr = 0 (offset 88)
    struct.pack_into("<f", arg_buf, 96, 1.0)    # alpha
    struct.pack_into("<f", arg_buf, 100, 0.0)   # beta
    # offsets all 0 (104-135)
    # Data strides
    struct.pack_into("<I", arg_buf, 136, d_N)
    struct.pack_into("<I", arg_buf, 140, d_C)
    struct.pack_into("<I", arg_buf, 144, d_H)
    # 148 reserved
    # Filter strides
    struct.pack_into("<I", arg_buf, 152, f_K)
    struct.pack_into("<I", arg_buf, 156, f_C)
    struct.pack_into("<I", arg_buf, 160, f_R)
    # 164 reserved
    # Output strides
    struct.pack_into("<I", arg_buf, 168, o_N)
    struct.pack_into("<I", arg_buf, 172, o_K)
    struct.pack_into("<I", arg_buf, 176, o_H)
    # 180 reserved
    struct.pack_into("<I", arg_buf, 184, group)  # G
    struct.pack_into("<I", arg_buf, 188, d_G)    # d_G_stride
    struct.pack_into("<I", arg_buf, 192, f_G)    # f_G_stride
    struct.pack_into("<I", arg_buf, 196, o_G)    # o_G_stride
    struct.pack_into("<B", arg_buf, 200, 0)      # activation_mode (identity)
    struct.pack_into("<B", arg_buf, 201, 0)      # sync_limit
    struct.pack_into("<B", arg_buf, 202, 0)      # sync_period

    # Create launch parameters using HIP_LAUNCH_PARAM extra args
    arg_array = (ctypes.c_char * len(arg_buf)).from_buffer(arg_buf)

    # Build the extra params array:
    # [HIP_LAUNCH_PARAM_BUFFER_POINTER, &arg_buf,
    #  HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END]
    HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
    HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
    HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)
    arg_size = ctypes.c_size_t(len(arg_buf))

    extra = (ctypes.c_void_p * 5)(
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        ctypes.cast(arg_array, ctypes.c_void_p),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p),
        HIP_LAUNCH_PARAM_END,
    )

    block_size = 768  # Rage on gfx942
    grid_x = n_groups * group

    print(f"Launching: grid=({grid_x},1,1) block=({block_size},1,1)")
    print(f"  Config: N={N} C={C} H={H}x{W} K={K} R={R}x{S} pad={pad_h} out={out_h}x{out_w}")

    # Warmup
    for _ in range(5):
        hip_check(hip.hipModuleLaunchKernel(
            kernel,
            grid_x, 1, 1,  # grid
            block_size, 1, 1,  # block
            65536,  # shared mem (64KB)
            None,  # stream
            None,  # kernel params
            extra   # extra args
        ))
    hip_check(hip.hipDeviceSynchronize())

    # Benchmark
    n_iters = 100
    start_event = hip_check(hip.hipEventCreate())
    stop_event = hip_check(hip.hipEventCreate())

    hip_check(hip.hipEventRecord(start_event, None))
    for _ in range(n_iters):
        hip_check(hip.hipModuleLaunchKernel(
            kernel,
            grid_x, 1, 1,
            block_size, 1, 1,
            65536,
            None,
            None,
            extra
        ))
    hip_check(hip.hipEventRecord(stop_event, None))
    hip_check(hip.hipEventSynchronize(stop_event))

    elapsed_ms = hip_check(hip.hipEventElapsedTime(start_event, stop_event))
    avg_us = elapsed_ms * 1000.0 / n_iters

    # Compute TFlops
    flops = 2.0 * N * C * K * out_h * out_w * R * S
    tflops = flops / (avg_us * 1e-6) / 1e12

    print(f"\n=== Winograd Rage v4.9 Results ===")
    print(f"  Avg time: {avg_us:.2f} us")
    print(f"  TFlops:   {tflops:.2f}")

    # Cleanup
    hip_check(hip.hipFree(input_dev))
    hip_check(hip.hipFree(filter_dev))
    hip_check(hip.hipFree(output_dev))
    hip_check(hip.hipModuleUnload(module))
    hip_check(hip.hipEventDestroy(start_event))
    hip_check(hip.hipEventDestroy(stop_event))

    return avg_us, tflops

def main():
    parser = argparse.ArgumentParser(description="Winograd kernel benchmark")
    parser.add_argument("--arch", default="gfx942")
    parser.add_argument("--dtype", default="f16", choices=["f16", "f32", "bf16"])
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--c", type=int, default=64)
    parser.add_argument("--h", type=int, default=56)
    parser.add_argument("--w", type=int, default=56)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--r", type=int, default=3)
    parser.add_argument("--s", type=int, default=3)
    parser.add_argument("--n_groups", type=int, default=120)
    parser.add_argument("--kernel-dir", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "lib", "Dialect",
        "Rock", "Winograd", "kernels"))
    args = parser.parse_args()

    run_winograd_benchmark(args)

if __name__ == "__main__":
    main()
