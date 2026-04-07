#!/usr/bin/env python3
"""Winograd vs GEMM performance comparison.

For each winograd-eligible conv config from tier1-conv-configs, runs
rocmlir-tuning-driver to explore both GEMM and Winograd perf configs,
then generates a comparison CSV showing the best of each.

Usage:
    ROCR_VISIBLE_DEVICES=1 python3 winograd_comparison.py \
        --build-dir /path/to/rocMLIR/build \
        --configs-file configs/tier1-conv-configs \
        --tuning-space quick \
        --output winograd_comparison.csv
"""

import argparse
import csv
import getopt
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class ConvParams:
    dtype_token: str
    datatype: str
    direction: str
    direction_flag: int
    n: int
    c: int
    hi: int
    wi: int
    k: int
    y: int
    x: int
    stride_h: int
    stride_w: int
    padding_h: int
    padding_w: int
    dilation_h: int
    dilation_w: int
    group: int
    filter_layout: str
    input_layout: str
    output_layout: str

    @property
    def ho(self):
        return math.floor(
            (self.hi + self.padding_h * 2 - (self.y - 1) * self.dilation_h - 1) / self.stride_h
        ) + 1

    @property
    def wo(self):
        return math.floor(
            (self.wi + self.padding_w * 2 - (self.x - 1) * self.dilation_w - 1) / self.stride_w
        ) + 1

    def compute_tflops(self, ns: float) -> float:
        if math.isnan(ns) or ns <= 0:
            return float('nan')
        return (
            2.0 * self.n * (self.c // self.group) * self.k * self.ho * self.wo * self.y * self.x
        ) / (ns * 1e-9) / 1e12

    @property
    def problem_key(self) -> str:
        return (
            f"{self.dtype_token} -F {self.direction_flag} "
            f"N{self.n}_C{self.c}_H{self.hi}_W{self.wi}_K{self.k}_"
            f"Y{self.y}_X{self.x}_SH{self.stride_h}_SW{self.stride_w}_"
            f"PH{self.padding_h}_PW{self.padding_w}_G{self.group}"
        )

    def rocmlir_gen_args(self, arch: str, num_cu: int) -> List[str]:
        dtype_map = {'conv': 'f32', 'convfp16': 'f16', 'convbfp16': 'bf16', 'convint8': 'i8'}
        dt = dtype_map[self.dtype_token]
        dir_map = {'fwd': 'conv', 'bwd': 'conv_bwd_data', 'wrw': 'conv_bwd_weight'}
        op = dir_map[self.direction]
        return [
            f'--operation={op}', '-t', dt,
            '--arch', arch, '--num_cu', str(num_cu),
            '--fil_layout', self.filter_layout,
            '--in_layout', self.input_layout,
            '--out_layout', self.output_layout,
            '--batchsize', str(self.n),
            '--in_channels', str(self.c),
            '--in_h', str(self.hi),
            '--in_w', str(self.wi),
            '--out_channels', str(self.k),
            '--fil_h', str(self.y),
            '--fil_w', str(self.x),
            '--dilation_h', str(self.dilation_h),
            '--dilation_w', str(self.dilation_w),
            '--conv_stride_h', str(self.stride_h),
            '--conv_stride_w', str(self.stride_w),
            '--padding_h', str(self.padding_h),
            '--padding_w', str(self.padding_w),
            '--groupsize', str(self.group),
        ]


def parse_config_line(line: str) -> Optional[ConvParams]:
    tokens = line.strip().split()
    if not tokens:
        return None
    dtype_token = tokens[0]
    dtype_map = {'conv': 'f32', 'convfp16': 'f16'}
    if dtype_token not in dtype_map:
        return None

    try:
        opts, _ = getopt.getopt(tokens[1:], "F:f:I:O:n:c:H:W:k:y:x:p:q:l:j:u:v:g:m:t:")
    except getopt.GetoptError:
        return None

    params: Dict[str, str] = {}
    for opt, arg in opts:
        params[opt] = arg

    direction_flag = int(params.get('-F', '0'))
    direction_map = {1: 'fwd', 2: 'bwd', 4: 'wrw'}
    direction = direction_map.get(direction_flag)
    if direction is None:
        return None

    return ConvParams(
        dtype_token=dtype_token,
        datatype=dtype_map[dtype_token],
        direction=direction,
        direction_flag=direction_flag,
        n=int(params.get('-n', '1')),
        c=int(params.get('-c', '1')),
        hi=int(params.get('-H', '1')),
        wi=int(params.get('-W', '1')),
        k=int(params.get('-k', '1')),
        y=int(params.get('-y', '1')),
        x=int(params.get('-x', '1')),
        stride_h=int(params.get('-u', '1')),
        stride_w=int(params.get('-v', '1')),
        padding_h=int(params.get('-p', '0')),
        padding_w=int(params.get('-q', '0')),
        dilation_h=int(params.get('-l', '1')),
        dilation_w=int(params.get('-j', '1')),
        group=int(params.get('-g', '1')),
        filter_layout=params.get('-f', 'NCHW').lower(),
        input_layout=params.get('-I', 'NCHW').lower(),
        output_layout=params.get('-O', 'NCHW').lower(),
    )


def is_winograd_eligible(p: ConvParams) -> bool:
    if p.y != 3 or p.x != 3:
        return False
    if p.dilation_h != 1 or p.dilation_w != 1:
        return False
    if p.group != 1:
        return False
    if p.stride_h > 2 or p.stride_w > 2:
        return False
    if p.direction_flag not in (1, 2):
        return False
    if p.dtype_token not in ('conv', 'convfp16'):
        return False
    return True


def filter_eligible_configs(config_path: str) -> List[Tuple[str, ConvParams]]:
    eligible = []
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            params = parse_config_line(line)
            if params and is_winograd_eligible(params):
                eligible.append((line, params))
    return eligible


@dataclass
class TuningEntry:
    perfconfig: str
    time_ns: float
    tflops: float
    is_winograd: bool


def run_tuning(
    config_line: str,
    params: ConvParams,
    rocmlir_gen: str,
    tuning_driver: str,
    arch: str,
    num_cu: int,
    num_chiplets: int,
    tuning_space: str,
    timeout: int,
) -> List[TuningEntry]:
    gen_args = [rocmlir_gen] + params.rocmlir_gen_args(arch, num_cu)
    driver_args = [
        tuning_driver,
        f'--tuning-space={tuning_space}',
        '--num-iterations=10',
        '--warmup-iterations=1',
        '--use-median',
        '--sleep-us=100',
    ]

    env = os.environ.copy()
    kernel_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'lib', 'Dialect', 'Rock', 'Winograd', 'kernels')
    kernel_dir = os.path.normpath(kernel_dir)
    if os.path.isdir(kernel_dir):
        env['ROCMLIR_WINOGRAD_KERNEL_DIR'] = kernel_dir

    try:
        p_gen = subprocess.Popen(gen_args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        p_driver = subprocess.Popen(
            driver_args, stdin=p_gen.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )
        p_gen.stdout.close()

        stdout, stderr = p_driver.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p_driver.kill()
        p_gen.kill()
        p_driver.communicate()
        print(f"  TIMEOUT after {timeout}s", file=sys.stderr)
        return []
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return []
    finally:
        for p in [p_gen, p_driver]:
            try:
                p.kill()
            except Exception:
                pass

    if p_driver.returncode != 0:
        print(f"  tuning-driver failed (rc={p_driver.returncode})", file=sys.stderr)
        if stderr:
            for l in stderr.decode('utf-8', errors='replace').strip().splitlines()[:5]:
                print(f"    {l}", file=sys.stderr)
        return []

    entries = []
    for line in stdout.decode('utf-8', errors='replace').splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        perfconfig = parts[0]
        time_str = parts[-1]
        try:
            if time_str == "N/A":
                tflops = float('nan')
                entries.append(TuningEntry(
                    perfconfig=perfconfig, time_ns=float('nan'),
                    tflops=tflops, is_winograd=perfconfig.startswith('winograd:'),
                ))
                continue
            ns = float(time_str)
        except ValueError:
            continue

        tflops = params.compute_tflops(ns)
        entries.append(TuningEntry(
            perfconfig=perfconfig,
            time_ns=ns,
            tflops=tflops,
            is_winograd=perfconfig.startswith('winograd:'),
        ))

    return entries


def find_best(entries: List[TuningEntry], winograd_only: bool) -> Optional[TuningEntry]:
    filtered = [e for e in entries if e.is_winograd == winograd_only and not math.isnan(e.tflops)]
    if not filtered:
        return None
    return max(filtered, key=lambda e: e.tflops)


def detect_gpu(gpu_id: int) -> Tuple[str, int, int]:
    env = os.environ.copy()
    env['ROCR_VISIBLE_DEVICES'] = str(gpu_id)
    env.pop('HIP_VISIBLE_DEVICES', None)

    try:
        out = subprocess.check_output(
            ['/opt/rocm/bin/rocminfo'], stderr=subprocess.PIPE, env=env
        ).decode('utf-8')
    except Exception as e:
        print(f"Failed to run rocminfo: {e}", file=sys.stderr)
        sys.exit(1)

    arch = None
    num_cu = None
    found_gpu = False
    for line in out.splitlines():
        if 'Name:' in line and 'gfx' in line:
            arch_str = line.split('Name:')[1].strip()
            if arch_str.startswith('amdgcn'):
                parts = arch_str.split('--')
                if len(parts) >= 2 and 'gfx' in parts[-1]:
                    arch = parts[-1].strip()
            elif arch_str.startswith('gfx'):
                arch = arch_str
            if arch and not arch.endswith('generic'):
                found_gpu = True
        if found_gpu and 'Compute Unit:' in line:
            num_cu = int(line.split('Compute Unit:')[1].strip())
            break

    if not arch or not num_cu:
        print("Failed to detect GPU arch/CU count", file=sys.stderr)
        sys.exit(1)

    num_chiplets = 1
    if 'gfx942' in arch and num_cu == 304:
        num_chiplets = 8
    elif 'gfx942' in arch and num_cu == 80:
        num_chiplets = 4
    elif 'gfx950' in arch:
        num_chiplets = 8

    return arch, num_cu, num_chiplets


def load_completed(output_path: str) -> set:
    if not os.path.exists(output_path):
        return set()
    completed = set()
    try:
        with open(output_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row.get('Config', ''))
    except Exception:
        pass
    return completed


CSV_COLUMNS = [
    'Config', 'Dtype', 'Direction', 'N', 'C', 'H', 'W', 'K', 'Y', 'X',
    'StrideH', 'StrideW', 'PadH', 'PadW',
    'BestGEMM_PerfConfig', 'BestGEMM_TFlops',
    'BestWino_PerfConfig', 'BestWino_TFlops',
    'Overall_Winner', 'Speedup_vs_GEMM',
    'NumGEMM_Tried', 'NumWino_Tried',
]


def main():
    parser = argparse.ArgumentParser(description='Winograd vs GEMM performance comparison')
    parser.add_argument('--build-dir', default=None,
                        help='rocMLIR build directory (auto-detected if not set)')
    parser.add_argument('--configs-file',
                        default=os.path.join(os.path.dirname(__file__), 'configs', 'tier1-conv-configs'))
    parser.add_argument('--tuning-space', default='quick', choices=['quick', 'full', 'exhaustive'])
    parser.add_argument('--output', '-o', default='winograd_comparison.csv')
    parser.add_argument('--gpu', type=int, default=1, help='GPU device ID')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout per config (seconds)')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of configs (0=all)')
    parser.add_argument('--resume', action='store_true', help='Resume from existing output file')
    args = parser.parse_args()

    if args.build_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.normpath(os.path.join(script_dir, '..', '..', '..', 'build'))
        if os.path.isdir(candidate):
            args.build_dir = candidate
        else:
            print("Could not auto-detect build dir. Use --build-dir.", file=sys.stderr)
            sys.exit(1)

    rocmlir_gen = os.path.join(args.build_dir, 'bin', 'rocmlir-gen')
    tuning_driver = os.path.join(args.build_dir, 'bin', 'rocmlir-tuning-driver')
    for tool in [rocmlir_gen, tuning_driver]:
        if not os.path.isfile(tool):
            print(f"Tool not found: {tool}", file=sys.stderr)
            sys.exit(1)

    os.environ['ROCR_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.pop('HIP_VISIBLE_DEVICES', None)

    print(f"Detecting GPU {args.gpu}...")
    arch, num_cu, num_chiplets = detect_gpu(args.gpu)
    print(f"  arch={arch}, num_cu={num_cu}, num_chiplets={num_chiplets}")

    print(f"Loading configs from {args.configs_file}...")
    eligible = filter_eligible_configs(args.configs_file)
    print(f"  {len(eligible)} winograd-eligible configs found")

    if args.limit > 0:
        eligible = eligible[:args.limit]
        print(f"  Limited to first {args.limit}")

    completed = set()
    write_header = True
    if args.resume:
        completed = load_completed(args.output)
        if completed:
            write_header = False
            print(f"  Resuming: {len(completed)} already completed")

    if write_header:
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()

    total = len(eligible)
    wino_wins = 0
    gemm_wins = 0
    ties = 0
    wino_applicable = 0
    errors = 0
    start_time = time.time()

    for idx, (config_line, params) in enumerate(eligible):
        if config_line in completed:
            continue

        elapsed = time.time() - start_time
        eta = (elapsed / max(idx, 1)) * (total - idx) if idx > 0 else 0
        print(f"\n[{idx+1}/{total}] ETA: {eta/60:.0f}min | {params.problem_key}", flush=True)

        entries = run_tuning(
            config_line, params, rocmlir_gen, tuning_driver,
            arch, num_cu, num_chiplets, args.tuning_space, args.timeout,
        )

        if not entries:
            errors += 1
            print(f"  No valid results", file=sys.stderr)
            row = {c: '' for c in CSV_COLUMNS}
            row['Config'] = config_line
            row['Dtype'] = params.datatype
            row['Direction'] = params.direction
            row.update({'N': params.n, 'C': params.c, 'H': params.hi, 'W': params.wi,
                        'K': params.k, 'Y': params.y, 'X': params.x,
                        'StrideH': params.stride_h, 'StrideW': params.stride_w,
                        'PadH': params.padding_h, 'PadW': params.padding_w})
            row['Overall_Winner'] = 'ERROR'
            with open(args.output, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)
            continue

        gemm_entries = [e for e in entries if not e.is_winograd]
        wino_entries = [e for e in entries if e.is_winograd]

        best_gemm = find_best(entries, winograd_only=False)
        best_wino = find_best(entries, winograd_only=True)

        winner = 'GEMM'
        speedup = 1.0
        if best_wino and best_gemm:
            wino_applicable += 1
            if best_wino.tflops > best_gemm.tflops:
                winner = 'WINOGRAD'
                speedup = best_wino.tflops / best_gemm.tflops
                wino_wins += 1
            else:
                speedup = best_gemm.tflops / (best_wino.tflops if best_wino.tflops > 0 else 1)
                gemm_wins += 1
        elif best_gemm:
            gemm_wins += 1
        elif best_wino:
            winner = 'WINOGRAD'
            wino_wins += 1
            wino_applicable += 1

        gemm_tflops_str = f"{best_gemm.tflops:.4f}" if best_gemm else 'N/A'
        wino_tflops_str = f"{best_wino.tflops:.4f}" if best_wino else 'N/A'
        speedup_str = f"{speedup:.3f}" if best_wino and best_gemm else 'N/A'

        print(f"  GEMM: {gemm_tflops_str} TFlops ({len(gemm_entries)} tried)  "
              f"Wino: {wino_tflops_str} TFlops ({len(wino_entries)} tried)  "
              f"Winner: {winner}" +
              (f"  Speedup: {speedup:.2f}x" if winner == 'WINOGRAD' else ''),
              flush=True)

        row = {
            'Config': config_line,
            'Dtype': params.datatype,
            'Direction': params.direction,
            'N': params.n, 'C': params.c, 'H': params.hi, 'W': params.wi,
            'K': params.k, 'Y': params.y, 'X': params.x,
            'StrideH': params.stride_h, 'StrideW': params.stride_w,
            'PadH': params.padding_h, 'PadW': params.padding_w,
            'BestGEMM_PerfConfig': best_gemm.perfconfig if best_gemm else '',
            'BestGEMM_TFlops': gemm_tflops_str,
            'BestWino_PerfConfig': best_wino.perfconfig if best_wino else '',
            'BestWino_TFlops': wino_tflops_str,
            'Overall_Winner': winner,
            'Speedup_vs_GEMM': speedup_str,
            'NumGEMM_Tried': len(gemm_entries),
            'NumWino_Tried': len(wino_entries),
        }

        with open(args.output, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"WINOGRAD vs GEMM COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"Arch: {arch} ({num_cu} CUs)")
    print(f"Total eligible configs: {total}")
    print(f"Winograd applicable (solver found entries): {wino_applicable}")
    print(f"Winograd wins: {wino_wins}")
    print(f"GEMM wins: {gemm_wins}")
    print(f"Errors/skipped: {errors}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {args.output}")

    if os.path.exists(args.output):
        print(f"\n--- Per-config results ---")
        with open(args.output, newline='') as f:
            reader = csv.DictReader(f)
            wino_speedups = []
            for row in reader:
                if row.get('Overall_Winner') == 'WINOGRAD':
                    try:
                        wino_speedups.append(float(row['Speedup_vs_GEMM']))
                    except (ValueError, KeyError):
                        pass
            if wino_speedups:
                import statistics
                print(f"Winograd speedup (where it wins):")
                print(f"  min:    {min(wino_speedups):.3f}x")
                print(f"  max:    {max(wino_speedups):.3f}x")
                print(f"  mean:   {statistics.mean(wino_speedups):.3f}x")
                print(f"  median: {statistics.median(wino_speedups):.3f}x")


if __name__ == '__main__':
    main()
