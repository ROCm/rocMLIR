#!/usr/bin/env python3

import sys
import numpy as np
import argparse

# This is a tool to analyze a given rocMLIR configuration to understand its bank conflict profile.
# The tool works by simulating the threads of a single workgroup, executing the following:
# - Load from global memory
# - Store in LDS
# - Read from LDS
# This tool has some restrictions:
# - We only simulate mfma used in its reduction form
# - We only consider kpack>1


# Mfma configuration, see
# https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-set-architecture.pdf
# For now we only support mfma computing a "reduction", i.e., the two blocks represent two sets of
# Ks that are added together
class Mfma:
    # Mfma we are using
    def __init__(self, mfma_d, mfma_k, kbase, blocks):
        self.mfma_d = mfma_d
        self.mfma_k = mfma_k
        self.kbase = kbase
        self.blocks = blocks

    @staticmethod
    def parse_mfma(str_mfma):
        list_mfma = str_mfma.split(",")
        mfma_mn = int(list_mfma[0])
        mfma_k = int(list_mfma[1])
        kbase = int(list_mfma[2])
        blocks = int(list_mfma[3])
        return Mfma(mfma_mn, mfma_k, kbase, blocks)


# Config class, this class compute all the necessary parameters throughout the simulation, taking into account:
# - The perf config, specified as (MPerBlock, NPerBlock, k_outer, MPerWave, NPerWave, k_pack)
# - The data_type (fp32, fp16, int8)
# - The mfma used (see above)
class Config:
    bank_size_bytes = 4

    @staticmethod
    def parse_config(str_config, data_type, mfma):
        list_config = str_config.split(",")
        block_size = int(list_config[0].strip())
        d = int(list_config[1].strip())
        k_outer = int(list_config[2].strip())
        d_per_wave = int(list_config[3].strip())
        k_pack = int(list_config[4].strip())
        return Config(block_size, d, d_per_wave, k_outer, k_pack, data_type, mfma)

    @staticmethod
    def compute_element_size_bytes(data_type):
        if data_type == "f32":
            return 4
        elif data_type == "f16":
            return 2
        elif data_type == "int8":
            return 1
        else:
            raise ValueError("Unsupported data type")

    def __init__(self, block_size, d, d_per_wave, k_outer, k_pack, data_type, mfma):
        # Store the original parameters
        self.d = d
        self.d_per_wave = d_per_wave
        self.k_outer = k_outer
        self.k_pack = k_pack
        self.mfma = mfma
        self.wave_size = 64

        # Store the parameters derived
        self.k = k_outer * k_pack
        self.element_size_bytes = Config.compute_element_size_bytes(data_type)
        self.block_size = block_size
        self.elementInABank = Config.bank_size_bytes // self.element_size_bytes
        self.copy_per_thread = d * self.k // self.block_size
        self.copyDPerThread = self.copy_per_thread // self.k_pack

    def __str__(self):
        return f"""
Perf Config:
BlockSize: {self.block_size}
DPerBlock: {self.d} elements
k_pack_per_block: {self.k_outer} elements
MperWave: {self.d_per_wave} elements
k_pack: {self.k_pack} elements

Parameters:
LDS size: {self.M*self.k*self.element_size_bytes} bytes
block_size: {self.blocksize} workitems
copy_per_thread : {self.copy_per_thread} elements
copy_per_threadM: {self.dataperthreadAlongM} elements
"""


# Given an LDS offset determine which bank it belongs to
def compute_bank(config, offset):
    offset_bytes = offset * config.element_size_bytes
    bank = (offset_bytes // config.bank_size_bytes) % 32
    row = (offset_bytes // config.bank_size_bytes) // 32
    return (row, bank)


# Apply a rotation on the "col" dimension, given that the condition is true
def rotate(config, row, col):
    new_col = (row + col) % config.d
    new_col = (new_col) * config.k_pack
    return new_col


def print_banks(config, waves_to_offset):
    print("LDS banks access per SIMD (16 threads):")
    for waveid in waves_to_offset:
        print("waveid:", waveid)
        for m in waves_to_offset[waveid]:
            print("m:", m)
            for k in waves_to_offset[waveid][m]:
                address = waves_to_offset[waveid][m][k]
                print("k:", k)

                for start_lane in range(0, config.wave_size, 16):
                    banks = []
                    conflicts = 32 * [0]
                    for lane in range(16):
                        bank = compute_bank(config, address[start_lane + lane])
                        banks.append((bank[0], bank[1]))
                    print(f"{banks} -> ", end="")
                    for b in banks:
                        conflicts[b[1]] += 1
                    print(f"{max(conflicts)}-way bank conflicts")


# This function is computing the write bank conflicts in LDS. The layout rocMLIR uses for LDS is the following: k_outer x MPerBLock x k_pack
def compute_write_bank_conflicts(config, is_k_major, disable_shuffle):
    # Each wave will write `copyDPerThread` data from global to LDS
    waves_to_banks = {}
    for wave in range(0, config.block_size // config.wave_size):
        waves_to_banks[wave] = {}
        for m in range(0, config.copyDPerThread):
            waves_to_banks[wave][m] = {}
            for k in range(0, 1):
                waves_to_banks[wave][m][k] = []

    # View of the lds in terms of global offset
    lds_to_offset = np.zeros([config.k_outer, config.d], dtype=np.int32)

    if is_k_major:
        for tid in range(0, config.block_size):
            for m in range(0, config.copyDPerThread):
                waveid = tid // config.wave_size
                tidk_pack = tid * config.k_pack

                # Compute global row/col and offset
                grow = m + config.copyDPerThread * (tidk_pack // config.k)
                gcol = tidk_pack % config.k
                goffset = grow * config.k + gcol

                # Compute LDS row/col and offset (note that we are transposing)
                lrow = (gcol // config.k_pack) % config.k_outer
                lcol = grow
                if not disable_shuffle:
                    lcol = rotate(config, lrow, lcol)
                loffset = lrow * config.d * config.k_pack + lcol

                # Fill data structures for analysis
                lds_to_offset[lrow, lcol // config.k_pack] = goffset
                waves_to_banks[waveid][m][0].append(loffset)
    else:
        for tid in range(0, config.block_size):
            for m in range(0, config.copyDPerThread):
                waveid = tid // config.wave_size

                # Compute global row/col and offset
                tid_d_per_thread = tid * config.copyDPerThread
                grow = (tid_d_per_thread // config.d) * config.k_pack
                gcol = (m + tid_d_per_thread) % config.d
                goffset = grow * config.d + gcol

                # Compute LDS row/col and offset (note that we are transposing)
                lrow = (grow // config.k_pack) % config.k_outer
                if disable_shuffle:
                    lcol = gcol * config.k_pack
                else:
                    d_threads = config.d // config.copyDPerThread
                    lcol = ((m * d_threads + tid) % config.d) * config.k_pack

                loffset = lrow * config.d * config.k_pack + lcol

                # Fill data structures for analysis
                lds_to_offset[lrow, lcol // config.k_pack] = goffset
                waves_to_banks[waveid][m][0].append(loffset)
    return (waves_to_banks, lds_to_offset)


# This function is computing the read bank conflicts in LDS. The layout rocMLIR uses for LDS is the following: k_outer x MPerBLock x k_pack
# Remember that the way each thread reads from LDS is different from the way they write to LDS. The indices accessed during a read by a
# thread also depends on the type of mfma used
def compute_read_bank_conflicts(config, is_k_major, disable_shuffle):
    kpackpermfma = config.k_outer // config.mfma.blocks
    d_repeats = config.d_per_wave // config.mfma.mfma_d

    waves_to_banks = {}
    for wave in range(0, config.block_size // config.wave_size):
        waves_to_banks[wave] = {}
        for m in range(0, d_repeats):
            waves_to_banks[wave][m] = {}
            for k in range(0, kpackpermfma):
                waves_to_banks[wave][m][k] = []

    # This is taking into account the MFMA layout. We consider only
    # reductions.
    for tid in range(0, config.block_size):
        waveid = tid // config.wave_size
        laneid = tid % config.wave_size
        for m in range(0, d_repeats):
            for k in range(0, kpackpermfma):
                loffset = laneid % config.mfma.mfma_d  # mOffset
                loffset += (
                    (laneid // config.mfma.mfma_d) * config.mfma.blocks * config.d)  # kOffset
                loffset += config.mfma.mfma_d * (waveid // d_repeats)  # nRepeat offset
                loffset += m * config.wave_size  # mRepeat offset
                loffset += k * config.d  # kRepeat offset

                # At this point the offset is a merge of [k_outer, M]
                # Let's unmerge it to compute the coordinate [k_outer, M]
                # apply the bankconflicts formula and combine it back
                lcol = loffset % config.d
                lrow = loffset // config.d
                if (not disable_shuffle) and is_k_major:
                    lcol = rotate(config, lrow, lcol)
                loffset = lrow * config.d + lcol
                waves_to_banks[waveid][m][k].append(loffset * config.k_pack)
    return waves_to_banks


def main(args=None):
    parser = argparse.ArgumentParser(
        prog="Bank Conflicts simulator",
        allow_abbrev=False,
    )

    # Example usage:
    # ./bank_conflicts.py --perf_config="256,128,4,64,4" --data-type=f32 --mfma="32,32,2,2"
    # ./bank_conflicts.py --perf_config="256,128,4,64,8" --data-type=f16 --mfma="32,32,8,2"
    # ./bank_conflicts.py --perf_config="256,64,8,32,8" --data-type=f16 --mfma="32,32,8,2"
    # ./bank_conflicts.py --perf_config="256,64,8,32,4" --data-type=f16 --mfma="32,32,8,2"
    parser.add_argument(
        "--kmajor",
        action="store_true",
        help="the global matrix is k-major (nxk or mxk)",
    )
    parser.add_argument("--read-conflicts", action="store_true", help="compute read conflicts")
    parser.add_argument(
        "--perf_config",
        type=str,
        help="perf configuration (block_size, D, k_outer, d_per_wave, k_pack)",
    )
    parser.add_argument("--mfma",
                        type=str,
                        help="mfma configuration (mfma_d, mfma_k, kBase, mfmaBlocks)")
    parser.add_argument("--data-type", type=str, default="f16", help="data type")
    parser.add_argument(
        "--show-offsets",
        action="store_true",
        help="print the global offsets stored in LDS",
    )
    parser.add_argument(
        "--show-conflicts",
        action="store_true",
        help="print the banks accessed by each thread",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="show the parameters for the configuration we are working on",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="don't use any bank-conflicts reduction algorithm",
    )

    parsed_args = parser.parse_args(args)
    if parsed_args.show_offsets and parsed_args.read_conflicts:
        raise ValueError("Offset can be only printed when evaluating the write-conflicts")

    # Config
    mfma = Mfma.parse_mfma(parsed_args.mfma)
    config = Config.parse_config(parsed_args.perf_config, parsed_args.data_type, mfma)

    # Show the configuration
    if parsed_args.show_config:
        print("Configuration")
        print(config)

    if parsed_args.read_conflicts:
        (waves_to_offset, _) = compute_read_bank_conflicts(config, parsed_args.kmajor,
                                                           parsed_args.no_shuffle)
    else:
        (waves_to_offset, _) = compute_write_bank_conflicts(config, parsed_args.kmajor,
                                                            parsed_args.no_shuffle)

    if parsed_args.show_conflicts:
        print_banks(config, waves_to_offset)

    if parsed_args.show_offsets:
        (_, lds_to_offset) = compute_write_bank_conflicts(config, parsed_args.kmajor,
                                                          parsed_args.no_shuffle)
        print("\nGlobal offset distribution (offsets are in element)")
        print(lds_to_offset)
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)


if __name__ == "__main__":
    sys.exit(main())
