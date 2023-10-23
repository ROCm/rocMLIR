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
    def __init__(self, mfmaD, mfmaK, kbase, blocks):
        self.mfmaD = mfmaD
        self.mfmaK = mfmaK
        self.kbase = kbase
        self.blocks = blocks

    def parseMfma(strMfma):
        list_mfma = strMfma.split(",")
        mfmaMN = int(list_mfma[0])
        mfmaK = int(list_mfma[1])
        kbase = int(list_mfma[2])
        blocks = int(list_mfma[3])
        return Mfma(mfmaMN, mfmaK, kbase, blocks)


# Config class, this class compute all the necessary parameters throughout the simulation, taking into account:
# - The perf config, specified as (MPerBlock, NPerBlock, Kouter, MPerWave, NPerWave, Kpack)
# - The dataType (fp32, fp16, int8)
# - The mfma used (see above)
class Config:
    bankSizeBytes = 4

    def parseConfig(strConfig, dataType, mfma):
        list_config = strConfig.split(",")
        blockSize = int(list_config[0].strip())
        D = int(list_config[1].strip())
        Kouter = int(list_config[2].strip())
        DPerWave = int(list_config[3].strip())
        Kpack = int(list_config[4].strip())
        return Config(blockSize, D, DPerWave, Kouter, Kpack, dataType, mfma)

    def computeElementSizeBytes(dataType):
        if dataType == "f32":
            return 4
        elif dataType == "f16":
            return 2
        elif dataType == "int8":
            return 1
        else:
            raise ValueError("Unsupported data type")

    def __init__(self, blockSize, D, DPerWave, Kouter, Kpack, dataType, mfma):
        # Store the original parameters
        self.D = D
        self.DPerWave = DPerWave
        self.Kouter = Kouter
        self.Kpack = Kpack
        self.mfma = mfma
        self.waveSize = 64

        # Store the parameters derived
        self.K = Kouter * Kpack
        self.elementSizeBytes = Config.computeElementSizeBytes(dataType)
        self.blockSize = blockSize
        self.elementInABank = Config.bankSizeBytes // self.elementSizeBytes
        self.copyPerThread = D * self.K // self.blockSize
        self.copyDPerThread = self.copyPerThread // self.Kpack

    def __str__(self):
        return f"""
Perf Config:
BlockSize: {self.blockSize}
DPerBlock: {self.D} elements
KpackPerBlock: {self.Kouter} elements
MperWave: {self.DPerWave} elements
Kpack: {self.Kpack} elements

Parameters:
LDS size: {self.M*self.K*self.elementSizeBytes} bytes
blockSize: {self.blocksize} workitems
copyPerThread : {self.copyPerThread} elements
copyPerThreadM: {self.dataperthreadAlongM} elements
"""


# Given an LDS offset determine which bank it belongs to
def computeBank(config, offset):
    offsetBytes = offset * config.elementSizeBytes
    bank = (offsetBytes // config.bankSizeBytes) % 32
    row = (offsetBytes // config.bankSizeBytes) // 32
    return (row, bank)


# Apply a rotation on the "col" dimension, given that the condition is true
def rotate(config, row, col):
    newCol = (row + col) % config.D
    newCol = (newCol) * config.Kpack
    return newCol


def printBanks(config, wavesToOffset):
    print("LDS banks access per SIMD (16 threads):")
    for waveid in wavesToOffset:
        print("waveid:", waveid)
        for m in wavesToOffset[waveid]:
            print("m:", m)
            for k in wavesToOffset[waveid][m]:
                address = wavesToOffset[waveid][m][k]
                print("k:", k)

                for l in range(0, config.waveSize, 16):
                    banks = []
                    conflicts = 32 * [0]
                    for lane in range(16):
                        bank = computeBank(config, address[l + lane])
                        banks.append((bank[0], bank[1]))
                    print(f"{banks} -> ", end="")
                    for b in banks:
                        conflicts[b[1]] += 1
                    print(f"{max(conflicts)}-way bank conflicts")


# This function is computing the write bank conflicts in LDS. The layout rocMLIR uses for LDS is the following: Kouter x MPerBLock x Kpack
def computeWriteBankConflicts(config, isKMajor, disableShuffle):
    # Each wave will write `copyDPerThread` data from global to LDS
    wavesToBanks = {}
    for wave in range(0, config.blockSize // config.waveSize):
        wavesToBanks[wave] = {}
        for m in range(0, config.copyDPerThread):
            wavesToBanks[wave][m] = {}
            for k in range(0, 1):
                wavesToBanks[wave][m][k] = []

    # View of the lds in terms of global offset
    ldsToOffset = np.zeros([config.Kouter, config.D], dtype=np.int32)

    if isKMajor:
        for tid in range(0, config.blockSize):
            for m in range(0, config.copyDPerThread):
                waveid = tid // config.waveSize
                tidKpack = tid * config.Kpack

                # Compute global row/col and offset
                grow = m + config.copyDPerThread * (tidKpack // config.K)
                gcol = tidKpack % config.K
                goffset = grow * config.K + gcol

                # Compute LDS row/col and offset (note that we are transposing)
                lrow = (gcol // config.Kpack) % config.Kouter
                lcol = grow
                if not disableShuffle:
                    lcol = rotate(config, lrow, lcol)
                loffset = lrow * config.D * config.Kpack + lcol

                # Fill data structures for analysis
                ldsToOffset[lrow, lcol // config.Kpack] = goffset
                wavesToBanks[waveid][m][0].append(loffset)
    else:
        for tid in range(0, config.blockSize):
            for m in range(0, config.copyDPerThread):
                waveid = tid // config.waveSize

                # Compute global row/col and offset
                tidDPerThread = tid * config.copyDPerThread
                grow = (tidDPerThread // config.D) * config.Kpack
                gcol = (m + tidDPerThread) % config.D
                goffset = grow * config.D + gcol

                # Compute LDS row/col and offset (note that we are transposing)
                lrow = (grow // config.Kpack) % config.Kouter
                if disableShuffle:
                    lcol = gcol * config.Kpack
                else:
                    dThreads = config.D // config.copyDPerThread
                    lcol = ((m * dThreads + tid) % config.D) * config.Kpack

                loffset = lrow * config.D * config.Kpack + lcol

                # Fill data structures for analysis
                ldsToOffset[lrow, lcol // config.Kpack] = goffset
                wavesToBanks[waveid][m][0].append(loffset)
    return (wavesToBanks, ldsToOffset)


# This function is computing the read bank conflicts in LDS. The layout rocMLIR uses for LDS is the following: Kouter x MPerBLock x Kpack
# Remember that the way each thread reads from LDS is different from the way they write to LDS. The indices accessed during a read by a
# thread also depends on the type of mfma used
def computeReadBankConflicts(config, isKMajor, disableShuffle):
    kpackpermfma = config.Kouter // config.mfma.blocks
    dRepeats = config.DPerWave // config.mfma.mfmaD

    wavesToBanks = {}
    for wave in range(0, config.blockSize // config.waveSize):
        wavesToBanks[wave] = {}
        for m in range(0, dRepeats):
            wavesToBanks[wave][m] = {}
            for k in range(0, kpackpermfma):
                wavesToBanks[wave][m][k] = []

    # This is taking into account the MFMA layout. We consider only
    # reductions.
    for tid in range(0, config.blockSize):
        waveid = tid // config.waveSize
        laneid = tid % config.waveSize
        for m in range(0, dRepeats):
            for k in range(0, kpackpermfma):
                loffset = laneid % config.mfma.mfmaD  # mOffset
                loffset += (
                    (laneid // config.mfma.mfmaD) * config.mfma.blocks * config.D
                )  # kOffset
                loffset += config.mfma.mfmaD * (waveid // dRepeats)  # nRepeat offset
                loffset += m * config.waveSize  # mRepeat offset
                loffset += k * config.D  # kRepeat offset

                # At this point the offset is a merge of [Kouter, M]
                # Let's unmerge it to compute the coordinate [Kouter, M]
                # apply the bankconflicts formula and combine it back
                lcol = loffset % config.D
                lrow = loffset // config.D
                if (not disableShuffle) and isKMajor:
                    lcol = rotate(config, lrow, lcol)
                loffset = lrow * config.D + lcol
                wavesToBanks[waveid][m][k].append(loffset * config.Kpack)
    return wavesToBanks


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
    parser.add_argument(
        "--read-conflicts", action="store_true", help="compute read conflicts"
    )
    parser.add_argument(
        "--perf_config",
        type=str,
        help="perf configuration (blockSize, D, Kouter, DPerWave, Kpack)",
    )
    parser.add_argument(
        "--mfma", type=str, help="mfma configuration (mfmaD, mfmaK, kBase, mfmaBlocks)"
    )
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
        raise ValueError(
            "Offset can be only printed when evaluating the write-conflicts"
        )

    # Config
    mfma = Mfma.parseMfma(parsed_args.mfma)
    config = Config.parseConfig(parsed_args.perf_config, parsed_args.data_type, mfma)

    # Show the configuration
    if parsed_args.show_config:
        print("Configuration")
        print(config)

    if parsed_args.read_conflicts:
        (wavesToOffset, _) = computeReadBankConflicts(
            config, parsed_args.kmajor, parsed_args.no_shuffle
        )
    else:
        (wavesToOffset, _) = computeWriteBankConflicts(
            config, parsed_args.kmajor, parsed_args.no_shuffle
        )

    if parsed_args.show_conflicts:
        printBanks(config, wavesToOffset)

    if parsed_args.show_offsets:
        (_, ldsToOffset) = computeWriteBankConflicts(
            config, parsed_args.kmajor, parsed_args.no_shuffle
        )
        print("\nGlobal offset distribution (offsets are in element)")
        print(ldsToOffset)
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)


if __name__ == "__main__":
    sys.exit(main())
