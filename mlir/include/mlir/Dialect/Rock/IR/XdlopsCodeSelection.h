//===- XdlopsCodeSelection.h - MLIR to C++ for Rock conversion
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements code selection logic for XDLOPS instructions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_XDLOPS_CODE_SELECTION_H
#define MLIR_XDLOPS_CODE_SELECTION_H

#include "mlir/Dialect/AMDGPU/AMDGPUDialect.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

using namespace mlir;

//===----------------------------------------------------------------------===//
// XDLOPS code selection logic.
//===----------------------------------------------------------------------===//
struct MFMAParams {
  // log_2 of the number of blocks to chop the lanes of A in to for broadcast
  uint32_t cbsz;
  // Which of the said groups should be broadcast
  uint32_t abid;
  amdgpu::MFMAPermB blgp;
};

struct XdlopsCodeSelection {
  int64_t mfmaNonKDim;
  // k for MFMA, and, by extension, for the xdlops as a whole
  int64_t k;
  int64_t blocksMfma;

  int64_t MRepeats;
  int64_t NRepeats;
  int64_t nResultVectors;

  VectorType vectorType;
  SmallVector<MFMAParams, 2> imms;
  Type argType;

  int64_t k_base;

  int64_t rowGroupSize;
  int64_t rowsPerMfmaOutput;
  int64_t blocksPerMfmaOutput;
  int64_t rowGroupsPerBlock;
  int64_t blocksInOutRegs;

  int64_t inputSpanLen;
  int64_t inputSpansPerMfmaIn;

  static void dumpUnsupported(const mlir::Type &dataType, int64_t MPerWave,
                              int64_t NPerWave) {
    llvm::errs() << "Unsupported case:\n";
    llvm::errs() << "MPerWave: " << MPerWave << "\n";
    llvm::errs() << "NPerWave: " << NPerWave << "\n";
    llvm::errs() << "dataType: ";
    dataType.dump();
    llvm::errs() << "\n";
  }

  static XdlopsCodeSelection get(mlir::Type dataType, int64_t MPerWave,
                                 int64_t NPerWave) {
    using amdgpu::MFMAPermB;
    constexpr int64_t waveSize = 64;
    // Determine which XDLOPS be used.
    int64_t mfmaNonKDim = 0, MRepeats = 0, NRepeats = 0, k = 0, blocksMfma = 0;
    SmallVector<MFMAParams, 2> imms;
    if (dataType.isF32()) {
      if (MPerWave == 128 && NPerWave == 64) {
        // instr = amdgpu::MFMAInstr::f32_32x32x1f32;
        mfmaNonKDim = 32;
        k = 1;
        blocksMfma = 2;

        MRepeats = 2;
        NRepeats = 1;
        imms.push_back({1, 0, MFMAPermB::none});
        imms.push_back({1, 1, MFMAPermB::none});
        imms.push_back({1, 0, MFMAPermB::none});
        imms.push_back({1, 1, MFMAPermB::none});
      } else if (MPerWave == 64 && NPerWave == 128) {
        // instr = amdgpu::MFMAInstr::f32_32x32x1f32;
        mfmaNonKDim = 32;
        k = 1;
        blocksMfma = 2;

        MRepeats = 1;
        NRepeats = 2;
        imms.push_back({1, 0, MFMAPermB::none});
        imms.push_back({1, 1, MFMAPermB::none});
        imms.push_back({1, 0, MFMAPermB::none});
        imms.push_back({1, 1, MFMAPermB::none});
      } else if (MPerWave == 64 && NPerWave == 64) {
        // instr = amdgpu::MFMAInstr::f32_32x32x1f32;
        mfmaNonKDim = 32;
        k = 1;
        blocksMfma = 2;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({1, 0, MFMAPermB::none});
        imms.push_back({1, 1, MFMAPermB::none});
      } else if (MPerWave == 64 && NPerWave == 32) {
        // instr = amdgpu::MFMAInstr::f32_32x32x1f32;
        mfmaNonKDim = 32;
        k = 1;
        blocksMfma = 2;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({0, 0, MFMAPermB::bcast_first_32});
      } else if (MPerWave == 32 && NPerWave == 64) {
        // instr = amdgpu::MFMAInstr::f32_32x32x1f32;
        mfmaNonKDim = 32;
        k = 1;
        blocksMfma = 2;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({1, 0, MFMAPermB::none});
      } else if (MPerWave == 64 && NPerWave == 16) {
        // instr = amdgpu::MFMAInstr::f32_16x16x1f32;
        mfmaNonKDim = 16;
        k = 1;
        blocksMfma = 4;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({0, 0, MFMAPermB::bcast_first_16});
      } else if (MPerWave == 16 && NPerWave == 64) {
        // instr = amdgpu::MFMAInstr::f32_16x16x1f32;
        mfmaNonKDim = 16;
        k = 1;
        blocksMfma = 4;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({2, 0, MFMAPermB::none});
      } else if (MPerWave == 8 && NPerWave == 64) {
        // instr = amdgpu::MFMAInstr::f32_4x4x1f32;
        mfmaNonKDim = 4;
        k = 1;
        blocksMfma = 16;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({4, 0, MFMAPermB::none});
        imms.push_back({4, 1, MFMAPermB::none});
      } else if (MPerWave == 4 && NPerWave == 64) {
        // instr = amdgpu::MFMAInstr::f32_4x4x1f32;
        mfmaNonKDim = 4;
        k = 1;
        blocksMfma = 16;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({4, 0, MFMAPermB::none});
      } else if (MPerWave == 32 && NPerWave == 32) {
        // instr = amdgpu::MFMAInstr::f32_32x32x2f32;
        mfmaNonKDim = 32;
        k = 2;
        blocksMfma = 1;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({0, 0, MFMAPermB::none});
      } else if (MPerWave == 16 && NPerWave == 16) {
        // instr = amdgpu::MFMAInstr::f32_16x16x4f32;
        mfmaNonKDim = 16;
        k = 4;
        blocksMfma = 1;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({0, 0, MFMAPermB::none});
      } else {
        dumpUnsupported(dataType, MPerWave, NPerWave);
        llvm_unreachable("Can't meaningfully continue xdlop generation");
      }
    } else if (dataType.isF16()) {
      if (MPerWave == 128 && NPerWave == 64) {
        // instr = amdgpu::MFMAInstr::f32_32x32x4f16;
        mfmaNonKDim = 32;
        k = 4;
        blocksMfma = 2;

        MRepeats = 2;
        NRepeats = 1;
        imms.push_back({1, 0, MFMAPermB::none});
        imms.push_back({1, 1, MFMAPermB::none});
        imms.push_back({1, 0, MFMAPermB::none});
        imms.push_back({1, 1, MFMAPermB::none});
      } else if (MPerWave == 64 && NPerWave == 128) {
        // instr = amdgpu::MFMAInstr::f32_32x32x4f16;
        mfmaNonKDim = 32;
        k = 4;
        blocksMfma = 2;

        MRepeats = 1;
        NRepeats = 2;
        imms.push_back({1, 0, MFMAPermB::none});
        imms.push_back({1, 1, MFMAPermB::none});
        imms.push_back({1, 0, MFMAPermB::none});
        imms.push_back({1, 1, MFMAPermB::none});
      } else if (MPerWave == 64 && NPerWave == 64) {
        // instr = amdgpu::MFMAInstr::f32_32x32x4f16;
        mfmaNonKDim = 32;
        k = 4;
        blocksMfma = 2;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({1, 0, MFMAPermB::none});
        imms.push_back({1, 1, MFMAPermB::none});
      } else if (MPerWave == 64 && NPerWave == 32) {
        // instr = amdgpu::MFMAInstr::f32_32x32x4f16;
        mfmaNonKDim = 32;
        k = 4;
        blocksMfma = 2;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({0, 0, MFMAPermB::bcast_first_32});
      } else if (MPerWave == 64 && NPerWave == 16) {
        // instr = amdgpu::MFMAInstr::f32_16x16x4f16;
        mfmaNonKDim = 16;
        k = 4;
        blocksMfma = 4;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({0, 0, MFMAPermB::bcast_first_16});
      } else if (MPerWave == 16 && NPerWave == 64) {
        // instr = amdgpu::MFMAInstr::f32_16x16x4f16;
        mfmaNonKDim = 16;
        k = 4;
        blocksMfma = 4;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({2, 0, MFMAPermB::none});
      } else if (MPerWave == 8 && NPerWave == 64) {
        // instr = amdgpu::MFMAInstr::f32_4x4x4f16;
        mfmaNonKDim = 4;
        k = 4;
        blocksMfma = 16;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({4, 0, MFMAPermB::none});
        imms.push_back({4, 1, MFMAPermB::none});
      } else if (MPerWave == 4 && NPerWave == 64) {
        // instr = amdgpu::MFMAInstr::f32_4x4x4f16;
        mfmaNonKDim = 4;
        k = 4;
        blocksMfma = 16;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({4, 0, MFMAPermB::none});
      } else if (MPerWave == 32 && NPerWave == 32) {
        // instr = amdgpu::MFMAInstr::f32_32x32x8f16;
        mfmaNonKDim = 32;
        k = 8;
        blocksMfma = 1;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({0, 0, MFMAPermB::none});
      } else if (MPerWave == 32 && NPerWave == 64) {
        // instr = amdgpu::MFMAInstr::f32_32x32x4f16;
        mfmaNonKDim = 32;
        k = 4;
        blocksMfma = 2;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({1, 0, MFMAPermB::none});
      } else if (MPerWave == 16 && NPerWave == 16) {
        // instr = amdgpu::MFMAInstr::f32_16x16x16f16;
        mfmaNonKDim = 16;
        k = 16;
        blocksMfma = 1;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({0, 0, MFMAPermB::none});
      } else {
        dumpUnsupported(dataType, MPerWave, NPerWave);
        llvm_unreachable("Can't meaningfully continue xdlop generation");
      }
    } else if (dataType.isBF16()) {
      if (MPerWave == 128 && NPerWave == 64) {
        // instr = amdgpu::MFMAInstr::f32_32x32x2bf16;
        mfmaNonKDim = 32;
        k = 2;
        blocksMfma = 2;

        MRepeats = 2;
        NRepeats = 1;
        imms.push_back({1, 0, MFMAPermB::none});
        imms.push_back({1, 1, MFMAPermB::none});
        imms.push_back({1, 0, MFMAPermB::none});
        imms.push_back({1, 1, MFMAPermB::none});
      } else if (MPerWave == 64 && NPerWave == 128) {
        // instr = amdgpu::MFMAInstr::f32_32x32x2bf16;
        mfmaNonKDim = 32;
        k = 2;
        blocksMfma = 2;

        MRepeats = 1;
        NRepeats = 2;
        imms.push_back({1, 0, MFMAPermB::none});
        imms.push_back({1, 1, MFMAPermB::none});
        imms.push_back({1, 0, MFMAPermB::none});
        imms.push_back({1, 1, MFMAPermB::none});
      } else if (MPerWave == 64 && NPerWave == 64) {
        // instr = amdgpu::MFMAInstr::f32_32x32x2bf16;
        mfmaNonKDim = 32;
        k = 2;
        blocksMfma = 2;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({1, 0, MFMAPermB::none});
        imms.push_back({1, 1, MFMAPermB::none});
      } else if (MPerWave == 64 && NPerWave == 32) {
        // instr = amdgpu::MFMAInstr::f32_32x32x2bf16;
        mfmaNonKDim = 32;
        k = 2;
        blocksMfma = 2;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({0, 0, MFMAPermB::bcast_first_32});
      } else if (MPerWave == 32 && NPerWave == 64) {
        // instr = amdgpu::MFMAInstr::f32_32x32x2bf16;
        mfmaNonKDim = 32;
        k = 2;
        blocksMfma = 2;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({1, 0, MFMAPermB::none});
      } else if (MPerWave == 64 && NPerWave == 16) {
        // instr = amdgpu::MFMAInstr::f32_16x16x2bf16;
        mfmaNonKDim = 16;
        k = 2;
        blocksMfma = 4;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({0, 0, MFMAPermB::bcast_first_16});
      } else if (MPerWave == 16 && NPerWave == 64) {
        // instr = amdgpu::MFMAInstr::f32_16x16x2bf16;
        mfmaNonKDim = 16;
        k = 2;
        blocksMfma = 4;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({2, 0, MFMAPermB::none});
      } else if (MPerWave == 8 && NPerWave == 64) {
        // instr = amdgpu::MFMAInstr::f32_4x4x2bf16;
        mfmaNonKDim = 4;
        k = 2;
        blocksMfma = 16;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({4, 0, MFMAPermB::none});
        imms.push_back({4, 1, MFMAPermB::none});
      } else if (MPerWave == 4 && NPerWave == 64) {
        // instr = amdgpu::MFMAInstr::f32_4x4x2bf16;
        mfmaNonKDim = 4;
        k = 2;
        blocksMfma = 16;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({4, 0, MFMAPermB::none});
      } else if (MPerWave == 32 && NPerWave == 32) {
        // instr = amdgpu::MFMAInstr::f32_32x32x4bf16;
        mfmaNonKDim = 32;
        k = 4;
        blocksMfma = 1;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({0, 0, MFMAPermB::none});
      } else if (MPerWave == 16 && NPerWave == 16) {
        // instr = amdgpu::MFMAInstr::f32_16x16x8bf16;
        mfmaNonKDim = 16;
        k = 8;
        blocksMfma = 1;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({0, 0, MFMAPermB::none});
      } else {
        dumpUnsupported(dataType, MPerWave, NPerWave);
        llvm_unreachable("Can't meaningfully continue xdlop generation");
      }
    } else if (dataType.isInteger(8)) {
      if (MPerWave == 32 && NPerWave == 32) {
        // instr = amdgpu::MFMAInstr::i32_32x32x8i8;
        mfmaNonKDim = 32;
        k = 8;
        blocksMfma = 1;

        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({0, 0, MFMAPermB::none});
      } else if (MPerWave == 16 && NPerWave == 16) {
        // instr = amdgpu::MFMAInstr::i32_16x16x16i8;
        mfmaNonKDim = 16;
        k = 16;
        blocksMfma = 1;
        MRepeats = 1;
        NRepeats = 1;
        imms.push_back({0, 0, MFMAPermB::none});
      } else {
        dumpUnsupported(dataType, MPerWave, NPerWave);
        llvm_unreachable("Can't meaningfully continue xdlop generation");
      }
    } else {
      llvm_unreachable("XDLOPs for unsupported data types should be rejected");
    }

    // Derived properties of the individual MFMA. These are computed here
    // and used in places throughout the code and may not all be needed.
    int64_t kPerMfmaInput =
        math_util::integer_divide_ceil(waveSize, mfmaNonKDim * blocksMfma);
    // k_base is the number of times you need to step in the k dimension on each
    // lane in a wave.
    int64_t k_base = k / kPerMfmaInput;

    // Number of logical values each thread needs to pass in to the MFMA in
    // order for the correct number of input values to be passed to the MFMA.
    int64_t nInputsToMfma = (mfmaNonKDim * blocksMfma * k) / waveSize;
    Type argType = nInputsToMfma == 1
                       ? dataType
                       : VectorType::get({nInputsToMfma}, dataType);

    int64_t nOutputsOfMfma =
        (mfmaNonKDim * mfmaNonKDim * blocksMfma) / waveSize;

    Builder builder(dataType.getContext());
    Type vectorElem;
    if (dataType.isa<IntegerType>())
      vectorElem = builder.getI32Type();
    else
      vectorElem = builder.getF32Type();
    VectorType vectorType = VectorType::get({nOutputsOfMfma}, vectorElem);

    constexpr int64_t rowGroupSize = 4;
    // The number of rows in each MFMA output item (usually a VGPR, except in
    // the case of double-precision). Note, a "row" is a complete output row of
    // one of blocks produced by the MFMA.

    // For most MFMAs, this is bounded by the number of retitions of n_mfma
    // that you can fit into a wave (ex. a 32x32x2 mfma can fit two rows per
    // result). However, in the case of 4x4xk (16 blocks) MFMAs, counting that
    // number would produce the inaccurate result 64 / 4 = 16 rows per output,
    // even though there is only one row available to place in each output
    // (remembering that the rows are first tiled into groups of group_size
    // outputs). Therefore, we need to impose the bound m_mfma / group_size on
    // the number of rows per output.
    int64_t rowsPerMfmaOutput = std::min(waveSize / /*n_mfma=*/mfmaNonKDim,
                                         /*m_mfma=*/mfmaNonKDim / rowGroupSize);
    // The number of blocks in each MFMA output. If rowsPerMfmaOutput followed
    // the typical case and was computed using waveSize / n_mfma, this will
    // be 1. However, in the 4x4 case, where we do have 16 blocks packed into
    // each output blocksPerOutput will be > 1 (namely 16).
    int64_t blocksPerMfmaOutput = math_util::integer_divide_ceil(
        waveSize, rowsPerMfmaOutput * /*n_mfma=*/mfmaNonKDim);
    // The number of register groups (of four rows) per block of output
    // Note that the inclusion of blocksPerOutput forces this value to be 1 in
    // the 4x4 case, as it should be.
    int64_t rowGroupsPerBlock = math_util::integer_divide_ceil(
        /*m_mfma=*/mfmaNonKDim,
        rowGroupSize * rowsPerMfmaOutput * blocksPerMfmaOutput);
    // Number of output blocks that can be accessed by going through the
    // registers on any given lane.
    int64_t blocksInOutRegs = blocksMfma / blocksPerMfmaOutput;

    // Properties of th selected bundle of MFMA instructions, or of how they
    // are used to implement GEMM.

    // Rename this field in future refactoring
    int64_t nResultVectors = imms.size();

    // Because the 4x4xk instructions are the only ones with blocksPerOutput > 1
    // and because they're only used for broadcast operations, we have
    // historically represented them as 4x64 operations that have one large
    // "block" instead of 16 tiny ones. So, the length of an input span
    // (row for A, column for B) is usually equal to mfmaNonKDim, but is
    // more generally equal to mfmaNonKDim * blocksPerOutput in order to
    // enable the math throughout our code to note break.
    int64_t inputSpanLen = mfmaNonKDim * blocksPerMfmaOutput;
    int64_t inputSpansPerMfmaIn = waveSize / inputSpanLen;

    // Populate result.
    XdlopsCodeSelection result;
    result.mfmaNonKDim = mfmaNonKDim;
    result.k = k;
    result.blocksMfma = blocksMfma;

    result.MRepeats = MRepeats;
    result.NRepeats = NRepeats;
    result.nResultVectors = nResultVectors;

    result.vectorType = vectorType;
    result.imms = imms;
    result.argType = argType;
    result.k_base = k_base;

    result.rowGroupSize = rowGroupSize;
    result.rowGroupsPerBlock = rowGroupsPerBlock;
    result.rowsPerMfmaOutput = rowsPerMfmaOutput;
    result.blocksPerMfmaOutput = blocksPerMfmaOutput;
    result.blocksInOutRegs = blocksInOutRegs;

    result.inputSpanLen = inputSpanLen;
    result.inputSpansPerMfmaIn = inputSpansPerMfmaIn;
    // llvm::errs() << "XDLOPS code selection result:\n";
    // llvm::errs() << "mfmaInstr: " << mfmaInstr << "\n";
    // llvm::errs() << "MPerXdlops: " << MPerXdlops << "\n";
    // llvm::errs() << "NPerXdlops: " << NPerXdlops << "\n";
    // llvm::errs() << "MRepeats: " << MRepeats << "\n";
    // llvm::errs() << "NRepeats: " << NRepeats << "\n";
    // llvm::errs() << "vectorType: " << vectorType << "\n";
    // llvm::errs() << "vectorNumber: " << vectorNumber << "\n";
    // llvm::errs() << "imms:\n";
    // for (auto imm : imms) {
    //         llvm::errs() << imm[0] << " " << imm[1] << " " << imm[2] << "\n";
    // }
    // llvm::errs() << "argType: " << argType << "\n";

    // llvm::errs() << "group_size: " << group_size << "\n";
    // llvm::errs() << "num_groups_blk: " << num_groups_blk << "\n";
    // llvm::errs() << "num_regs_blk: " << num_regs_blk << "\n";
    // llvm::errs() << "num_threads_blk: " << num_threads_blk << "\n";
    // llvm::errs() << "num_input_blks: " << num_input_blks << "\n";
    // llvm::errs() << "num_output_blks: " << num_output_blks << "\n";
    // llvm::errs() << "num_regs_xdlops: " << num_regs_xdlops << "\n";
    // llvm::errs() << "m: " << m << "\n";
    // llvm::errs() << "n: " << n << "\n";
    // llvm::errs() << "k: " << k << "\n";
    // llvm::errs() << "cycles: " << cycles << "\n";
    // llvm::errs() << "k_base: " << k_base << "\n";

    return result;
  }

  // Checks whether given XDLOp config is valid for a given
  // KPack and KPerBlock values.
  bool isValid(int64_t KPack, int64_t KPerBlock) {
    bool IsKReduction = (blocksInOutRegs == 1) && (inputSpansPerMfmaIn > 1);
    if (KPack > 1) {
      if (KPack < k_base) {
        // llvm::dbgs()
        //     << "Should pack at least k_base elements and avoid waste
        //     xdlopsgemm cycles\n";
        return false;
      }
      if (IsKReduction && KPerBlock < inputSpansPerMfmaIn) {
        // llvm::dbgs()
        //     << " When reduction, KPerBlock must be at least
        //     num_input_blks\n";
        return false;
      }
      return true;
    } else {
      if (!IsKReduction && KPerBlock < k_base) {
        // llvm::dbgs()
        //     << "When non-reduction, KPerBlock must be at least k_base\n";
        return false;
      }
      if (IsKReduction && KPerBlock < k_base * inputSpansPerMfmaIn) {
        // llvm::dbgs()
        //     << "When reduction, KPerBlock must be at least k_base *
        //     num_input_blks\n";
        return false;
      }
      return true;
    }
  }
};
#endif
