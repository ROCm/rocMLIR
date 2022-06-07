//===- XdlopsCodeSelection.h - MLIR to C++ for MIOpen conversion
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

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "llvm/Support/ErrorHandling.h"
#include <stdint.h>

using namespace mlir;

//===----------------------------------------------------------------------===//
// XDLOPS code selection logic.
//===----------------------------------------------------------------------===//
struct XdlopsCodeSelection {
  StringRef mfmaInstr;
  int64_t MPerXdlops;
  int64_t NPerXdlops;
  int64_t MRepeats;
  int64_t NRepeats;
  VectorType vectorType;
  int64_t vectorNumber;
  SmallVector<SmallVector<unsigned, 3>, 2> imms;
  Type argType;

  int64_t group_size;
  int64_t num_groups_blk;
  int64_t num_regs_blk;
  int64_t num_threads_blk;
  int64_t num_input_blks;
  int64_t num_output_blks;
  int64_t num_regs_xdlops;
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t cycles;
  int64_t k_base;

  static void dumpUnsupported(const mlir::Type &dataType, int64_t MPerWave,
                              int64_t NPerWave) {
    llvm::errs() << "Unsupported case:\n";
    llvm::errs() << "MPerWave: " << MPerWave << "\n";
    llvm::errs() << "NPerWave: " << NPerWave << "\n";
    llvm::errs() << "dataType: ";
    dataType.dump();
    llvm::errs() << "\n";
    llvm_unreachable("Can't meaningfully continue xdlop generation");
  }

  static XdlopsCodeSelection get(mlir::Type &dataType, int64_t MPerWave,
                                 int64_t NPerWave, OpBuilder &b) {
    // Determine which XDLOPS be used.
    int64_t MPerXdlops = 0, NPerXdlops = 0, MRepeats = 0, NRepeats = 0;
    StringRef mfmaInstr = "";
    VectorType vectorType;
    int64_t vectorNumber;
    SmallVector<SmallVector<unsigned, 3>, 2> imms;
    Type argType;

    if (dataType == b.getF32Type()) {
      if (MPerWave == 128 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_32x32x1f32";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 2;
        NRepeats = 1;
        vectorType = VectorType::get({32}, b.getF32Type());
        vectorNumber = 4;
        imms.push_back({1, 0, 0});
        imms.push_back({1, 1, 0});
        imms.push_back({1, 0, 0});
        imms.push_back({1, 1, 0});
        argType = b.getF32Type();
      } else if (MPerWave == 64 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_32x32x1f32";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 2;
        vectorType = VectorType::get({32}, b.getF32Type());
        vectorNumber = 4;
        imms.push_back({1, 0, 0});
        imms.push_back({1, 1, 0});
        imms.push_back({1, 0, 0});
        imms.push_back({1, 1, 0});
        argType = b.getF32Type();
      } else if (MPerWave == 64 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_32x32x1f32";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({32}, b.getF32Type());
        vectorNumber = 2;
        imms.push_back({1, 0, 0});
        imms.push_back({1, 1, 0});
        argType = b.getF32Type();
      } else if (MPerWave == 64 && NPerWave == 32) {
        mfmaInstr = "mfma_f32_32x32x1f32";
        MPerXdlops = 64;
        NPerXdlops = 32;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({32}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({0, 0, 1});
        argType = b.getF32Type();
      } else if (MPerWave == 32 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_32x32x1f32";
        MPerXdlops = 32;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({32}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({1, 0, 0});
        argType = b.getF32Type();
      } else if (MPerWave == 64 && NPerWave == 16) {
        mfmaInstr = "mfma_f32_16x16x1f32";
        MPerXdlops = 64;
        NPerXdlops = 16;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({16}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({0, 0, 4});
        argType = b.getF32Type();
      } else if (MPerWave == 16 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_16x16x1f32";
        MPerXdlops = 16;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({16}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({2, 0, 0});
        argType = b.getF32Type();
      } else if (MPerWave == 8 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_4x4x1f32";
        MPerXdlops = 8;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({4}, b.getF32Type());
        vectorNumber = 2;
        imms.push_back({4, 0, 0});
        imms.push_back({4, 1, 0});
        argType = b.getF32Type();
      } else if (MPerWave == 4 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_4x4x1f32";
        MPerXdlops = 4;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({4}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({4, 0, 0});
        argType = b.getF32Type();
      } else if (MPerWave == 32 && NPerWave == 32) {
        mfmaInstr = "mfma_f32_32x32x2f32";
        MPerXdlops = 32;
        NPerXdlops = 32;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({16}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({0, 0, 0});
        argType = b.getF32Type();
      } else if (MPerWave == 16 && NPerWave == 16) {
        mfmaInstr = "mfma_f32_16x16x4f32";
        MPerXdlops = 16;
        NPerXdlops = 16;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({4}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({0, 0, 0});
        argType = b.getF32Type();
      } else {
        dumpUnsupported(dataType, MPerWave, NPerWave);
      }
    } else if (dataType == b.getF16Type()) {
      if (MPerWave == 128 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_32x32x4f16";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 2;
        NRepeats = 1;
        vectorType = VectorType::get({32}, b.getF32Type());
        vectorNumber = 4;
        imms.push_back({1, 0, 0});
        imms.push_back({1, 1, 0});
        imms.push_back({1, 0, 0});
        imms.push_back({1, 1, 0});
        argType = VectorType::get({4}, b.getF16Type());
      } else if (MPerWave == 64 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_32x32x4f16";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 2;
        vectorType = VectorType::get({32}, b.getF32Type());
        vectorNumber = 4;
        imms.push_back({1, 0, 0});
        imms.push_back({1, 1, 0});
        imms.push_back({1, 0, 0});
        imms.push_back({1, 1, 0});
        argType = VectorType::get({4}, b.getF16Type());
      } else if (MPerWave == 64 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_32x32x4f16";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({32}, b.getF32Type());
        vectorNumber = 2;
        imms.push_back({1, 0, 0});
        imms.push_back({1, 1, 0});
        argType = VectorType::get({4}, b.getF16Type());
      } else if (MPerWave == 64 && NPerWave == 32) {
        mfmaInstr = "mfma_f32_32x32x4f16";
        MPerXdlops = 64;
        NPerXdlops = 32;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({32}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({0, 0, 1});
        argType = VectorType::get({4}, b.getF16Type());
      } else if (MPerWave == 64 && NPerWave == 16) {
        mfmaInstr = "mfma_f32_16x16x4f16";
        MPerXdlops = 64;
        NPerXdlops = 16;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({16}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({0, 0, 4});
        argType = VectorType::get({4}, b.getF16Type());
      } else if (MPerWave == 16 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_16x16x4f16";
        MPerXdlops = 16;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({16}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({2, 0, 0});
        argType = VectorType::get({4}, b.getF16Type());
      } else if (MPerWave == 8 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_4x4x4f16";
        MPerXdlops = 8;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({4}, b.getF32Type());
        vectorNumber = 2;
        imms.push_back({4, 0, 0});
        imms.push_back({4, 1, 0});
        argType = VectorType::get({4}, b.getF16Type());
      } else if (MPerWave == 4 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_4x4x4f16";
        MPerXdlops = 4;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({4}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({4, 0, 0});
        argType = VectorType::get({4}, b.getF16Type());
      } else if (MPerWave == 32 && NPerWave == 32) {
        mfmaInstr = "mfma_f32_32x32x8f16";
        MPerXdlops = 32;
        NPerXdlops = 32;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({16}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({0, 0, 0});
        argType = VectorType::get({4}, b.getF16Type());
      } else if (MPerWave == 32 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_32x32x4f16";
        MPerXdlops = 32;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({32}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({1, 0, 0});
        argType = VectorType::get({4}, b.getF16Type());
      } else if (MPerWave == 16 && NPerWave == 16) {
        mfmaInstr = "mfma_f32_16x16x16f16";
        MPerXdlops = 16;
        NPerXdlops = 16;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({4}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({0, 0, 0});
        argType = VectorType::get({4}, b.getF16Type());
      } else {
        dumpUnsupported(dataType, MPerWave, NPerWave);
      }
    } else if (dataType == b.getBF16Type()) {
      if (MPerWave == 128 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_32x32x2bf16";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 2;
        NRepeats = 1;
        vectorType = VectorType::get({32}, b.getF32Type());
        vectorNumber = 4;
        imms.push_back({1, 0, 0});
        imms.push_back({1, 1, 0});
        imms.push_back({1, 0, 0});
        imms.push_back({1, 1, 0});
        argType = VectorType::get({2}, b.getBF16Type());
      } else if (MPerWave == 64 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_32x32x2bf16";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 2;
        vectorType = VectorType::get({32}, b.getF32Type());
        vectorNumber = 4;
        imms.push_back({1, 0, 0});
        imms.push_back({1, 1, 0});
        imms.push_back({1, 0, 0});
        imms.push_back({1, 1, 0});
        argType = VectorType::get({2}, b.getBF16Type());
      } else if (MPerWave == 64 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_32x32x2bf16";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({32}, b.getF32Type());
        vectorNumber = 2;
        imms.push_back({1, 0, 0});
        imms.push_back({1, 1, 0});
        argType = VectorType::get({2}, b.getBF16Type());
      } else if (MPerWave == 64 && NPerWave == 32) {
        mfmaInstr = "mfma_f32_32x32x2bf16";
        MPerXdlops = 64;
        NPerXdlops = 32;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({32}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({0, 0, 1});
        argType = VectorType::get({2}, b.getBF16Type());
      } else if (MPerWave == 32 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_32x32x2bf16";
        MPerXdlops = 32;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({32}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({1, 0, 0});
        argType = VectorType::get({2}, b.getBF16Type());
      } else if (MPerWave == 64 && NPerWave == 16) {
        mfmaInstr = "mfma_f32_16x16x2bf16";
        MPerXdlops = 64;
        NPerXdlops = 16;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({16}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({0, 0, 4});
        argType = VectorType::get({2}, b.getBF16Type());
      } else if (MPerWave == 16 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_16x16x2bf16";
        MPerXdlops = 16;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({16}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({2, 0, 0});
        argType = VectorType::get({2}, b.getBF16Type());
      } else if (MPerWave == 8 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_4x4x2bf16";
        MPerXdlops = 8;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({4}, b.getF32Type());
        vectorNumber = 2;
        imms.push_back({4, 0, 0});
        imms.push_back({4, 1, 0});
        argType = VectorType::get({2}, b.getBF16Type());
      } else if (MPerWave == 4 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_4x4x2bf16";
        MPerXdlops = 4;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({4}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({4, 0, 0});
        argType = VectorType::get({2}, b.getBF16Type());
      } else if (MPerWave == 32 && NPerWave == 32) {
        mfmaInstr = "mfma_f32_32x32x4bf16";
        MPerXdlops = 32;
        NPerXdlops = 32;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({16}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({0, 0, 0});
        argType = VectorType::get({2}, b.getBF16Type());
      } else if (MPerWave == 16 && NPerWave == 16) {
        mfmaInstr = "mfma_f32_16x16x8bf16";
        MPerXdlops = 16;
        NPerXdlops = 16;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({4}, b.getF32Type());
        vectorNumber = 1;
        imms.push_back({0, 0, 0});
        argType = VectorType::get({2}, b.getBF16Type());
      } else {
        dumpUnsupported(dataType, MPerWave, NPerWave);
      }
    } else if (dataType == b.getIntegerType(8)) {
      if (MPerWave == 32 && NPerWave == 32) {
        mfmaInstr = "mfma_i32_32x32x8i8";
        MPerXdlops = 32;
        NPerXdlops = 32;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({16}, b.getI32Type());
        vectorNumber = 1;
        imms.push_back({0, 0, 0});
        argType = VectorType::get({4}, b.getIntegerType(8));
      } else if (MPerWave == 16 && NPerWave == 16) {
        mfmaInstr = "mfma_i32_16x16x16i8";
        MPerXdlops = 16;
        NPerXdlops = 16;
        MRepeats = 1;
        NRepeats = 1;
        vectorType = VectorType::get({4}, b.getI32Type());
        vectorNumber = 1;
        imms.push_back({0, 0, 0});
        argType = VectorType::get({4}, b.getIntegerType(8));
      } else {
        dumpUnsupported(dataType, MPerWave, NPerWave);
      }
    } else {
      llvm_unreachable("XDLOPs for unsupported data types should be rejected");
    }

    // Obtain properties of MFMA instructions.
    int64_t group_size, num_groups_blk, num_regs_blk, num_threads_blk,
        num_output_blks, num_regs_xdlops, m, n, k, cycles, k_base;
    if (mfmaInstr == "mfma_f32_32x32x1f32") {
      group_size = 4;
      num_groups_blk = 4;
      num_regs_blk = group_size * num_groups_blk;
      num_threads_blk = 32;
      num_output_blks = 2;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m = 32;
      n = 32;
      k = 1;
      cycles = 64;
      k_base = 1;
    } else if (mfmaInstr == "mfma_f32_32x32x2f32") {
      group_size = 4;
      num_groups_blk = 4;
      num_regs_blk = group_size * num_groups_blk;
      num_threads_blk = 32;
      num_output_blks = 1;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m = 32;
      n = 32;
      k = 2;
      cycles = 64;
      k_base = 1;
    } else if (mfmaInstr == "mfma_f32_16x16x4f32") {
      group_size = 4;
      num_groups_blk = 1;
      num_regs_blk = group_size * num_groups_blk;
      num_threads_blk = 16;
      num_output_blks = 1;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m = 16;
      n = 16;
      k = 4;
      cycles = 32;
      k_base = 1;
    } else if (mfmaInstr == "mfma_f32_16x16x1f32") {
      group_size = 4;
      num_groups_blk = 1;
      num_regs_blk = group_size * num_groups_blk;
      num_threads_blk = 16;
      num_output_blks = 4;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m = 16;
      n = 16;
      k = 1;
      cycles = 32;
      k_base = 1;
    } else if (mfmaInstr == "mfma_f32_4x4x1f32") {
      group_size = 4;
      num_groups_blk = 1;
      num_regs_blk = group_size * num_groups_blk;
      num_threads_blk = 64;
      num_output_blks = 1;
      num_regs_xdlops = 4;
      m = 4;
      n = 64;
      k = 1;
      cycles = 8;
      k_base = 1;
    } else if (mfmaInstr == "mfma_f32_32x32x4f16") {
      group_size = 4;
      num_groups_blk = 4;
      num_regs_blk = group_size * num_groups_blk;
      num_threads_blk = 32;
      num_output_blks = 2;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m = 32;
      n = 32;
      k = 4;
      cycles = 64;
      k_base = 4;
    } else if (mfmaInstr == "mfma_f32_32x32x8f16") {
      group_size = 4;
      num_groups_blk = 4;
      num_regs_blk = group_size * num_groups_blk;
      num_threads_blk = 32;
      num_output_blks = 1;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m = 32;
      n = 32;
      k = 8;
      cycles = 64;
      k_base = 4;
    } else if (mfmaInstr == "mfma_f32_16x16x16f16") {
      group_size = 4;
      num_groups_blk = 1;
      num_regs_blk = group_size * num_groups_blk;
      num_threads_blk = 16;
      num_output_blks = 1;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m = 16;
      n = 16;
      k = 16;
      cycles = 32;
      k_base = 4;
    } else if (mfmaInstr == "mfma_f32_16x16x4f16") {
      group_size = 4;
      num_groups_blk = 1;
      num_regs_blk = group_size * num_groups_blk;
      num_threads_blk = 16;
      num_output_blks = 4;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m = 16;
      n = 16;
      k = 4;
      cycles = 32;
      k_base = 4;
    } else if (mfmaInstr == "mfma_f32_4x4x4f16") {
      group_size = 4;
      num_groups_blk = 1;
      num_regs_blk = group_size * num_groups_blk;
      num_threads_blk = 64;
      num_output_blks = 1;
      num_regs_xdlops = 4;
      m = 4;
      n = 64;
      k = 4;
      cycles = 8;
      k_base = 4;
    } else if (mfmaInstr == "mfma_f32_32x32x2bf16") {
      group_size = 4;
      num_groups_blk = 4;
      num_regs_blk = group_size * num_groups_blk;
      num_threads_blk = 32;
      num_output_blks = 2;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m = 32;
      n = 32;
      k = 2;
      cycles = 64;
      k_base = 2;
    } else if (mfmaInstr == "mfma_f32_32x32x4bf16") {
      group_size = 4;
      num_groups_blk = 4;
      num_regs_blk = group_size * num_groups_blk;
      num_threads_blk = 32;
      num_output_blks = 1;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m = 32;
      n = 32;
      k = 4;
      cycles = 64;
      k_base = 2;
    } else if (mfmaInstr == "mfma_f32_16x16x8bf16") {
      group_size = 4;
      num_groups_blk = 1;
      num_regs_blk = group_size * num_groups_blk;
      num_threads_blk = 16;
      num_output_blks = 1;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m = 16;
      n = 16;
      k = 8;
      cycles = 32;
      k_base = 2;
    } else if (mfmaInstr == "mfma_f32_32x32x2xf32") {
      group_size = 4;
      num_groups_blk = 1;
      num_regs_blk = group_size * num_groups_blk;
      num_threads_blk = 16;
      num_output_blks = 4;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m = 16;
      n = 16;
      k = 2;
      cycles = 32;
      k_base = 2;
    } else if (mfmaInstr == "mfma_f32_4x4x2bf16") {
      group_size = 4;
      num_groups_blk = 1;
      num_regs_blk = group_size * num_groups_blk;
      num_threads_blk = 64;
      num_output_blks = 1;
      num_regs_xdlops = 4;
      m = 4;
      n = 64;
      k = 2;
      cycles = 8;
      k_base = 2;
    } else if (mfmaInstr == "mfma_i32_32x32x8i8") {
      group_size = 4;
      num_groups_blk = 4;
      num_regs_blk = group_size * num_groups_blk;
      num_threads_blk = 32;
      num_output_blks = 1;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m = 32;
      n = 32;
      k = 8;
      cycles = 64;
      k_base = 4;
    } else if (mfmaInstr == "mfma_i32_16x16x16i8") {
      group_size = 4;
      num_groups_blk = 1;
      num_regs_blk = group_size * num_groups_blk;
      num_threads_blk = 16;
      num_output_blks = 1;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m = 16;
      n = 16;
      k = 16;
      cycles = 32;
      k_base = 4;
    } else {
      llvm_unreachable("Unsupported case as mfmaInstr not selected!\n");
    }

    constexpr int64_t wave_size = 64;
    // Populate result.
    XdlopsCodeSelection result;
    result.mfmaInstr = mfmaInstr;
    result.MPerXdlops = MPerXdlops;
    result.NPerXdlops = NPerXdlops;
    result.MRepeats = MRepeats;
    result.NRepeats = NRepeats;
    result.vectorType = vectorType;
    result.vectorNumber = vectorNumber;
    result.imms = imms;
    result.argType = argType;

    result.group_size = group_size;
    result.num_groups_blk = num_groups_blk;
    result.num_regs_blk = num_regs_blk;
    result.num_threads_blk = num_threads_blk;
    result.num_input_blks = wave_size / num_threads_blk;
    result.num_output_blks = num_output_blks;
    result.num_regs_xdlops = num_regs_xdlops;
    result.m = m;
    result.n = n;
    result.k = k;
    result.cycles = cycles;
    result.k_base = k_base;

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
};
#endif
