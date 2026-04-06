//===-- WinogradArgLayout.h - Winograd kernel arg layout ---------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//
//
// Describes the kernel argument buffer layout for Winograd assembly kernels.
// Two distinct ABIs are supported:
//   V1 (v21/v30/v40 kernels) - 248 bytes
//   V2 (Fury/Rage kernels)   - 232 bytes
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_WINOGRAD_WINOGRAD_ARG_LAYOUT_H
#define MLIR_DIALECT_ROCK_WINOGRAD_WINOGRAD_ARG_LAYOUT_H

#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
namespace rock {
namespace winograd {

struct WinogradConvProblem;

struct ArgField {
  std::string name;
  int64_t offset;
  int64_t size;
  enum Kind { U32, I32, U64, F32, U8, U16 } kind;
  bool isPointer;
  int tensorIndex; // 0=data, 1=filter, 2=output, -1=other
};

struct PointerSlot {
  int64_t offset;
  int tensorIndex; // 0=input/data, 1=filter/weights, 2=output
  std::string name;
};

class WinogradArgLayout {
public:
  static WinogradArgLayout createV1();
  static WinogradArgLayout createV2();

  int64_t getTotalSize() const;
  const std::vector<ArgField> &getFields() const;
  std::vector<PointerSlot> getPointerSlots() const;

  std::vector<uint8_t> buildTemplate(const WinogradConvProblem &problem,
                                     int64_t nGroups, uint32_t flags) const;

  static uint32_t computeFlagsV1(bool isForward);
  static uint64_t computeFlagsV2(bool isForward, bool hasBias,
                                 bool groupedConv, bool useActivation);

  struct TensorStrides {
    uint32_t d_N, d_C, d_H, d_W, d_G;
    uint32_t f_K, f_C, f_R, f_S, f_G;
    uint32_t o_N, o_K, o_H, o_W, o_G;
  };
  static TensorStrides computeStrides(const WinogradConvProblem &problem);

private:
  std::vector<ArgField> fields;
  int64_t totalSize = 0;
  int abiVersion = 0;
};

} // namespace winograd
} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_WINOGRAD_WINOGRAD_ARG_LAYOUT_H
