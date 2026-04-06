//===-- WinogradArgLayout.cpp - Winograd kernel arg layout ----------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Winograd/WinogradArgLayout.h"
#include "mlir/Dialect/Rock/Winograd/WinogradConvProblem.h"

#include <cassert>
#include <cstring>

namespace mlir {
namespace rock {
namespace winograd {

// ---------------------------------------------------------------------------
// Helpers to write little-endian scalars into a byte buffer.
// ---------------------------------------------------------------------------

static void writeU32(std::vector<uint8_t> &buf, int64_t off, uint32_t v) {
  assert(off >= 0 && off + sizeof(v) <= buf.size() && "writeU32 out of bounds");
  std::memcpy(buf.data() + off, &v, sizeof(v));
}

static void writeI32(std::vector<uint8_t> &buf, int64_t off, int32_t v) {
  assert(off >= 0 && off + sizeof(v) <= buf.size() && "writeI32 out of bounds");
  std::memcpy(buf.data() + off, &v, sizeof(v));
}

static void writeU64(std::vector<uint8_t> &buf, int64_t off, uint64_t v) {
  assert(off >= 0 && off + sizeof(v) <= buf.size() && "writeU64 out of bounds");
  std::memcpy(buf.data() + off, &v, sizeof(v));
}

static void writeF32(std::vector<uint8_t> &buf, int64_t off, float v) {
  assert(off >= 0 && off + sizeof(v) <= buf.size() && "writeF32 out of bounds");
  std::memcpy(buf.data() + off, &v, sizeof(v));
}

// ---------------------------------------------------------------------------
// ArgField construction helpers.
// ---------------------------------------------------------------------------

static ArgField scalar(const char *name, int64_t off, int64_t sz,
                       ArgField::Kind k) {
  return {name, off, sz, k, /*isPointer=*/false, /*tensorIndex=*/-1};
}

static ArgField pointer(const char *name, int64_t off, int tidx) {
  return {name, off, 8, ArgField::U64, /*isPointer=*/true, /*tensorIndex=*/tidx};
}

// ---------------------------------------------------------------------------
// createV1 - V1 ABI (v21/v30/v40 kernels), 248 bytes.
// ---------------------------------------------------------------------------

WinogradArgLayout WinogradArgLayout::createV1() {
  WinogradArgLayout l;
  l.abiVersion = 1;
  l.totalSize = 248;

  auto &f = l.fields;
  f.reserve(48);

  // Convolution dimensions.
  f.push_back(scalar("N",        0,  4, ArgField::U32));
  f.push_back(scalar("C",        4,  4, ArgField::U32));
  f.push_back(scalar("H",        8,  4, ArgField::U32));
  f.push_back(scalar("W",       12,  4, ArgField::U32));
  f.push_back(scalar("K",       16,  4, ArgField::U32));
  f.push_back(scalar("n_groups",20,  4, ArgField::U32));
  f.push_back(scalar("flags",   24,  4, ArgField::U32));
  f.push_back(scalar("reserved",28,  4, ArgField::U32));

  // Tensor pointers.
  f.push_back(pointer("data_addr",  32, 0));
  f.push_back(pointer("filter_addr",40, 1));
  f.push_back(pointer("output_addr",48, 2));
  f.push_back(pointer("reserved_ptr",56, -1));

  // Filter spatial dims and padding.
  f.push_back(scalar("R",     64,  4, ArgField::U32));
  f.push_back(scalar("S",     68,  4, ArgField::U32));
  f.push_back(scalar("pad_h", 72,  4, ArgField::I32));
  f.push_back(scalar("pad_w", 76,  4, ArgField::I32));
  f.push_back(scalar("out_h", 80,  4, ArgField::U32));
  f.push_back(scalar("out_w", 84,  4, ArgField::U32));

  // Bias pointer and scaling factors.
  f.push_back(pointer("bias_addr", 88, -1));
  f.push_back(scalar("alpha",  96, 4, ArgField::F32));
  f.push_back(scalar("beta",  100, 4, ArgField::F32));

  // Tensor offsets.
  f.push_back(scalar("d_offset",104, 8, ArgField::U64));
  f.push_back(scalar("f_offset",112, 8, ArgField::U64));
  f.push_back(scalar("o_offset",120, 8, ArgField::U64));
  f.push_back(scalar("b_offset",128, 8, ArgField::U64));

  // Data (input) strides.
  f.push_back(scalar("d_N_stride",136, 4, ArgField::U32));
  f.push_back(scalar("d_C_stride",140, 4, ArgField::U32));
  f.push_back(scalar("d_H_stride",144, 4, ArgField::U32));
  f.push_back(scalar("d_W_stride",148, 4, ArgField::U32));

  // Filter strides.
  f.push_back(scalar("f_K_stride",152, 4, ArgField::U32));
  f.push_back(scalar("f_C_stride",156, 4, ArgField::U32));
  f.push_back(scalar("f_R_stride",160, 4, ArgField::U32));
  f.push_back(scalar("f_S_stride",164, 4, ArgField::U32));

  // Output strides.
  f.push_back(scalar("o_N_stride",168, 4, ArgField::U32));
  f.push_back(scalar("o_K_stride",172, 4, ArgField::U32));
  f.push_back(scalar("o_H_stride",176, 4, ArgField::U32));
  f.push_back(scalar("o_W_stride",180, 4, ArgField::U32));

  // Group count and group strides.
  f.push_back(scalar("G",         184, 4, ArgField::U32));
  f.push_back(scalar("d_G_stride",188, 4, ArgField::U32));
  f.push_back(scalar("f_G_stride",192, 4, ArgField::U32));
  f.push_back(scalar("o_G_stride",196, 4, ArgField::U32));

  // Activation and reserved.
  f.push_back(scalar("activation_mode",200, 1, ArgField::U8));
  f.push_back(scalar("reserved_u8",    201, 1, ArgField::U8));
  f.push_back(scalar("reserved_u16",   202, 2, ArgField::U16));
  f.push_back(scalar("reserved_u32",   204, 4, ArgField::U32));

  // Hidden arguments (filled by the runtime/dispatch).
  f.push_back(scalar("hidden_global_offset_x",208, 8, ArgField::U64));
  f.push_back(scalar("hidden_global_offset_y",216, 8, ArgField::U64));
  f.push_back(scalar("hidden_global_offset_z",224, 8, ArgField::U64));
  f.push_back(scalar("hidden_none_1",         232, 8, ArgField::U64));
  f.push_back(scalar("hidden_none_2",         240, 8, ArgField::U64));

  return l;
}

// ---------------------------------------------------------------------------
// createV2 - V2 ABI (Fury/Rage kernels), 232 bytes.
// ---------------------------------------------------------------------------

WinogradArgLayout WinogradArgLayout::createV2() {
  WinogradArgLayout l;
  l.abiVersion = 2;
  l.totalSize = 232;

  auto &f = l.fields;
  f.reserve(40);

  // Convolution dimensions.
  f.push_back(scalar("N",        0,  4, ArgField::U32));
  f.push_back(scalar("C",        4,  4, ArgField::U32));
  f.push_back(scalar("H",        8,  4, ArgField::U32));
  f.push_back(scalar("W",       12,  4, ArgField::U32));
  f.push_back(scalar("K",       16,  4, ArgField::U32));
  f.push_back(scalar("n_groups",20,  4, ArgField::U32));
  f.push_back(scalar("flags64", 24,  8, ArgField::U64));

  // Tensor pointers.
  f.push_back(pointer("data_addr",  32, 0));
  f.push_back(pointer("filter_addr",40, 1));
  f.push_back(pointer("output_addr",48, 2));
  f.push_back(scalar("reserved_u64",56, 8, ArgField::U64));

  // Filter spatial dims and padding.
  f.push_back(scalar("R",     64,  4, ArgField::U32));
  f.push_back(scalar("S",     68,  4, ArgField::U32));
  f.push_back(scalar("pad_h", 72,  4, ArgField::I32));
  f.push_back(scalar("pad_w", 76,  4, ArgField::I32));
  f.push_back(scalar("out_h", 80,  4, ArgField::U32));
  f.push_back(scalar("out_w", 84,  4, ArgField::U32));

  // Bias pointer and scaling factors.
  f.push_back(pointer("bias_addr", 88, -1));
  f.push_back(scalar("alpha",  96, 4, ArgField::F32));
  f.push_back(scalar("beta",  100, 4, ArgField::F32));

  // Tensor offsets.
  f.push_back(scalar("d_offset",104, 8, ArgField::U64));
  f.push_back(scalar("f_offset",112, 8, ArgField::U64));
  f.push_back(scalar("o_offset",120, 8, ArgField::U64));
  f.push_back(scalar("b_offset",128, 8, ArgField::U64));

  // Data (input) strides -- V2 omits d_W_stride (reserved).
  f.push_back(scalar("d_N_stride",136, 4, ArgField::U32));
  f.push_back(scalar("d_C_stride",140, 4, ArgField::U32));
  f.push_back(scalar("d_H_stride",144, 4, ArgField::U32));
  f.push_back(scalar("reserved_d",148, 4, ArgField::U32));

  // Filter strides -- V2 omits f_S_stride (reserved).
  f.push_back(scalar("f_K_stride",152, 4, ArgField::U32));
  f.push_back(scalar("f_C_stride",156, 4, ArgField::U32));
  f.push_back(scalar("f_R_stride",160, 4, ArgField::U32));
  f.push_back(scalar("reserved_f",164, 4, ArgField::U32));

  // Output strides -- V2 omits o_W_stride (reserved).
  f.push_back(scalar("o_N_stride",168, 4, ArgField::U32));
  f.push_back(scalar("o_K_stride",172, 4, ArgField::U32));
  f.push_back(scalar("o_H_stride",176, 4, ArgField::U32));
  f.push_back(scalar("reserved_o",180, 4, ArgField::U32));

  // Group count and group strides.
  f.push_back(scalar("G",         184, 4, ArgField::U32));
  f.push_back(scalar("d_G_stride",188, 4, ArgField::U32));
  f.push_back(scalar("f_G_stride",192, 4, ArgField::U32));
  f.push_back(scalar("o_G_stride",196, 4, ArgField::U32));

  // Activation, sync control, and reserved.
  f.push_back(scalar("activation_mode",200, 1, ArgField::U8));
  f.push_back(scalar("sync_limit",     201, 1, ArgField::U8));
  f.push_back(scalar("sync_period",    202, 1, ArgField::U8));
  f.push_back(scalar("reserved_u8",    203, 1, ArgField::U8));
  f.push_back(scalar("reserved_u32",   204, 4, ArgField::U32));

  // Sync/accumulator pointers and offset.
  f.push_back(pointer("sync_addr",208, -1));
  f.push_back(pointer("acc_addr", 216, -1));
  f.push_back(scalar("a_offset",  224, 8, ArgField::U64));

  return l;
}

// ---------------------------------------------------------------------------
// Accessors.
// ---------------------------------------------------------------------------

int64_t WinogradArgLayout::getTotalSize() const { return totalSize; }

const std::vector<ArgField> &WinogradArgLayout::getFields() const {
  return fields;
}

std::vector<PointerSlot> WinogradArgLayout::getPointerSlots() const {
  std::vector<PointerSlot> slots;
  for (const auto &f : fields) {
    if (f.isPointer && f.tensorIndex >= 0)
      slots.push_back({f.offset, f.tensorIndex, f.name});
  }
  return slots;
}

// ---------------------------------------------------------------------------
// computeStrides - NCHW / KCRS / NKHW element strides.
// ---------------------------------------------------------------------------

WinogradArgLayout::TensorStrides
WinogradArgLayout::computeStrides(const WinogradConvProblem &p) {
  TensorStrides s{};

  // Data tensor [N, C, H, W] (NCHW).
  s.d_W = 1;
  s.d_H = static_cast<uint32_t>(p.W);
  s.d_C = static_cast<uint32_t>(p.H * p.W);
  s.d_N = static_cast<uint32_t>(p.C * p.H * p.W);
  s.d_G = static_cast<uint32_t>(p.C * p.H * p.W);

  // Filter tensor [K, C, R, S] (KCRS).
  s.f_S = 1;
  s.f_R = static_cast<uint32_t>(p.S);
  s.f_C = static_cast<uint32_t>(p.R * p.S);
  s.f_K = static_cast<uint32_t>(p.C * p.R * p.S);
  s.f_G = static_cast<uint32_t>(p.K * p.C * p.R * p.S);

  // Output tensor [N, K, out_h, out_w] (NKHW).
  s.o_W = 1;
  s.o_H = static_cast<uint32_t>(p.outW);
  s.o_K = static_cast<uint32_t>(p.outH * p.outW);
  s.o_N = static_cast<uint32_t>(p.K * p.outH * p.outW);
  s.o_G = static_cast<uint32_t>(p.K * p.outH * p.outW);

  return s;
}

// ---------------------------------------------------------------------------
// computeFlagsV1 - flag word for V1 ABI kernels.
// ---------------------------------------------------------------------------

uint32_t WinogradArgLayout::computeFlagsV1(bool isForward) {
  constexpr uint32_t F_REVERSE_R = 1u << 0;
  constexpr uint32_t F_REVERSE_S = 1u << 1;
  constexpr uint32_t F_FLIP_K_C = 1u << 2;
  constexpr uint32_t F_NKC_STRIDES = 1u << 9;

  uint32_t flags = 0;
  if (!isForward)
    flags |= F_REVERSE_R | F_REVERSE_S | F_FLIP_K_C; // 0x7
  flags |= F_NKC_STRIDES;
  return flags;
}

// ---------------------------------------------------------------------------
// computeFlagsV2 - 64-bit flag word for V2 ABI kernels.
// ---------------------------------------------------------------------------

uint64_t WinogradArgLayout::computeFlagsV2(bool isForward, bool hasBias,
                                           bool groupedConv,
                                           bool useActivation) {
  constexpr uint64_t F_REVERSE_R = 1ull << 0;
  constexpr uint64_t F_REVERSE_S = 1ull << 1;
  constexpr uint64_t F_BIAS = 1ull << 7;
  constexpr uint64_t F_NKCHR_STRIDES = 1ull << 9;
  constexpr uint64_t F_GROUPED_CONVOLUTION = 1ull << 10;
  constexpr uint64_t F_TENSOR_OFFSETS = 1ull << 13;
  constexpr uint64_t F_USE_ACTIVATION_MODE = 1ull << 14;
  constexpr uint64_t F_USE_EXTENDED_FLAGS_64 = 1ull << 15;

  uint64_t flags = 0;
  if (!isForward)
    flags |= F_REVERSE_R | F_REVERSE_S; // 0x3
  flags |= F_NKCHR_STRIDES | F_TENSOR_OFFSETS | F_USE_EXTENDED_FLAGS_64;
  if (hasBias)
    flags |= F_BIAS;
  if (groupedConv)
    flags |= F_GROUPED_CONVOLUTION;
  if (useActivation)
    flags |= F_USE_ACTIVATION_MODE;
  return flags;
}

// ---------------------------------------------------------------------------
// buildTemplate - fill a byte buffer with all compile-time-known values.
// ---------------------------------------------------------------------------

std::vector<uint8_t>
WinogradArgLayout::buildTemplate(const WinogradConvProblem &problem,
                                 int64_t nGroups, uint32_t flags) const {
  std::vector<uint8_t> buf(totalSize, 0);

  // Convolution dimensions.
  writeU32(buf, 0,  static_cast<uint32_t>(problem.N));
  writeU32(buf, 4,  static_cast<uint32_t>(problem.C));
  writeU32(buf, 8,  static_cast<uint32_t>(problem.H));
  writeU32(buf, 12, static_cast<uint32_t>(problem.W));
  writeU32(buf, 16, static_cast<uint32_t>(problem.K));
  writeU32(buf, 20, static_cast<uint32_t>(nGroups));

  // Flags: V1 uses u32 at offset 24; V2 uses u64 at offset 24.
  if (this->abiVersion == 1) {
    writeU32(buf, 24, flags);
  } else {
    writeU64(buf, 24, static_cast<uint64_t>(flags));
  }

  // Pointer slots (offsets 32, 40, 48) are left as zero.

  // Filter spatial dimensions and padding.
  writeU32(buf, 64, static_cast<uint32_t>(problem.R));
  writeU32(buf, 68, static_cast<uint32_t>(problem.S));
  writeI32(buf, 72, static_cast<int32_t>(problem.padH));
  writeI32(buf, 76, static_cast<int32_t>(problem.padW));
  writeU32(buf, 80, static_cast<uint32_t>(problem.outH));
  writeU32(buf, 84, static_cast<uint32_t>(problem.outW));

  // Scaling factors: alpha = 1.0, beta = 0.0 (standard defaults).
  writeF32(buf, 96, 1.0f);
  writeF32(buf, 100, 0.0f);

  // Tensor strides.
  TensorStrides st = computeStrides(problem);

  writeU32(buf, 136, st.d_N);
  writeU32(buf, 140, st.d_C);
  writeU32(buf, 144, st.d_H);
  if (this->abiVersion == 1)
    writeU32(buf, 148, st.d_W);

  writeU32(buf, 152, st.f_K);
  writeU32(buf, 156, st.f_C);
  writeU32(buf, 160, st.f_R);
  if (this->abiVersion == 1)
    writeU32(buf, 164, st.f_S);

  writeU32(buf, 168, st.o_N);
  writeU32(buf, 172, st.o_K);
  writeU32(buf, 176, st.o_H);
  if (this->abiVersion == 1)
    writeU32(buf, 180, st.o_W);

  // Group count and group strides.
  writeU32(buf, 184, static_cast<uint32_t>(problem.groupCount));
  writeU32(buf, 188, st.d_G);
  writeU32(buf, 192, st.f_G);
  writeU32(buf, 196, st.o_G);

  return buf;
}

} // namespace winograd
} // namespace rock
} // namespace mlir
