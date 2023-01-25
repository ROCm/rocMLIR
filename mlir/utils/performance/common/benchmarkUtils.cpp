//===------- benchmarkUtils.cpp - common benchmark utility functions ------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

#include "benchmarkUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/MemAlloc.h"

#include "hip/hip_runtime.h"

llvm::APFloat::Semantics getLlvmFltSemantics(DataType dataType) {
  switch (dataType) {
  case DataType::F32:
    return llvm::APFloat::S_IEEEsingle;
  case DataType::F16:
    return llvm::APFloat::S_IEEEhalf;
  case DataType::BF16:
    return llvm::APFloat::S_BFloat;
  case DataType::I8:
    assert(0 && "Can't have i8 floats");
  }
}

void *allocAndFill(DataType dataType, size_t byteSize, bool isOut) {
  uint8_t *ret = reinterpret_cast<uint8_t *>(llvm::safe_malloc(byteSize));
  std::vector<llvm::APInt> intPattern;
  if (dataType != DataType::I8) {
    std::vector<llvm::APFloat> pattern = {
        llvm::APFloat(0.5), llvm::APFloat(-1.0), llvm::APFloat(0.75)};

    llvm::APFloat::Semantics sem = getLlvmFltSemantics(dataType);

    for (auto &flt : pattern) {
      bool dontCare = false;
      flt.convert(llvm::APFloat::EnumToSemantics(sem),
                  llvm::APFloat::rmNearestTiesToEven, &dontCare);
      intPattern.push_back(flt.bitcastToAPInt());
    }
  } else { // int8
    size_t bitWidth = (isOut ? 32 : 8);
    for (int64_t i : {1, -1, 2}) {
      intPattern.emplace_back(bitWidth, i);
    }
  }

  size_t bytesPerElem = intPattern[0].getBitWidth() / 8;
  size_t elems = byteSize / bytesPerElem;
  for (size_t i = 0; i < elems; ++i) {
    const llvm::APInt &elem = intPattern[i % intPattern.size()];
    for (size_t byte = 0; i < bytesPerElem; ++i) {
      uint8_t value = elem.extractBitsAsZExtValue(8, byte * 8);
      ret[byte + bytesPerElem * i] = value;
    }
  }
  return ret;
}

size_t getByteSize(DataType dataType, size_t elems, bool isOut) {
  switch (dataType) {
  case DataType::F32:
    return elems * 4;
  case DataType::F16:
  case DataType::BF16:
    return elems * 2;
  case DataType::I8:
    return elems * (isOut ? 4 : 1);
  }
}

void *getGpuBuffer(const void *hostMem, size_t byteSize) {
  void *gpuBuffer;
  HIP_ABORT_IF_FAIL(hipMalloc(&gpuBuffer, byteSize));
  HIP_ABORT_IF_FAIL(
      hipMemcpy(gpuBuffer, hostMem, byteSize, hipMemcpyHostToDevice));
  return gpuBuffer;
}