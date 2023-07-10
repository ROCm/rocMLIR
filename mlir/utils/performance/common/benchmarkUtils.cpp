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

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace benchmark;

namespace {

/// Get the identifier of the current device
int get_device_id() {
  int device;
  auto status = hipGetDevice(&device);
  if (status != hipSuccess)
    assert(0 && "No device found");
  return device;
}

// Conversion helpers for F16 and BF16

// BF16 conversion
// Reference: mlir/tools/rocmlir-gen/bf16convert.hpp
typedef union cvt_bf16_fp32 {
  uint32_t u32;
  unsigned short ushortvec[2];
  float f32;
} cvt_bf16_fp32_t;

uint16_t float_to_bfloat16(float src_val) {
  cvt_bf16_fp32_t target_val;
  target_val.f32 = src_val;
  if ((~target_val.u32 & 0x7f800000) == 0) // Inf or NaN
  {
    if ((target_val.u32 & 0xffff) != 0) {
      target_val.u32 |= 0x10000; // Preserve signaling NaN
    }
  } else {
    target_val.u32 += (0x7fff + (target_val.ushortvec[1] &
                                 1)); // Round to nearest, round to even
  }
  return target_val.ushortvec[1];
}

// F16 conversion (does not support Inf or NaN)
// Reference-1: https://stackoverflow.com/a/1659563/4066096
// Reference-2: https://arxiv.org/pdf/2112.08926.pdf (page 28)
uint16_t float_to_float16(float flt) {
  union {
    float f;
    uint32_t u;
  } x{flt};

  const uint32_t b = x.u + 0x00001000;          // round-to-nearest-even
  const uint32_t e = (b & 0x7F800000) >> 23;    // exponent
  const uint32_t m = b & 0x007FFFFF;            // mantissa
  const uint32_t sign = (b & 0x80000000) >> 16; // sign

  if (e > 112)
    // normalized case
    return sign | (((e - 112) << 10) & 0x7C00) | m >> 13;

  if ((e > 101) && (e < 113))
    // denormalized case
    return sign | ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1);

  if (e > 143)
    // saturate
    return 0x7FFF;

  return sign;
}

// Print the help message
void printUsage(const std::string &name) {
  std::cout << "Usage: \n"
            << name
            << " -g numGroups -m numOutRows -n numOutCols -k numReductions -t "
               "(f32|f16|bf16|i8) \n [-transA=(True|False)] "
               "[-transB=(True|False)] \n "
               "[--kernel-repeats numKernelRepeats]\n";
}

// Get a pattern to fill the input tensors. This is because we want to avoid
// testing things with random data or very simple patterns like all 0s or all 1s
std::vector<uint8_t> getPattern(DataType dataType, bool isOut) {
  std::vector<float> patternFlt = {0.5f, -1.0f, 0.75f};
  std::vector<int> patternInt{1, -1, 2};
  std::vector<uint8_t> res;
  switch (dataType) {
  case DataType::F32:
    for (auto flt : patternFlt) {
      auto *p = reinterpret_cast<unsigned char const *>(&flt);
      res.push_back(p[0]);
      res.push_back(p[1]);
      res.push_back(p[2]);
      res.push_back(p[3]);
    }
    break;
  case DataType::F16:
    for (auto flt : patternFlt) {
      ushort f16flt = float_to_float16(flt);
      auto *p = reinterpret_cast<unsigned char const *>(&f16flt);
      res.push_back(p[0]);
      res.push_back(p[1]);
    }
    break;
  case DataType::BF16:
    for (auto flt : patternFlt) {
      ushort bf16flt = float_to_bfloat16(flt);
      auto *p = reinterpret_cast<unsigned char const *>(&bf16flt);
      res.push_back(p[0]);
      res.push_back(p[1]);
    }
    break;
  case DataType::I8:
    for (auto i : patternInt) {
      auto *p = reinterpret_cast<unsigned char const *>(&i);
      res.push_back(p[0]);
      if (isOut) {
        res.push_back(p[1]);
        res.push_back(p[2]);
        res.push_back(p[3]);
      }
    }
    break;
  case DataType::UNKNOWN:
    break;
  }
  return res;
}

// Utility function to convert a string to its correspondent DataType
DataType strToDataType(const std::string &dataTypeStr) {
  if (dataTypeStr == "f16") {
    return DataType::F16;
  } else if (dataTypeStr == "f32") {
    return DataType::F32;
  } else if (dataTypeStr == "bf16") {
    return DataType::BF16;
  } else if (dataTypeStr == "i8") {
    return DataType::I8;
  } else {
    return DataType::UNKNOWN;
  }
}

// Utility function to convert a DataType to its string representation
std::string dataTypeToStr(DataType dataType) {
  switch (dataType) {
  case DataType::F32:
    return "f32";
  case DataType::F16:
    return "f16";
  case DataType::BF16:
    return "bf16";
  case DataType::I8:
    return "i8";
  default:
    return "unknown";
  }
}

// Utility function to convert "true"/"false" to boolean true/false
bool atob(const std::string &arg) {
  auto lowercaseArg = arg;
  std::transform(lowercaseArg.begin(), lowercaseArg.end(), lowercaseArg.begin(),
                 ::tolower);
  return (lowercaseArg == "true" ? true : false);
}

} // namespace

namespace benchmark {

BenchmarkArgs parseCommandLine(const std::string &name, int argc, char **argv) {
  // Note: this parsing function is only meant to parse arguments in this
  // specific form:
  //
  // -operation gemm -t dataType --arch arch -g G -m M -k K -n K
  // -transA={True/False} -transB={True/False} --kernel-repeats=reps
  // --perf_config=
  //
  // issued by the perfRunner.py script
  BenchmarkArgs res;
  int i = 1;
  while (i < argc) {
    std::string arg = argv[i];
    if (arg == "-g") {
      res.gemmG = atoi(argv[++i]);
    } else if (arg == "-m") {
      res.gemmM = atoi(argv[++i]);
    } else if (arg == "-k") {
      res.gemmK = atoi(argv[++i]);
    } else if (arg == "-n") {
      res.gemmN = atoi(argv[++i]);
    } else if (arg == "-t") {
      res.dataType = strToDataType(argv[++i]);
    } else if (arg.rfind("-transA=", 0) == 0) {
      const int lenTransA = std::string("-transA=").length();
      std::string value = arg.substr(lenTransA);
      res.transposeA = atob(value);
    } else if (arg.rfind("-transB=", 0) == 0) {
      const int lenTransB = std::string("-transB=").length();
      std::string value = arg.substr(lenTransB);
      res.transposeB = atob(value);
    } else if (arg == "--perf_config=" || arg == "--arch" ||
               arg == "-operation") {
      i++;
    } else if (arg == "--kernel-repeats") {
      res.kernelRepeats = atoi(argv[++i]);
    } else {
      std::cerr << "Invalid argument!\n";
      printUsage(name);
      exit(1);
      break;
    }
    i++;
  }

  return res;
}

void printProblem(BenchmarkArgs args) {
  std::cout << "G:" << args.gemmG << "\n"
            << "M: " << args.gemmM << "\n"
            << "N: " << args.gemmN << "\n"
            << "K: " << args.gemmK << "\n"
            << "transA: " << (args.transposeA ? "true" : "false") << "\n"
            << "transB: " << (args.transposeB ? "true" : "false") << "\n"
            << "DataType: " << dataTypeToStr(args.dataType) << "\n";
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
  default:
    return 0;
  }
}

size_t getBytesPerElement(DataType dataType, bool isOut) {
  switch (dataType) {
  case DataType::F32:
    return 4;
  case DataType::F16:
  case DataType::BF16:
    return 2;
  case DataType::I8:
    return (isOut ? 4 : 1);
  default:
    assert(0 && "Data type unknown");
  }
}

void *allocAndFill(DataType dataType, size_t byteSize, bool isOut) {
  uint8_t *ret = reinterpret_cast<uint8_t *>(malloc(byteSize));
  std::vector<uint8_t> pattern = getPattern(dataType, isOut);
  size_t bytesPerElem = getBytesPerElement(dataType, isOut);
  size_t patternLen = (pattern.size() / bytesPerElem);
  size_t elems = byteSize / bytesPerElem;
  for (size_t i = 0; i < elems; ++i) {
    for (size_t byte = 0; byte < bytesPerElem; ++byte) {
      int elem = pattern[(i % patternLen) * bytesPerElem];
      ret[bytesPerElem * i + byte] = elem;
    }
  }
  return ret;
}

void *makeHostConstant(float flt, DataType dataType) {
  union {
    float f;
    uint32_t u;
    int32_t i;
  } bytes{0};
  switch (dataType) {
  case DataType::F32:
    bytes.f = flt;
    break;
  case DataType::F16:
    bytes.u = float_to_float16(flt);
    break;
  case DataType::BF16:
    bytes.u = float_to_bfloat16(flt);
    break;
  case DataType::I8:
    bytes.i = flt;
    break;
  default:
    break;
  }
  uint32_t *ret = reinterpret_cast<uint32_t *>(malloc(4));
  *ret = bytes.u;
  return ret;
}

void *getGpuBuffer(const void *hostMem, size_t byteSize) {
  void *gpuBuffer;
  HIP_ABORT_IF_FAIL(hipMalloc(&gpuBuffer, byteSize));
  HIP_ABORT_IF_FAIL(
      hipMemcpy(gpuBuffer, hostMem, byteSize, hipMemcpyHostToDevice));
  return gpuBuffer;
}

std::string get_device_name() {
  hipDeviceProp_t props{};
  auto status = hipGetDeviceProperties(&props, get_device_id());
  if (status != hipSuccess)
    assert(0 && "Device unknown");
  return std::string(props.gcnArchName);
}

} // namespace benchmark
