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
#include "hip_f8_impl.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
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

uint8_t float_to_float8(float flt) {
  return benchmark::cast_to_f8<4, 3, float, false, false>(flt, false, 0);
}

// Print the help message
void printUsage(const std::string &name) {
  std::cout << "Usage: \n"
            << name
            << " -g numGroups -m numOutRows -n numOutCols -k numReductions -t "
               "(f32|f16|bf16|i8) \n [-transA=(True|False)] "
               "[-transB=(True|False)] \n "
               "[--kernel-repeats numKernelRepeats]\n"
               "[--fusion=(fastgelu_add_add)]\n"
               "[-split-k-factor]\n"
               "[-v]\n";
}

// Get a pattern to fill the input tensors. This is because we want to avoid
// testing things with random data or very simple patterns like all 0s or all 1s
std::vector<uint8_t> getPattern(DataType dataType) {
  std::vector<float> patternFlt = {0.5f, -1.0f, 0.75f};
  // For the benchmarking we just use random data and don't really care about
  // the values. Choose some random patterns of values that can be represened by
  // FP4 and F8E8M0FNU.
  std::vector<uint8_t> patternFp4 = {2, 4, 8, 10};
  std::vector<uint8_t> patternF8E8M0FNU = {1, 2, 4, 8};
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
  case DataType::I32:
    for (auto i : patternInt) {
      auto *p = reinterpret_cast<unsigned char const *>(&i);
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
  case DataType::F8:
    for (auto flt : patternFlt)
      res.push_back(float_to_float8(flt));
    break;
  case DataType::F8E8M0FNU:
    for (auto flt : patternF8E8M0FNU) {
      res.push_back(flt);
    }
    break;
  case DataType::I8:
    for (auto i : patternInt) {
      auto *p = reinterpret_cast<unsigned char const *>(&i);
      res.push_back(p[0]);
    }
    break;
  case DataType::F4:
    // getPattern is used to fill GPU buffers. GPU buffers are allocated in
    // terms of bytes. Therefore FP4 values need to be packed into 8-bit values.
    for (size_t i = 0; i < patternFp4.size(); i = i + 2) {
      uint8_t packedF4 =
          (patternFp4[i] & 0x0F) | ((patternFp4[i + 1] & 0x0F) << 4);
      res.push_back(packedF4);
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
  } else if (dataTypeStr == "i32") {
    return DataType::I32;
  } else if (dataTypeStr == "bf16") {
    return DataType::BF16;
  } else if (dataTypeStr == "i8") {
    return DataType::I8;
  } else if (dataTypeStr == "fp8") {
    return DataType::F8;
  } else if (dataTypeStr == "f8E8M0FNU") {
    return DataType::F8E8M0FNU;
  } else if (dataTypeStr == "f4E2M1FN") {
    return DataType::F4;
  } else {
    return DataType::UNKNOWN;
  }
}

// Utility function to convert a DataType to its string representation
std::string dataTypeToStr(DataType dataType) {
  switch (dataType) {
  case DataType::F32:
    return "f32";
  case DataType::I32:
    return "i32";
  case DataType::F16:
    return "f16";
  case DataType::BF16:
    return "bf16";
  case DataType::I8:
    return "i8";
  case DataType::F8:
    return "fp8";
  case DataType::F8E8M0FNU:
    return "f8E8M0FNU";
  case DataType::F4:
    return "f4E2M1FN";
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
  // -operation gemm -t dataType --arch arch -out_datatype dataType --num_cu
  // numCU -g G -m M -k K -n N -transA={True/False} -transB={True/False}
  // --kernel-repeats=reps --fusion --perf_config=
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
    } else if (arg.rfind("-transScaleA=", 0) == 0) {
      const int lenTransScaleA = std::string("-transScaleA=").length();
      std::string value = arg.substr(lenTransScaleA);
      res.transScaleA = atob(value);
    } else if (arg.rfind("-transScaleB=", 0) == 0) {
      const int lenTransScaleB = std::string("-transScaleB=").length();
      std::string value = arg.substr(lenTransScaleB);
      res.transScaleB = atob(value);
    } else if (arg.rfind("-scale_a_dtype=", 0) == 0) {
      const int lenScaleADType = std::string("-scale_a_dtype=").length();
      std::string value = arg.substr(lenScaleADType);
      res.scaleADataType = strToDataType(value);
    } else if (arg.rfind("-scale_b_dtype=", 0) == 0) {
      const int lenScaleBDType = std::string("-scale_b_dtype=").length();
      std::string value = arg.substr(lenScaleBDType);
      res.scaleBDataType = strToDataType(value);
    } else if (arg == "--perf_config=" || arg == "--arch" ||
               arg == "--num_cu" || arg == "-operation" ||
               arg == "--scaledGemm") {
      i++;
    } else if (arg == "--kernel-repeats") {
      res.kernelRepeats = atoi(argv[++i]);
    } else if (arg.rfind("--fusion=", 0) == 0) {
      const int lenTransB = std::string("--fusion=").length();
      std::string value = arg.substr(lenTransB);
      res.fusion = value;
    } else if (arg.rfind("-out_datatype", 0) == 0) {
      res.outDataType = strToDataType(argv[++i]);
    } else if (arg.rfind("-split-k-factor", 0) == 0) {
      res.splitKFactor = atoi(argv[++i]);
      if (res.splitKFactor < 1) {
        std::cerr << "`-split-k-factor` must be greater than 0\n";
        printUsage(name);
        exit(1);
      }
    } else if (arg.rfind("-v", 0) == 0) {
      res.verbose = true;
    } else {
      std::cerr << "Invalid argument!\n";
      printUsage(name);
      exit(1);
      break;
    }
    i++;
  }
  // By default, output datatype is the same as input datatype
  if (res.outDataType == DataType::UNKNOWN)
    res.outDataType = res.dataType;

  return res;
}

void printProblem(BenchmarkArgs args) {
  std::cout << "G: " << args.gemmG << "\n"
            << "M: " << args.gemmM << "\n"
            << "N: " << args.gemmN << "\n"
            << "K: " << args.gemmK << "\n"
            << "transA: " << (args.transposeA ? "true" : "false") << "\n"
            << "transB: " << (args.transposeB ? "true" : "false") << "\n"
            << "transScaleA: " << (args.transScaleA ? "true" : "false") << "\n"
            << "transScaleB: " << (args.transScaleB ? "true" : "false") << "\n"
            << "scaleADataType: " << dataTypeToStr(args.scaleADataType) << "\n"
            << "scaleBDataType: " << dataTypeToStr(args.scaleBDataType) << "\n"
            << "DataType: " << dataTypeToStr(args.dataType) << "\n"
            << "OutDataType: " << dataTypeToStr(args.outDataType) << "\n"
            << "SplitK Factor: " << args.splitKFactor << std::endl;
}

/*
Given a data type and the number of elements, return the number of bytes
required to store the data.
*/
size_t getByteSize(DataType dataType, size_t elems) {
  switch (dataType) {
  case DataType::F32:
  case DataType::I32:
    return elems * 4;
  case DataType::F16:
  case DataType::BF16:
    return elems * 2;
  case DataType::I8:
  case DataType::F8:
  case DataType::F8E8M0FNU:
    return elems;
  case DataType::F4:
    return (elems + 1) / 2; // ceilDiv
  default:
    return 0;
  }
}

size_t getBytesPerElement(DataType dataType) {
  switch (dataType) {
  case DataType::F32:
  case DataType::I32:
    return 4;
  case DataType::F16:
  case DataType::BF16:
    return 2;
  case DataType::I8:
  case DataType::F8:
  case DataType::F8E8M0FNU:
  case DataType::F4:
    return 1;
  default:
    assert(0 && "Data type unknown");
  }
}

void *allocAndFill(DataType dataType, size_t byteSize) {
  uint8_t *ret = reinterpret_cast<uint8_t *>(malloc(byteSize));
  std::vector<uint8_t> pattern = getPattern(dataType);
  // note that fp4 dtype, getPattern returns vector of packed values
  size_t bytesPerElem = getBytesPerElement(dataType);
  size_t patternLen = (pattern.size() / bytesPerElem);
  size_t elems = byteSize / bytesPerElem;
  for (size_t i = 0; i < elems; ++i) {
    for (size_t byte = 0; byte < bytesPerElem; ++byte) {
      uint8_t elem = pattern[(i % patternLen) * bytesPerElem + byte];
      ret[bytesPerElem * i + byte] = elem;
    }
  }
  return ret;
}

void *makeHostConstant(float flt, DataType computeDataType) {
  switch (computeDataType) {
  case DataType::F32: {
    float *ret = reinterpret_cast<float *>(malloc(4));
    *ret = flt;
    return ret;
  }
  case DataType::I32: {
    int32_t *ret = reinterpret_cast<int32_t *>(malloc(4));
    *ret = int32_t(flt);
    return ret;
  }
  case DataType::F16: {
    uint16_t *ret = reinterpret_cast<uint16_t *>(malloc(2));
    *ret = float_to_bfloat16(flt);
    return ret;
  }
  case DataType::BF16: {
    uint16_t *ret = reinterpret_cast<uint16_t *>(malloc(2));
    *ret = float_to_float16(flt);
    return ret;
  }
  case DataType::I8: {
    int8_t *ret = reinterpret_cast<int8_t *>(malloc(1));
    *ret = int8_t(flt);
    return ret;
  }
  case DataType::F8: {
    uint8_t *ret = reinterpret_cast<uint8_t *>(malloc(1));
    *ret = float_to_float8(flt);
    return ret;
  }
  case DataType::F8E8M0FNU: {
    uint8_t *ret = reinterpret_cast<uint8_t *>(malloc(1));
    // extract exponent from float
    *ret = (*(reinterpret_cast<uint32_t *>(&flt)) >> 23) & 0xFF;
    return ret;
  }
  // fp4 is not supported yet, this is used for rocMLIR benchmarking only
  case DataType::F4:
  default:
    return nullptr;
  }
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
