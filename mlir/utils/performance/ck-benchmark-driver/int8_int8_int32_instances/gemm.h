//===------------------------------- gemm.h -------------------------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//
//
// Copy of library/include/ck/library/tensor_operation_instance/gpu/gemm.hpp
// from CK to enable int8->int32 gemms
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdlib>
#include <memory>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_c_shuffle_i8_i8_i32_km_kn_mn_instances(
    std::vector<
        std::unique_ptr<DeviceGemm<Col, Row, Row, int8_t, int8_t, int8_t,
                                   PassThrough, PassThrough, PassThrough>>>
        &instances);

void add_device_gemm_xdl_c_shuffle_i8_i8_i32_km_nk_mn_instances(
    std::vector<
        std::unique_ptr<DeviceGemm<Col, Col, Row, int8_t, int8_t, int8_t,
                                   PassThrough, PassThrough, PassThrough>>>
        &instances);

void add_device_gemm_xdl_c_shuffle_i8_i8_i32_mk_kn_mn_instances(
    std::vector<
        std::unique_ptr<DeviceGemm<Row, Row, Row, int8_t, int8_t, int8_t,
                                   PassThrough, PassThrough, PassThrough>>>
        &instances);

void add_device_gemm_xdl_c_shuffle_i8_i8_i32_mk_nk_mn_instances(
    std::vector<
        std::unique_ptr<DeviceGemm<Row, Col, Row, int8_t, int8_t, int8_t,
                                   PassThrough, PassThrough, PassThrough>>>
        &instances);

template <typename ALayout, typename BLayout, typename CLayout>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGemm<
    ALayout, BLayout, CLayout, int8_t, int8_t, int32_t,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>> {
  using DeviceOp =
      DeviceGemm<ALayout, BLayout, CLayout, int8_t, int8_t, int32_t,
                 ck::tensor_operation::element_wise::PassThrough,
                 ck::tensor_operation::element_wise::PassThrough,
                 ck::tensor_operation::element_wise::PassThrough>;

  static auto GetInstances() {
    std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

    if constexpr (is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                  is_same_v<CLayout, Row>) {
      add_device_gemm_xdl_c_shuffle_i8_i8_i32_mk_kn_mn_instances(op_ptrs);
    } else if constexpr (is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>) {
      add_device_gemm_xdl_c_shuffle_i8_i8_i32_mk_nk_mn_instances(op_ptrs);
    } else if constexpr (is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>) {
      add_device_gemm_xdl_c_shuffle_i8_i8_i32_km_kn_mn_instances(op_ptrs);
    } else if constexpr (is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>) {
      add_device_gemm_xdl_c_shuffle_i8_i8_i32_km_nk_mn_instances(op_ptrs);
    }

    return op_ptrs;
  }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
