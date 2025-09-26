//===- AmdArchDbBindings.cpp - Python bindings for the AMD arch database
//------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"

namespace py = pybind11;

PYBIND11_MODULE(amd_arch_db, m) {
  m.doc() = "Database of AMD GPU features";

  py::enum_<mlir::rock::GemmFeatures>(m, "GemmFeatures")
      .value("NONE", mlir::rock::GemmFeatures::none)
      .value("MFMA", mlir::rock::GemmFeatures::mfma)
      .value("WMMA", mlir::rock::GemmFeatures::wmma)
      .value("DOT", mlir::rock::GemmFeatures::dot)
      .value("ATOMIC_ADD", mlir::rock::GemmFeatures::atomic_add)
      .value("ATOMIC_ADD_BF16", mlir::rock::GemmFeatures::atomic_add_bf16)
      .value("ATOMIC_ADD_F16", mlir::rock::GemmFeatures::atomic_add_f16)
      .value("ATOMIC_FMAX_F32", mlir::rock::GemmFeatures::atomic_fmax_f32);

  py::class_<mlir::rock::AmdArchInfo>(m, "AmdArchInfo")
      .def_readonly("default_features",
                    &mlir::rock::AmdArchInfo::defaultFeatures)
      .def_readonly("wave_size", &mlir::rock::AmdArchInfo::waveSize)
      .def_readonly("max_waves_per_eu", &mlir::rock::AmdArchInfo::maxWavesPerEU)
      .def_readonly("total_sgpr_per_eu",
                    &mlir::rock::AmdArchInfo::totalSGPRPerEU)
      .def_readonly("total_vgpr_per_eu",
                    &mlir::rock::AmdArchInfo::totalVGPRPerEU)
      .def_readonly("total_shared_mem_per_cu",
                    &mlir::rock::AmdArchInfo::totalSharedMemPerCU)
      .def_readonly("max_shared_mem_per_wg",
                    &mlir::rock::AmdArchInfo::maxSharedMemPerWG)
      .def_readonly("num_eu_per_cu", &mlir::rock::AmdArchInfo::numEUPerCU)
      .def_readonly("min_num_cu", &mlir::rock::AmdArchInfo::minNumCU)
      .def_readonly("has_fp8_conversion_instrs",
                    &mlir::rock::AmdArchInfo::hasFp8ConversionInstrs)
      .def_readonly("has_ocp_fp8_conversion_instrs",
                    &mlir::rock::AmdArchInfo::hasOcpFp8ConversionInstrs)
      .def_readonly("max_num_xcc", &mlir::rock::AmdArchInfo::maxNumXCC);

  m.def("lookup_arch_info", [](const std::string &arch) {
    return mlir::rock::lookupArchInfo(arch);
  });
}
