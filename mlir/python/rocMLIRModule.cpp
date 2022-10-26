//===-- TorchBind.td - Torch dialect bind ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir-c/Dialect/MIGraphX.h"
#include "mlir-c/RegisterRocMLIR.h"

namespace py = pybind11;

PYBIND11_MODULE(_rocMlir, m) {
  mlirRegisterRocMLIRPasses();

  m.doc() = "rocmlir main python extension";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirRegisterRocMLIRDialects(registry);
  });
}
