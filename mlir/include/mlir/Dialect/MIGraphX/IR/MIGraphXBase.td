//===- MIGraphXBase.td - MIGraphX dialect definitions ---*- tablegen -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2023 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR MIGraphX dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MIGRAPHX_BASE
#define MIGRAPHX_BASE

include "mlir/IR/OpBase.td"

def MIGraphX_Dialect : Dialect {
  let name = "migraphx";
  let cppNamespace = "mlir::migraphx";

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
}

#endif // MIGRAPHX_BASE
