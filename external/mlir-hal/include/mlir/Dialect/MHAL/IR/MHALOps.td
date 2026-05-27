//===- MHALOps.td - MHAL dialect TableGen entry point ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TableGen entry point referenced by `add_mlir_dialect(MHALOps mhal)`. The
// MHAL dialect currently exposes only attributes (e.g. mhal.targets,
// mhal.write_access) -- no ops, no types. The dialect class itself is defined
// in MHALBase.td; attribute definitions live in MHALAttrDefs.td.
//
//===----------------------------------------------------------------------===//

#ifndef MHAL_OPS
#define MHAL_OPS

include "mlir/Dialect/MHAL/IR/MHALBase.td"
include "mlir/Dialect/MHAL/IR/MHALAttrDefs.td"

#endif // MHAL_OPS
