//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MIOPEN_CONVERSION_PASSDETAIL_H_
#define MIOPEN_CONVERSION_PASSDETAIL_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class AffineDialect;

// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace complex {
class ComplexDialect;
} // end namespace complex

namespace gpu {
class GPUDialect;
class GPUModuleOp;
} // end namespace gpu

namespace linalg {
class LinalgDialect;
} // namespace linalg

namespace LLVM {
class LLVMArmNeonDialect;
class LLVMArmSVEDialect;
class LLVMAVX512Dialect;
class LLVMDialect;
} // end namespace LLVM

namespace migraphx {
class MIGraphXDialect;
} // namespace migraphx

namespace miopen {
class MIOpenDialect;
} // namespace miopen

namespace NVVM {
class NVVMDialect;
} // end namespace NVVM

namespace omp {
class OpenMPDialect;
} // end namespace omp

namespace pdl_interp {
class PDLInterpDialect;
} // end namespace pdl_interp

namespace ROCDL {
class ROCDLDialect;
} // end namespace ROCDL

namespace scf {
class SCFDialect;
} // end namespace scf

namespace spirv {
class SPIRVDialect;
} // end namespace spirv

namespace vector {
class VectorDialect;
} // end namespace vector

#define GEN_PASS_CLASSES
#include "mlir/Conversion/MIOpenPasses.h.inc"

} // end namespace mlir

#endif // MIOPEN_CONVERSION_PASSDETAIL_H_
