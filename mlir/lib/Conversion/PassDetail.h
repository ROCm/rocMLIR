//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCMLIR_CONVERSION_PASSDETAIL_H_
#define ROCMLIR_CONVERSION_PASSDETAIL_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace amdgpu {
class AMDGPUDialect;
} // end namespace amdgpu

namespace cf {
class ControlFlowDialect;
} // namespace cf

namespace func {
class FuncDialect;
} // end namespace func

namespace gpu {
class GPUDialect;
class GPUModuleOp;
} // end namespace gpu

namespace linalg {
class LinalgDialect;
} // namespace linalg

namespace migraphx {
class MIGraphXDialect;
} // namespace migraphx

namespace rock {
class RockDialect;
} // namespace rock

namespace tosa {
class TosaDialect;
} // end namespace tosa

namespace vector {
class VectorDialect;
} // end namespace vector

#define GEN_PASS_CLASSES
#include "mlir/Conversion/RocMLIRPasses.h.inc"

} // end namespace mlir

#endif // ROCMLIR_CONVERSION_PASSDETAIL_H_
