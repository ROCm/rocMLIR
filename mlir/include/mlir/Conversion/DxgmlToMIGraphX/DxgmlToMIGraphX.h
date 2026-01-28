//===- DxgmlToMIGraphX.h - DXGML to MIGraphX conversion --------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the pass to convert DXGML dialect to MIGraphX dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_DXGMLTOMIGRAPHX_DXGMLTOMIGRAPHX_H
#define MLIR_CONVERSION_DXGMLTOMIGRAPHX_DXGMLTOMIGRAPHX_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class TypeConverter;
class RewritePatternSet;
class Pass;

/// Populate DXGML to MIGraphX conversion patterns.
void populateDxgmlToMIGraphXConversionPatterns(TypeConverter &typeConverter,
                                                 RewritePatternSet &patterns);

/// Create a pass to convert DXGML dialect to MIGraphX dialect.
std::unique_ptr<Pass> createConvertDxgmlToMIGraphXPass();

} // namespace mlir

#endif // MLIR_CONVERSION_DXGMLTOMIGRAPHX_DXGMLTOMIGRAPHX_H
