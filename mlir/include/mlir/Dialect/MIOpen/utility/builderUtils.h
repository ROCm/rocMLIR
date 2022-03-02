#ifndef MLIR_DIALECT_MIOPEN_UTILITY_BUILDERUTILS_H
#define MLIR_DIALECT_MIOPEN_UTILITY_BUILDERUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace miopen {
// Utility function to emit constant zero op. Can return scalars or vectors.
Value createZeroConstantOp(OpBuilder &b, Location loc, Type type);

// Utility function to emit type conversion ops.
Value createTypeConversionOp(OpBuilder &b, Location loc, Value source,
                             Type sourceType, Type destType);

// Utility function to collapse an multi-dimensional memref to 1D.
Value createCollapseShapeOp(OpBuilder &b, Location loc, Value source);
} // namespace miopen
} // namespace mlir

#endif // MLIR_DIALECT_MIOPEN_BUILDERUTILS_H
