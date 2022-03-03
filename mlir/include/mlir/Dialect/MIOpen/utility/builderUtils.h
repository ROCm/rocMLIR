#ifndef MLIR_DIALECT_MIOPEN_UTILITY_BUILDERUTILS_H
#define MLIR_DIALECT_MIOPEN_UTILITY_BUILDERUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace miopen {
// Utility op to emit constant float op
Value createConstantFloatOp(OpBuilder &b, Location loc, Type type,
                            Type elemType, float value);

// Utility op to emit constant float op
Value createConstantIntOp(OpBuilder &b, Location loc, Type type, Type elemType,
                          int64_t value);

// Utility function to emit constant zero op. Can return scalars or vectors.
Value createZeroConstantOp(OpBuilder &b, Location loc, Type type);

} // namespace miopen
} // namespace mlir

#endif // MLIR_DIALECT_MIOPEN_BUILDERUTILS_H
