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

} // namespace miopen
} // namespace mlir

#endif // MLIR_DIALECT_MIOPEN_BUILDERUTILS_H
