#ifndef MLIR_DIALECT_MIOPEN_UTILITY_COMMON_H
#define MLIR_DIALECT_MIOPEN_UTILITY_COMMON_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace miopen {
Value createConstantFloatOp(OpBuilder &b, Location loc, Type elementType,
                            float value);
Value createZeroConstantFloatOp(OpBuilder &b, Location loc, Type type);
} // namespace miopen
} // namespace mlir

#endif // MLIR_DIALECT_MIOPEN_UTILITY_COMMON_H
