#ifndef MLIR_DIALECT_ROCK_UTILITY_BUILDERUTILS_H
#define MLIR_DIALECT_ROCK_UTILITY_BUILDERUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace rock {
/// Utility op to emit constant float op
Value createConstantFloatOp(OpBuilder &b, Location loc, Type type,
                            Type elemType, float value);

/// Utility op to emit constant int op
Value createConstantIntOp(OpBuilder &b, Location loc, Type type, Type elemType,
                          int64_t value);

/// Utility function to emit constant zero op. Can return scalars or vectors.
Value createZeroConstantOp(OpBuilder &b, Location loc, Type type);

/// Utility function to emit type conversion ops.
Value createTypeConversionOp(OpBuilder &b, Location loc, Value source,
                             Type destType);

// Utility function to perform cast
// and copy to another memref using a Linalg Generic.
void createTypeConversionLaGeneric(PatternRewriter &rewriter, Location loc,
                                   Value src, Value dst);

// Utility function to perform cast
// and copy to another memref using a vector store.
void createTypeConversionStore(PatternRewriter &rewriter, Location loc,
                               Value src, Value dst);

/// Utility function to collapse an multi-dimensional memref to 1D.
Value createCollapseShapeOp(OpBuilder &b, Location loc, Value source);

/// Utility function to get the number of bytes a value of type `type` takes up.
int64_t getByteWidth(Type type);

// Utility function to get a MemRef as a tensor
Value getAsTensor(OpBuilder &builder, Location loc, mlir::Value value,
                  bool isWritable = false);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_BUILDERUTILS_H
