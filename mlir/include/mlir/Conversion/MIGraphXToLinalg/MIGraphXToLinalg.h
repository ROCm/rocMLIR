#ifndef MLIR_CONVERSION_MIGRAPHXTOLINALG_H
#define MLIR_CONVERSION_MIGRAPHXTOLINALG_H

#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DECL_MIGRAPHXTOLINALGPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"

namespace linalg {
class MIXRShapedToTensorConverter : public TypeConverter {
public:
  MIXRShapedToTensorConverter();
};

class BoundaryTypeConverter : public TypeConverter {
public:
  BoundaryTypeConverter();
};

/// Populates conversion passes from MIGraphX dialect to Linalg dialect.
void populateMIGraphXToLinalgConversionPatterns(TypeConverter &converter,
                                                RewritePatternSet &patterns);

/// Populates conversion patterns for function boundaries, including
/// migraphx.mlir.as_logical_shape and migraphx.mlir.as_underlying_shape.
void populateMIGraphXFuncBoundaryToLinalgConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter);
} // namespace linalg
} // namespace mlir

#endif
