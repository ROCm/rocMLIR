#ifndef MLIR_CONVERSION_LINALGTOROCK_H
#define MLIR_CONVERSION_LINALGTOROCK_H

#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DECL_LINALGTOROCKPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"

namespace rock {
void populateLinalgToRockConversionPattern(RewritePatternSet &pattern,
                                           MLIRContext *context);
}
} // namespace mlir

#endif
