#include "mlir/Dialect/MIOpen/Tuning/GemmContext.h"

#include "mlir/Dialect/MIOpen/Tuning/ConvContext.h"

using namespace mlir;
using namespace mlir::miopen;

GemmContext GemmContext::fromConvolution(ConvOpType type,
                                         ConvolutionDims sizes) {
  int64_t gemmMSize, gemmKSize, gemmNSize;
  switch (type) {
  case ConvOpType::Fwd:
    gemmMSize = sizes.k;
    gemmKSize = sizes.c * sizes.y * sizes.x;
    gemmNSize = sizes.n * sizes.ho * sizes.wo;
    break;
  case ConvOpType::BwdData:
    gemmMSize = sizes.c;
    gemmKSize = sizes.k * sizes.y * sizes.x;
    gemmNSize = sizes.n * sizes.ho * sizes.wo;
    break;
  case ConvOpType::BwdWeight:
    gemmMSize = sizes.k;
    gemmKSize = sizes.n * sizes.ho * sizes.wo;
    gemmNSize = sizes.c * sizes.y * sizes.x;
    break;
  }
  return GemmContext(gemmMSize, gemmKSize, gemmNSize);
}
