
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MHAL/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace mhal {
#define GEN_PASS_DEF_MHALDROPBINARYMETADATAPASS
#include "mlir/Dialect/MHAL/Transforms/Passes.h.inc"
} // namespace mhal
} // namespace mlir

#define DEBUG_TYPE "mhal-prefill"

using namespace mlir;

namespace {
class MHALDropBinaryMetadataPass
    : public mhal::impl::MHALDropBinaryMetadataPassBase<
          MHALDropBinaryMetadataPass> {
public:
  // Inspect each gpu::BinaryOp and drop all the metadata.
  void runOnOperation() override;
};
} // namespace

// Inspect each gpu::BinaryOp and drop all the metadata.
void MHALDropBinaryMetadataPass::runOnOperation() {
  Builder b(&getContext());
  for (gpu::BinaryOp binary :
       getOperation().getBody()->getOps<gpu::BinaryOp>()) {
    // Drop all discardable attributes.
    binary->setDiscardableAttrs(b.getDictionaryAttr({}));
    SmallVector<Attribute, 10> objects;
    for (auto objRaw : binary.getObjects()) {
      auto object = cast<gpu::ObjectAttr>(objRaw);
      // Drop the property dictionary.
      objects.push_back(
          b.getAttr<gpu::ObjectAttr>(object.getTarget(), object.getFormat(),
                                     object.getObject(), nullptr, nullptr));
    }
    binary.setObjectsAttr(b.getArrayAttr(objects));
  }
}
