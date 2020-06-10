#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"

using namespace mlir;

static constexpr int kConv2DTensorDimension = 4;
static constexpr StringLiteral kVarName[3] = {"weight", "input", "output"};

static void EmitLayoutString(llvm::raw_ostream &output,
                             llvm::ArrayRef<mlir::Attribute> &layoutArrayAttr,
                             llvm::StringRef prefix, llvm::StringRef suffix,
                             llvm::StringRef delimiter = "") {
  for (int i = 0; i < kConv2DTensorDimension; ++i) {
    auto attr = layoutArrayAttr[i];
    if (auto strAttr = attr.dyn_cast<StringAttr>()) {
      output << prefix << strAttr.getValue() << suffix;
    }
    if (i < kConv2DTensorDimension - 1) {
      output << delimiter;
    }
  }
}

static void ObtainConvDirection(FuncOp &f, miopen::ConvOpType &opType) {
  f.walk([&opType](miopen::GridwiseGemmOp op) {
    auto kernel_algorithm = op.getAttrOfType<StringAttr>("kernel_algorithm");
    if (kernel_algorithm.getValue().find(StringRef("backward_data")) !=
        StringRef::npos) {
      opType = miopen::ConvOpType::Conv2DBwdDataOpType;
    } else if (kernel_algorithm.getValue().find(StringRef("backward_weight")) !=
               StringRef::npos) {
      opType = miopen::ConvOpType::Conv2DBwdWeightOpType;
    } else {
      opType = miopen::ConvOpType::Conv2DOpType;
    }
  });
}

static void
populateDimVal(const ArrayAttr &layoutAttr, const ArrayAttr &dimAttr,
               llvm::StringMap<std::pair<size_t, int64_t>> &dimIndexVal) {
  assert(layoutAttr.size() == dimAttr.size());
  size_t dimValSize = layoutAttr.size();
  for (size_t i = 0; i < dimValSize; ++i) {
    auto key = layoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();
    auto value = dimAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
    dimIndexVal[key] = std::make_pair(i, value);
  }
}

static void populateSeqVal(const ArrayAttr &seqAttr,
                           llvm::SmallVector<int64_t, 0> &seqVal) {
  size_t seqValSize = seqAttr.size();
  for (size_t i = 0; i < seqValSize; ++i) {
    // Not nested array, push back the value and be done
    if (seqAttr.getValue()[i].dyn_cast<ArrayAttr>() == nullptr) {
      seqVal.push_back(seqAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt());
      continue;
    }
    // There is nested values, continue to populate those
    for (size_t j = 0; j < seqAttr.getValue()[i].dyn_cast<ArrayAttr>().size();
         ++j) {
      seqVal.push_back(seqAttr.getValue()[i]
                           .dyn_cast<ArrayAttr>()
                           .getValue()[j]
                           .dyn_cast<IntegerAttr>()
                           .getInt());
    }
  }
}
