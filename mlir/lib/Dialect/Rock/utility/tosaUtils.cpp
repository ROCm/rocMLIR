//===- tosa Utility Functions -===//
//
// Copyright 2025 Advanced Micro Devices.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================

#include "mlir/Dialect/Rock/utility/tosaUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tosa-utils"

namespace mlir { 
namespace rock {
    
bool isSpecificValueAttribute(Attribute value, double target) {
  if (auto intValue = dyn_cast<IntegerAttr>(value))
    return target == 0.0 ? intValue.getValue().isZero()
                         : intValue.getValue() == target;
  if (auto fpValue = dyn_cast<FloatAttr>(value))
    return fpValue.getValue().isExactlyValue(target);
  if (auto splatValue = dyn_cast<SplatElementsAttr>(value))
    return isSpecificValueAttribute(splatValue.getSplatValue<Attribute>(),
                                    target);
  if (auto elementsValue = dyn_cast<ElementsAttr>(value))
    return llvm::all_of(elementsValue.getValues<Attribute>(),
                        [target](Attribute attr) {
                          return isSpecificValueAttribute(attr, target);
                        });
  if (auto elementsValue = dyn_cast<DenseElementsAttr>(value))
    return llvm::all_of(elementsValue.getValues<Attribute>(),
                        [target](Attribute attr) {
                          return isSpecificValueAttribute(attr, target);
                        });
  if (auto arrayValue = dyn_cast<ArrayAttr>(value))
    return llvm::all_of(arrayValue.getValue(), [target](Attribute attr) {
      return isSpecificValueAttribute(attr, target);
    });
  return false;
}

bool isConstantValue(Value v, double target) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>())
    return isSpecificValueAttribute(cst.getValue(), target);
  if (auto cst = v.getDefiningOp<tosa::ConstOp>())
    return isSpecificValueAttribute(cst.getValuesAttr(), target);
  return false;
}

bool isConstantZero(Value v) {
  auto elementTy = getElementTypeOrSelf(cast<ShapedType>(v.getType()));
  if (isa<Float8E8M0FNUType>(elementTy)) {
    // zero is not representable in Float8E8M0FNUType
    LLVM_DEBUG(
        llvm::dbgs()
        << "Encountered Float8E8M0FNUType, which cannot represent zero.\n");
    return false;
  }
  return isConstantValue(v, 0.0);
}

bool isConstantOne(Value v) { return isConstantValue(v, 1.0); }

bool isConstNegInf(Value v) {
  return isConstantValue(v, -std::numeric_limits<double>::infinity());
}

static bool isIntAttrSame(Attribute value, int64_t expectedVal) {
  if (auto intValue = dyn_cast<IntegerAttr>(value)) {
    return isSpecificValueAttribute(value, expectedVal); 
  }
  return false;
}

static bool isConstRangeAttribute(Attribute value) {
  if (auto splatValue = dyn_cast<SplatElementsAttr>(value))
    return false;
  if (auto elementsValue = dyn_cast<ElementsAttr>(value))
    return llvm::all_of(llvm::enumerate(elementsValue.getValues<Attribute>()),
                        [](const auto &indexedAttr) {
                          return isIntAttrSame(indexedAttr.value(),
                                               indexedAttr.index());
                        });
  if (auto elementsValue = dyn_cast<DenseElementsAttr>(value))
    return llvm::all_of(llvm::enumerate(elementsValue.getValues<Attribute>()),
                        [](const auto &indexedAttr) {
                          return isIntAttrSame(indexedAttr.value(),
                                               indexedAttr.index());
                        });
  if (auto arrayValue = dyn_cast<ArrayAttr>(value))
    return llvm::all_of(
        llvm::enumerate(arrayValue.getValue()), [](const auto &indexedAttr) {
          return isIntAttrSame(indexedAttr.value(), indexedAttr.index());
        });

  return false;
}

bool isConstRange(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>())
    return isConstRangeAttribute(cst.getValue());
  if (auto cst = v.getDefiningOp<tosa::ConstOp>())
    return isConstRangeAttribute(cst.getValuesAttr());
  return false;
}

} // namespace rock
} // namespace mlir
