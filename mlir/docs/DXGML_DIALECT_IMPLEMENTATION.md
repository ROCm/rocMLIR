# DXGML Dialect Implementation for rocMLIR

## Overview

This document tracks the implementation of DXGML (DirectX Machine Learning) dialects for rocMLIR, enabling compilation of DXML models to AMD GPU binaries.

## Phase 1: Core Foundation (IN PROGRESS)

### Completed

#### 1. **Dxgml Core Dialect** - TableGen Definitions ✅
- **Location**: `mlir/include/mlir/Dialect/Dxgml/IR/`
- **Files Created**:
  - `DxgmlBase.td` - Dialect base definition
  - `DxgmlTypes.td` - Type system (25+ scalar types + TensorType)
  - `DxgmlAttrs.td` - Attributes (Integer, Float, Bool, Str, DenseElements, ConstantResource)
  - `Dxgml.td` - Core operations (module, entry_point, function, return, invoke)
  - `Dxgml.h` - C++ header declarations

**Types Implemented:**
- Scalars: bool, int2/4/8/16/32/64, uint2/4/8/16/32/64
- Floats: float16/32/64, bfloat16
- Special floats: float4e2m1fn, float8e4m3fn/fnuz, float8e5m2fnuz, float8e8m0fnu
- Containers: TensorType<shape, elementType>
- Null type for optional values

**Operations Implemented:**
- `dxgml.module` - Top-level container
- `dxgml.entry_point` - Entry point function
- `dxgml.function` - Function definition
- `dxgml.return` - Return terminator
- `dxgml.invoke` - Function/subgraph invocation

#### 2. **DxgmlOp Operations Dialect** - TableGen Definitions ✅
- **Location**: `mlir/include/mlir/Dialect/DxgmlOp/IR/`
- **Files Created**:
  - `DxgmlOpBase.td` - Dialect base + enums
  - `DxgmlOp.td` - Essential ML operations

**Operations Implemented (~20 ops):**

*Constant:*
- `dxgml_op.constant` - Create constant tensors/scalars

*Convolution:*
- `dxgml_op.convolution` - 2D/3D convolution with bias
- `dxgml_op.gemm` - General matrix multiply

*Elementwise Unary:*
- `dxgml_op.relu`, `sigmoid`, `tanh` - Activations
- `dxgml_op.abs`, `exp`, `log`, `sqrt`, `negate` - Math operations

*Elementwise Binary:*
- `dxgml_op.add`, `subtract`, `multiply`, `divide`
- `dxgml_op.max`, `min`

*Shape Operations:*
- `dxgml_op.reshape`, `transpose`, `slice`, `concat`, `broadcast`

*Depth/Space:*
- `dxgml_op.depth_to_space`, `space_to_depth`

*Utility:*
- `dxgml_op.cast`, `identity`

### Remaining Work for Phase 1

#### 3. **C++ Implementation Files** ⏳
Need to create:

**Dxgml Dialect:**
```cpp
mlir/lib/Dialect/Dxgml/IR/Dxgml.cpp
```
- Dialect initialization
- Type parsing/printing (especially TensorType)
- Attribute parsing/printing  
- Custom assembly format for function ops
- Type getters (BoolAttr::getType())

**DxgmlOp Dialect:**
```cpp
mlir/lib/Dialect/DxgmlOp/IR/DxgmlOp.cpp
```
- Dialect initialization
- Constant folding for ConstantOp
- Any custom verification logic

#### 4. **CMake Build Integration** ⏳

**Files to Create/Update:**

`mlir/include/mlir/Dialect/Dxgml/IR/CMakeLists.txt`:
```cmake
add_mlir_dialect(Dxgml dxgml)
add_mlir_doc(Dxgml DxgmlDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS DxgmlTypes.td)
mlir_tablegen(DxgmlTypes.h.inc -gen-typedef-decls)
mlir_tablegen(DxgmlTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRDxgmlTypesIncGen)

set(LLVM_TARGET_DEFINITIONS DxgmlAttrs.td)
mlir_tablegen(DxgmlAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(DxgmlAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRDxgmlAttrsIncGen)
```

`mlir/lib/Dialect/Dxgml/CMakeLists.txt`:
```cmake
add_rocmlir_dialect_library(MLIRDxgmlDialect
  IR/Dxgml.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Dxgml

  DEPENDS
  MLIRDxgmlIncGen
  MLIRDxgmlTypesIncGen
  MLIRDxgmlAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFunctionInterfaces
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
)
```

Similar files needed for `DxgmlOp` dialect.

Update `mlir/lib/Dialect/CMakeLists.txt`:
```cmake
add_subdirectory(Rock)
add_subdirectory(MIGraphX)
add_subdirectory(Dxgml)      # Add this
add_subdirectory(DxgmlOp)    # Add this
```

#### 5. **Dialect Registration** ⏳

Update `mlir/include/mlir/InitRocMLIRDialects.h`:
```cpp
#include "mlir/Dialect/Dxgml/IR/Dxgml.h"
#include "mlir/Dialect/DxgmlOp/IR/DxgmlOp.h"

inline void registerRocMLIRDialects(DialectRegistry &registry) {
  registry.insert<rock::RockDialect, 
                  migraphx::MIGraphXDialect,
                  dxgml::DxgmlDialect,          // Add
                  dxgml_op::DxgmlOpDialect>();  // Add
  // ... rest of registration
}
```

#### 6. **Basic Tests** ⏳

Create test files in `mlir/test/Dialect/Dxgml/`:
- `ops.mlir` - Basic operation tests
- `types.mlir` - Type system tests
- `parse.mlir` - Parsing tests
- `model1.mlir` - Test with actual model1

## Phase 2: Model Compilation Pipeline

### DXML to Rock/GPU Lowering

Need to implement conversion passes:

```
DXML IR (dxgml + dxgml_op dialects)
    ↓
  [DxgmlToMIGraphX or DxgmlToRock conversion pass]
    ↓
Rock/MIGraphX IR
    ↓
  [Existing rocMLIR pipeline]
    ↓
GPU IR (gpu dialect)
    ↓
  [Existing rocMLIR pipeline]
    ↓
ROCDL
    ↓
AMD GCN Binary (.hsaco)
```

**Conversion Pass Locations:**
```
mlir/lib/Conversion/DxgmlToRock/
mlir/lib/Conversion/DxgmlToMIGraphX/
```

**Required Conversions:**
- `dxgml_op.convolution` → `rock.conv` or `migraphx.convolution`
- `dxgml_op.relu` → `rock.relu` or elementwise ops
- `dxgml_op.add` → `rock.add` or arith ops
- `dxgml_op.gemm` → `rock.gemm` or `migraphx.dot`
- Shape ops → Standard MLIR shape/tensor ops

### Driver Integration

Update `rocmlir-driver` to support DXGML pipeline:
```bash
rocmlir-driver model1.mlir \
  -kernel-pipeline dxgml,migraphx,highlevel,gpu,binary \
  --arch=gfx1150 \
  -o model1.hsaco
```

## Testing with Existing Models

Located in: `C:\Users\hisha\Documents\shared_drive\DxML\DXGML-Drop3.7\Models\`

**Models Available:**
- `model1/` - Conv + ReLU residual blocks (image upscaling)
- `model2/` - TBD
- `model3/` - TBD
- `llama32/` - Large language model
- `nemotron/` - Large language model
- `audio2face/` - Audio processing

**Test Strategy:**
1. Start with `model1` (simplest - only uses conv, relu, add, depth_to_space)
2. Ensure it parses correctly
3. Implement conversion to Rock/MIGraphX
4. Compile to GPU binary
5. Validate output
6. Expand to other models

## Implementation Estimates

- ✅ **Phase 1A: TableGen Definitions** - COMPLETE (2 hours)
- ⏳ **Phase 1B: C++ Implementation** - 1-2 days
- ⏳ **Phase 1C: CMake Integration** - 0.5 day
- ⏳ **Phase 1D: Basic Tests** - 0.5 day
- ⏳ **Phase 2: Conversion Pipeline** - 2-3 days
- ⏳ **Phase 3: End-to-End Testing** - 1 day

**Total Remaining**: ~5-7 days

## Current Status

**✅ COMPLETED:**
- Dxgml dialect TableGen definitions
- DxgmlOp dialect TableGen definitions (essential ops)
- Header structure

**⏳ IN PROGRESS:**
- C++ implementation files
- CMake build integration

**📋 TODO:**
- Complete C++ implementations
- Build integration
- Dialect registration
- Basic parsing tests
- Conversion passes (DXML → Rock/MIGraphX)
- End-to-end compilation testing

## Operations Still Needed for Full DXML Support

Based on model files, additional operations needed in future phases:

**Phase 2 Additions (~30 more ops):**
- Pooling: `average_pooling`, `max_pooling`
- Activations: `gelu`, `swish`, `leaky_relu`, `elu`
- Normalization: `batch_normalization`, `layer_normalization`
- More elementwise: `pow`, `clip`, `ceil`, `floor`, `round`
- Comparisons: `logical_equals`, `logical_greater_than`, etc.
- Reductions: Full `reduce` op support

**Phase 3 Additions (Advanced, ~50+ ops):**
- Attention: `multihead_attention`, `group_query_attention`
- RNN/LSTM/GRU operations
- Quantization ops
- Advanced shape ops
- ROI operations

## Next Immediate Steps

1. Create `Dxgml.cpp` implementation
2. Create `DxgmlOp.cpp` implementation
3. Add CMakeLists.txt for both dialects
4. Register in `InitRocMLIRDialects.h`
5. Build and test parsing of `model1.mlir`
6. Begin conversion pass development

## References

- **DXML Spec**: `C:\Users\hisha\Documents\shared_drive\DxML\DXGML-Drop3.7\Spec\`
- **Test Models**: `C:\Users\hisha\Documents\shared_drive\DxML\DXGML-Drop3.7\Models\`
- **MIGraphX Dialect**: `mlir/include/mlir/Dialect/MIGraphX/` (reference implementation)
- **Rock Dialect**: `mlir/include/mlir/Dialect/Rock/` (target for lowering)
