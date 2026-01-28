# DXGML Dialect Tests

This directory contains tests for the DXGML (DirectX Machine Learning) dialects in rocMLIR.

## Test Files

### Basic Tests

- **`types.mlir`** - Type system tests
  - Scalar types (int8-64, uint8-64, float16/32/64, bfloat16)
  - Special float types (float8 variants, float4)
  - Tensor types
  - Bool and null types

- **`ops.mlir`** - Operations tests
  - Constants
  - Elementwise unary operations (relu, sigmoid, tanh, abs, etc.)
  - Elementwise binary operations (add, multiply, max, etc.)
  - Convolution
  - Depth/space transformations

### Model Tests

- **`model1.mlir`** - Image upscaling model
  - Uses: convolution, relu, add, depth_to_space
  - Pattern: Conv → ReLU residual blocks
  - Source: `C:\Users\hisha\Documents\shared_drive\DxML\DXGML-Drop3.7\Models\model1\`

## Running Tests

### Run all DXGML tests:
```bash
cd build
ninja check-dxgml
```

### Run specific test:
```bash
./bin/llvm-lit ../mlir/test/Dialect/Dxgml/types.mlir -v
```

### Run with rocmlir-opt directly:
```bash
./bin/rocmlir-opt ../mlir/test/Dialect/Dxgml/ops.mlir
```

## Test Model Sources

The test models are derived from actual DXML models located in:
```
C:\Users\hisha\Documents\shared_drive\DxML\DXGML-Drop3.7\Models\
├── model1/       - Image upscaling (Conv + ReLU)
├── model2/       - TBD
├── model3/       - TBD  
├── llama32/      - LLM decoder and pre-fill
├── nemotron/     - LLM decoder and pre-fill
└── audio2face/   - Audio processing
```

## Expected Test Results

All tests should:
1. Parse successfully with `rocmlir-opt`
2. Round-trip (parse → print → parse) without changes
3. Type-check correctly
4. Match FileCheck patterns

## Adding New Tests

To add a new test:

1. Create a `.mlir` file in this directory
2. Add `// RUN: rocmlir-opt %s | FileCheck %s` at the top
3. Add `// CHECK:` patterns to verify operations
4. The test will be automatically discovered by lit

## Future Tests

When conversion passes are implemented, add:
- `dxgml-to-migraphx.mlir` - Test conversion to MIGraphX dialect
- `dxgml-to-rock.mlir` - Test conversion to Rock dialect
- `end-to-end/` - Full compilation pipeline tests
