# MIGraphX Dialect Examples

This directory contains examples demonstrating the **MIGraphX MLIR Dialect** and how to use `rocmlir-driver` to compile them.

## Overview

The MIGraphX dialect is an MLIR dialect used in the rocMLIR project for representing operations commonly used in deep learning models. It's primarily used by [MIGraphX](https://github.com/ROCm/AMDMIGraphX) but can be used standalone.

## Prerequisites

Before running these examples, you need to:

1. **Build rocMLIR** following the instructions in the main README.md:
   ```bash
   mkdir build && cd build
   cmake -G Ninja .. \
     -DCMAKE_BUILD_TYPE=RelWithDebInfo \
     -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
     -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
   ninja
   ```

2. Have a **ROCm-capable GPU** (or use CPU-only mode for IR inspection)

## Examples

### 1. Simple GEMM + ReLU (`migraphx_simple_example.mlir`)

**Operations demonstrated:**
- `migraphx.dot` - Matrix multiplication
- `migraphx.add` - Element-wise addition
- `migraphx.relu` - ReLU activation function

**Pattern:** `output = relu(dot(A, B) + bias)`

This is a fundamental building block in neural networks.

### 2. Convolution Network (`migraphx_convolution_example.mlir`)

**Operations demonstrated:**
- `migraphx.convolution` - 2D convolution
- `migraphx.batch_norm_inference` - Batch normalization
- `migraphx.relu` - ReLU activation
- `migraphx.pooling` - Max pooling

**Pattern:** Conv → BatchNorm → ReLU → MaxPool

This represents a typical convolutional neural network layer.

## Running the Examples

### Quick Start (Windows)

From the `examples/migraphx-dialect` directory, run:

```cmd
REM Validate MLIR syntax
run_examples.bat simple gfx1150 validate

REM Run GPU lowering pipeline
run_examples.bat simple gfx1150 gpu

REM Full compilation
run_examples.bat simple gfx1150 full

REM Run convolution example
run_examples.bat conv gfx1150 full
```

### Quick Start (Linux/Unix)

```bash
# Validate MLIR syntax
./run_examples.sh simple gfx1150 validate

# Run GPU lowering pipeline
./run_examples.sh simple gfx1150 gpu

# Full compilation
./run_examples.sh simple gfx1150 full

# Run convolution example
./run_examples.sh conv gfx1150 full
```

### Manual Execution

All commands below should be run from the `build` directory.

### Basic Pipeline Stages

#### 1. Parse and Validate (rocmlir-opt)
```bash
./bin/rocmlir-opt ../examples/migraphx-dialect/migraphx_simple_example.mlir
```
This verifies the MLIR syntax is correct.

#### 2. GPU Lowering Pipeline
```bash
./bin/rocmlir-driver ../examples/migraphx-dialect/migraphx_simple_example.mlir \
  --kernel-pipeline=gpu \
  --arch=gfx1150
```
This lowers MIGraphX operations to GPU dialect.

#### 3. ROCDL Lowering
```bash
./bin/rocmlir-driver ../examples/migraphx-dialect/migraphx_simple_example.mlir \
  --kernel-pipeline=gpu,rocdl \
  --arch=gfx1150
```
This further lowers to ROCDL (AMD GPU-specific LLVM IR).

#### 4. Binary Generation
```bash
./bin/rocmlir-driver ../examples/migraphx-dialect/migraphx_simple_example.mlir \
  --kernel-pipeline=gpu,binary \
  --arch=gfx1150
```
This generates the final GPU binary.

#### 5. Full Compilation (Shorthand)
```bash
./bin/rocmlir-driver ../examples/migraphx-dialect/migraphx_simple_example.mlir \
  -c \
  --arch=gfx1150
```
The `-c` flag is equivalent to `--kernel-pipeline=full --host-pipeline=runner`.

### Viewing Intermediate Representations

To see the IR at each stage, pipe through `rocmlir-opt`:

```bash
# View after GPU lowering
./bin/rocmlir-driver ../examples/migraphx-dialect/migraphx_simple_example.mlir \
  --kernel-pipeline=gpu \
  --arch=gfx1150 | ./bin/rocmlir-opt

# View after ROCDL lowering
./bin/rocmlir-driver ../examples/migraphx-dialect/migraphx_simple_example.mlir \
  --kernel-pipeline=gpu,rocdl \
  --arch=gfx1150 | ./bin/rocmlir-opt
```

### Architecture-Specific Compilation

Different AMD GPU architectures require different flags:

```bash
# For gfx1150 (default)
./bin/rocmlir-driver ../examples/migraphx-dialect/migraphx_simple_example.mlir \
  --arch=gfx1150 -c

# For MI100 (gfx908)
./bin/rocmlir-driver ../examples/migraphx-dialect/migraphx_simple_example.mlir \
  --arch=gfx908 -c

# For MI200 series (gfx90a)
./bin/rocmlir-driver ../examples/migraphx-dialect/migraphx_simple_example.mlir \
  --arch=gfx90a -c

# For RDNA3 (gfx1100)
./bin/rocmlir-driver ../examples/migraphx-dialect/migraphx_simple_example.mlir \
  --arch=gfx1100 -c
```

### Advanced Options

#### Enable Pass Verification
```bash
./bin/rocmlir-driver ../examples/migraphx-dialect/migraphx_simple_example.mlir \
  --kernel-pipeline=gpu \
  --verify-passes \
  --arch=gfx1150
```

#### Debug Output
```bash
# Show kernel binary serialization
./bin/rocmlir-driver ../examples/migraphx-dialect/migraphx_simple_example.mlir \
  -c \
  --arch=gfx1150 \
  --debug-only=serialize-to-blob 2>&1
```

## Understanding MIGraphX Shaped Types

MIGraphX uses a special type system: `!migraphx.shaped<DxDx...xType, S1xS2x...>`

- **First part (DxDx...)**: Logical shape dimensions
- **Second part (S1xS2x...)**: Memory strides

Example: `!migraphx.shaped<2x4xf32, 4x1>`
- Shape: 2 rows × 4 columns
- Strides: 4 (row stride), 1 (column stride)
- Type: f32 (32-bit float)
- This represents standard row-major layout

Non-standard strides allow for:
- **Transposed views**: `<4x2xf32, 1x4>` (column-major)
- **Broadcasted dimensions**: `<4x3xf32, 0x1>` (first dimension broadcasted)
- **Padded layouts**: Custom memory layouts for performance

## Complete Pipeline Example

Here's a complete example showing all stages:

```bash
# 1. Parse the input
./bin/rocmlir-opt ../examples/migraphx-dialect/migraphx_simple_example.mlir > /tmp/step1.mlir

# 2. Lower to GPU dialect
./bin/rocmlir-driver /tmp/step1.mlir --kernel-pipeline=gpu --arch=gfx1150 > /tmp/step2.mlir

# 3. Lower to ROCDL
./bin/rocmlir-driver /tmp/step1.mlir --kernel-pipeline=gpu,rocdl --arch=gfx1150 > /tmp/step3.mlir

# 4. Translate to LLVM IR
./bin/rocmlir-translate /tmp/step3.mlir -gpu-module-to-rocdlir > /tmp/step4.ll

# 5. Optimize with LLVM
opt /tmp/step4.ll -passes='default<O3>,strip' -S > /tmp/step5.ll

# 6. Generate assembly
llc /tmp/step5.ll -mcpu=gfx1150 > /tmp/step6.s
```

## MIGraphX Dialect Operations Reference

For a complete list of operations, see:
- `mlir/include/mlir/Dialect/MIGraphX/IR/MIGraphX.td` - Operation definitions
- `mlir/test/Dialect/MIGraphX/` - Test cases with examples
- `mlir/test/fusion/` - Fusion examples
- `mlir/test/migraphx_models/` - Real model examples (ResNet50, etc.)

### Common Operations

| Operation | Description | Example |
|-----------|-------------|---------|
| `migraphx.dot` | Matrix multiplication | GEMM, attention mechanisms |
| `migraphx.quant_dot` | Quantized matrix multiplication | Low-precision inference |
| `migraphx.convolution` | 2D/3D convolution | CNNs |
| `migraphx.add` | Element-wise addition | Residual connections |
| `migraphx.mul` | Element-wise multiplication | Gating mechanisms |
| `migraphx.relu` | ReLU activation | Activation layers |
| `migraphx.sigmoid` | Sigmoid activation | Binary classification |
| `migraphx.tanh` | Tanh activation | LSTMs, GRUs |
| `migraphx.softmax` | Softmax | Classification output |
| `migraphx.reshape` | Reshape tensor | Shape transformations |
| `migraphx.transpose` | Transpose dimensions | Layout changes |
| `migraphx.broadcast` | Broadcast tensor | Broadcasting operations |
| `migraphx.reduce_sum` | Sum reduction | Aggregations |
| `migraphx.pooling` | Pooling operation | Downsampling |

## Troubleshooting

### "Unknown operation" error
Make sure the MIGraphX dialect is registered. This should be automatic in rocmlir-driver.

### Architecture mismatch
Ensure you're using `--arch` flag matching your GPU. Use `rocminfo` to check your GPU architecture.

### Build errors
Ensure you've built with the correct LLVM/Clang version (ROCm's clang).

## Additional Resources

- [rocMLIR Documentation](../../README.md)
- [MIGraphX Project](https://github.com/ROCm/AMDMIGraphX)
- [MLIR Documentation](https://mlir.llvm.org/)
- [ROCm Documentation](https://rocm.docs.amd.com/)

## Testing Your Examples

To run the test suite that validates these operations:

```bash
# From build directory
ninja check-rocmlir

# Run specific MIGraphX tests
./bin/llvm-lit ../mlir/test/Dialect/MIGraphX/ -v
