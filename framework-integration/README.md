
# Integrating Other Framework with rocMLIR

This is all the jupyter notebook output in one file:

## jax

### Overview

jax -> stablehlo -> linalg -> rock

### Compiling jax using rocMLIR

Run the following python code to generate the file: 

```
# MLIR Based on the following: https://openxla.org/stablehlo/tutorials/jax-export
#!pip install -U jax jaxlib flax transformers tf-nightly
import jax
from jax import export
import jax.numpy as jnp
import numpy as np
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir import passmanager as pm

# stablehlo-opt -stablehlo-legalize-to-linalg <filename.mlir>

# Returns prettyprint of StableHLO module without large constants
def get_stablehlo_asm(module_str):
  with jax_mlir.make_ir_context():
    stablehlo_module = ir.Module.parse(module_str, context=jax_mlir.make_ir_context())
    return stablehlo_module.operation.get_asm(large_elements_limit=20)

# Disable logging for better tutorial rendering
import logging
logging.disable(logging.WARNING)

# crearing GEMM gemm add
@jax.jit
def gemm_add(a, b, d):
  return jnp.add(jnp.matmul(a, b), d)
inputs = (
    np.ones((10, 10), dtype=np.float32),
    np.ones((10, 10), dtype=np.float32),
    np.ones((10, 10), dtype=np.float32),
)
input_shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in inputs]
stablehlo_gemm = export.export(gemm_add)(*input_shapes).mlir_module()
print(get_stablehlo_asm(stablehlo_gemm))
```

Starting with the output from above:

```
module @jit_gemm_add attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>, %arg2: tensor<10x10xf32>) -> (tensor<10x10xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<10x10xf32>
    return %1 : tensor<10x10xf32>
  }
}
```

Run the following command to go from StableHLO to Rock Dialect

```
./bin/stablehlo-opt  --stablehlo-legalize-to-linalg=enable-primitive-ops |\
      /home/vhe/rocMLIR/build-release/bin/rocmlir-driver --kernel-pipeline=highlevel
```

Gives the following rock output:

```
map = affine_map<(d0, d1) -> (d0, d1)>
module @jit_gemm_add attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
func.func public @main(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>, %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32> {jax.result_info = "result"}) attributes {arch = "gfx950", kernel} {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<10x10xf32>
    rock.gemm %alloc = %arg0 * %arg1 storeMethod =  set : memref<10x10xf32> = memref<10x10xf32> * memref<10x10xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<10x10xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc, %arg2 : memref<10x10xf32>, memref<10x10xf32>) outs(%alloc_0 : memref<10x10xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %0 = arith.addf %in, %in_1 : f32
         linalg.yield %0 : f32
    }
  memref.copy %alloc_0, %arg3 : memref<10x10xf32> to memref<10x10xf32>
    return
}
```

## iree (Backend)

### Overview

migraphx -> linalg -> iree


### Compiling MIGraphX to Use iree

Starting with migraphx

```
func.func @testing(%first :!migraphx.shaped<10x10xf32, 10x1>, %second: !migraphx.shaped<10x10xf32, 10x1>) -> !migraphx.shaped<10x10xf32, 10x1> {
  %result = migraphx.dot %first, %second: <10x10xf32, 10x1>, <10x10xf32, 10x1> -> <10x10xf32, 10x1>
  func.return %result : !migraphx.shaped<10x10xf32, 10x1>
}
```

We can run to get into linalg using `rocmlir-opt --migraphx-to-linalg <filename>` to get:

```
module {
  func.func @testing(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> tensor<100xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x10x10xf32>
    %expanded = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [1, 10, 10] : tensor<100xf32> into tensor<1x10x10xf32>
    %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [1, 10, 10] : tensor<100xf32> into tensor<1x10x10xf32>
    %0 = linalg.batch_matmul ins(%expanded, %expanded_0 : tensor<1x10x10xf32>, tensor<1x10x10xf32>) outs(%cst : tensor<1x10x10xf32>) -> tensor<1x10x10xf32>
    %collapsed = tensor.collapse_shape %0 [[0, 1, 2]] : tensor<1x10x10xf32> into tensor<100xf32>
    return %collapsed : tensor<100xf32>
  }
}
```

Run the following two command to run code on the GPU

```
/opt/iree/bin/iree-compile linalg.mlir --iree-hal-target-device=hip --iree-rocm-target=gfx950 -o result.vmfb  --iree-opt-level=O3
/opt/iree/bin/iree-benchmark-module --module=result.vmfb --device=hip://GPU-62363665-6661-6533-3832-363633636434 --function=testing  --input=100xf32=2 --input=100xf32=3
```

Terminal output:

```
2026-04-07T18:36:31+00:00
Running /opt/iree/bin/iree-benchmark-module
Run on (384 X 1200 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x192)
  L1 Instruction 32 KiB (x192)
  L2 Unified 1024 KiB (x192)
  L3 Unified 32768 KiB (x24)
Load Average: 4.09, 4.08, 3.74
***WARNING*** ASLR is enabled, the results may have unreproducible noise in them.
***WARNING*** Library was built as DEBUG. Timings may be affected.
--------------------------------------------------------------------------------------------
Benchmark                                  Time             CPU   Iterations UserCounters...
--------------------------------------------------------------------------------------------
BM_testing/process_time/real_time       1.38 ms         1.85 ms          859 items_per_second=723.4/s
```

## onnx-mlir

### Overview

onnx file -> onnx ir -> linalg -> rock

### ONNX-MLIR using rocMLIR as a backend
First, let us generate the onnx file:

```
import onnx
from onnx import helper, TensorProto
# Inputs
a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [10, 10])
b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [10, 10])
d = helper.make_tensor_value_info("D", TensorProto.FLOAT, [10, 10])
# Output
y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [10, 10])
# Nodes
matmul_node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["Y"])
#add_node = helper.make_node("Add", inputs=["AB", "D"], outputs=["Y"])
# Graph & Model
graph = helper.make_graph(
    [matmul_node],
    "gemm_add",
    inputs=[a, b],
    outputs=[y],
)

# going to export model here
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
model.ir_version = 10
onnx.checker.check_model(model)
onnx.save(model, "gemm_add.onnx")

print(onnx.printer.to_text(model))
```

Running this command `/home/vhe/frameworks/onnx-mlir/build/Release/bin/onnx-mlir gemm_add.onnx --EmitONNXBasic to get the following IR:`, we get this IR: 

```
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "gemm_add"} {
  func.func @main_graph(%arg0: tensor<10x10xf32> {onnx.name = "A"}, %arg1: tensor<10x10xf32> {onnx.name = "B"}) -> (tensor<10x10xf32> {onnx.name = "Y"}) {
    %0 = "onnx.MatMul"(%arg0, %arg1) {onnx_node_name = "onnx.MatMul_0"} : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    return %0 : tensor<10x10xf32>
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
```

Then we can run the following to convert the ONNX IR into Linalg ` /home/vhe/frameworks/onnx-mlir/build/Release/bin/onnx-mlir-opt gemm_add.onnx.mlir --convert-onnx-to-linalg`: 

```
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "gemm_add"} {
  func.func @main_graph(%arg0: tensor<10x10xf32> {onnx.name = "A"}, %arg1: tensor<10x10xf32> {onnx.name = "B"}) -> (tensor<10x10xf32> {onnx.name = "Y"}) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<10x10xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<10x10xf32>) -> tensor<10x10xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<10x10xf32>, tensor<10x10xf32>) outs(%1 : tensor<10x10xf32>) -> tensor<10x10xf32>
    return %2 : tensor<10x10xf32>
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
```

Deleting all the `onnx` dependencies manually, we have: 

```
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "gemm_add"} {
  func.func @main_graph(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>) -> (tensor<10x10xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<10x10xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<10x10xf32>) -> tensor<10x10xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<10x10xf32>, tensor<10x10xf32>) outs(%1 : tensor<10x10xf32>) -> tensor<10x10xf32>
    return %2 : tensor<10x10xf32>
  }
}
```

We then use rocmlir-driver and rocmlir-gen to compile the mlir from above using:

```
~/rocMLIR/build-release/bin/rocmlir-gen rocmlir.mlir -arch=gfx950 -fut main_graph -clone-harness | ~/rocMLIR/build-release/bin/rocmlir-driver --kernel-pipeline=highlevel
```

Output: 

```
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "gemm_add"} {
  func.func @main_graph(%arg0: memref<10x10xf32> {mhal.read_access}, %arg1: memref<10x10xf32> {mhal.read_access}, %arg2: memref<10x10xf32> {mhal.write_access}) {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<10x10xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%alloc : memref<10x10xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<10x10xf32>, memref<10x10xf32>) outs(%alloc : memref<10x10xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    }
    memref.copy %alloc, %arg2 : memref<10x10xf32> to memref<10x10xf32>
    return
  }
  func.func @main_graph_wrapper(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>, %arg2: memref<10x10xf32>) {
    %alloc = memref.alloc() : memref<10x10xf32>
    %token = mhal.launch @main_graph (%arg0, %arg1, %alloc) : (memref<10x10xf32>, memref<10x10xf32>, memref<10x10xf32>)
    mhal.await %token : !mhal.token
    memref.copy %alloc, %arg2 : memref<10x10xf32> to memref<10x10xf32>
    return
  }
  module @__xmodule_ attributes {mhal.arch = "gfx950", mhal.module} {
    func.func @main_graph(%arg0: memref<10x10xf32> {mhal.read_access}, %arg1: memref<10x10xf32> {mhal.read_access}, %arg2: memref<10x10xf32> {mhal.write_access}) attributes {kernel, original_func = @main_graph} {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<10x10xf32>
      rock.gemm %alloc = %arg0 * %arg1 storeMethod =  set : memref<10x10xf32> = memref<10x10xf32> * memref<10x10xf32>
      memref.copy %alloc, %arg2 : memref<10x10xf32> to memref<10x10xf32>
      return
    }
  }
}
```

## torch-mlir

### Overview 

pytorch (nn.Module) -> torch-mlir -> linalg -> rock

### Pytorch using rocMLIR as backend

```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_mlir import fx

def export_model(model, data, filename="result.txt"):
    module = fx.export_and_import(
        model,
        random_data,
        output_type="linalg-on-tensors",
    )
    with open(filename, "w") as f:
        print(module.operation.get_asm())
        f.write(module.operation.get_asm())
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # in, out, kernel_size
        self.lienar = nn.Linear(10, 10)
    
    # x represents our data
    def forward(self, x):
        # Apply softmax to x
        output = self.lienar(x)
        output = output + x
        return output

# Equates to one random 28x28 image
random_data = torch.rand((20, 10))
network = Net()
export_model(network, random_data, filename="reuslt.txt")
```

Gives the following IR

```
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%arg0: tensor<20x10xf32>) -> tensor<20x10xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense_resource<torch_tensor_10_10_torch.float32> : tensor<10x10xf32>
    %cst_1 = arith.constant dense_resource<torch_tensor_10_torch.float32> : tensor<10xf32>
    %0 = tensor.empty() : tensor<10x10xf32>
    %transposed = linalg.transpose ins(%cst_0 : tensor<10x10xf32>) outs(%0 : tensor<10x10xf32>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<20x10xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<20x10xf32>) -> tensor<20x10xf32>
    %3 = linalg.matmul ins(%arg0, %transposed : tensor<20x10xf32>, tensor<10x10xf32>) outs(%2 : tensor<20x10xf32>) -> tensor<20x10xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %cst_1 : tensor<20x10xf32>, tensor<10xf32>) outs(%1 : tensor<20x10xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.addf %in, %in_2 : f32
      linalg.yield %6 : f32
    } -> tensor<20x10xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %arg0 : tensor<20x10xf32>, tensor<20x10xf32>) outs(%1 : tensor<20x10xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.addf %in, %in_2 : f32
      linalg.yield %6 : f32
    } -> tensor<20x10xf32>
    return %5 : tensor<20x10xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_10_10_torch.float32: "0x040000005A1A323C95BD453EAA55773ED4C4AABC7A088E3EE0E30B3E066E84BE7E362BBE5C5773BE55DBDC3D705298BED22951BD85E393BD876A31BCB2FD86BE8A423DBE9D90EE3D5C52683E77329E3EE040593DA4B0D63DC8163B3E821C92BD8F757D3E89308DBE5868E63D70692FBE964E00BEF2546BBE06DB69BE1D09E8BD536DE93D9041933E7F87203E4DC094BE658FF23D3AA4C23D8134333EB893E8BDFB23563C8DCC0E3E7486D1BDF8F80F3E29BBED3DE6948FBE53A9B83DE65BA0BCF50EF7BDCCEB893D8522503E10FD8EBD3ED20FBD40281EBEA35F92BEFA0E77BDD3CB153E683E06BE794F8ABEE4D416BE6A3C7C3E391A0DBDBD1462BE8D4581BBCDB0D3BD6D50A0BE84E0133EBC624BBECB5C263E3C163E3CF4719CBE30E593BE037A893EDD3A5C3D58ED8BBE8E1C62BEC83795BE2A56433E0668953E71A3213E66C5383E7BE6093E2461EE3D48246DBE2910773EC1C8993EDD4BF6BDDE43A73DA302563E978B573E41A311BE0B68EF3D34B01ABE1DE67E3D549C0B3E41852B3EE818753E04D989BECEF58ABEC91418BE75E02D3E",
      torch_tensor_10_torch.float32: "0x04000000606B153E87D6773E84B39ABE00CD123E4D6805BE946BB1BDB77A27BE472D97BE402375BEBC3E81BD"
    }
  }
#-}
```

We can run the following to compile to rock:

```
~/rocMLIR/build-release/bin/rocmlir-gen result.mlir -arch=gfx950 -fut main -clone-harness | \
        ~/rocMLIR/build-release/bin/rocmlir-driver --kernel-pipeline=highlevel
```

output: 

```
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>
#map5 = affine_map<(d0, d1) -> (d1, d0)>
#transform_map = #rock.transform_map<#map5 by [<PassThrough ["dim1", "dim0"] at [0, 1] -> ["dim1", "dim0"] at [1, 0]>] bounds = [10, 10] -> [10, 10]>
module {
  memref.global "private" constant @__constant_10xf32 : memref<10xf32> = dense_resource<torch_tensor_10_torch.float32> {alignment = 64 : i64}
  memref.global "private" constant @__constant_10x10xf32 : memref<10x10xf32> = dense_resource<torch_tensor_10_10_torch.float32> {alignment = 64 : i64}
  func.func @main(%arg0: memref<20x10xf32> {mhal.read_access}, %arg1: memref<20x10xf32> {mhal.write_access}) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_10x10xf32 : memref<10x10xf32>
    %1 = memref.get_global @__constant_10xf32 : memref<10xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<20x10xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%alloc : memref<20x10xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %0 : memref<20x10xf32>, memref<10x10xf32>) outs(%alloc : memref<20x10xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.mulf %in, %in_0 : f32
      %3 = arith.addf %out, %2 : f32
      linalg.yield %3 : f32
    }
    linalg.generic {indexing_maps = [#map, #map4, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc, %1, %arg0 : memref<20x10xf32>, memref<10xf32>, memref<20x10xf32>) outs(%alloc : memref<20x10xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      %3 = arith.addf %2, %in_1 : f32
      linalg.yield %3 : f32
    }
    memref.copy %alloc, %arg1 : memref<20x10xf32> to memref<20x10xf32>
    return
  }
  func.func @main_wrapper(%arg0: memref<20x10xf32>, %arg1: memref<20x10xf32>) {
    %alloc = memref.alloc() : memref<20x10xf32>
    %token = mhal.launch @main (%arg0, %alloc) : (memref<20x10xf32>, memref<20x10xf32>)
    mhal.await %token : !mhal.token
    memref.copy %alloc, %arg1 : memref<20x10xf32> to memref<20x10xf32>
    return
  }
  module @__xmodule_ attributes {mhal.arch = "gfx950", mhal.module} {
    memref.global "private" constant @__constant_10xf32 : memref<10xf32> = dense_resource<torch_tensor_10_torch.float32> {alignment = 64 : i64}
    memref.global "private" constant @__constant_10x10xf32 : memref<10x10xf32> = dense_resource<torch_tensor_10_10_torch.float32> {alignment = 64 : i64}
    func.func @main(%arg0: memref<20x10xf32> {mhal.read_access}, %arg1: memref<20x10xf32> {mhal.write_access}) attributes {kernel, original_func = @main} {
      %0 = memref.get_global @__constant_10x10xf32 : memref<10x10xf32>
      %1 = memref.get_global @__constant_10xf32 : memref<10xf32>
      %2 = rock.transform %0 by #transform_map : memref<10x10xf32> to memref<10x10xf32>
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<20x10xf32>
      %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<20x10xf32>
      rock.gemm %alloc_0 = %arg0 * %2 storeMethod =  set : memref<20x10xf32> = memref<20x10xf32> * memref<10x10xf32>
      linalg.generic {indexing_maps = [#map, #map4, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_0, %1, %arg0 : memref<20x10xf32>, memref<10xf32>, memref<20x10xf32>) outs(%alloc : memref<20x10xf32>) {
      ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32):
        %3 = arith.addf %in, %in_1 : f32
        %4 = arith.addf %3, %in_2 : f32
        linalg.yield %4 : f32
      }
      memref.copy %alloc, %arg1 : memref<20x10xf32> to memref<20x10xf32>
      return
    }
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_10_torch.float32: "0x040000006D8581BEC69643BE4A65FB3DD8424B3DE032C53D44A64EBEC7E7F83D14F6133C5F8C223ED542273E",
      torch_tensor_10_10_torch.float32: "0x0400000019C28ABE851BF3BC61CADD3C6C1840BC4DF9953E417C3FBEDFB857BD51E3B93DD6D289BD9687283A179E103D6F6FFFBD9687623D3229B23D8437B1BD9502943EDA0680BEE03A5FBEA40D973E95343E3E14A0223E18DA0D3D93826ABC4C7787BE6C579CBEC0AF673E2F6DB2BD85254FBE3D6D653E1B213E3D77B0113E932A0E3E7F02DF3DEBCA1FBEE946B73DB43D343E2EAF2C3EE12F11BE37A4B03D6330F23D812B9BBC28376CBE11A333BE879CCDBD511F96BE6B365F3EECB92CBE95F59D3E7B5C183E9FEDC73D04DF20BDAF13A43D2480CCB9A5D16ABE272A79BEC43F423C66B74C3D01F43F3E3A6801BE7EC9DA3D3F5B0DBEEA423EBEC17189BD7E5A92BEDB291A3CE1DB21BD49F0583E6F7B6CBC5AAA98BDB7D072BED818DB3CFE179D3EA73854BEFCF2D5BDB2C21FBEA07C963DF5DE0A3EBC1DD6BC8A2C66BDB444E83D689F74BEC20034BEAF3DE83D7640E3BD4E221DBE531AA7BDE11EBC3CDACB97BE544E5ABE52D5B2BC8B410BBEAE1482BE8902DCBD1639053E0CCE323E20A593BEB8D835BD30C371BE37185FBEE2BC3D3E"
    }
  }
#-}
```


## Tensorflow

### Overview

Tensorflow is a bit more complicated.

Tensorflow -> tf (Dialect) -> stablehlo -> linalg -> rock

### Tensorflow using rocMLIR as a backend

Generating Tensorflow: 

```
import tensorflow as tf
import numpy as np
class GemmAdd(tf.Module):
    @tf.function(input_signature=[
        tf.TensorSpec([10, 10], tf.float32),
        tf.TensorSpec([10, 10], tf.float32),
        tf.TensorSpec([10, 10], tf.float32),
    ])
    def forward(self, a, b, d):
        return tf.add(tf.linalg.matmul(a, b), d)
        
model = GemmAdd()
cf = model.forward.get_concrete_function()
print(tf.mlir.experimental.convert_function(cf))
tf.saved_model.save(model, "gemm_add_savedmodel",
                    signatures={"serving_default": model.forward})
```

Outputs: 

```
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 2474 : i32}} {
  func.func @__inference_forward_88(%arg0: tensor<10x10xf32> {tf._user_specified_name = "a"}, %arg1: tensor<10x10xf32> {tf._user_specified_name = "b"}, %arg2: tensor<10x10xf32> {tf._user_specified_name = "d"}) -> tensor<10x10xf32> attributes {allow_soft_placement = false, tf.entry_function = {control_outputs = "", inputs = "a,b,d", outputs = "identity_RetVal"}} {
    %0 = "tf.MatMul"(%arg0, %arg1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> {device = ""} : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    %1 = "tf.AddV2"(%0, %arg2) {device = ""} : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    %2 = "tf.Identity"(%1) {device = ""} : (tensor<10x10xf32>) -> tensor<10x10xf32>
    return %2 : tensor<10x10xf32>
  }
}
```

Running the following command to go from tensorflow to stablehlo (note that the stablehlo is in byte code format)

```
iree-import-tf --tf-import-type=savedmodel_v2  \
    --tf-savedmodel-exported-names=forward   \
    gemm_add_savedmodel -o gemm_add_stablehlo.mlir
```

We can see the following IR in stablehlo: 

```
module {
  func.func @forward(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>, %arg2: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %0 = stablehlo.dot %arg0, %arg1, precision = [DEFAULT, DEFAULT] : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<10x10xf32>
    return %1 : tensor<10x10xf32>
  }
}

```

Running the following command to go from stablehlo into linalg

```
~/frameworks/stablehlo/build/bin/stablehlo-opt gemm_add_stablehlo.mlir --stablehlo-legalize-to-linalg=enable-primitive-ops
```

Output: 

```
module {
  func.func @forward(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>, %arg2: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %0 = tensor.empty() : tensor<10x10xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<10x10xf32>) -> tensor<10x10xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<10x10xf32>, tensor<10x10xf32>) outs(%1 : tensor<10x10xf32>) -> tensor<10x10xf32>
    %3 = tensor.empty() : tensor<10x10xf32>
    %mapped = linalg.map { arith.addf } ins(%2, %arg2 : tensor<10x10xf32>, tensor<10x10xf32>) outs(%3 : tensor<10x10xf32>)
    return %mapped : tensor<10x10xf32>
  }
}
```

Finally, outputting rocMLIR from linalg:

```
~/rocMLIR/build-release/bin/rocmlir-gen rocmlir.mlir --clone-harness --fut forward --arch gfx950  | ~/rocMLIR/build-release/bin/rocmlir-driver --kernel-pipeline=highlevel
```

We have the following code

```
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @forward(%arg0: memref<10x10xf32> {mhal.read_access}, %arg1: memref<10x10xf32> {mhal.read_access}, %arg2: memref<10x10xf32> {mhal.read_access}, %arg3: memref<10x10xf32> {mhal.write_access}) {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<10x10xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%alloc : memref<10x10xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<10x10xf32>, memref<10x10xf32>) outs(%alloc : memref<10x10xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %0 = arith.mulf %in, %in_1 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<10x10xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc, %arg2 : memref<10x10xf32>, memref<10x10xf32>) outs(%alloc_0 : memref<10x10xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %0 = arith.addf %in, %in_1 : f32
      linalg.yield %0 : f32
    }
    memref.copy %alloc_0, %arg3 : memref<10x10xf32> to memref<10x10xf32>
    return
  }
  func.func @forward_wrapper(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>, %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32>) {
    %alloc = memref.alloc() : memref<10x10xf32>
    %token = mhal.launch @forward (%arg0, %arg1, %arg2, %alloc) : (memref<10x10xf32>, memref<10x10xf32>, memref<10x10xf32>, memref<10x10xf32>)
    mhal.await %token : !mhal.token
    memref.copy %alloc, %arg3 : memref<10x10xf32> to memref<10x10xf32>
    return
  }
  module @__xmodule_ attributes {mhal.arch = "gfx950", mhal.module} {
    func.func @forward(%arg0: memref<10x10xf32> {mhal.read_access}, %arg1: memref<10x10xf32> {mhal.read_access}, %arg2: memref<10x10xf32> {mhal.read_access}, %arg3: memref<10x10xf32> {mhal.write_access}) attributes {kernel, original_func = @forward} {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<10x10xf32>
      rock.gemm %alloc = %arg0 * %arg1 storeMethod =  set : memref<10x10xf32> = memref<10x10xf32> * memref<10x10xf32>
      %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<10x10xf32>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc, %arg2 : memref<10x10xf32>, memref<10x10xf32>) outs(%alloc_0 : memref<10x10xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %0 = arith.addf %in, %in_1 : f32
        linalg.yield %0 : f32
      }
      memref.copy %alloc_0, %arg3 : memref<10x10xf32> to memref<10x10xf32>
      return
    }
  }
}
```