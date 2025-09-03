// Check that we can properly use `mlirIsModuleFusible` on ReduceOps without
// having GemmFeatures set on the func

// clang-format off
// RUN: mlir-reduce-fusible-test
// clang-format on

#include "mlir-c/Dialect/Rock.h"
#include "mlir-c/RegisterRocMLIR.h"

#include <iostream>
#include <string>

static bool testReduceFusible(MlirContext ctx) {
  // clang-format off
  const char *mlirModuleText = R"mlir(
    module {
      func.func @mlir_convolution_reshape_reshape_broadcast_add_mul_reshape_reduce_sum_reshape_mul_mul_reshape_reduce_sum_reshape(%arg0: memref<32768xf32>, %arg1: memref<11520xf32>, %arg2: memref<320xf32>, %arg3: memref<64xf32> {mhal.read_access, rock.prefill = 0.000000e+00 : f32}, %arg4: memref<64xf32> {mhal.read_access, rock.prefill = 0.000000e+00 : f32}, %arg5: memref<2621440xf32>) attributes {arch = "gfx942:sramecc+:xnack-", enable_splitk_for_tuning, kernel = "mixr", num_cu = 304 : i64} {
        %cst = arith.constant 2.44140629E-5 : f32
        %0 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3) -> (((d0 * 4 + d1) * 3 + d2) * 3 + d3)> by [<Unmerge{320, 4, 3, 3} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [320, 4, 3, 3] -> [11520]> : memref<11520xf32> to memref<320x4x3x3xf32>
        %1 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> (((d0 * 4 + d1) * 64 + d2) * 64 + d3)> by [<Unmerge{2, 4, 64, 64} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [2, 4, 64, 64] -> [32768]> : memref<32768xf32> to memref<2x4x64x64xf32>
        %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x320x64x64xf32>
        %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 4 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 4} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [2, 1, 4, 64, 64] -> [2, 4, 64, 64]> : memref<2x4x64x64xf32> to memref<2x1x4x64x64xf32>
        %3 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0 * 320 + d1, d2, d3, d4)> by [<PassThrough ["c", "y", "x"] at [2, 3, 4] -> ["c", "y", "x"] at [1, 2, 3]>, <Unmerge{1, 320} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 320, 4, 3, 3] -> [320, 4, 3, 3]> : memref<320x4x3x3xf32> to memref<1x320x4x3x3xf32>
        %4 = rock.transform %alloc by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 320 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 320} ["g", "k"] at [1, 2] -> ["k"] at [1]>] bounds = [2, 1, 320, 64, 64] -> [2, 320, 64, 64]> : memref<2x320x64x64xf32> to memref<2x1x320x64x64xf32>
        %5 = rock.transform %3 by <affine_map<(d0, d1, d2, d3, d4) -> (d3, d0, d1, d2, d4)> by [<PassThrough ["dim1", "dim2", "dim3", "dim0", "dim4"] at [0, 1, 2, 3, 4] -> ["dim1", "dim2", "dim3", "dim0", "dim4"] at [1, 2, 3, 0, 4]>] bounds = [320, 4, 3, 1, 3] -> [1, 320, 4, 3, 3]> : memref<1x320x4x3x3xf32> to memref<320x4x3x1x3xf32>
        %6 = rock.transform %2 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)> by [<PassThrough ["dim0", "dim2", "dim3", "dim1", "dim4"] at [0, 1, 2, 3, 4] -> ["dim0", "dim2", "dim3", "dim1", "dim4"] at [0, 2, 3, 1, 4]>] bounds = [2, 4, 64, 1, 64] -> [2, 1, 4, 64, 64]> : memref<2x1x4x64x64xf32> to memref<2x4x64x1x64xf32>
        rock.conv(%3, %2, %4) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index]} : memref<1x320x4x3x3xf32>, memref<2x1x4x64x64xf32>, memref<2x1x320x64x64xf32>
        %7 = rock.transform %alloc by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 10 + d2, d3, d4)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Unmerge{32, 10} ["exp1", "exp2"] at [1, 2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [4] -> ["dim3"] at [3]>] bounds = [2, 32, 10, 64, 64] -> [2, 320, 64, 64]> : memref<2x320x64x64xf32> to memref<2x32x10x64x64xf32>
        %8 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (d1 * 10 + d2)> by [<Unmerge{32, 10} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit3"] at [3] -> [] at []>, <AddDim{1} ["unit4"] at [4] -> [] at []>] bounds = [1, 32, 10, 1, 1] -> [320]> : memref<320xf32> to memref<1x32x10x1x1xf32>
        %9 = rock.transform %8 by <affine_map<(d0, d1, d2, d3, d4) -> (0, d1, d2, 0, 0)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <Broadcast{1} ["dim3"] at [3] -> ["dim3"] at [3]>, <Broadcast{1} ["dim4"] at [4] -> ["dim4"] at [4]>] bounds = [2, 32, 10, 64, 64] -> [1, 32, 10, 1, 1]> : memref<1x32x10x1x1xf32> to memref<2x32x10x64x64xf32>
        %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x32x10x64x64xf32>
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%7, %9 : memref<2x32x10x64x64xf32>, memref<2x32x10x64x64xf32>) outs(%alloc_0 : memref<2x32x10x64x64xf32>) {
        ^bb0(%in: f32, %in_5: f32, %out: f32):
        %15 = arith.addf %in, %in_5 : f32
        linalg.yield %15 : f32
        }
        %10 = rock.transform %alloc_0 by <affine_map<(d0) -> (d0 floordiv 1310720, (d0 mod 1310720) floordiv 40960, (d0 mod 40960) floordiv 4096, (d0 mod 4096) floordiv 64, d0 mod 64)> by [<Merge{2, 32, 10, 64, 64} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3", "col4"] at [0, 1, 2, 3, 4]>] bounds = [2621440] -> [2, 32, 10, 64, 64]> : memref<2x32x10x64x64xf32> to memref<2621440xf32>
        %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<2x32x10x64x64xf32>
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%alloc_0 : memref<2x32x10x64x64xf32>) outs(%alloc_1 : memref<2x32x10x64x64xf32>) {
        ^bb0(%in: f32, %out: f32):
        %15 = arith.mulf %in, %cst : f32
        linalg.yield %15 : f32
        }
        %11 = rock.transform %alloc_1 by <affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 4096, (d2 mod 4096) floordiv 64, d2 mod 64)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Merge{10, 64, 64} ["dim2"] at [2] -> ["col2", "col3", "col4"] at [2, 3, 4]>] bounds = [2, 32, 40960] -> [2, 32, 10, 64, 64]> : memref<2x32x10x64x64xf32> to memref<2x32x40960xf32>
        %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<2x32x1xf32>
        rock.reduce  sum %11 into %alloc_2 {axis = 2 : index, blockSize = 256 : i32, gridSize = 6080 : i32} : memref<2x32x40960xf32> into memref<2x32x1xf32>
        %12 = rock.transform %alloc_2 by <affine_map<(d0) -> (d0 floordiv 32, d0 mod 32, 0)> by [<Merge{2, 32, 1} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [64] -> [2, 32, 1]> : memref<2x32x1xf32> to memref<64xf32>
        %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<2x32x10x64x64xf32>
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%alloc_0 : memref<2x32x10x64x64xf32>) outs(%alloc_3 : memref<2x32x10x64x64xf32>) {
        ^bb0(%in: f32, %out: f32):
        %15 = arith.mulf %in, %in : f32
        %16 = arith.mulf %15, %cst : f32
        linalg.yield %16 : f32
        }
        %13 = rock.transform %alloc_3 by <affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 4096, (d2 mod 4096) floordiv 64, d2 mod 64)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Merge{10, 64, 64} ["dim2"] at [2] -> ["col2", "col3", "col4"] at [2, 3, 4]>] bounds = [2, 32, 40960] -> [2, 32, 10, 64, 64]> : memref<2x32x10x64x64xf32> to memref<2x32x40960xf32>
        %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<2x32x1xf32>
        rock.reduce  sum %13 into %alloc_4 {axis = 2 : index, blockSize = 256 : i32, gridSize = 6080 : i32} : memref<2x32x40960xf32> into memref<2x32x1xf32>
        %14 = rock.transform %alloc_4 by <affine_map<(d0) -> (d0 floordiv 32, d0 mod 32, 0)> by [<Merge{2, 32, 1} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [64] -> [2, 32, 1]> : memref<2x32x1xf32> to memref<64xf32>
        memref.copy %12, %arg3 : memref<64xf32> to memref<64xf32>
        memref.copy %14, %arg4 : memref<64xf32> to memref<64xf32>
        memref.copy %10, %arg5 : memref<2621440xf32> to memref<2621440xf32>
        return
      }
    }
    )mlir";
  // clang-format on

  // Parse the module from the string
  MlirStringRef moduleStr = mlirStringRefCreateFromCString(mlirModuleText);
  MlirModule moduleOp = mlirModuleCreateParse(ctx, moduleStr);

  if (mlirModuleIsNull(moduleOp)) {
    std::cerr << "Failed to parse module" << std::endl;
    return false;
  }

  // Create performance configuration string
  std::string perfConfigStr = "v3:64,64,16,32,32,4,4,1,2,1,1";
  MlirStringRef perfStr = mlirStringRefCreateFromCString(perfConfigStr.c_str());

  // Test whether the module is fusible
  const bool isFusible = mlirIsModuleFusible(moduleOp, perfStr);
 
  // Clean up
  mlirModuleDestroy(moduleOp);

  return !isFusible;
}
int main(int argc, char *argv[]) {
  // Create MLIR context and register dialects
  MlirContext ctx = mlirContextCreate();
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirRegisterRocMLIRDialects(registry);
  mlirRegisterRocMLIRPasses();
  mlirContextAppendDialectRegistry(ctx, registry);
  mlirContextLoadAllAvailableDialects(ctx);
  mlirDialectRegistryDestroy(registry);

  // Test the module fusibility
  bool isOk = testReduceFusible(ctx);

  // Clean up
  mlirContextDestroy(ctx);

  if (!isOk) {
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  return 0;
}
