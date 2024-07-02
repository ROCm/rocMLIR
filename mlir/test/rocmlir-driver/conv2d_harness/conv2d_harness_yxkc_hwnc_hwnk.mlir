// RUN: rocmlir-gen --arch %arch -p -fil_layout=gyxkc -in_layout=hwngc -out_layout=hwngk %s | FileCheck %s --check-prefix=HARNESS
// RUN: rocmlir-gen --arch %arch -p -fil_layout=gyxkc -in_layout=hwngc -out_layout=hwngk %s | rocmlir-driver -c | FileCheck %s --check-prefix=LOWERING
// RUN: rocmlir-gen --arch %arch -p -fil_layout=gyxkc -in_layout=hwngc -out_layout=hwngk %s | rocmlir-driver -c | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=E2E

func.func private @rock_conv_g01kc_01ngc_01ngk_0(%filter : memref<9216xf32>, %input : memref<1048576xf32>, %output : memref<14745600xf32>) -> ()

// HARNESS: module
// HARNESS: func.func @rock_conv_g01kc_01ngc_01ngk_0([[FILTER_MEMREF:%.*]]: memref<9216xf32>, [[INPUT_MEMREF:%.*]]: memref<1048576xf32>, [[OUTPUT_MEMREF:%.*]]: memref<14745600xf32>)
// LOWERING: module
// LOWERING: gpu.binary @rock_conv_g01kc_01ngc_01ngk_0_module

func.func @main() {
  // memref.allocate CPU memory.
  %0 = memref.alloc() : memref<9216xf32>
  %1 = memref.alloc() : memref<1048576xf32>
  %2 = memref.alloc() : memref<14745600xf32>

  // populate initial values.
  %cst = arith.constant 1.0 : f32
  linalg.fill ins(%cst : f32) outs(%0 : memref<9216xf32>)
  linalg.fill ins(%cst : f32) outs(%1 : memref<1048576xf32>)
  linalg.fill ins(%cst : f32) outs(%2 : memref<14745600xf32>)

  // memref.allocate GPU memory.
  %filter = gpu.alloc  () : memref<9216xf32>
  %input = gpu.alloc  () : memref<1048576xf32>
  %output = gpu.alloc  () : memref<14745600xf32>

  // transfer data CPU -> GPU.
  gpu.memcpy  %filter, %0 : memref<9216xf32>, memref<9216xf32>
  gpu.memcpy  %input, %1 : memref<1048576xf32>, memref<1048576xf32>
  gpu.memcpy  %output, %2 : memref<14745600xf32>, memref<14745600xf32>

  // launch kernel.
  call @rock_conv_g01kc_01ngc_01ngk_0(%filter, %input, %output) : (memref<9216xf32>, memref<1048576xf32>, memref<14745600xf32>) -> ()

  // transfer data GPU -> CPU.
  gpu.memcpy  %2, %output : memref<14745600xf32>,memref<14745600xf32>

  // verify result.
  // TBD. Add more verifying logic.
  %6 = memref.cast %2 : memref<14745600xf32> to memref<*xf32>
  call @printMemrefF32(%6) : (memref<*xf32>) -> ()

  // dellocate GPU memory.
  gpu.dealloc  %filter : memref<9216xf32>
  gpu.dealloc  %input : memref<1048576xf32>
  gpu.dealloc  %output : memref<14745600xf32>

  // memref.deallocate CPU memory.
  memref.dealloc %0 : memref<9216xf32>
  memref.dealloc %1 : memref<1048576xf32>
  memref.dealloc %2 : memref<14745600xf32>

  return
}

func.func private @printMemrefF32(%ptr : memref<*xf32>)
// E2E: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [14745600] strides = [1] data =
