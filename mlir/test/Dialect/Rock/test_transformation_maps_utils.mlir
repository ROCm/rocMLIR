// RUN: rocmlir-opt -rock-transform-maps-utils-test -allow-unregistered-dialect --mlir-print-local-scope %s | FileCheck %s

#tr0_ex0 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1, d0, d2)> by [<PassThrough ["B", "A", "C"] at [0, 1, 2] -> ["A", "B", "C"] at [1, 0, 2]>] bounds = [1, 4, 32] -> [4, 1, 32]>
#tr1_ex0 = #rock.transform_map<affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 4} ["A"] at [0] -> ["A", "B"] at [0, 1]>, <PassThrough ["C"] at [1] -> ["C"] at [2]>] bounds = [4, 32] -> [1, 4, 32]>

// CHECK-LABEL: @transform_ex0_t0
// CHECK-SAME: ([[orig:%.*]]: [[orig_shape:memref<32xf32>]])
func.func @transform_ex0_t0(%arg0: memref<4x1x32xf32>) {
  // CHECK-NEXT: [[next0:%.*]] = rock.transform [[orig]] by <affine_map<(d0) -> (d0)> by [<PassThrough ["C"] at [0] -> ["C"] at [0]>] bounds = [32] -> [32]> : [[orig_shape]] to memref<32xf32>
  %0 = rock.transform %arg0 by #tr0_ex0 : memref<4x1x32xf32> to memref<1x4x32xf32>
  // CHECK-NEXT: [[next1:%.*]] = rock.transform [[next0]] by <affine_map<(d0) -> (d0)> by [<PassThrough ["C"] at [0] -> ["C"] at [0]>] bounds = [32] -> [32]> : memref<32xf32> to memref<32xf32>
  %1 = rock.transform %0 by #tr1_ex0 : memref<1x4x32xf32> to memref<4x32xf32>
  "remove_dims"(%1) {names_to_drop = ["A"]} : (memref<4x32xf32>) -> ()
  return
}

// CHECK-LABEL: @transform_ex0_t1
// CHECK-SAME: ([[orig:%.*]]: [[orig_shape:memref<4x1xf32>]])
func.func @transform_ex0_t1(%arg0: memref<4x1x32xf32>) {
  // CHECK-NEXT: [[next0:%.*]] = rock.transform [[orig]] by <affine_map<(d0, d1) -> (d1, d0)> by [<PassThrough ["B", "A"] at [0, 1] -> ["A", "B"] at [1, 0]>] bounds = [1, 4] -> [4, 1]> : [[orig_shape]] to memref<1x4xf32>
  %0 = rock.transform %arg0 by #tr0_ex0 : memref<4x1x32xf32> to memref<1x4x32xf32>
  // CHECK-NEXT: [[next1:%.*]] = rock.transform [[next0]] by <affine_map<(d0) -> (0, d0)> by [<Merge{1, 4} ["A"] at [0] -> ["A", "B"] at [0, 1]>] bounds = [4] -> [1, 4]> : memref<1x4xf32> to memref<4xf32>
  %1 = rock.transform %0 by #tr1_ex0 : memref<1x4x32xf32> to memref<4x32xf32>
  "remove_dims"(%1) {names_to_drop = ["C"]} : (memref<4x32xf32>) -> ()
  return
}

// CHECK-LABEL: @transform_ex0_t2
// CHECK-SAME: ([[orig:%.*]]: [[orig_shape:memref<4x1x32xf32>]])
func.func @transform_ex0_t2(%arg0: memref<4x1x32xf32>) {
  // CHECK-NEXT: [[next0:%.*]] = rock.transform [[orig]] by <affine_map<(d0, d1, d2) -> (d1, d0, d2)> by [<PassThrough ["B", "A", "C"] at [0, 1, 2] -> ["A", "B", "C"] at [1, 0, 2]>] bounds = [1, 4, 32] -> [4, 1, 32]> : [[orig_shape]] to memref<1x4x32xf32>
  %0 = rock.transform %arg0 by #tr0_ex0 : memref<4x1x32xf32> to memref<1x4x32xf32>
  // CHECK-NEXT: [[next1:%.*]] = rock.transform [[next0]] by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 4} ["A"] at [0] -> ["A", "B"] at [0, 1]>, <PassThrough ["C"] at [1] -> ["C"] at [2]>] bounds = [4, 32] -> [1, 4, 32]> : memref<1x4x32xf32> to memref<4x32xf32>
  %1 = rock.transform %0 by #tr1_ex0 : memref<1x4x32xf32> to memref<4x32xf32>
  "remove_dims"(%1) {names_to_drop = ["X"]} : (memref<4x32xf32>) -> ()
  return
}


#tr0_ex1 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 1]>] bounds = [1, 1024, 64] -> [1, 64, 1024]>
#tr1_ex1 = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (d0, d1 * 256 + d2, d3)> by [<PassThrough ["A", "C"] at [0, 3] -> ["A", "C"] at [0, 2]>, <Unmerge{4, 256} ["D", "B"] at [1, 2] -> ["B"] at [1]>] bounds = [1, 4, 256, 64] -> [1, 1024, 64]>

// CHECK-LABEL: @transform_ex1_t0
// CHECK-SAME: ([[orig:%.*]]: [[orig_shape:memref<1x64x256xf32>]])
func.func @transform_ex1_t0(%arg0: memref<1x64x1024xf32>) {
  // CHECK-NEXT: [[next0:%.*]] = rock.transform [[orig]] by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 1]>] bounds = [1, 256, 64] -> [1, 64, 256]> : [[orig_shape]] to memref<1x256x64xf32>
  %0 = rock.transform %arg0 by #tr0_ex1 : memref<1x64x1024xf32> to memref<1x1024x64xf32>
  // CHECK-NEXT: [[next1:%.*]] = rock.transform [[next0]] by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["A", "C"] at [0, 2] -> ["A", "C"] at [0, 2]>, <Unmerge{256} ["B"] at [1] -> ["B"] at [1]>] bounds = [1, 256, 64] -> [1, 256, 64]> : memref<1x256x64xf32> to memref<1x256x64xf32>
  %1 = rock.transform %0 by #tr1_ex1 : memref<1x1024x64xf32> to memref<1x4x256x64xf32>
  "remove_dims"(%1) {names_to_drop = ["D"]} : (memref<1x4x256x64xf32>) -> ()
  return
}

// CHECK-LABEL: @transform_ex1_t1
// CHECK-SAME: ([[orig:%.*]]: [[orig_shape:memref<1x64x4xf32>]])
func.func @transform_ex1_t1(%arg0: memref<1x64x1024xf32>) {
  // CHECK-NEXT: [[next0:%.*]] = rock.transform [[orig]] by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 1]>] bounds = [1, 4, 64] -> [1, 64, 4]> : [[orig_shape]] to memref<1x4x64xf32>
  %0 = rock.transform %arg0 by #tr0_ex1 : memref<1x64x1024xf32> to memref<1x1024x64xf32>
  // CHECK-NEXT: [[next1:%.*]] = rock.transform [[next0]] by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["A", "C"] at [0, 2] -> ["A", "C"] at [0, 2]>, <Unmerge{4} ["D"] at [1] -> ["B"] at [1]>] bounds = [1, 4, 64] -> [1, 4, 64]> : memref<1x4x64xf32> to memref<1x4x64xf32>
  %1 = rock.transform %0 by #tr1_ex1 : memref<1x1024x64xf32> to memref<1x4x256x64xf32>
  "remove_dims"(%1) {names_to_drop = ["B"]} : (memref<1x4x256x64xf32>) -> ()
  return
}

// CHECK-LABEL: @transform_ex1_t2
// CHECK-SAME: ([[orig:%.*]]: [[orig_shape:memref<1x64xf32>]])
func.func @transform_ex1_t2(%arg0: memref<1x64x1024xf32>) {
  // CHECK-NEXT: [[next0:%.*]] = rock.transform [[orig]] by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["C"] at [1] -> ["C"] at [1]>] bounds = [1, 64] -> [1, 64]> : [[orig_shape]] to memref<1x64xf32>
  %0 = rock.transform %arg0 by #tr0_ex1 : memref<1x64x1024xf32> to memref<1x1024x64xf32>
  // CHECK-NEXT: [[next1:%.*]] = rock.transform [[next0]] by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["A", "C"] at [0, 1] -> ["A", "C"] at [0, 1]>] bounds = [1, 64] -> [1, 64]> : memref<1x64xf32> to memref<1x64xf32>
  %1 = rock.transform %0 by #tr1_ex1 : memref<1x1024x64xf32> to memref<1x4x256x64xf32>
  "remove_dims"(%1) {names_to_drop = ["B", "D"]} : (memref<1x4x256x64xf32>) -> ()
  return
}

// CHECK-LABEL: @transform_ex1_t3
// CHECK-SAME: ([[orig:%.*]]: [[orig_shape:memref<1024xf32>]])
func.func @transform_ex1_t3(%arg0: memref<1x64x1024xf32>) {
  // CHECK-NEXT: [[next0:%.*]] = rock.transform [[orig]] by <affine_map<(d0) -> (d0)> by [<PassThrough ["B"] at [0] -> ["B"] at [0]>] bounds = [1024] -> [1024]> : [[orig_shape]] to memref<1024xf32>
  %0 = rock.transform %arg0 by #tr0_ex1 : memref<1x64x1024xf32> to memref<1x1024x64xf32>
  // CHECK-NEXT: [[next1:%.*]] = rock.transform [[next0]] by <affine_map<(d0, d1) -> (d0 * 256 + d1)> by [<Unmerge{4, 256} ["D", "B"] at [0, 1] -> ["B"] at [0]>] bounds = [4, 256] -> [1024]> : memref<1024xf32> to memref<4x256xf32>
  %1 = rock.transform %0 by #tr1_ex1 : memref<1x1024x64xf32> to memref<1x4x256x64xf32>
  "remove_dims"(%1) {names_to_drop = ["A", "C"]} : (memref<1x4x256x64xf32>) -> ()
  return
}


#tr0_ex2 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 1]>] bounds = [1, 1024, 64] -> [1, 64, 1024]>
#tr1_ex2 = #rock.transform_map<affine_map<(d0, d1, d2) -> (0, d1, d2)> by [<Broadcast{32} ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [1, 2]>] bounds = [32, 1024, 64] -> [1, 1024, 64]>

// CHECK-LABEL: @transform_ex2_t0
// CHECK-SAME: ([[orig:%.*]]: [[orig_shape:memref<64x1024xf32>]])
func.func @transform_ex2_t0(%arg0: memref<1x64x1024xf32>) {
  // CHECK-NEXT: [[next0:%.*]] = rock.transform [[orig]] by <affine_map<(d0, d1) -> (d1, d0)> by [<PassThrough ["B", "C"] at [0, 1] -> ["B", "C"] at [1, 0]>] bounds = [1024, 64] -> [64, 1024]> : [[orig_shape]] to memref<1024x64xf32>
  %0 = rock.transform %arg0 by #tr0_ex2 : memref<1x64x1024xf32> to memref<1x1024x64xf32>
  // CHECK-NEXT: [[next1:%.*]] = rock.transform [[next0]] by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["B", "C"] at [0, 1] -> ["B", "C"] at [0, 1]>] bounds = [1024, 64] -> [1024, 64]> : memref<1024x64xf32> to memref<1024x64xf32>
  %1 = rock.transform %0 by #tr1_ex2 : memref<1x1024x64xf32> to memref<32x1024x64xf32>
  "remove_dims"(%1) {names_to_drop = ["A"]} : (memref<32x1024x64xf32>) -> ()
  return
}

// CHECK-LABEL: @transform_ex2_t1
// CHECK-SAME: ([[orig:%.*]]: [[orig_shape:memref<1x1024xf32>]])
func.func @transform_ex2_t1(%arg0: memref<1x64x1024xf32>) {
  // CHECK-NEXT: [[next0:%.*]] = rock.transform [[orig]] by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B"] at [1] -> ["B"] at [1]>] bounds = [1, 1024] -> [1, 1024]> : [[orig_shape]] to memref<1x1024xf32>
  %0 = rock.transform %arg0 by #tr0_ex2 : memref<1x64x1024xf32> to memref<1x1024x64xf32>
  // CHECK-NEXT: [[next1:%.*]] = rock.transform [[next0]] by <affine_map<(d0, d1) -> (d0 mod 32, d1)> by [<Broadcast{32} ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B"] at [1] -> ["B"] at [1]>] bounds = [32, 1024] -> [1, 1024]> : memref<1x1024xf32> to memref<32x1024xf32>
  %1 = rock.transform %0 by #tr1_ex2 : memref<1x1024x64xf32> to memref<32x1024x64xf32>
  "remove_dims"(%1) {names_to_drop = ["C"]} : (memref<32x1024x64xf32>) -> ()
  return
}


#tr0_ex3 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 1]>] bounds = [1, 1024, 64] -> [1, 64, 1024]>
#tr1_ex3 = #rock.transform_map<affine_map<(d0, d2) -> (d0, 0, d2)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <ConstDim{0, 1} [] at [] -> ["B"] at [1]>, <PassThrough ["C"] at [1] -> ["C"] at [2]>] bounds = [1, 64] -> [1, 1024, 64]>

// CHECK-LABEL: @transform_ex3_t0
// CHECK-SAME: ([[orig:%.*]]: [[orig_shape:memref<64x1024xf32>]])
func.func @transform_ex3_t0(%arg0: memref<1x64x1024xf32>) {
  // CHECK-NEXT: [[next0:%.*]] = rock.transform [[orig]] by <affine_map<(d0, d1) -> (d1, d0)> by [<PassThrough ["B", "C"] at [0, 1] -> ["B", "C"] at [1, 0]>] bounds = [1024, 64] -> [64, 1024]> : [[orig_shape]] to memref<1024x64xf32>
  %0 = rock.transform %arg0 by #tr0_ex3 : memref<1x64x1024xf32> to memref<1x1024x64xf32>
  // CHECK-NEXT: [[next1:%.*]] = rock.transform [[next0]] by <affine_map<(d0) -> (0, d0)> by [<ConstDim{0, 1} [] at [] -> ["B"] at [0]>, <PassThrough ["C"] at [0] -> ["C"] at [1]>] bounds = [64] -> [1024, 64]> : memref<1024x64xf32> to memref<64xf32>
  %1 = rock.transform %0 by #tr1_ex3 : memref<1x1024x64xf32> to memref<1x64xf32>
  "remove_dims"(%1) {names_to_drop = ["A"]} : (memref<1x64xf32>) -> ()
  return
}

// CHECK-LABEL: @transform_ex3_t1
// CHECK-SAME: ([[orig:%.*]]: [[orig_shape:memref<1x1024xf32>]])
func.func @transform_ex3_t1(%arg0: memref<1x64x1024xf32>) {
  // CHECK-NEXT: [[next0:%.*]] = rock.transform [[orig]] by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B"] at [1] -> ["B"] at [1]>] bounds = [1, 1024] -> [1, 1024]> : [[orig_shape]] to memref<1x1024xf32>
  %0 = rock.transform %arg0 by #tr0_ex3 : memref<1x64x1024xf32> to memref<1x1024x64xf32>
  // CHECK-NEXT: [[next1:%.*]] = rock.transform [[next0]] by <affine_map<(d0) -> (d0, 0)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <ConstDim{0, 1} [] at [] -> ["B"] at [1]>] bounds = [1] -> [1, 1024]> : memref<1x1024xf32> to memref<1xf32>
  %1 = rock.transform %0 by #tr1_ex3 : memref<1x1024x64xf32> to memref<1x64xf32>
  "remove_dims"(%1) {names_to_drop = ["C"]} : (memref<1x64xf32>) -> ()
  return
}

// CHECK-LABEL: @transform_ex3_t2
// CHECK-SAME: ([[orig:%.*]]: [[orig_shape:memref<1024xf32>]])
func.func @transform_ex3_t2(%arg0: memref<1x64x1024xf32>) {
  // CHECK-NEXT: [[next0:%.*]] = rock.transform [[orig]] by <affine_map<(d0) -> (d0)> by [<PassThrough ["B"] at [0] -> ["B"] at [0]>] bounds = [1024] -> [1024]> : [[orig_shape]] to memref<1024xf32>
  %0 = rock.transform %arg0 by #tr0_ex3 : memref<1x64x1024xf32> to memref<1x1024x64xf32>
  // CHECK-NEXT: [[next1:%.*]] = rock.transform [[next0]] by <affine_map<() -> (0)> by [<ConstDim{0, 1} [] at [] -> ["B"] at [0]>] bounds = [] -> [1024]> : memref<1024xf32> to memref<f32>
  %1 = rock.transform %0 by #tr1_ex3 : memref<1x1024x64xf32> to memref<1x64xf32>
  "remove_dims"(%1) {names_to_drop = ["A", "C"]} : (memref<1x64xf32>) -> ()
  return
}


#tr0_ex4 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 1]>] bounds = [1, 1024, 64] -> [1, 64, 1024]>
#tr1_ex4 = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)> by [<AddDim{8} ["X"] at [1] -> [] at []>, <PassThrough ["A", "B", "C"] at [0, 2, 3] -> ["A", "B", "C"] at [0, 1, 2]>] bounds = [1, 8, 1024, 64] -> [1, 1024, 64]>

// CHECK-LABEL: @transform_ex4_t0
// CHECK-SAME: ([[orig:%.*]]: [[orig_shape:memref<1x64x1024xf32>]])
func.func @transform_ex4_t0(%arg0: memref<1x64x1024xf32>) {
  // CHECK-NEXT: [[next0:%.*]] = rock.transform [[orig]] by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 1]>] bounds = [1, 1024, 64] -> [1, 64, 1024]> : [[orig_shape]] to memref<1x1024x64xf32>
  %0 = rock.transform %arg0 by #tr0_ex4 : memref<1x64x1024xf32> to memref<1x1024x64xf32>
  // CHECK-NEXT: [[next1:%.*]] = rock.transform [[next0]] by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["A", "B", "C"] at [0, 1, 2] -> ["A", "B", "C"] at [0, 1, 2]>] bounds = [1, 1024, 64] -> [1, 1024, 64]> : memref<1x1024x64xf32> to memref<1x1024x64xf32>
  %1 = rock.transform %0 by #tr1_ex4 : memref<1x1024x64xf32> to memref<1x8x1024x64xf32>
  "remove_dims"(%1) {names_to_drop = ["X"]} : (memref<1x8x1024x64xf32>) -> ()
  return
}

// CHECK-LABEL: @transform_ex4_t1
// CHECK-SAME: ([[orig:%.*]]: [[orig_shape:memref<f32>]])
func.func @transform_ex4_t1(%arg0: memref<1x64x1024xf32>) {
  %0 = rock.transform %arg0 by #tr0_ex4 : memref<1x64x1024xf32> to memref<1x1024x64xf32>
  // CHECK-NEXT: [[next0:%.*]] = rock.transform [[orig]] by <affine_map<(d0) -> ()> by [<AddDim{8} ["X"] at [0] -> [] at []>] bounds = [8] -> []> : [[orig_shape]] to memref<8xf32>
  %1 = rock.transform %0 by #tr1_ex4 : memref<1x1024x64xf32> to memref<1x8x1024x64xf32>
  "remove_dims"(%1) {names_to_drop = ["A", "B", "C"]} : (memref<1x8x1024x64xf32>) -> ()
  return
}
