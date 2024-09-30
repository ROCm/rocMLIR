// RUN: rocmlir-opt -rock-transform-maps-utils-test -allow-unregistered-dialect --mlir-print-local-scope %s | FileCheck %s

#tr0_ex0 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1, d0, d2)> by [<PassThrough ["B", "A", "C"] at [0, 1, 2] -> ["A", "B", "C"] at [1, 0, 2]>] bounds = [1, 4, 32] -> [4, 1, 32]>
#tr1_ex0 = #rock.transform_map<affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 4} ["A"] at [0] -> ["A", "B"] at [0, 1]>, <PassThrough ["C"] at [1] -> ["C"] at [2]>] bounds = [4, 32] -> [1, 4, 32]>

#tr2_ex0 = #rock.transform_map<affine_map<(d0) -> (d0 floordiv 32, d0 mod 32)> by [<Merge{4, 32} ["A"] at [0] -> ["A", "C"] at [0, 1]>] bounds = [128] -> [4, 32]>
#tr3_ex0 = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 16 + d1)> by [<Unmerge{8, 16} ["A1", "A2"] at [0, 1] -> ["A"] at [0]>] bounds = [8, 16] -> [128]>

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

// CHECK-LABEL: @transform_unmerge_merge_unmerge_pt
func.func @transform_unmerge_merge_unmerge_pt(%arg0: memref<4x1x32xf32>) {
  // CHECK: %[[TR1:.+]] = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1, d0, d2)> by [<PassThrough ["B", "A", "C"] at [0, 1, 2] -> ["A", "B", "C"] at [1, 0, 2]>] bounds = [1, 1, 16] -> [1, 1, 16]> : memref<1x1x16xf32> to memref<1x1x16xf32>
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1, d0, d2)> by [<PassThrough ["B", "A", "C"] at [0, 1, 2] -> ["A", "B", "C"] at [1, 0, 2]>] bounds = [1, 4, 32] -> [4, 1, 32]> : memref<4x1x32xf32> to memref<1x4x32xf32>
  // CHECK-NEXT: %[[TR2:.+]] = rock.transform %[[TR1]] by <affine_map<(d0, d1) -> (0, 0, d1)> by [<Merge{1, 1} ["A"] at [0] -> ["A", "B"] at [0, 1]>, <PassThrough ["C"] at [1] -> ["C"] at [2]>] bounds = [1, 16] -> [1, 1, 16]> : memref<1x1x16xf32> to memref<1x16xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 4} ["A"] at [0] -> ["A", "B"] at [0, 1]>, <PassThrough ["C"] at [1] -> ["C"] at [2]>] bounds = [4, 32] -> [1, 4, 32]> : memref<1x4x32xf32> to memref<4x32xf32>
  // CHECK-NEXT: %[[TR3:.+]] = rock.transform %[[TR2]] by <affine_map<(d0) -> (0, d0)> by [<Merge{1, 16} ["A"] at [0] -> ["A", "C"] at [0, 1]>] bounds = [16] -> [1, 16]> : memref<1x16xf32> to memref<16xf32>
  %2 = rock.transform %1 by <affine_map<(d0) -> (d0 floordiv 32, d0 mod 32)> by [<Merge{4, 32} ["A"] at [0] -> ["A", "C"] at [0, 1]>] bounds = [128] -> [4, 32]> : memref<4x32xf32> to memref<128xf32>
  // CHECK-NEXT: %[[TR4:.+]] = rock.transform %[[TR3]] by <affine_map<(d0) -> (d0)> by [<Unmerge{16} ["A2"] at [0] -> ["A"] at [0]>] bounds = [16] -> [16]> : memref<16xf32> to memref<16xf32>
  %3 = rock.transform %2 by <affine_map<(d0, d1) -> (d0 * 16 + d1)> by [<Unmerge{8, 16} ["A1", "A2"] at [0, 1] -> ["A"] at [0]>] bounds = [8, 16] -> [128]> : memref<128xf32> to memref<8x16xf32>
  "remove_dims"(%3) {names_to_drop = ["A1"]} : (memref<8x16xf32>) -> ()
  return
}

// CHECK-LABEL: @transform_unmerge_merge_pt_unmerge_merge
func.func @transform_unmerge_merge_pt_unmerge_merge(%arg0: memref<16x16x16xf32>) {
  // CHECK: %[[MERGE:.+]] = rock.transform %arg0 by <affine_map<(d0) -> (d0 floordiv 64, (d0 mod 64) floordiv 16, d0 mod 16)> by [<Merge{2, 4, 16} ["C"] at [0] -> ["C1", "C2", "C3"] at [0, 1, 2]>] bounds = [128] -> [2, 4, 16]> : memref<2x4x16xf32> to memref<128xf32>
  %merge = rock.transform %arg0 by <affine_map<(d0) -> (d0 floordiv (16 * 16), (d0 mod (16 * 16)) floordiv 16, d0 mod 16)> by [<Merge{16, 16, 16} ["C"] at [0] -> ["C1", "C2", "C3"] at [0, 1, 2]>] bounds = [4096] -> [16, 16, 16]> : memref<16x16x16xf32> to memref<4096xf32>
  // CHECK: %[[UNMERGE:.+]] = rock.transform %[[MERGE]] by <affine_map<(d0, d1, d2) -> ((d0 * 8 + d1) * 16 + d2)> by [<Unmerge{1, 8, 16} ["B2", "B1", "B3"] at [0, 1, 2] -> ["C"] at [0]>] bounds = [1, 8, 16] -> [128]> : memref<128xf32> to memref<1x8x16xf32>
  %unmerge =  rock.transform %merge by <affine_map<(d0, d1, d2) -> (d0 * 32 * 16 + d1 * 32 + d2)> by [<Unmerge{8, 16, 32} ["B2", "B1", "B3"] at [0, 1, 2] -> ["C"] at [0]>] bounds = [8, 16, 32] -> [4096]>  : memref<4096xf32> to memref<8x16x32xf32>
  // CHECK: %[[TRANSPOSED:.+]] = rock.transform %[[UNMERGE]] by <affine_map<(d0, d1, d2) -> (d1, d0, d2)> by [<PassThrough ["B1", "B2", "B3"] at [0, 1, 2] -> ["B2", "B1", "B3"] at [1, 0, 2]>] bounds = [8, 1, 16] -> [1, 8, 16]> : memref<1x8x16xf32> to memref<8x1x16xf32>
  %transposed = rock.transform %unmerge by <affine_map<(d0, d1, d2) -> (d1, d0, d2)> by [<PassThrough ["B1", "B2", "B3"] at [0, 1, 2] -> ["B2", "B1", "B3"] at [1, 0, 2]>] bounds = [16, 8, 32] -> [8, 16, 32]> : memref<8x16x32xf32> to memref<16x8x32xf32>
  // CHECK: %[[MERGE2:.+]] = rock.transform %[[TRANSPOSED]] by <affine_map<(d0) -> (d0 floordiv 16, 0, d0 mod 16)> by [<Merge{8, 1, 16} ["A"] at [0] -> ["B1", "B2", "B3"] at [0, 1, 2]>] bounds = [128] -> [8, 1, 16]> : memref<8x1x16xf32> to memref<128xf32>
  %merge2 = rock.transform %transposed by <affine_map<(d0) -> (d0 floordiv (8 * 32), (d0 mod (8 * 32)) floordiv 32, d0 mod 32)> by [<Merge{16, 8, 32} ["A"] at [0] -> ["B1", "B2", "B3"] at [0, 1, 2]>] bounds = [4096] -> [16, 8, 32]> : memref<16x8x32xf32> to memref<4096xf32>
  // CHECK: %[[UNMERGE2:.+]] = rock.transform %[[MERGE2]] by <affine_map<(d0, d1) -> (d0 * 16 + d1)> by [<Unmerge{8, 16} ["A1", "A3"] at [0, 1] -> ["A"] at [0]>] bounds = [8, 16] -> [128]> : memref<128xf32> to memref<8x16xf32>
  %unmerge2 = rock.transform %merge2 by <affine_map<(d0, d1, d2) -> (d0 * 16 * 32 + d1 * 16 + d2)> by [<Unmerge{8, 32, 16} ["A1", "A2", "A3"] at [0, 1, 2] -> ["A"] at [0]>] bounds = [8, 32, 16] -> [4096]> : memref<4096xf32> to memref<8x32x16xf32>
  "remove_dims"(%unmerge2) {names_to_drop = ["A2"]} : (memref<8x32x16xf32>) -> ()
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

// CHECK-LABEL: @mfma_out_thread_subtile
func.func @mfma_out_thread_subtile(%arg0: memref<1x64x384xf32>) {
  // CHECK : %[[TR1:.+]] = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0, d1)> by [<Unmerge{4} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{2} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [4, 2] -> [4, 2]> : memref<4x2xf32> to memref<4x2xf32>
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 32 + d3, d2 * 64 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{2, 32} ["m_block", "gemmBlockM"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{6, 64} ["n_block", "gemmBlockN"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 2, 6, 32, 64] -> [1, 64, 384]> : memref<1x64x384xf32> to memref<1x2x6x32x64xf32>
  // CHECK : %[[TR2:.+]] = rock.transform %[[TR1]] by <affine_map<(d0, d1, d2) -> (d1 + d0, d2)> by [<Unmerge{4, 1} ["m_tid", "m_iter"] at [1, 0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [2] -> ["gemmBlockN"] at [1]>] bounds = [1, 4, 2] -> [4, 2]> : memref<4x2xf32> to memref<1x4x2xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4 + d3, d5)> by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>, <Unmerge{32, 1} ["m_tid", "m_iter"] at [4, 3] -> ["gemmBlockM"] at [3]>, <PassThrough ["gemmBlockN"] at [5] -> ["gemmBlockN"] at [4]>] bounds = [1, 2, 6, 1, 32, 64] -> [1, 2, 6, 32, 64]> : memref<1x2x6x32x64xf32> to memref<1x2x6x1x32x64xf32>
  // CHECK : %[[TR3:.+]] = rock.transform %[[TR2]] by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 4} ["gemmBlockM"] at [0] -> ["m_iter", "m_tid"] at [0, 1]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [2]>] bounds = [4, 2] -> [1, 4, 2]> : memref<1x4x2xf32> to memref<4x2xf32>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d3, d4)> by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>, <Merge{1, 32} ["gemmBlockM"] at [3] -> ["m_iter", "m_tid"] at [3, 4]>, <PassThrough ["gemmBlockN"] at [4] -> ["gemmBlockN"] at [5]>] bounds = [1, 2, 6, 32, 64] -> [1, 2, 6, 1, 32, 64]> : memref<1x2x6x1x32x64xf32> to memref<1x2x6x32x64xf32>
  // CHECK : %[[TR4:.+]] = rock.transform %[[TR3]] by <affine_map<(d0, d1, d2, d3, d4, d5) -> ((d0 + d2 + d4) * 4 + d5, d1 + d3)> by [<Unmerge{1, 1, 1, 4} ["m_i", "blk_row", "vec_group", "vec_item"] at [0, 2, 4, 5] -> ["gemmBlockM"] at [0]>, <Unmerge{2, 1} ["n_i", "blk_col"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 2, 1, 1, 1, 4] -> [4, 2]> : memref<4x2xf32> to memref<1x2x1x1x1x4xf32>
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12) -> (d0, d1, d2, ((d7 * 2 + d3 + d9 + d11) * 4 + d5) * 4 + d12, (d8 * 2 + d4 + d10) * 16 + d6)> by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>, <Unmerge{1, 2, 1, 1, 4, 4} ["m_i", "wave_m", "blk_row", "vec_group", "m_tid", "vec_item"] at [7, 3, 9, 11, 5, 12] -> ["gemmBlockM"] at [3]>, <Unmerge{2, 2, 1, 16} ["n_i", "wave_n", "blk_col", "n_tid"] at [8, 4, 10, 6] -> ["gemmBlockN"] at [4]>] bounds = [1, 2, 6, 2, 2, 4, 16, 1, 2, 1, 1, 1, 4] -> [1, 2, 6, 32, 64]> : memref<1x2x6x32x64xf32> to memref<1x2x6x2x2x4x16x1x2x1x1x1x4xf32>
  // CHECK : %[[TR5:.+]] = rock.transform %[[TR4]] by <affine_map<(d0, d1, d2, d3) -> (0, d0, 0, 0, d2, d3)> by [<Merge{1, 2} ["i"] at [0] -> ["m_i", "n_i"] at [0, 1]>, <Merge{1, 1} ["j"] at [1] -> ["blk_row", "blk_col"] at [2, 3]>, <PassThrough ["vec_group", "vec_item"] at [2, 3] -> ["vec_group", "vec_item"] at [4, 5]>] bounds = [2, 1, 1, 4] -> [1, 2, 1, 1, 1, 4]> : memref<1x2x1x1x1x4xf32> to memref<2x1x1x4xf32>
  %4 = rock.transform %3 by <affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3 floordiv 2, d3 mod 2, d4, d5, 0, d6, 0, 0, d8, d9)> by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>, <Merge{2, 2} ["wave"] at [3] -> ["wave_m", "wave_n"] at [3, 4]>, <PassThrough ["m_tid", "n_tid"] at [4, 5] -> ["m_tid", "n_tid"] at [5, 6]>, <Merge{1, 2} ["i"] at [6] -> ["m_i", "n_i"] at [7, 8]>, <Merge{1, 1} ["j"] at [7] -> ["blk_row", "blk_col"] at [9, 10]>, <PassThrough ["vec_group", "vec_item"] at [8, 9] -> ["vec_group", "vec_item"] at [11, 12]>] bounds = [1, 2, 6, 4, 4, 16, 2, 1, 1, 4] -> [1, 2, 6, 2, 2, 4, 16, 1, 2, 1, 1, 1, 4]> : memref<1x2x6x2x2x4x16x1x2x1x1x1x4xf32> to memref<1x2x6x4x4x16x2x1x1x4xf32>
  // CHECK : %[[TR6:.+]] = rock.transform %[[TR5]] by <affine_map<(d0) -> (d0 floordiv 4, 0, 0, d0 mod 4)> by [<Merge{2, 1, 1, 4} ["item"] at [0] -> ["i", "j", "vec_group", "vec_item"] at [0, 1, 2, 3]>] bounds = [8] -> [2, 1, 1, 4]> : memref<2x1x1x4xf32> to memref<8xf32>
  %5 = rock.transform %4 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 floordiv 64, (d3 mod 64) floordiv 16, d3 mod 16, d4 floordiv 4, 0, 0, d4 mod 4)> by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>, <Merge{4, 4, 16} ["tid"] at [3] -> ["wave", "m_tid", "n_tid"] at [3, 4, 5]>, <Merge{2, 1, 1, 4} ["item"] at [4] -> ["i", "j", "vec_group", "vec_item"] at [6, 7, 8, 9]>] bounds = [1, 2, 6, 256, 8] -> [1, 2, 6, 4, 4, 16, 2, 1, 1, 4]> : memref<1x2x6x4x4x16x2x1x1x4xf32> to memref<1x2x6x256x8xf32>
  "remove_dims"(%5) {names_to_drop = ["g_block", "m_block", "n_block", "tid"]} : (memref<1x2x6x256x8xf32>) -> ()
}

// CHECK-LABEL: @mfma_in_thread_subtile
func.func @mfma_in_thread_subtile(%arg0: memref<1x64x384xf32>) {
  // CHECK : %[[TR1:.+]] = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d2 * 8 + d0, d1)> by [<Unmerge{2, 8} ["k", "kpack"] at [2, 0] -> ["K"] at [0]>, <Unmerge{2} ["d"] at [1] -> ["D"] at [1]>] bounds = [8, 2, 2] -> [16, 2]> : memref<16x2xf32> to memref<8x2x2xf32>
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, (d0 * 8 + d5) * 8 + d3, d2 * 64 + d4)> by [<PassThrough ["g_block"] at [1] -> ["G"] at [0]>, <Unmerge{1, 8, 8} ["k_loop", "k", "kpack"] at [0, 5, 3] -> ["K"] at [1]>, <Unmerge{6, 64} ["n_block", "d"] at [2, 4] -> ["D"] at [2]>] bounds = [1, 1, 6, 8, 64, 8] -> [1, 64, 384]> : memref<1x64x384xf32> to memref<1x1x6x8x64x8xf32>
  // CHECK : %[[TR2:.+]] = rock.transform %[[TR1]] by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["kpack"] at [0] -> ["kpack"] at [0]>, <Unmerge{2} ["d_iter"] at [1] -> ["d"] at [1]>, <Unmerge{2} ["k_iter"] at [2] -> ["k"] at [2]>] bounds = [8, 2, 2] -> [8, 2, 2]> : memref<8x2x2xf32> to memref<8x2x2xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, (d8 * 2 + d5) * 16 + d7, d6 * 2 + d9)> by [<PassThrough ["k_loop", "g_block"] at [0, 1] -> ["k_loop", "g_block"] at [0, 1]>, <PassThrough ["n_block"] at [2] -> ["n_block"] at [2]>, <PassThrough ["kpack"] at [3] -> ["kpack"] at [3]>, <Unmerge{2, 2, 16} ["d_iter", "wave_n", "blk_td"] at [8, 5, 7] -> ["d"] at [4]>, <Unmerge{4, 2} ["blk_id", "k_iter"] at [6, 9] -> ["k"] at [5]>, <AddDim{2} ["wave_m"] at [4] -> [] at []>] bounds = [1, 1, 6, 8, 2, 2, 4, 16, 2, 2] -> [1, 1, 6, 8, 64, 8]> : memref<1x1x6x8x64x8xf32> to memref<1x1x6x8x2x2x4x16x2x2xf32>
  // CHECK : %[[TR3:.+]] = rock.transform %[[TR2]] by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["kpack"] at [0] -> ["kpack"] at [0]>, <PassThrough ["d_iter", "k_iter"] at [1, 2] -> ["d_iter", "k_iter"] at [1, 2]>] bounds = [8, 2, 2] -> [8, 2, 2]> : memref<8x2x2xf32> to memref<8x2x2xf32>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4 floordiv 2, d4 mod 2, d5, d6, d7, d8)> by [<PassThrough ["k_loop", "g_block"] at [0, 1] -> ["k_loop", "g_block"] at [0, 1]>, <PassThrough ["n_block"] at [2] -> ["n_block"] at [2]>, <PassThrough ["kpack"] at [3] -> ["kpack"] at [3]>, <Merge{2, 2} ["wave_id"] at [4] -> ["wave_m", "wave_n"] at [4, 5]>, <PassThrough ["blk_id", "blk_td", "d_iter", "k_iter"] at [5, 6, 7, 8] -> ["blk_id", "blk_td", "d_iter", "k_iter"] at [6, 7, 8, 9]>] bounds = [1, 1, 6, 8, 4, 4, 16, 2, 2] -> [1, 1, 6, 8, 2, 2, 4, 16, 2, 2]> : memref<1x1x6x8x2x2x4x16x2x2xf32> to memref<1x1x6x8x4x4x16x2x2xf32>
  // CHECK : %[[TR4:.+]] = rock.transform %[[TR3]] by <affine_map<(d0, d1, d2) -> (d2, d0, d1)> by [<PassThrough ["kpack"] at [2] -> ["kpack"] at [0]>, <PassThrough ["drepeat", "kpack_iter"] at [0, 1] -> ["d_iter", "k_iter"] at [1, 2]>] bounds = [2, 2, 8] -> [8, 2, 2]> : memref<8x2x2xf32> to memref<2x2x8xf32>
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d3, d7, d4 floordiv 64, (d4 mod 64) floordiv 16, d4 mod 16, d5, d6)> by [<PassThrough ["k_loop", "g_block"] at [0, 1] -> ["k_loop", "g_block"] at [0, 1]>, <PassThrough ["n_block"] at [3] -> ["n_block"] at [2]>, <PassThrough ["kpack"] at [7] -> ["kpack"] at [3]>, <Merge{4, 4, 16} ["tid"] at [4] -> ["wave_id", "blk_id", "blk_td"] at [4, 5, 6]>, <PassThrough ["drepeat", "kpack_iter"] at [5, 6] -> ["d_iter", "k_iter"] at [7, 8]>] bounds = [1, 1, 12, 6, 256, 2, 2, 8] -> [1, 1, 6, 8, 4, 4, 16, 2, 2]> : memref<1x1x6x8x4x4x16x2x2xf32> to memref<1x1x12x6x256x2x2x8xf32>
  // CHECK : %[[TR5:.+]] = rock.transform %[[TR4]] by <affine_map<(d0) -> (d0 floordiv 16, (d0 mod 16) floordiv 8, d0 mod 8)> by [<Merge{2, 2, 8} ["iter"] at [0] -> ["drepeat", "kpack_iter", "kpack"] at [0, 1, 2]>] bounds = [32] -> [2, 2, 8]> : memref<2x2x8xf32> to memref<32xf32>
  %4 = rock.transform %3 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5 floordiv 16, (d5 mod 16) floordiv 8, d5 mod 8)> by [<PassThrough ["k_loop", "g_block", "m_block", "n_block", "tid"] at [0, 1, 2, 3, 4] -> ["k_loop", "g_block", "m_block", "n_block", "tid"] at [0, 1, 2, 3, 4]>, <Merge{2, 2, 8} ["iter"] at [5] -> ["drepeat", "kpack_iter", "kpack"] at [5, 6, 7]>] bounds = [1, 1, 12, 6, 256, 32] -> [1, 1, 12, 6, 256, 2, 2, 8]> : memref<1x1x12x6x256x2x2x8xf32> to memref<1x1x12x6x256x32xf32>
  "remove_dims"(%4) {names_to_drop = ["k_loop", "g_block", "m_block", "n_block", "tid"]} : (memref<1x1x12x6x256x32xf32>) -> ()
}

// CHECK-LABEL: @padding_no_overlap
func.func @padding_no_overlap(%arg0: memref<320xf32>) {
  // CHECK : %[[TR1:.+]] = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0 * 10 + d1)> by [<Unmerge{4, 10} ["B1", "B2"] at [0, 1] -> ["A"] at [0]>] bounds = [4, 10] -> [40]> : memref<40xf32> to memref<4x10xf32>
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0 * 10 + d1)> by [<Unmerge{32, 10} ["B1", "B2"] at [0, 1] -> ["A"] at [0]>] bounds = [32, 10] -> [320]> : memref<320xf32> to memref<32x10xf32>
  // CHECK : %[[TR2:.+]] = rock.transform %[[TR1]] by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["C1"] at [0] -> ["B1"] at [0]>, <Pad{0, 6} ["C2"] at [1] -> ["B2"] at [1]>] bounds = [4, 16] -> [4, 10]> : memref<4x10xf32> to memref<4x16xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d1) -> (d0, d1 - 6)> by [<PassThrough ["C1"] at [0] -> ["B1"] at [0]>, <Pad{0, 6} ["C2"] at [1] -> ["B2"] at [1]>] bounds = [32, 16] -> [32, 10]> : memref<32x10xf32> to memref<32x16xf32>
  // CHECK : %[[TR3:.+]] = rock.transform %[[TR2]] by <affine_map<(d0) -> (d0 floordiv 16, d0 mod 16)> by [<Merge{4, 16} ["D"] at [0] -> ["C1", "C2"] at [0, 1]>] bounds = [64] -> [4, 16]> : memref<4x16xf32> to memref<64xf32>
  %2 = rock.transform %1 by <affine_map<(d0) -> (d0 floordiv 16, d0 mod 16)> by [<Merge{32, 16} ["D"] at [0] -> ["C1", "C2"] at [0, 1]>] bounds = [512] -> [32, 16]> : memref<32x16xf32> to memref<512xf32>
  // CHECK : %[[TR4:.+]] = rock.transform %[[TR3]] by <affine_map<(d0) -> (d0)> by [<Unmerge{64} ["E2"] at [0] -> ["D"] at [0]>] bounds = [64] -> [64]> : memref<64xf32> to memref<64xf32>
  %3 = rock.transform %2 by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{8, 64} ["E1", "E2"] at [0, 1] -> ["D"] at [0]>] bounds = [8, 64] -> [512]> : memref<512xf32> to memref<8x64xf32>
  "remove_dims"(%3) {names_to_drop = ["E1"]} : (memref<8x64xf32>) -> ()
}

// CHECK-LABEL: @padding_overlap
func.func @padding_overlap(%arg0: memref<320xf32>) {
  // CHECK : %[[TR1:.+]] = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0 * 4 + d1)> by [<Unmerge{8, 4} ["B1", "B2"] at [0, 1] -> ["A"] at [0]>] bounds = [8, 4] -> [32]> : memref<32xf32> to memref<8x4xf32>
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0 * 10 + d1)> by [<Unmerge{32, 10} ["B1", "B2"] at [0, 1] -> ["A"] at [0]>] bounds = [32, 10] -> [320]> : memref<320xf32> to memref<32x10xf32>
  // CHECK : %[[TR2:.+]] = rock.transform %[[TR1]] by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["C1"] at [0] -> ["B1"] at [0]>, <Pad{0, 0} ["C2"] at [1] -> ["B2"] at [1]>] bounds = [8, 4] -> [8, 4]> : memref<8x4xf32> to memref<8x4xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d1) -> (d0, d1 - 6)> by [<PassThrough ["C1"] at [0] -> ["B1"] at [0]>, <Pad{0, 6} ["C2"] at [1] -> ["B2"] at [1]>] bounds = [32, 16] -> [32, 10]> : memref<32x10xf32> to memref<32x16xf32>
  // CHECK : %[[TR3:.+]] = rock.transform %[[TR2]] by <affine_map<(d0) -> (d0 floordiv 4, d0 mod 4)> by [<Merge{8, 4} ["D"] at [0] -> ["C1", "C2"] at [0, 1]>] bounds = [32] -> [8, 4]> : memref<8x4xf32> to memref<32xf32>
  %2 = rock.transform %1 by <affine_map<(d0) -> (d0 floordiv 16, d0 mod 16)> by [<Merge{32, 16} ["D"] at [0] -> ["C1", "C2"] at [0, 1]>] bounds = [512] -> [32, 16]> : memref<32x16xf32> to memref<512xf32>
  // CHECK : %[[TR4:.+]] = rock.transform %[[TR3]] by <affine_map<(d0, d1) -> (d0 * 4 + d1)> by [<Unmerge{8, 4} ["E1", "E3"] at [0, 1] -> ["D"] at [0]>] bounds = [8, 4] -> [32]> : memref<32xf32> to memref<8x4xf32>
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (d0 * (16 * 4) + d1 * 4 + d2)> by [<Unmerge{8, 16, 4} ["E1", "E2", "E3"] at [0, 1, 2] -> ["D"] at [0]>] bounds = [8, 16, 4] -> [512]> : memref<512xf32> to memref<8x16x4xf32>
  "remove_dims"(%3) {names_to_drop = ["E2"]} : (memref<8x16x4xf32>) -> ()
}

