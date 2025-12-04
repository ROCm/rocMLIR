// RUN: rocmlir-opt --rock-detect-flash-decoding %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 128 + d2)>
#map1 = affine_map<(d0, d1, d2) -> ((d0 * 128 + d1) * 256 + d2)>
#map2 = affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 128 + d2)>
#transform_map = #rock.transform_map<#map by [<Unmerge{12, 256, 128} ["b", "m", "k"] at [0, 1, 2] -> ["flat"] at [0]>] bounds = [12, 256, 128] -> [393216]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{12, 128, 256} ["b", "k", "n"] at [0, 1, 2] -> ["flat"] at [0]>] bounds = [12, 128, 256] -> [393216]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{12, 256, 128} ["b", "n", "d"] at [0, 1, 2] -> ["flat"] at [0]>] bounds = [12, 256, 128] -> [393216]>
module {
  // Test 1: No LSE output: flash decoding requires LSE for correctness
  // CHECK-LABEL: @no_lse_output
  func.func @no_lse_output(%arg0: tensor<393216xf16>, %arg1: tensor<393216xf16>, %arg2: tensor<393216xf16>) -> tensor<393216xf16> attributes {arch = "gfx942", kernel = "mixr"} {
    %q = rock.transform %arg0 by #transform_map : tensor<393216xf16> to tensor<12x256x128xf16>
    %k = rock.transform %arg1 by #transform_map1 : tensor<393216xf16> to tensor<12x128x256xf16>
    %v = rock.transform %arg2 by #transform_map2 : tensor<393216xf16> to tensor<12x256x128xf16>
    %out = bufferization.alloc_tensor() : tensor<12x256x128xf16>

    // CHECK: rock.attention
    // CHECK: splitKV = 1

    %result = rock.attention{
     qk = %q * %k : tensor<12x256x128xf16>, tensor<12x128x256xf16>
     %out = softmax(qk) * %v : tensor<12x256x128xf16> -> tensor<12x256x128xf16>
    } {firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32, storeMethod = #rock<StoreMethod set>} -> tensor<12x256x128xf16>
    %flat = rock.transform %result by #transform_map : tensor<12x256x128xf16> to tensor<393216xf16>
    return %flat : tensor<393216xf16>
  }

  // Test 2: No broadcast pattern: regular attention without splitKV dimension
  // CHECK-LABEL: @no_broadcast_pattern
  func.func @no_broadcast_pattern(%arg0: tensor<393216xf16>, %arg1: tensor<393216xf16>, %arg2: tensor<393216xf16>) -> (tensor<393216xf16>, tensor<3072xf32>) attributes {arch = "gfx942", kernel = "mixr"} {
    %q = rock.transform %arg0 by #transform_map : tensor<393216xf16> to tensor<12x256x128xf16>
    %k = rock.transform %arg1 by #transform_map1 : tensor<393216xf16> to tensor<12x128x256xf16>
    %v = rock.transform %arg2 by #transform_map2 : tensor<393216xf16> to tensor<12x256x128xf16>
    %out = bufferization.alloc_tensor() : tensor<12x256x128xf16>
    %lse = bufferization.alloc_tensor() : tensor<12x256xf32>

    // CHECK: rock.attention
    // CHECK: splitKV = 1

    %result, %lseOut = rock.attention{
     qk = %q * %k : tensor<12x256x128xf16>, tensor<12x128x256xf16>
     lse = %lse : tensor<12x256xf32>
     %out = softmax(qk) * %v : tensor<12x256x128xf16> -> tensor<12x256x128xf16>
    } {firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32, storeMethod = #rock<StoreMethod set>} -> tensor<12x256x128xf16>, tensor<12x256xf32>
    %flat_result = rock.transform %result by #transform_map : tensor<12x256x128xf16> to tensor<393216xf16>
    %flat_lse = rock.transform %lseOut by <affine_map<(d0) -> (d0 floordiv 256, d0 mod 256)> by [<Merge{12, 256} ["flat"] at [0] -> ["b", "m"] at [0, 1]>] bounds = [3072] -> [12, 256]> : tensor<12x256xf32> to tensor<3072xf32>
    return %flat_result, %flat_lse : tensor<393216xf16>, tensor<3072xf32>
  }

  // Test 3: Different tensor dimensions
  // CHECK-LABEL: @different_dimensions  
  func.func @different_dimensions(%arg0: tensor<196608xf16>, %arg1: tensor<196608xf16>, %arg2: tensor<196608xf16>) -> (tensor<196608xf16>, tensor<1536xf32>) attributes {arch = "gfx942", kernel = "mixr"} {
    %q = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 128 + d1) * 128 + d2)> by [<Unmerge{12, 128, 128} ["b", "m", "k"] at [0, 1, 2] -> ["flat"] at [0]>] bounds = [12, 128, 128] -> [196608]> : tensor<196608xf16> to tensor<12x128x128xf16>
    %k = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 128 + d1) * 128 + d2)> by [<Unmerge{12, 128, 128} ["b", "k", "n"] at [0, 1, 2] -> ["flat"] at [0]>] bounds = [12, 128, 128] -> [196608]> : tensor<196608xf16> to tensor<12x128x128xf16>
    %v = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 128 + d1) * 128 + d2)> by [<Unmerge{12, 128, 128} ["b", "n", "d"] at [0, 1, 2] -> ["flat"] at [0]>] bounds = [12, 128, 128] -> [196608]> : tensor<196608xf16> to tensor<12x128x128xf16>
    %out = bufferization.alloc_tensor() : tensor<12x128x128xf16>
    %lse = bufferization.alloc_tensor() : tensor<12x128xf32>

    // CHECK: rock.attention
    // CHECK: splitKV = 1

    %result, %lseOut = rock.attention{
     qk = %q * %k : tensor<12x128x128xf16>, tensor<12x128x128xf16>
     lse = %lse : tensor<12x128xf32>
     %out = softmax(qk) * %v : tensor<12x128x128xf16> -> tensor<12x128x128xf16>
    } {firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32, storeMethod = #rock<StoreMethod set>} -> tensor<12x128x128xf16>, tensor<12x128xf32>
    %flat_result = rock.transform %result by <affine_map<(d0) -> (d0 floordiv 16384, (d0 mod 16384) floordiv 128, d0 mod 128)> by [<Merge{12, 128, 128} ["flat"] at [0] -> ["b", "m", "d"] at [0, 1, 2]>] bounds = [196608] -> [12, 128, 128]> : tensor<12x128x128xf16> to tensor<196608xf16>
    %flat_lse = rock.transform %lseOut by <affine_map<(d0) -> (d0 floordiv 128, d0 mod 128)> by [<Merge{12, 128} ["flat"] at [0] -> ["b", "m"] at [0, 1]>] bounds = [1536] -> [12, 128]> : tensor<12x128xf32> to tensor<1536xf32>
    return %flat_result, %flat_lse : tensor<196608xf16>, tensor<1536xf32>
  }

  // Test 4: Mismatched tensor dimensions
  // CHECK-LABEL: @mismatched_splitkv
  func.func @mismatched_splitkv(%arg0: tensor<786432xf16>, %arg1: tensor<393216xf16>, %arg2: tensor<393216xf16>) -> (tensor<786432xf16>, tensor<3072xf32>) attributes {arch = "gfx942", kernel = "mixr"} {
    %q = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 256 + d2)> by [<Unmerge{12, 256, 256} ["b", "m", "k"] at [0, 1, 2] -> ["flat"] at [0]>] bounds = [12, 256, 256] -> [786432]> : tensor<786432xf16> to tensor<12x256x256xf16>
    %k = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 128 + d2)> by [<Unmerge{12, 256, 128} ["b", "k", "n"] at [0, 1, 2] -> ["flat"] at [0]>] bounds = [12, 256, 128] -> [393216]> : tensor<393216xf16> to tensor<12x256x128xf16>
    %v = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 128 + d1) * 256 + d2)> by [<Unmerge{12, 128, 256} ["b", "n", "d"] at [0, 1, 2] -> ["flat"] at [0]>] bounds = [12, 128, 256] -> [393216]> : tensor<393216xf16> to tensor<12x128x256xf16>
    %out = bufferization.alloc_tensor() : tensor<12x256x256xf16>
    %lse = bufferization.alloc_tensor() : tensor<12x256xf32>

    // CHECK: rock.attention
    // CHECK: splitKV = 1

    %result, %lseOut = rock.attention{
     qk = %q * %k : tensor<12x256x256xf16>, tensor<12x256x128xf16>
     lse = %lse : tensor<12x256xf32>
     %out = softmax(qk) * %v : tensor<12x128x256xf16> -> tensor<12x256x256xf16>
    } {firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32, storeMethod = #rock<StoreMethod set>} -> tensor<12x256x256xf16>, tensor<12x256xf32>
    %flat_result = rock.transform %result by <affine_map<(d0) -> (d0 floordiv 65536, (d0 mod 65536) floordiv 256, d0 mod 256)> by [<Merge{12, 256, 256} ["flat"] at [0] -> ["b", "m", "n"] at [0, 1, 2]>] bounds = [786432] -> [12, 256, 256]> : tensor<12x256x256xf16> to tensor<786432xf16>
    %flat_lse = rock.transform %lseOut by <affine_map<(d0) -> (d0 floordiv 256, d0 mod 256)> by [<Merge{12, 256} ["flat"] at [0] -> ["b", "m"] at [0, 1]>] bounds = [3072] -> [12, 256]> : tensor<12x256xf32> to tensor<3072xf32>
    return %flat_result, %flat_lse : tensor<786432xf16>, tensor<3072xf32>
  }
}

