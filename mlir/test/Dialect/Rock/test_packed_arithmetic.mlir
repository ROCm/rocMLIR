// RUN: rocmlir-opt --rock-affix-params --rock-gemm-to-gridwise --rock-regularize \
// RUN: --rock-gridwise-gemm-to-blockwise --rock-linalg-align \
// RUN: --convert-linalg-to-affine-loops --rock-vectorize-fusions %s | FileCheck %s --check-prefix=VECTORIZE
// RUN:  rocmlir-driver --kernel-pipeline=gpu,rocdl --arch=gfx942 %s | FileCheck %s --check-prefix=ROCDL
// RUN:  rocmlir-driver --kernel-pipeline=gpu,rocdl --arch=gfx942 %s | \
// RUN:  rocmlir-translate --gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | FileCheck %s --check-prefix=LLVM
#map = affine_map<(d0, d1) -> (0, d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0 * 128 + d1, d2)>
#transform_map = #rock.transform_map<#map by [<Merge{1, 128} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [128, 128] -> [1, 128, 128]>
#transform_map1 = #rock.transform_map<#map2 by [<Unmerge{1, 128} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 128, 128] -> [128, 128]>
// VECTORIZE: affine.for
// VECTORIZE-SAME: step 2
// VECTORIZE: %[[vec:.*]] = vector.transfer_read
// VECTORIZE: %[[trunc:.*]] = arith.truncf %[[vec]]
// VECTORIZE: vector.transfer_write %[[trunc]]
// ROCDL:  %[[pkrtz:.*]] = rocdl.cvt.pkrtz {{.*}}, {{.*}} : vector<2xf16>
// ROCDL:  llvm.store %[[pkrtz]], {{.*}} : vector<2xf16>, !llvm.ptr<5>
// LLVM: %[[extract0:.*]] = extractelement <16 x float> {{.*}}, i64 0
// LLVM: %[[extract1:.*]] = extractelement <16 x float> {{.*}}, i64 1
// LLVM: tail call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %[[extract0]], float %[[extract1]])
// LLVM: %[[extract2:.*]] = extractelement <16 x float> {{.*}}, i64 2
// LLVM: %[[extract3:.*]] = extractelement <16 x float> {{.*}}, i64 3
// LLVM: tail call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %[[extract2]], float %[[extract3]])
// LLVM: %[[extract14:.*]] = extractelement <16 x float> {{.*}}, i64 14
// LLVM: %[[extract15:.*]] = extractelement <16 x float> {{.*}}, i64 15
// LLVM: tail call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %[[extract14]], float %[[extract15]])
module {
  func.func @test_fusion(%arg0: memref<1x128x128xf16> {func.read_access}, %arg1: memref<1x128x128xf16> {func.read_access}, %arg2: memref<1x128x128xf16> {func.read_access}, %arg3: memref<1x128x128xf16> {func.write_access}) attributes {arch = "gfx942", kernel} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x128xf16>
    rock.gemm %alloc = %arg0 * %arg1 features =  mfma|dot|atomic_add storeMethod =  set {arch = "gfx942"} : memref<1x128x128xf16> = memref<1x128x128xf16> * memref<1x128x128xf16>
    %0 = rock.transform %alloc by #transform_map : memref<1x128x128xf16> to memref<128x128xf16>
    %1 = rock.transform %arg2 by #transform_map : memref<1x128x128xf16> to memref<128x128xf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<128x128xf16>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : memref<128x128xf16>, memref<128x128xf16>) outs(%alloc_0 : memref<128x128xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %3 = arith.addf %in, %in_1 : f16
      linalg.yield %3 : f16
    }
    %2 = rock.transform %alloc_0 by #transform_map1 : memref<128x128xf16> to memref<1x128x128xf16>
    memref.copy %2, %arg3 : memref<1x128x128xf16> to memref<1x128x128xf16>
    return
  }
}
