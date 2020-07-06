// RUN: mlir-opt -convert-miopen-to-gpu="kernel-name=mfma" %s | FileCheck %s

module {
  func @mfma(%a : f32, %b : f32, %c : vector<32xf32>) {
    %c0 = constant 0 : i32
    %c1 = constant 1 : i32

    %d = miopen.mfma(%a, %b, %c, %c0, %c0, %c1) : vector<32xf32>
    // CHECK: %{{.*}} = "gpu.mfma"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (f32, f32, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    return
  }
}
