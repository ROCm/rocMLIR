// RUN: mlir-opt -convert-miopen-to-gpu="kernel-name=mfma" %s | FileCheck %s
// RUN: mlir-opt -convert-miopen-to-gpu="kernel-name=mfma" -convert-gpu-to-rocdl %s | FileCheck %s --check-prefix=ROCDL

module {
  func @mfma(%a : f32, %b : f32, %c : vector<32xf32>) {
    %d = miopen.mfma(%a, %b, %c) { m_per_wave = 64, n_per_wave = 64 }: vector<32xf32>
    // CHECK: %{{.*}} = gpu.mfma(%{{.*}}, %{{.*}}, %{{.*}}) {m_per_wave = 64 : i64, n_per_wave = 64 : i64} : vector<32xf32>
    // ROCDL: %{{.*}} = rocdl.mfma.f32.32x32x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!llvm.float, !llvm.float, !llvm<"<32 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<32 x float>">

    return
  }
}
