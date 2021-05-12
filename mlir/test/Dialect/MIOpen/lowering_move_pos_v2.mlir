// RUN: mlir-opt -miopen-lowering-step3 %s | FileCheck %s

// CHECK-LABEL: @miopen_lowering_move_pos_v2_i32
func @miopen_lowering_move_pos_v2_i32(%vector_i32 : vector<2xi32>) -> vector<2xi32> {
  %deltaY_i32 = constant 16 : i32
  %deltaX_i32 = constant 8 : i32
  // CHECK: %{{.*}} = vector.extractelement %{{.*}}[%{{.*}} : i32] : vector<2xi32>
  // CHECK: %{{.*}} = addi %{{.*}}, %{{.*}} : i32
  // CHECK: %[[VECTOR0:.*]] = vector.insertelement %{{.*}}, %{{.*}}[%{{.*}} : i32] : vector<2xi32>
  // CHECK: %{{.*}} = vector.extractelement %[[VECTOR0]][%{{.*}} : i32] : vector<2xi32>
  // CHECK: %{{.*}} = addi %{{.*}}, %{{.*}} : i32
  // CHECK: %[[VECTOR1:.*]] = vector.insertelement %{{.*}}, %{{.*}}[%{{.*}} : i32] : vector<2xi32>
  %output = miopen.move_pos_v2(%vector_i32, %deltaY_i32, %deltaX_i32) : vector<2xi32>
  // CHECK: return %[[VECTOR1]] : vector<2xi32>
  return %output : vector<2xi32>
}

// CHECK-LABEL: @miopen_lowering_move_pos_v2_f32
func @miopen_lowering_move_pos_v2_f32(%vector_f32 : vector<2xf32>) -> vector<2xf32> {
  %deltaY_f32 = constant 16.0 : f32
  %deltaX_f32 = constant 8.0 : f32
  // CHECK: %{{.*}} = vector.extractelement %{{.*}}[%{{.*}} : i32] : vector<2xf32>
  // CHECK: %{{.*}} = addf %{{.*}}, %{{.*}} : f32
  // CHECK: %[[VECTOR0:.*]] = vector.insertelement %{{.*}}, %{{.*}}[%{{.*}} : i32] : vector<2xf32>
  // CHECK: %{{.*}} = vector.extractelement %[[VECTOR0]][%{{.*}} : i32] : vector<2xf32>
  // CHECK: %{{.*}} = addf %{{.*}}, %{{.*}} : f32
  // CHECK: %[[VECTOR1:.*]] = vector.insertelement %{{.*}}, %{{.*}}[%{{.*}} : i32] : vector<2xf32>
  %output = miopen.move_pos_v2(%vector_f32, %deltaY_f32, %deltaX_f32) : vector<2xf32>
  // CHECK: return %[[VECTOR1]] : vector<2xf32>
  return %output : vector<2xf32>
}
