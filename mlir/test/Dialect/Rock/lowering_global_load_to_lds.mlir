// RUN: sed s/##TOKEN_ARCH##/gfx942/g %s | rocmlir-opt --rock-sugar-to-loops | FileCheck %s --check-prefixes=CHECK,GFX942
// RUN: sed s/##TOKEN_ARCH##/gfx1250/g %s | rocmlir-opt --rock-sugar-to-loops | FileCheck %s --check-prefixes=CHECK,GFX1250

module {
// CHECK-LABEL: func.func @load_scalar_in_bounds
// CHECK-SAME: (%[[mem:.*]]: memref<192xf32>)
func.func @load_scalar_in_bounds(%mem: memref<192xf32>) attributes {kernel, arch = "##TOKEN_ARCH##"} {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %lds = rock.alloc() : memref<64xi8, #gpu.address_space<workgroup>>
    %lds_view = memref.view %lds[%c0][] : memref<64xi8, #gpu.address_space<workgroup>> to memref<4xf32, #gpu.address_space<workgroup>>
    // CHECK: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // GFX942: amdgpu.gather_to_lds %[[cast]]
    // GFX942-SAME: f32, memref<192xf32, #gpu.address_space<global>>, memref<4xf32, #gpu.address_space<workgroup>>
    // GFX1250: amdgpu.async_load_to_lds %[[cast]]
    // GFX1250-SAME: f32, memref<192xf32, #gpu.address_space<global>>, memref<4xf32, #gpu.address_space<workgroup>>
    rock.global_load_to_lds %mem[%c0] -> %lds_view[%c0]  if %true {transferType = f32} : memref<192xf32> -> memref<4xf32, #gpu.address_space<workgroup>>
    return
}

// CHECK-LABEL: func.func @load_scalar_in_bounds_force_oob
// CHECK-SAME: (%[[mem:.*]]: memref<192xf32>)
func.func @load_scalar_in_bounds_force_oob(%mem: memref<192xf32>) attributes {kernel, arch = "##TOKEN_ARCH##"} {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %lds = rock.alloc() : memref<64xi8, #gpu.address_space<workgroup>>
    %lds_view = memref.view %lds[%c0][] : memref<64xi8, #gpu.address_space<workgroup>> to memref<4xf32, #gpu.address_space<workgroup>>
    // CHECK: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // GFX942: %[[fatBuff:.*]] = amdgpu.fat_raw_buffer_cast %[[cast]] : memref<192xf32, #gpu.address_space<global>> to memref<192xf32, #amdgpu.address_space<fat_raw_buffer>>
    // GFX942: amdgpu.gather_to_lds %[[fatBuff]]
    // GFX942-SAME: f32, memref<192xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<4xf32, #gpu.address_space<workgroup>>
    // GFX1250: amdgpu.async_load_to_lds %[[cast]]
    // GFX1250-SAME: f32, memref<192xf32, #gpu.address_space<global>>, memref<4xf32, #gpu.address_space<workgroup>>
    rock.global_load_to_lds %mem[%c0] -> %lds_view[%c0]  if %true {transferType = f32, canReadOffEnd} : memref<192xf32> -> memref<4xf32, #gpu.address_space<workgroup>>
    return
}

// CHECK-LABEL: func.func @load_scalar
// CHECK-SAME: (%[[mem:.*]]: memref<f32>, %[[idx:.*]]: index)
func.func @load_scalar_empty_mem(%mem: memref<f32>, %idx: index) attributes {kernel, arch = "##TOKEN_ARCH##"} {
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %lds = rock.alloc() : memref<64xi8, #gpu.address_space<workgroup>>
    %lds_view = memref.view %lds[%c0][] : memref<64xi8, #gpu.address_space<workgroup>> to memref<4xf32, #gpu.address_space<workgroup>>
    // CHECK: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // GFX942: amdgpu.gather_to_lds %[[cast]]
    // GFX942-SAME: f32, memref<f32, #gpu.address_space<global>>, memref<4xf32, #gpu.address_space<workgroup>>
    // GFX1250: amdgpu.async_load_to_lds %[[cast]]
    // GFX1250-SAME: f32, memref<f32, #gpu.address_space<global>>, memref<4xf32, #gpu.address_space<workgroup>>
    rock.global_load_to_lds %mem[] -> %lds_view[%c0] if %true {transferType = f32} 
        : memref<f32> -> memref<4xf32, #gpu.address_space<workgroup>>
    return
}

// CHECK-LABEL: func.func @load_scalar_in_bounds_large
// CHECK-SAME: (%[[mem:.*]]: memref<1073741825xf32>)
func.func @load_scalar_in_bounds_large(%mem: memref<1073741825xf32>) attributes {kernel, arch = "##TOKEN_ARCH##"} {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %lds = rock.alloc() : memref<64xi8, #gpu.address_space<workgroup>>
    %lds_view = memref.view %lds[%c0][] : memref<64xi8, #gpu.address_space<workgroup>> to memref<4xf32, #gpu.address_space<workgroup>>
    // CHECK: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // GFX942: amdgpu.gather_to_lds %[[cast]][%c0] 
    // GFX942-SAME: f32, memref<1073741825xf32, #gpu.address_space<global>>, memref<4xf32, #gpu.address_space<workgroup>>
    // GFX1250: amdgpu.async_load_to_lds %[[cast]][%c0]
    // GFX1250-SAME: f32, memref<1073741825xf32, #gpu.address_space<global>>, memref<4xf32, #gpu.address_space<workgroup>>
    rock.global_load_to_lds %mem[%c0] -> %lds_view[%c0] if %true {transferType = f32, needs64BitIdx}
        : memref<1073741825xf32> -> memref<4xf32, #gpu.address_space<workgroup>>
    return
}

// CHECK-LABEL: func.func @load_4bit_boundary_case_to_lds
// CHECK-SAME: (%[[mem:.*]]: memref<4294967295xi4>)
func.func @load_4bit_boundary_case_to_lds(%mem: memref<4294967295xi4>) attributes {kernel, arch = "##TOKEN_ARCH##"} {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %lds = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %lds_view = memref.view %lds[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<8xi4, #gpu.address_space<workgroup>>
    // 4294967295 elements * 4 bits = 17179869180 bits
    // ceil(17179869180 / 8) = 2147483648 bytes = exactly 2^31 bytes
    // This triggers numBytes.trunc(32).isNegative() without needs64BitIdx
    // Forces buffer operations due to size exceeding 2GB threshold
    // CHECK: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // GFX942: %[[fatBuff:.*]] = amdgpu.fat_raw_buffer_cast %[[cast]]
    // GFX942-SAME: memref<4294967295xi4, #gpu.address_space<global>> to memref<4294967295xi4, #amdgpu.address_space<fat_raw_buffer>>
    // GFX942: amdgpu.gather_to_lds %[[fatBuff]]
    // GFX942-SAME: i32, memref<4294967295xi4, #amdgpu.address_space<fat_raw_buffer>>, memref<8xi4, #gpu.address_space<workgroup>>
    // GFX1250: amdgpu.async_load_to_lds %[[cast]]
    // GFX1250-SAME: i32, memref<4294967295xi4, #gpu.address_space<global>>, memref<8xi4, #gpu.address_space<workgroup>>
    rock.global_load_to_lds %mem[%c0] -> %lds_view[%c0] if %true {transferType = i32}
        : memref<4294967295xi4> -> memref<8xi4, #gpu.address_space<workgroup>>
    return
}

// CHECK-LABEL: func.func @load_4bit_boundary_case_to_lds_f4E2M1FN
// CHECK-SAME: (%[[mem:.*]]: memref<4294967295xf4E2M1FN>)
func.func @load_4bit_boundary_case_to_lds_f4E2M1FN(%mem: memref<4294967295xf4E2M1FN>) attributes {kernel, arch = "##TOKEN_ARCH##"} {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %lds = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %lds_view = memref.view %lds[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<8xf4E2M1FN, #gpu.address_space<workgroup>>
    // Same boundary case with f4E2M1FN type
    // 4294967295 elements * 4 bits = 2147483648 bytes after ceiling division
    // CHECK: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // GFX942: %[[fatBuff:.*]] = amdgpu.fat_raw_buffer_cast %[[cast]]
    // GFX942-SAME: memref<4294967295xf4E2M1FN, #gpu.address_space<global>> to memref<4294967295xf4E2M1FN, #amdgpu.address_space<fat_raw_buffer>>
    // GFX942: amdgpu.gather_to_lds %[[fatBuff]]
    // GFX942-SAME: f32, memref<4294967295xf4E2M1FN, #amdgpu.address_space<fat_raw_buffer>>, memref<8xf4E2M1FN, #gpu.address_space<workgroup>>
    // GFX1250: amdgpu.async_load_to_lds %[[cast]]
    // GFX1250-SAME: f32, memref<4294967295xf4E2M1FN, #gpu.address_space<global>>, memref<8xf4E2M1FN, #gpu.address_space<workgroup>>
    rock.global_load_to_lds %mem[%c0] -> %lds_view[%c0] if %true {transferType = f32}
        : memref<4294967295xf4E2M1FN> -> memref<8xf4E2M1FN, #gpu.address_space<workgroup>>
    return
}

// CHECK-LABEL: func.func @load_conditionally_oob_checks
// CHECK-SAME: (%[[mem:.*]]: memref<512xf16>)
func.func @load_conditionally_oob_checks(%arg0: memref<512xf16>) attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %c5 = arith.constant 5 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = rock.alloc() : memref<4096xf16, #gpu.address_space<workgroup>>
  // Check that we generate a buffer_load for gfx950 and a global_load with oob checks for gfx1250
  // GFX1250: %[[zerosvec:.*]] = arith.constant dense<0.000000e+00> : vector<1xf128>
  // CHECK: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
  // CHECK-SAME: #gpu.address_space<global>
  // GFX942: %[[fatBuff:.*]] = amdgpu.fat_raw_buffer_cast %[[cast]]
  // GFX942-SAME: memref<512xf16, #gpu.address_space<global>> to memref<512xf16, #amdgpu.address_space<fat_raw_buffer>>
  // CHECK: scf.for %[[index:.*]] = {{.*}}
  scf.for %arg3 = %c0 to %c5 step %c1 {
    // CHECK: %[[cond:.*]] = arith.index_cast %[[index]] : index to i1
    %1 = arith.index_cast %arg3 : index to i1
    %2 = arith.addi %arg3, %arg3 : index
    // GFX942: amdgpu.gather_to_lds %[[fatBuff]]
    // GFX942-SAME: f128, memref<512xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<4096xf16, #gpu.address_space<workgroup>>
    // GFX1250: scf.if %[[cond]] {
    // GFX1250:   amdgpu.async_load_to_lds %[[cast]]
    // GFX1250-SAME: f128, memref<512xf16, #gpu.address_space<global>>, memref<4096xf16, #gpu.address_space<workgroup>>
    // GFX1250: else {
    // GFX1250:  %[[bitcast:.*]] = vector.bitcast %[[zerosvec]] : vector<1xf128> to vector<8xf16>
    // GFX1250:  %[[extracted1:.*]] = vector.extract %[[bitcast]][0] : f16 from vector<8xf16>
    // GFX1250:  memref.store %[[extracted1]], {{.*}} : memref<4096xf16, #gpu.address_space<workgroup>>
    // GFX1250:  %[[extracted7:.*]] = vector.extract %[[bitcast]][7] : f16 from vector<8xf16>
    // GFX1250:  memref.store %[[extracted7]], {{.*}} : memref<4096xf16, #gpu.address_space<workgroup>>
    rock.global_load_to_lds %arg0[%arg3] -> %0[%2] if %1 {transferType = f128} : memref<512xf16> -> memref<4096xf16, #gpu.address_space<workgroup>>
  }
  return
}

}
