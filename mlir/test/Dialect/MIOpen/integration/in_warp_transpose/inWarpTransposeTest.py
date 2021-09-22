#!/usr/bin/env python3
# RUN: %python %s | miopen-opt \
# RUN: --convert-linalg-to-affine-loops --lower-affine \
# RUN: --miopen-lowering-step4 --miopen-lowering-step5 \
# RUN: --tensor-constant-bufferize --finalizing-bufferize \
# RUN:  | mlir-rocm-runner \
# RUN: --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
# RUN: --entry-point-result=i32 | FileCheck %s

from collections import namedtuple
from typing import List

TestSpec = namedtuple('TestSpec', ['size', 'perm', 'name'])
DEFAULT_SPECS: List[TestSpec] = [TestSpec(4, [0, 1, 2, 3], "tr_4x4"),
    TestSpec(2, [0, 1, 2, 3], "tr_2x2"),
    TestSpec(2, [0, 2, 1, 3], "tr_2x2_0213")]

def gpuCode(spec: TestSpec) -> str:
    """Generates the gpu-side code for calling a particular inWarpTranspose test"""
    ret = [f"""
gpu.module @{spec.name} {{
gpu.func @{spec.name}_kern(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) -> () kernel
attributes {{block_size = 64 : i32, grid_size = 64 : i32}} {{
  %cst0 = constant 0 : index
  %cst0_i32 = constant 0 : i32
  %cst1 = constant 1 : index
  %cst16 = constant 16 : index
  %cst64 = constant 64 : index
  %cst64_i32 = constant 64 : i32

  %workitem = "gpu.thread_id"() {{ dimension = "x" }} : () -> (index)
  %tid = remi_unsigned %workitem, %cst64 : index
  %vec_0 = constant dense<-1> : vector<16xi32>"""]
    for i in range(16):
        if i != 0:
            ret.append(f"%cst{i}_i32 = constant {i} : i32")
        load_idx = "%tid" if i == 0 else f"%load_idx_{i}"
        ret.append(f"""
%v_{i} = memref.load %arg0[{load_idx}] : memref<1024xi32>
%vec_{i+1} = vector.insertelement %v_{i}, %vec_{i}[ %cst{i}_i32 : i32 ] : vector<16xi32>
%load_idx_{i+1} = addi {load_idx}, %cst64 : index""")
    ret.append(f"""gpu.barrier
%transposed = miopen.in_warp_transpose {{ size = {spec.size} : i32,
  inGroupPerm = [ {', '.join(str(i) + " : i32" for i in spec.perm)}] }} %vec_16, %tid
  : vector<16xi32>, index
gpu.barrier""")
    for i in range(16):
        store_idx = "%tid" if i == 0 else f"%store_idx_{i}"
        ret.append(f"""
%e_{i} = vector.extractelement %transposed[%cst{i}_i32 : i32] : vector<16xi32>
memref.store %e_{i}, %arg1[{store_idx}] : memref<1024xi32>
%store_idx_{i+1} = addi {store_idx}, %cst64 : index""")
    ret.append("""
gpu.return
}
}""")
    return '\n'.join(ret)

def swizzleMap(spec: TestSpec, i: int, j: int) -> int:
    size = spec.size
    outI = size * (i // size) + (j % size)
    outJ = size * (j // size) + (i % size)
    return outJ + 64 * outI

def swizzleGoal(spec: TestSpec) -> List[int]:
    unpermuted = [swizzleMap(spec, r, t) for r in range(16) for t in range(64)]
    ret = [unpermuted[4 * (i // 4) + spec.perm[i % 4]]
        for i in range(len(unpermuted))]
    return ret

def hostCode(spec: TestSpec) -> str:
    """Generates the code needed to call a given inWarpTranspose kernel"""
    init = [t + 64 * r for r in range(16) for t in range(64)]
    goal = swizzleGoal(spec)

    ret = f"""
func @host_{spec.name}_run(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {{
    %cst64 = constant 64 : index
    %cst = constant 1 : index
    gpu.launch_func @{spec.name}::@{spec.name}_kern
        blocks in (%cst, %cst, %cst)
        threads in (%cst64, %cst, %cst)
        args(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>)
    return
}}

func @host_{spec.name}() -> i1 {{
    %init = constant dense<{init}> : tensor<1024xi32>
    %goal = constant dense<{goal}> :  tensor<1024xi32>
    %cst1_i32 = constant 1 : i32
    %cst2_i32 = constant 2 : i32
    %cst_deadbeef_i32 = constant 0xdeadbeef : i32

    %init_mem = memref.buffer_cast %init : memref<1024xi32>
    %goal_mem = memref.buffer_cast %goal : memref<1024xi32>

    %arg = memref.alloc() : memref<1024xi32>
    %res = memref.alloc() : memref<1024xi32>

    linalg.copy(%init_mem, %arg) : memref<1024xi32>, memref<1024xi32>
    %arg_dyn = memref.cast %arg : memref<1024xi32> to memref<?xi32>
    %arg_gpu_dyn = call @mgpuMemAllocInt32(%arg_dyn) : (memref<?xi32>) -> (memref<?xi32>)
    call @mgpuMemCopyInt32(%arg_dyn, %arg_gpu_dyn, %cst1_i32) : (memref<?xi32>, memref<?xi32>, i32) -> ()
    %arg_gpu = memref.cast %arg_gpu_dyn : memref<?xi32> to memref<1024xi32>

    %res_dyn = memref.cast %res : memref<1024xi32> to memref<?xi32>
    %res_gpu_dyn = call @mgpuMemAllocInt32(%res_dyn) : (memref<?xi32>) -> (memref<?xi32>)
    call @mgpuMemSetInt32(%res_gpu_dyn, %cst_deadbeef_i32) : (memref<?xi32>, i32) -> ()
    %res_gpu = memref.cast %res_gpu_dyn : memref<?xi32> to memref<1024xi32>

    call @host_{spec.name}_run(%arg_gpu, %res_gpu) : (memref<1024xi32>, memref<1024xi32>) -> ()

    call @mgpuMemCopyInt32(%res_gpu_dyn, %res_dyn, %cst2_i32) : (memref<?xi32>, memref<?xi32>, i32) -> ()

    %true = constant 1 : i1
    %ret = affine.for %i = 0 to 1024 iter_args(%state = %true) -> (i1) {{
        %goal_e = affine.load %goal_mem[%i] : memref<1024xi32>
        %arg_e = affine.load %res[%i] : memref<1024xi32>
        %current = cmpi "eq", %goal_e, %arg_e : i32
        %next = and %state, %current : i1
        affine.yield %next : i1
    }}
    scf.if %ret {{
        scf.yield
    }} else {{
        %res_no_shape = memref.cast %res : memref<1024xi32> to memref<*xi32>
        call @print_memref_i32(%res_no_shape) : (memref<*xi32>) -> ()
    }}
    call @mgpuMemDeallocInt32(%arg_gpu_dyn) : (memref<?xi32>) -> ()
    call @mgpuMemDeallocInt32(%res_gpu_dyn) : (memref<?xi32>) -> ()
    return %ret : i1
}}
"""
    return ret

def genTestProgram(specs: List[TestSpec]):
    ret: List[str] = [f"""
module attributes {{gpu.container_module}} {{
    func private @mgpuMemAllocInt32(%ptr : memref<?xi32>) -> (memref<?xi32>)
    func private @mgpuMemDeallocInt32(%ptr : memref<?xi32>)
    func private @mgpuMemSetInt32(%ptr : memref<?xi32>, %value: i32)
    func private @mgpuMemCopyInt32(%src : memref<?xi32>, %dest : memref<?xi32>, %dir: i32)
    func private @print_memref_i32(memref<*xi32>)
"""]
    for spec in specs:
        ret.append(gpuCode(spec))
    for spec in specs:
        ret.append(hostCode(spec))
    ret.append("""
func @main() -> i32 {
    %cst0_i32 = constant 0 : i32
    %cst1_i32 = constant 1 : i32
""")
    for spec in specs:
        ret.append(f"%{spec.name} = call @host_{spec.name}() : () -> i1")
    ret.append("%0 = constant 1 : i1")
    for i, spec in enumerate(specs):
        ret.append(f"%{i + 1} = and %{i}, %{spec.name} : i1")
    ret.append(f"%ret = select %{len(specs)}, %cst0_i32, %cst1_i32 : i32")
    ret.append("""
    return %ret : i32
    }
    }""")
    return "\n".join(ret)

if __name__ == '__main__':
    print(genTestProgram(DEFAULT_SPECS))
# CHECK: {{^}}0{{$}}

