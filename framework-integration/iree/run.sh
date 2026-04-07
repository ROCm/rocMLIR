/home/vhe/rocMLIR/build-release/bin/rocmlir-opt /home/vhe/DotFiles/work/scripts/test.mlir --migraphx-to-linalg --canonicalize --cse | tee linalg.mlir
/opt/iree/bin/iree-compile linalg.mlir --iree-hal-target-device=hip --iree-rocm-target=gfx950 -o result.vmfb  --iree-opt-level=O3
/opt/iree/bin/iree-benchmark-module --module=result.vmfb --device=hip://GPU-62363665-6661-6533-3832-363633636434 --function=testing  --input=100xf32=2 --input=100xf32=3
