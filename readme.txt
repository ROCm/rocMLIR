./build/bin/mlir-opt  -miopen-affine-transform -miopen-affix-params group_test.mlir
./build/bin/mlir-opt  -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 group_test.mlir
./build/bin/mlir-opt  -miopen-affine-transform -miopen-affix-params group_test.mlir -miopen-lowering-step2 -miopen-lowering-step3
./build/bin/mlir-opt  -miopen-affine-transform -miopen-affix-params group_test.mlir -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4
