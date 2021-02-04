./build/bin/mlir-opt  -miopen-affine-transform -miopen-affix-params group_test.mlir
./build/bin/mlir-opt  -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 group_test.mlir
