set_target_properties(MLIRMIOpenUnitTests PROPERTIES FOLDER "MLIR MIOpen Tests")

function(add_mlir_miopen_unittest test_dirname)
  add_unittest(MLIRMIOpenUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Dialect)
