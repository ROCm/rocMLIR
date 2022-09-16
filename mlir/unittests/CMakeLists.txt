set_target_properties(MLIRRockUnitTests PROPERTIES FOLDER "MLIR Rock Tests")

function(add_mlir_rock_unittest test_dirname)
  add_unittest(MLIRRockUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Dialect)
