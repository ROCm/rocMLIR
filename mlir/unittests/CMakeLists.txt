set_target_properties(RocMLIRUnitTests PROPERTIES FOLDER "MLIR Rock Tests")

function(add_mlir_rock_unittest test_dirname)
  add_unittest(RocMLIRUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Dialect)
