set_target_properties(RocMLIRUnitTests PROPERTIES FOLDER "MLIR Rock Tests")

function(add_rocmlir_unittest test_dirname)
  add_unittest(RocMLIRUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Dialect)
