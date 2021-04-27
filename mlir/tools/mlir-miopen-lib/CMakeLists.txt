get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  LLVMAMDGPUAsmParser
  LLVMX86AsmParser
  LLVMROCmBackendUtils
  MLIRAnalysis
  MLIRExecutionEngine
  MLIRMIOpenOps
  MLIRParser
  MLIRPass
  MLIRQuant
  MLIRShape
  MLIRSDBM
  MLIRTransforms
  MLIRSupport
  MLIRIR
  MLIRTargetMIOpenCppTranslation
)

llvm_add_library(MLIRMIOpenThin
PARTIAL_SOURCES_INTENDED

  mlir-miopen-lib.cpp

  LINK_LIBS
  ${LIBS}
  )

add_llvm_executable(mlir-miopen-lib-test
  PARTIAL_SOURCES_INTENDED

  mlir-miopen-lib-test.cpp

  DEPENDS
  MLIRMIOpenThin
  ${LIBS}
)

llvm_update_compile_flags(mlir-miopen-lib-test)
target_link_libraries(mlir-miopen-lib-test PRIVATE MLIRMIOpenThin ${LIBS})
mlir_check_link_libraries(mlir-miopen-lib-test)

install(FILES "Miir.h"
  DESTINATION include)

# Static library target, enabled only when building static libs
if( NOT BUILD_SHARED_LIBS )
  function(combine_archives output_archive)
    set(mri_file ${CMAKE_CURRENT_BINARY_DIR}/${output_archive}.mri)
    set(full_output_path ${LLVM_LIBRARY_DIR}/lib${output_archive}.a)
    set(output_archive_dummy_file ./${output_archive}.dummy.cpp)
    set(install_path /opt/rocm/lib/lib${output_archive}.a)
    set(header_path /opt/rocm/include/)

    # Step one: construct mri file.
    add_custom_command(OUTPUT ${output_archive_dummy_file}
                       COMMAND if [ -f ${output_archive}.mri ]\; then rm ${output_archive}.mri\; fi
                       COMMAND touch ${output_archive}.mri
                       COMMAND echo "create ${full_output_path}" >> ${output_archive}.mri
                       COMMAND for archive in ${LLVM_LIBRARY_DIR}/*.a\;
                       do echo "addlib $$archive" >> ${output_archive}.mri \; done
                       COMMAND echo "save" >> ${output_archive}.mri
                       COMMAND echo "end" >> ${output_archive}.mri
                       COMMAND touch ${output_archive_dummy_file}
                       DEPENDS MLIRMIOpenThin)

    # Step two: use mri file to generate the fat library.
    llvm_add_library(${output_archive}
      PARTIAL_SOURCES_INTENDED
      STATIC ${output_archive_dummy_file})
    add_custom_command(TARGET ${output_archive}
                       POST_BUILD
                       COMMAND ${CMAKE_AR} -M < ${mri_file}
                       COMMAND ${CMAKE_COMMAND} -E copy ${full_output_path} ${install_path}
                       COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Miir.h ${header_path}
                       DEPENDS ${output_archive_dummy_file})
  endfunction(combine_archives)

  combine_archives(MLIRMIOpen)

  add_custom_target(libMLIRMIOpen ALL
    DEPENDS
    MLIRMIOpen
    )

  install(TARGETS MLIRMIOpen
    ARCHIVE DESTINATION lib)

endif()
