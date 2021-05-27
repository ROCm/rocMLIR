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

llvm_canonicalize_cmake_booleans(BUILD_FAT_LIBMLIRMIOPEN)
# Static library target, enabled only when building static libs
if( BUILD_FAT_LIBMLIRMIOPEN )
  function(combine_archives output_archive)
    set(mri_file ${CMAKE_CURRENT_BINARY_DIR}/${output_archive}.mri)
    set(full_output_path ${LLVM_LIBRARY_DIR}/lib${output_archive}.a)
    set(output_archive_dummy_file ./${output_archive}.dummy.cpp)

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
                       DEPENDS ${output_archive_dummy_file})

    install(FILES ${full_output_path}
            TYPE LIB
            COMPONENT libMLIRMIOpen EXCLUDE_FROM_ALL)
    install(FILES Miir.h
            TYPE INCLUDE
            COMPONENT libMLIRMIOpen EXCLUDE_FROM_ALL)
  endfunction(combine_archives)

  combine_archives(MLIRMIOpen)

  add_custom_target(libMLIRMIOpen ALL
    DEPENDS
    MLIRMIOpen
    )

  install(TARGETS MLIRMIOpen
    ARCHIVE DESTINATION lib)

endif()
