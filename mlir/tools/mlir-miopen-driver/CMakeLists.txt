set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRAnalysis
  MLIRMIOpenOps
  MLIRParser
  MLIRPass
  MLIRQuant
  MLIRShape
  MLIRSDBM
  MLIRStandardOps
  MLIRTransforms
  MLIRSupport
  MLIRIR
  MLIRTargetMIOpenCppTranslation
)

add_llvm_executable(mlir-miopen-driver
  PARTIAL_SOURCES_INTENDED

  mlir-miopen-driver.cpp

  DEPENDS
  ${LIBS}
)

#llvm_add_library(MLIRMIOpenThin STATIC
llvm_add_library(MLIRMIOpenThin STATIC
#add_llvm_executable(MLIRMIOpen 
PARTIAL_SOURCES_INTENDED

  mlir-miopen-lib.cpp

  #DEPENDS
  LINK_LIBS
  ${LIBS}
  )

llvm_update_compile_flags(mlir-miopen-driver)
target_link_libraries(mlir-miopen-driver PRIVATE ${LIBS})
mlir_check_link_libraries(mlir-miopen-driver)

llvm_update_compile_flags(MLIRMIOpenThin)
target_link_libraries(MLIRMIOpenThin PUBLIC ${LIBS})
mlir_check_link_libraries(MLIRMIOpenThin)

add_custom_command(
  OUTPUT libMLIRMIOpen.a
  COMMAND ${CMAKE_AR} -cqT libMLIRMIOpen.a ${LLVM_LIBRARY_DIR}/*.a
  COMMAND cp libMLIRMIOpen.a ${LLVM_LIBRARY_DIR}
  )

add_custom_target(libMLIRMIOpen ALL
  DEPENDS 
  libMLIRMIOpen.a
  MLIRMIOpenThin
  )
