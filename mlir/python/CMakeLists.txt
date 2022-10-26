include(AddMLIRPython)

#add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=rocmlir.")

# Disables generation of "version soname" (i.e. libFoo.so.<version>), which
# causes pure duplication as part of Python wheels.
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)


declare_mlir_python_extension(rocMLIRPythonExtension
  MODULE_NAME _rocMlir
  ROOT_DIR "${CMAKE_SOURCE_DIR}/mlir/python"
  SOURCES
    rocMLIRModule.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIRegisterRocMLIR
  PRIVATE_LINK_LIBS
    LLVMSupport
)

add_mlir_python_common_capi_library(rocMLIRAggregateCAPI
  INSTALL_COMPONENT rocMLIRPythonModules
  INSTALL_DESTINATION python_packages/rocmlir/rocmlir/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_packages/rocmlir/rocmlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    rocMLIRPythonExtension
)

add_mlir_python_modules(rocMLIRPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python_packages/rocmlir/rocmlir"
  INSTALL_PREFIX "python_packages/rocmlir/rocmlir"
  DECLARED_SOURCES
    rocMLIRPythonExtension
  COMMON_CAPI_LINK_LIBS
    rocMLIRAggregateCAPI
  )
