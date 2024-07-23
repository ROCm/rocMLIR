add_rocmlir_dialect_library(MLIRRocTarget
  Target.cpp

  LINK_LIBS PUBLIC
  MLIRROCDLTarget
  MLIRLLVMToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  )

llvm_canonicalize_cmake_booleans(
    MLIR_ENABLE_ROCM_CONVERSIONS
    )

if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  if (NOT ("lld" IN_LIST LLVM_ENABLE_PROJECTS))
    message(SEND_ERROR
      "LLD is not enabled, please reconfigure llvm build")
  endif()

  ### rocMLIR-specific incantations to embed device libraries statically.
  ### This is meant to let us embed the device libraries into a static library.
  if (NOT DEFINED ROCM_PATH)
    if (NOT DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCm has been installed")
    else()
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCm has been installed")
    endif()
  endif()
  message(VERBOSE "MLIR Default ROCM toolkit path: ${ROCM_PATH}")

  target_compile_definitions(obj.MLIRRocTarget
    PRIVATE
    __DEFAULT_ROCM_PATH__="${ROCM_PATH}"
  )
  # A lot of the ROCm CMake files expect to find their own dependencies in
  # CMAKE_PREFIX_PATH and don't respect PATHS or HINTS :( .
  # Therefore, temporarily add the ROCm path to CMAKE_PREFIX_PATH so we can
  # find the device libraries, then remove it
  set(REAL_CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} "${ROCM_PATH}/hip")
  find_package(AMDDeviceLibs CONFIG)
  set(CMAKE_PREFIX_PATH "${REAL_CMAKE_PREFIX_PATH}")
  if(AMDDeviceLibs_FOUND)
    set(device_lib_targets ${AMD_DEVICE_LIBS_TARGETS})
    # Filter out control constants, since we set those inline
    list(FILTER device_lib_targets EXCLUDE REGEX "^oclc_")
    # None of our stuff uses opencl functions or functionality during
    # compilation, and the only time that device library gets pulled in is for
    # printf(), which we use the hip version of anyway.
    list(REMOVE_ITEM device_lib_targets "opencl")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/AmdDeviceLibs.cpp.inc
      COMMAND ${Python3_EXECUTABLE}
      ARGS
        ${CMAKE_CURRENT_SOURCE_DIR}/AmdDeviceLibsIncGen.py
        ${CMAKE_CURRENT_BINARY_DIR}/AmdDeviceLibs.cpp.inc
        ${AMD_DEVICE_LIBS_PREFIX}
        ${device_lib_targets}
      DEPENDS
        AmdDeviceLibsIncGen.py
        ${AMD_DEVICE_LIBS_TARGETS}
      COMMENT "Generating device libraries include package"
    )
    set_property(DIRECTORY APPEND PROPERTY MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/AmdDeviceLibs.cpp.inc)
    add_custom_target(AmdDeviceLibsIncGen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/AmdDeviceLibs.cpp.inc)
    set_property(SOURCE Target.cpp APPEND
      PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/AmdDeviceLibs.cpp.inc)
    add_dependencies(obj.MLIRRocTarget AmdDeviceLibsIncGen)
    target_compile_definitions(obj.MLIRRocTarget
      PRIVATE
      ROCMLIR_DEVICE_LIBS_PACKAGED=1
    )

  else()
    message(NOTICE "Device libraries not found during builds, assuming we can find them at runtime")
  endif()
  ### End rocMLIR-specific hackery

  target_include_directories(obj.MLIRRocTarget
    PUBLIC
    ${LLVM_EXTERNAL_SRC_DIR}/lld/include
  )
  target_link_libraries(MLIRRocTarget
    PRIVATE
    lldELF
    lldCommon
  )

  # Link lldELF also to libmlir.so. Create an alias that starts with LLVM
  # because LINK_COMPONENTS elements are implicitly prefixed with LLVM.
  add_library(LLVMAliasTolldELF ALIAS lldELF)
  set_property(GLOBAL APPEND PROPERTY MLIR_LLVM_LINK_COMPONENTS AliasTolldELF)
  endif()
