get_property(rocmlir_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)
get_property(rocmlir_capi_libs GLOBAL PROPERTY ROCMLIR_PUBLIC_C_API_LIBS)
get_property(rocmlir_conversion_libs GLOBAL PROPERTY ROCMLIR_CONVERSION_LIBS)

set(LIBS
  ${rocmlir_dialect_libs}
  ${rocmlir_capi_libs}
  ${rocmlir_conversion_libs}
)

set(CMAKE_BUILD_RPATH ${CMAKE_BUILD_DIR}/external/llvm-project/llvm/lib)

llvm_add_library(MLIRRockThin
PARTIAL_SOURCES_INTENDED

  rocmlir-lib.cpp

  LINK_LIBS
  ${LIBS}
  )

add_llvm_executable(rocmlir-lib-test
  PARTIAL_SOURCES_INTENDED

  rocmlir-lib-test.cpp

  DEPENDS
  MLIRRockThin
  ${LIBS}
)

llvm_update_compile_flags(rocmlir-lib-test)
target_link_libraries(rocmlir-lib-test PRIVATE MLIRRockThin ${LIBS})
mlir_check_link_libraries(rocmlir-lib-test)

llvm_canonicalize_cmake_booleans(BUILD_FAT_LIBROCKCOMPILER)
# Static library target, enabled only when building static libs
if(BUILD_FAT_LIBROCKCOMPILER)
  find_package(ROCM 0.8 REQUIRED PATHS /opt/rocm)
  include(ROCMInstallTargets)
  include(ROCMCreatePackage)

  set(ROCMCHECKS_WARN_TOOLCHAIN_VAR OFF)
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

  string(TOLOWER ${CMAKE_PROJECT_NAME} CMAKE_PROJECT_NAME_LOWER)
  set(PACKAGE_NAME ${CMAKE_PROJECT_NAME_LOWER})

  set(LIBRARY_NAME rockCompiler)
  set(__library_name ${CMAKE_STATIC_LIBRARY_PREFIX}${LIBRARY_NAME})

  # Static libraries on Windows have a 4GB size limit. The Debug version of the FAT library
  # exceeds that limit significantly. Therefore, the change is to expose CMake's INTERFACE
  # library instead as the interface to clients. Underneath, on Linux, the single monolithic
  # library is delivered (as it was before). However, we deliver a series of smaller static
  # libraries on Windows due to the size limit mentioned.
  # The client interface has not changed and is OS agnostic. Both MIOpen and MIGraphX do not
  # need to modify their CMake code to find the rocMLIR package and build with it.
  add_library(${LIBRARY_NAME} INTERFACE)

  set(__llvm_targets
    lldCommon
    lldELF
    LLVMAggressiveInstCombine
    LLVMAMDGPUAsmParser
    LLVMAMDGPUCodeGen
    LLVMAMDGPUDesc
    LLVMAMDGPUDisassembler
    LLVMAMDGPUInfo
    LLVMAMDGPUUtils
    LLVMAnalysis
    LLVMAsmParser
    LLVMAsmPrinter
    LLVMBinaryFormat
    LLVMBitReader
    LLVMBitstreamReader
    LLVMBitWriter
    LLVMCodeGen
    LLVMCodeGenTypes
    LLVMCore
    LLVMCoroutines
    LLVMDebugInfoCodeView
    LLVMDebugInfoDWARF
    LLVMDebugInfoMSF
    LLVMDebugInfoPDB
    LLVMDemangle
    LLVMExtensions
    LLVMFrontendOpenMP
    LLVMGlobalISel
    LLVMInstCombine
    LLVMInstrumentation
    LLVMipo
    LLVMIRPrinter
    LLVMIRReader
    LLVMLinker
    LLVMLTO
    LLVMMC
    LLVMMCDisassembler
    LLVMMCParser
    LLVMMIRParser
    LLVMObjCARCOpts
    LLVMObject
    LLVMOption
    LLVMPasses
    LLVMProfileData
    LLVMRemarks
    LLVMScalarOpts
    LLVMSelectionDAG
    LLVMSupport
    LLVMSymbolize
    LLVMTableGen
    LLVMTableGenGlobalISel
    LLVMTarget
    LLVMTargetParser
    LLVMTextAPI
    LLVMTransformUtils
    LLVMVectorize
    MLIRAffineAnalysis
    MLIRAffineDialect
    MLIRAffineToStandard
    MLIRAffineTransforms
    MLIRAffineUtils
    MLIRAMDGPUDialect
    MLIRAMDGPUToROCDL
    MLIRAMDGPUTransforms
    MLIRAMDGPUUtils
    MLIRAMXDialect
    MLIRAMXTransforms
    MLIRAnalysis
    MLIRArithAttrToLLVMConversion
    MLIRArithDialect
    MLIRArithToAMDGPU
    MLIRArithToLLVM
    MLIRArithTransforms
    MLIRArithUtils
    MLIRArmNeonDialect
    MLIRArmSMEDialect
    MLIRArmSMETransforms
    MLIRArmSVEDialect
    MLIRArmSVETransforms
    MLIRAsmParser
    MLIRAsyncDialect
    MLIRAsyncToLLVM
    MLIRAsyncTransforms
    MLIRBufferizationDialect
    MLIRBufferizationTransformOps
    MLIRBufferizationTransforms
    MLIRBuiltinToLLVMIRTranslation
    MLIRBytecodeOpInterface
    MLIRBytecodeReader
    MLIRBytecodeWriter
    MLIRCallInterfaces
    MLIRCAPIIR
    MLIRCastInterfaces
    MLIRComplexDialect
    MLIRControlFlowDialect
    MLIRControlFlowInterfaces
    MLIRControlFlowToLLVM
    MLIRCopyOpInterface
    MLIRDataLayoutInterfaces
    MLIRDestinationStyleOpInterface
    MLIRDialect
    MLIRDialectUtils
    MLIRDLTIDialect
    MLIRExecutionEngineUtils
    MLIRFuncDialect
    MLIRFuncToLLVM
    MLIRFuncTransforms
    MLIRGPUDialect
    MLIRGPUToGPURuntimeTransforms
    MLIRGPUToLLVMIRTranslation
    MLIRGPUToROCDLTransforms
    MLIRGPUTransforms
    MLIRIndexDialect
    MLIRInferIntRangeCommon
    MLIRInferIntRangeInterface
    MLIRInferTypeOpInterface
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgToLLVM
    MLIRLinalgTransforms
    MLIRLinalgUtils
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRLLVMIRTransforms
    MLIRLLVMToLLVMIRTranslation
    MLIRLoopLikeInterface
    MLIRMaskableOpInterface
    MLIRMaskingOpInterface
    MLIRMathDialect
    MLIRMathToLibm
    MLIRMathToLLVM
    MLIRMemorySlotInterfaces
    MLIRMemRefDialect
    MLIRMemRefToLLVM
    MLIRMemRefTransforms
    MLIRMemRefUtils
    MLIRNVGPUDialect
    MLIRNVVMDialect
    MLIRParallelCombiningOpInterface
    MLIRParser
    MLIRPass
    MLIRPDLDialect
    MLIRPDLInterpDialect
    MLIRPDLToPDLInterp
    MLIRPresburger
    MLIRQuantDialect
    MLIRQuantUtils
    MLIRReconcileUnrealizedCasts
    MLIRRewrite
    MLIRROCDLDialect
    MLIRROCDLToLLVMIRTranslation
    MLIRRuntimeVerifiableOpInterface
    MLIRSCFDialect
    MLIRSCFToControlFlow
    MLIRSCFTransforms
    MLIRSCFUtils
    MLIRShapedOpInterfaces
    MLIRSideEffectInterfaces
    MLIRSparseTensorDialect
    MLIRSupport
    MLIRTableGen
    MLIRTargetLLVMIRExport
    MLIRTblgenLib
    MLIRTensorDialect
    MLIRTensorInferTypeOpInterfaceImpl
    MLIRTensorTilingInterfaceImpl
    MLIRTensorToLinalg
    MLIRTensorTransforms
    MLIRTensorUtils
    MLIRTilingInterface
    MLIRTosaDialect
    MLIRTosaToArith
    MLIRTosaToLinalg
    MLIRTosaToSCF
    MLIRTosaToTensor
    MLIRTosaTransforms
    MLIRTransformDialect
    MLIRTransformDialectUtils
    MLIRTransforms
    MLIRTransformUtils
    MLIRTranslateLib
    MLIRValueBoundsOpInterface
    MLIRVectorDialect
    MLIRVectorInterfaces
    MLIRVectorToLLVM
    MLIRVectorToSCF
    MLIRVectorTransforms
    MLIRVectorUtils
    MLIRViewLikeInterface
    MLIRX86VectorDialect
    MLIRX86VectorTransforms
  )

  set(__rocmlir_targets
    GpuModuleToRocdlirTranslation
    MLIRCAPIMIGraphX
    MLIRCAPIRegisterRocMLIR
    MLIRCAPIRock
    MLIRGPUToMIGraphX
    MLIRMHAL
    MLIRMHALPipeline
    MLIRMHALSupport
    MLIRMHALToCPU
    MLIRMHALToGPU
    MLIRMHALTransforms
    MLIRMIGraphX
    MLIRMIGraphXPipeline
    MLIRMIGraphXToTosa
    MLIRRockConv2dGenerator
    MLIRRockOps
    MLIRRockPipeline
    MLIRRockTestPasses
    MLIRRockThin
    MLIRRockToGPU
    MLIRRockTransforms
    MLIRRockTuning
    MLIRRockUtility
    MLIRTosaToRock
    RocmlirFp8ExtToTables
  )

  if (WIN32)

    # Setup build and installation interface for LLVM libraries
    foreach(__target ${__llvm_targets})
        list(APPEND __llvm_libraries ${LLVM_EXTERNAL_LIB_DIR}/${__target}.lib)
        target_link_libraries(${LIBRARY_NAME}
            INTERFACE
                $<BUILD_INTERFACE:${__target}>
                $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/lib/llvm/${__target}.lib>)
    endforeach()

    # Setup build and installation interface for rocMLIR libraries
    foreach(__target ${__rocmlir_targets})
        target_link_libraries(${LIBRARY_NAME}
            INTERFACE
                $<BUILD_INTERFACE:${__target}>
                $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/lib/${__target}.lib>)
    endforeach()

    add_dependencies(${LIBRARY_NAME} MLIRRockThin MLIRRockTestPasses GpuModuleToRocdlirTranslation)

    rocm_install(FILES ${__llvm_libraries}
            DESTINATION lib/llvm)

    rocm_install(FILES ${ROCMLIR_LIB_DIR}/MLIRRockThin.lib
            DESTINATION lib)

    unset(__llvm_libraries)

    # Compatibility target with Linux
    add_custom_target(lib${LIBRARY_NAME} ALL DEPENDS ${LIBRARY_NAME})

  else()

    set(full_output_path ${ROCMLIR_LIB_DIR}/${__library_name}${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(mri_file ${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}.mri)

    # Step one: construct mri file.
    add_custom_command(OUTPUT ${mri_file}
                       COMMAND echo "create ${full_output_path}" > ${mri_file}
                       COMMAND for archive in ${ROCMLIR_LIB_DIR}/*.a ${LLVM_EXTERNAL_LIB_DIR}/*.a \; do echo "addlib $$archive" >> ${mri_file} \; done
                       COMMAND echo "save" >> ${mri_file}
                       COMMAND echo "end" >> ${mri_file}
                       DEPENDS MLIRRockThin MLIRRockTestPasses GpuModuleToRocdlirTranslation)

    # Step two: use mri file to generate the fat library.
    add_custom_command(OUTPUT ${full_output_path}
                       COMMAND ${CMAKE_AR} -M < ${mri_file}
                       DEPENDS ${mri_file})

    add_custom_target(__generate_fat_library DEPENDS ${full_output_path})
    add_dependencies(${LIBRARY_NAME} __generate_fat_library)

    target_link_libraries(${LIBRARY_NAME}
        INTERFACE
            $<BUILD_INTERFACE:${full_output_path}>
            $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/lib/${__library_name}${CMAKE_STATIC_LIBRARY_SUFFIX}>)

    rocm_install(FILES ${full_output_path}
            DESTINATION lib)

    # Backward compatibility to provide `librockCompiler` build target.
    # INTERFACE libraries are not exposed as build targets.
    add_custom_target(${__library_name} ALL DEPENDS ${LIBRARY_NAME})

    unset(mri_file)
    unset(full_output_path)

  endif()

  # Install Miir.h to ${CMAKE_INSTALL_PREFIX}/include/${PACKAGE_NAME}/
  # as part of component rockCompiler
  rocm_install(FILES Miir.h
    DESTINATION include/${PACKAGE_NAME})

  # Install a target, i.e. ${package_name}::${component_name}, as part of
  # devel component and export it to be searchable by find_package().
  # The include directories are propagated to the downstream targets
  # that links to ${package_name}::${component_name}
  # Be default, include/${package_name} is added. If extra include directories
  # are needed, add them using the keyword EXTRA_INCLUDES.
  #
  # As an example, when CMAKE_INSTALL_PREFIX is set to /a/dummy/ and
  # CMAKE_BUILD_TYPE is set to CONFIG
  # The following files are generated and installed as a result of
  # cmake --install . --component ${component_name} --prefix /a/dummy
  # /a/dummy/lib/${package_name}/${component_name}.a
  # /a/dummy/lib/cmake/${package_name}/${package_name}-targets.cmake
  #   This file is autogenerated by install(EXPORT).
  #   It creates the imported library as a target that can be used by
  #   target_link_libraries()
  #   It also sets the INTERFACE_INCLUDE_DIRECTORIES property of the target
  # /a/dummy/lib/cmake/${package_name}/${package_name}-targets-release.cmake
  #   This file is autogenerated by install(EXPORT).
  #   It sets the location of the library as a property of the imported target
  # /a/dummy/lib/cmake/${package_name}/${package_name}-config.cmake
  #   This file is generated by rocm_export_targets and is searched
  #   and loaded by find_package(). It includes ${package_name}-targets.cmake.
  # /a/dummy/lib/cmake/${package_name}/${package_name}-config-version.cmake
  #   This file is autogenerated by rocm_export_targets and is used by
  #   find_package to implement a package version matching mechanism.
  #
  # The only input to this exporting process is cmake/${package_name}Config.cmake.in,
  # which is used to auto generate ${package_name}Config.cmake
  function(export_target package_name lib_name)
    cmake_parse_arguments(ARG "" "" "EXTRA_INCLUDES" ${ARGN})

    # Install ${component_name} to ${CMAKE_INSTALL_PREFIX}/lib/${package_name}/
    # and group it into the ${export-set}
    # The INTERFACE_INCLUDE_DIRECTORIES will be set as the property of the exported library
    # so that when consumer libraries (Rock) link in ${COMPONENT_NAME}, ${INCLUDE_DIR} is
    # automatically propagated to the consumer library's include_dirs

    # Call install() instead of rocm_install() here in order to install
    # header files in a subdirectory.
    install(TARGETS ${lib_name}
      EXPORT ${package_name}-targets
      ARCHIVE DESTINATION lib
      COMPONENT devel
      INCLUDES DESTINATION include/${package_name} ${ARG_EXTRA_INCLUDES}
    )

    # Generate package config and version file
    rocm_export_targets(
      NAMESPACE ${CMAKE_PROJECT_NAME}::
      NAME ${package_name}
      TARGETS ${package_name}
      COMPATIBILITY AnyNewerVersion
      )

  endfunction()

  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
    rocm_install(DIRECTORY
      ${PROJECT_SOURCE_DIR}/mlir/include/mlir
      ${PROJECT_SOURCE_DIR}/mlir/include/mlir-c
      DESTINATION include/${PACKAGE_NAME}
      FILES_MATCHING
      PATTERN "*.def"
      PATTERN "*.h"
      PATTERN "*.inc"
      PATTERN "*.td"
      PATTERN "LICENSE.TXT"
    )

    rocm_install(DIRECTORY ${MLIR_INCLUDE_DIRS}
      DESTINATION include/${PACKAGE_NAME}/external
      FILES_MATCHING
      PATTERN "*.def"
      PATTERN "*.h"
      PATTERN "*.gen"
      PATTERN "*.inc"
      PATTERN "*.td"
      PATTERN "CMakeFiles" EXCLUDE
      PATTERN "config.h" EXCLUDE
    )

    rocm_install(DIRECTORY ${PROJECT_BINARY_DIR}/mlir/include
      DESTINATION include/${PACKAGE_NAME}/build
      FILES_MATCHING
      PATTERN "*.def"
      PATTERN "*.h"
      PATTERN "*.gen"
      PATTERN "*.inc"
      PATTERN "*.td"
      PATTERN "CMakeFiles" EXCLUDE
      PATTERN "config.h" EXCLUDE
    )

    set(extra_include_dir include/${PACKAGE_NAME}/external/include)
  endif()

  # Install ${LIBRARY_NAME} as component of ${PACKAGE_NAME}
  # and export it to be searchable by find_package()
  export_target(${PACKAGE_NAME} # package name
    ${LIBRARY_NAME}             # library name
    EXTRA_INCLUDES              # extra include directories
    ${extra_include_dir}
  )

  rocm_create_package(
    NAME ${CMAKE_PROJECT_NAME}
    DESCRIPTION "MLIR packages"
    MAINTAINER "rocMLIR Dev Team dl.dl-mlir@amd.com"
    HEADER_ONLY
  )

  # Validate the correctness of the rockCompiler targets list
  if(WIN32)
    cmake_path(NATIVE_PATH LLVM_EXTERNAL_LIB_DIR __llvm_external_lib_dir)
    add_custom_command(OUTPUT llvm_libs.txt
                       COMMAND del /f llvm_libs.txt 2> nul && for %f in (${__llvm_external_lib_dir}\\*.lib) do @echo %~nf>> llvm_libs.txt
                       DEPENDS MLIRRockThin MLIRRockTestPasses GpuModuleToRocdlirTranslation)
    cmake_path(NATIVE_PATH ROCMLIR_LIB_DIR __rocmlir_lib_dir)
    add_custom_command(OUTPUT rocmlir_libs.txt
                       COMMAND del /f rocmlir_libs.txt 2> nul && for %f in (${__rocmlir_lib_dir}\\*.lib) do @echo %~nf>> rocmlir_libs.txt
                       DEPENDS MLIRRockThin MLIRRockTestPasses GpuModuleToRocdlirTranslation)
  else()
    add_custom_command(OUTPUT rocmlir_libs.txt
                       COMMAND (for f in ${ROCMLIR_LIB_DIR}/*.a \; do printf '%s\\n' "\$\${f%.a}" \; done) > rocmlir_libs.txt
                       DEPENDS MLIRRockThin MLIRRockTestPasses GpuModuleToRocdlirTranslation)
    add_custom_command(OUTPUT llvm_libs.txt
                       COMMAND (for f in ${LLVM_EXTERNAL_LIB_DIR}/*.a \; do printf '%s\\n' "\$\${f%.a}" \; done) > llvm_libs.txt
                       DEPENDS MLIRRockThin MLIRRockTestPasses GpuModuleToRocdlirTranslation)
  endif()

  find_package(Python 3.6 REQUIRED COMPONENTS Interpreter)

  list(SORT __llvm_targets)
  string(REPLACE ";" "\n" __llvm_targets "${__llvm_targets}")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/llvm_targets.txt ${__llvm_targets}\n)

  add_custom_target(__check_llvm_libs DEPENDS llvm_libs.txt)
  add_custom_command(TARGET __check_llvm_libs POST_BUILD
                     COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/checkTargets.py ${LIBRARY_NAME}
                                  __llvm_targets llvm_targets.txt llvm_libs.txt)

  list(SORT __rocmlir_targets)
  string(REPLACE ";" "\n" __rocmlir_targets "${__rocmlir_targets}")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/rocmlir_targets.txt ${__rocmlir_targets}\n)

  add_custom_target(__check_rocmlir_libs DEPENDS rocmlir_libs.txt)
  add_custom_command(TARGET __check_rocmlir_libs POST_BUILD
                     COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/checkTargets.py ${LIBRARY_NAME}
                                  __rocmlir_targets rocmlir_targets.txt rocmlir_libs.txt)

  add_custom_target(check-lib${LIBRARY_NAME} DEPENDS __check_rocmlir_libs __check_llvm_libs)
  add_dependencies(check-rocmlir check-lib${LIBRARY_NAME})

  unset(__library_name)
  unset(__llvm_targets)
  unset(__rocmlir_targets)
endif()
