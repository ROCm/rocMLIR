get_property(rocmlir_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)
get_property(rocmlir_capi_libs GLOBAL PROPERTY ROCMLIR_PUBLIC_C_API_LIBS)

set(LIBS
  ${rocmlir_dialect_libs}
  ${rocmlir_capi_libs}
)

if(NOT WIN32)
    set(CMAKE_BUILD_RPATH ${CMAKE_BUILD_DIR}/external/llvm-project/llvm/lib)
endif()

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
  find_package(ROCM 0.8 REQUIRED)
  include(ROCMInstallTargets)
  include(ROCMCreatePackage)

  set(ROCMCHECKS_WARN_TOOLCHAIN_VAR OFF)
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

  set(PACKAGE_NAME ${CMAKE_PROJECT_NAME})
  set(LIBRARY_NAME rockCompiler)
  set(COMPONENT_NAME ${LIBRARY_NAME})

  set(__rocmlir_names
          GpuModuleToRocdlirTranslation
          MLIRCAPIMIGraphX
          MLIRCAPIRegisterRocMLIR
          MLIRCAPIRock
          MLIRGPUToMIGraphX
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
          MLIRTosaToRock)

  set(__llvm_names
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
          LLVMCFGuard
          LLVMCodeGen
          LLVMCore
          LLVMCoroutines
          LLVMDebugInfoCodeView
          LLVMDebugInfoDWARF
          LLVMDebugInfoMSF
          LLVMDebugInfoPDB
          LLVMDemangle
          LLVMExecutionEngine
          LLVMExtensions
          LLVMFrontendOpenMP
          LLVMGlobalISel
          LLVMInstCombine
          LLVMInstrumentation
          LLVMipo
          LLVMIRPrinter
          LLVMIRReader
          LLVMJITLink
          LLVMLinker
          LLVMLTO
          LLVMMC
          LLVMMCDisassembler
          LLVMMCParser
          LLVMMIRParser
          LLVMObjCARCOpts
          LLVMObject
          LLVMOption
          LLVMOrcJIT
          LLVMOrcShared
          LLVMOrcTargetProcess
          LLVMPasses
          LLVMProfileData
          LLVMRemarks
          LLVMRuntimeDyld
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
          LLVMWindowsDriver
          LLVMX86AsmParser
          LLVMX86CodeGen
          LLVMX86Desc
          LLVMX86Disassembler
          LLVMX86Info
          MLIRAffineAnalysis
          MLIRAffineDialect
          MLIRAffineToStandard
          MLIRAffineTransforms
          MLIRAffineUtils
          MLIRAMDGPUDialect
          MLIRAMDGPUToROCDL
          MLIRAMDGPUUtils
          MLIRAMXDialect
          MLIRAMXToLLVMIRTranslation
          MLIRAMXTransforms
          MLIRAnalysis
          MLIRArithAttrToLLVMConversion
          MLIRArithDialect
          MLIRArithToLLVM
          MLIRArithTransforms
          MLIRArithUtils
          MLIRArmNeonDialect
          MLIRArmNeonToLLVMIRTranslation
          MLIRArmSVEDialect
          MLIRArmSVEToLLVMIRTranslation
          MLIRArmSVETransforms
          MLIRAsmParser
          MLIRAsyncDialect
          MLIRAsyncToLLVM
          MLIRAsyncTransforms
          MLIRBufferizationDialect
          MLIRBufferizationTransformOps
          MLIRBufferizationTransforms
          MLIRBytecodeReader
          MLIRBytecodeWriter
          MLIRCallInterfaces
          MLIRCAPIGPU
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
          MLIREmitCDialect
          MLIRExecutionEngine
          MLIRExecutionEngineUtils
          MLIRFromLLVMIRTranslationRegistration
          MLIRFuncDialect
          MLIRFuncToLLVM
          MLIRFuncTransforms
          MLIRGPUOps
          MLIRGPUToGPURuntimeTransforms
          MLIRGPUToROCDLTransforms
          MLIRGPUTransforms
          MLIRInferIntRangeInterface
          MLIRInferTypeOpInterface
          MLIRIR
          MLIRLinalgAnalysis
          MLIRLinalgDialect
          MLIRLinalgToLLVM
          MLIRLinalgTransforms
          MLIRLinalgUtils
          MLIRLLVMCommonConversion
          MLIRLLVMDialect
          MLIRLLVMIRToLLVMTranslation
          MLIRLLVMIRTransforms
          MLIRLLVMToLLVMIRTranslation
          MLIRLoopLikeInterface
          MLIRLspServerLib
          MLIRLspServerSupportLib
          MLIRMaskableOpInterface
          MLIRMaskingOpInterface
          MLIRMathDialect
          MLIRMathToLibm
          MLIRMathToLLVM
          MLIRMemRefDialect
          MLIRMemRefToLLVM
          MLIRMemRefTransforms
          MLIRNVVMDialect
          MLIRNVVMToLLVMIRTranslation
          MLIROpenACCDialect
          MLIROpenACCToLLVM
          MLIROpenACCToLLVMIRTranslation
          MLIROpenMPDialect
          MLIROpenMPToLLVMIRTranslation
          MLIROptLib
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
          MLIRSPIRVBinaryUtils
          MLIRSPIRVDeserialization
          MLIRSPIRVDialect
          MLIRSPIRVSerialization
          MLIRSPIRVTranslateRegistration
          MLIRSupport
          MLIRSupportIndentedOstream
          MLIRTableGen
          MLIRTargetCpp
          MLIRTargetLLVMIRExport
          MLIRTargetLLVMIRImport
          MLIRTblgenLib
          MLIRTensorDialect
          MLIRTensorInferTypeOpInterfaceImpl
          MLIRTensorTilingInterfaceImpl
          MLIRTensorToLinalg
          MLIRTensorTransforms
          MLIRTensorUtils
          MLIRTilingInterface
          MLIRToLLVMIRTranslationRegistration
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
          MLIRVectorDialect
          MLIRVectorInterfaces
          MLIRVectorToLLVM
          MLIRVectorToSCF
          MLIRVectorTransforms
          MLIRVectorUtils
          MLIRViewLikeInterface
          MLIRX86VectorDialect
          MLIRX86VectorToLLVMIRTranslation
          MLIRX86VectorTransforms
          MLIRXModel
          MLIRXModelPipeline
          MLIRXModelToGPU
          MLIRXModelTransforms)

  foreach(__name ${__llvm_names})
      list(APPEND __llvm_libraries ${CMAKE_STATIC_LIBRARY_PREFIX}${__name}${CMAKE_STATIC_LIBRARY_SUFFIX})
  endforeach()

  foreach(__name ${__rocmlir_names})
      list(APPEND __rocmlir_libraries ${CMAKE_STATIC_LIBRARY_PREFIX}${__name}${CMAKE_STATIC_LIBRARY_SUFFIX})
  endforeach()

  if (MSVC)
    add_library(${LIBRARY_NAME} INTERFACE)

    foreach(__library ${__llvm_libraries})
      list(APPEND __llvm_install ${LLVM_EXTERNAL_LIB_DIR}/${__library})
      target_link_libraries(${LIBRARY_NAME}
          INTERFACE
              $<BUILD_INTERFACE:${LLVM_EXTERNAL_LIB_DIR}/${__library}>
              $<INSTALL_INTERFACE:\${_IMPORT_PREFIX}/lib/${PACKAGE_NAME}/llvm/${__library}>)
    endforeach()

    foreach(__library ${__rocmlir_libraries})
      list(APPEND __rocmlir_install ${ROCMLIR_LIB_DIR}/${__library})
      target_link_libraries(${LIBRARY_NAME}
          INTERFACE
              $<BUILD_INTERFACE:${ROCMLIR_LIB_DIR}/${__library}>
              $<INSTALL_INTERFACE:\${_IMPORT_PREFIX}/lib/${PACKAGE_NAME}/${__library}>)
    endforeach()

    add_dependencies(${LIBRARY_NAME} ${__rocmlir_names})

    install(FILES ${__rocmlir_install}
         DESTINATION lib/${PACKAGE_NAME}
         COMPONENT ${COMPONENT_NAME})

    install(FILES ${__llvm_install}
         DESTINATION lib/${PACKAGE_NAME}/llvm
         COMPONENT ${COMPONENT_NAME})

    unset(__llvm_libraries)
    unset(__rocmlir_libraries)

  else()

    set(output_archive_dummy_file ${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}.dummy.cpp)
    file(TOUCH ${output_archive_dummy_file})
    set(full_output_path ${ROCMLIR_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${LIBRARY_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(mri_file ${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}.mri)

    list(TRANSFORM __llvm_libraries PREPEND ${LLVM_EXTERNAL_LIB_DIR}/)
    list(TRANSFORM __rocmlir_libraries PREPEND ${ROCMLIR_LIB_DIR}/)

    # Step one: construct mri file.
    add_custom_command(OUTPUT ${mri_file}
                       COMMAND echo "create ${full_output_path}" >> ${mri_file}
                       COMMAND for archive in ${__rocmlir_libraries} ${__llvm_libraries} \; do echo "addlib $$archive" >> ${mri_file} \; done
                       COMMAND echo "save" >> ${mri_file}
                       COMMAND echo "end" >> ${mri_file}
                       DEPENDS ${__rocmlir_names})

    # Step two: use mri file to generate the fat library.
    llvm_add_library(${LIBRARY_NAME}
      PARTIAL_SOURCES_INTENDED
      STATIC ${output_archive_dummy_file})

    target_sources(${LIBRARY_NAME} PRIVATE ${mri_file})

    add_custom_command(TARGET ${LIBRARY_NAME}
                       POST_BUILD
                       COMMAND ${CMAKE_AR} -M < ${mri_file})

    unset(mri_file)
    unset(output_archive_dummy_file)
    unset(full_output_path)

  endif()

  rocm_package_setup_component(${LIBRARY_NAME}
        LIBRARY_NAME ${LIBRARY_NAME}
        PACKAGE_NAME ${PACKAGE_NAME})

  rocm_install(FILES LICENSE
    DESTINATION share/doc/${PACKAGE_NAME}
    COMPONENT ${COMPONENT_NAME})

  # Install Miir.h to ${CMAKE_INSTALL_PREFIX}/include/${PACKAGE_NAME}/
  # as part of component rockCompiler
  rocm_install(FILES Miir.h
    DESTINATION include/${PACKAGE_NAME}
    COMPONENT ${COMPONENT_NAME})

  # Install a target, i.e. ${package_name}::${component_name}, as part of
  # ${component_name} and export it to be searchable by find_package().
  # The include directories are propagated to the downstream targets
  # that links to ${package_name}::${component_name}
  # Be default, include/${package_name} is added. If extra include directories
  # are needed, add them using the keyword EXTRA_INCLUDES.
  #
  # As an example, when CMAKE_INSTALL_PREFIX is set to /a/dummy/ and
  # CMAKE_BUILD_TYPE is set to CONFIG
  # The following files is generated and installed as a result of
  # cmake --install . --component ${component_name} --prefix /a/dummy
  # /a/dummy/lib/${package_name}/${component_name}.a
  # /a/dummy/lib/cmake/${package_name}/${package_name}Targets.cmake
  #   This file is autogenerated by install(EXPORT).
  #   It creates the imported library as a target that can be used by
  #   target_link_libraries()
  #   It also sets the INTERFACE_INCLUDE_DIRECTORIES property of the target
  # /a/dummy/lib/cmake/${package_name}/${package_name}Targets-CONFIG.cmake
  #   This file is autogenerated by install(EXPORT).
  #   It sets the location of the library as a property of the imported target
  # /a/dummy/lib/cmake/${package_name}/${package_name}Config.cmake
  #   This file is generated by configure_package_config_file and is searched
  #   and loaded by find_package(). It includes ${package_name}Targets.cmake.
  # /a/dummy/lib/cmake/${package_name}/${package_name}ConfigVersion.cmake
  #   This file is autogenerated by write_basic_package_version_file and is used by
  #   find_package to implement a package version matching mechanism.
  #
  # The only input to this exporting process is cmake/${package_name}Config.cmake.in,
  # which is used to auto generate ${package_name}Config.cmake
  function(export_target package_name lib_name component_name)
    cmake_parse_arguments(ARG "" "" "EXTRA_INCLUDES" ${ARGN})
    set(export_set ${package_name}Targets)
    set(pkg_config_file ${package_name}Config.cmake)
    set(pkg_version_file ${package_name}ConfigVersion.cmake)
    set(INCLUDE_DIR include/${package_name}/)
    set(LIB_CMAKE_DIR lib/cmake/${package_name}/)

    # Install ${component_name} to ${CMAKE_INSTALL_PREFIX}/lib/${package_name}/
    # and group it into the ${export-set}
    # The INTERFACE_INCLUDE_DIRECTORIES will be set as the property of the exported library
    # so that when consumer libraries (Rock) link in ${COMPONENT_NAME}, ${INCLUDE_DIR} is
    # automatically propagated to the consumer library's include_dirs
    set(EXTRA_INCLUDES "")
    if(ARG_EXTRA_INCLUDES)
      set(EXTRA_INCLUDES ${ARG_EXTRA_INCLUDES})
    endif()
    # Call install() instead of rocm_install() here in order to install
    # the package in a specific direcotry. 
    install(TARGETS ${lib_name}
      EXPORT ${export_set}
      ARCHIVE DESTINATION lib/${package_name}
      COMPONENT ${component_name}
      INCLUDES DESTINATION ${INCLUDE_DIR} ${EXTRA_INCLUDES}
    )

    # Generate package config and version file
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
      ${CMAKE_CURRENT_BINARY_DIR}/${pkg_version_file}
      VERSION ${CMAKE_PROJECT_VERSION}
      COMPATIBILITY AnyNewerVersion
    )

    configure_package_config_file(cmake/${pkg_config_file}.in
      ${CMAKE_CURRENT_BINARY_DIR}/${pkg_config_file}
      INSTALL_DESTINATION lib/cmake/${package_name}
      PATH_VARS LIB_CMAKE_DIR
      NO_SET_AND_CHECK_MACRO
      NO_CHECK_REQUIRED_COMPONENTS_MACRO
    )

    # Generate Targets.cmake file for the export-set to contain imported targets
    rocm_install(EXPORT ${export_set}
      NAMESPACE ${package_name}::
      DESTINATION ${LIB_CMAKE_DIR}
      COMPONENT ${component_name}
    )

    # Install package configuration and version files
    rocm_install(
      FILES
      ${CMAKE_CURRENT_BINARY_DIR}/${pkg_config_file}
      ${CMAKE_CURRENT_BINARY_DIR}/${pkg_version_file}
      DESTINATION ${LIB_CMAKE_DIR}
      COMPONENT ${component_name}
    )

  endfunction()

  set(extra_include_dir "")

  if ((NOT LLVM_INSTALL_TOOLCHAIN_ONLY) AND EXPORT_ALL_HEADERS)
  rocm_package_setup_component(${LIBRARY_NAME}-headers
        LIBRARY_NAME ${LIBRARY_NAME}-headers
        PACKAGE_NAME ${PACKAGE_NAME}
  )
    rocm_install(DIRECTORY
      ${PROJECT_SOURCE_DIR}/mlir/include/mlir
      ${PROJECT_SOURCE_DIR}/mlir/include/mlir-c
      DESTINATION include/${PACKAGE_NAME}
      COMPONENT ${LIBRARY_NAME}-headers
      FILES_MATCHING
      PATTERN "*.def"
      PATTERN "*.h"
      PATTERN "*.inc"
      PATTERN "*.td"
      PATTERN "LICENSE.TXT"
    )

    rocm_install(DIRECTORY ${MLIR_INCLUDE_DIRS}
      DESTINATION include/${PACKAGE_NAME}/external
      COMPONENT ${LIBRARY_NAME}-headers
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
      COMPONENT ${LIBRARY_NAME}-headers
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

  # Install ${COMPONENT_NAME}.a as part of component lib${LIBRARY_NAME}
  # and export it to be searchable by find_package()
  export_target(${PACKAGE_NAME} # package name
    ${LIBRARY_NAME}             # library name
    ${COMPONENT_NAME}           # component name
    EXTRA_INCLUDES              # extra include directories
    ${extra_include_dir})


  rocm_create_package(
    NAME ${PACKAGE_NAME}
    DESCRIPTION "MLIR packages"
    MAINTAINER "rocMLIR Dev Team dl.dl-mlir@amd.com")
endif()
