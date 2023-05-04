get_property(rocmlir_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)
get_property(rocmlir_capi_libs GLOBAL PROPERTY ROCMLIR_PUBLIC_C_API_LIBS)

set(LIBS
  ${rocmlir_dialect_libs}
  ${rocmlir_capi_libs}
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
  find_package(ROCM 0.8 REQUIRED)
  include(ROCMInstallTargets)
  include(ROCMCreatePackage)

  set(ROCMCHECKS_WARN_TOOLCHAIN_VAR OFF)
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

  set(PACKAGE_NAME ${CMAKE_PROJECT_NAME})
  set(LIBRARY_NAME rockCompiler)
  set(COMPONENT_NAME ${CMAKE_STATIC_LIBRARY_PREFIX}${LIBRARY_NAME})

  if (MSVC)
    add_library(${LIBRARY_NAME} INTERFACE)
    set(__rocmlir_libraries
            MLIRCAPIMIGraphX.lib
            MLIRCAPIRegisterRocMLIR.lib
            MLIRCAPIRock.lib
            MLIRGPUToMIGraphX.lib
            MLIRMIGraphX.lib
            MLIRMIGraphXPipeline.lib
            MLIRMIGraphXToTosa.lib
            MLIRRockConv2dGenerator.lib
            MLIRRockOps.lib
            MLIRRockPipeline.lib
            MLIRRockThin.lib
            MLIRRockToGPU.lib
            MLIRRockTransforms.lib
            MLIRRockTuning.lib
            MLIRRockUtility.lib
            MLIRTosaToRock.lib)
    set(__llvm_mlir_libraries
            lldCommon.lib
            lldELF.lib
            LLVMAggressiveInstCombine.lib
            LLVMAMDGPUAsmParser.lib
            LLVMAMDGPUCodeGen.lib
            LLVMAMDGPUDesc.lib
            LLVMAMDGPUDisassembler.lib
            LLVMAMDGPUInfo.lib
            LLVMAMDGPUUtils.lib
            LLVMAnalysis.lib
            LLVMAsmParser.lib
            LLVMAsmPrinter.lib
            LLVMBinaryFormat.lib
            LLVMBitReader.lib
            LLVMBitstreamReader.lib
            LLVMBitWriter.lib
            LLVMCFGuard.lib
            LLVMCodeGen.lib
            LLVMCore.lib
            LLVMCoroutines.lib
            LLVMDebugInfoCodeView.lib
            LLVMDebugInfoDWARF.lib
            LLVMDebugInfoMSF.lib
            LLVMDebugInfoPDB.lib
            LLVMDemangle.lib
            LLVMExtensions.lib
            LLVMFrontendOpenMP.lib
            LLVMGlobalISel.lib
            LLVMInstCombine.lib
            LLVMInstrumentation.lib
            LLVMipo.lib
            LLVMIRPrinter.lib
            LLVMIRReader.lib
            LLVMLinker.lib
            LLVMLTO.lib
            LLVMMC.lib
            LLVMMCDisassembler.lib
            LLVMMCParser.lib
            LLVMMIRParser.lib
            LLVMObjCARCOpts.lib
            LLVMObject.lib
            LLVMOption.lib
            LLVMPasses.lib
            LLVMProfileData.lib
            LLVMRemarks.lib
            LLVMScalarOpts.lib
            LLVMSelectionDAG.lib
            LLVMSupport.lib
            LLVMSymbolize.lib
            LLVMTableGen.lib
            LLVMTableGenGlobalISel.lib
            LLVMTarget.lib
            LLVMTargetParser.lib
            LLVMTextAPI.lib
            LLVMTransformUtils.lib
            LLVMVectorize.lib
            LLVMX86AsmParser.lib
            LLVMX86CodeGen.lib
            LLVMX86Desc.lib
            LLVMX86Disassembler.lib
            LLVMX86Info.lib
            MLIRAffineAnalysis.lib
            MLIRAffineDialect.lib
            MLIRAffineToStandard.lib
            MLIRAffineTransforms.lib
            MLIRAffineUtils.lib
            MLIRAMDGPUDialect.lib
            MLIRAMDGPUToROCDL.lib
            MLIRAMXDialect.lib
            MLIRAMXToLLVMIRTranslation.lib
            MLIRAMXTransforms.lib
            MLIRAnalysis.lib
            MLIRArithAttrToLLVMConversion.lib
            MLIRArithDialect.lib
            MLIRArithToLLVM.lib
            MLIRArithTransforms.lib
            MLIRArithUtils.lib
            MLIRArmNeonDialect.lib
            MLIRArmNeonToLLVMIRTranslation.lib
            MLIRArmSVEDialect.lib
            MLIRArmSVEToLLVMIRTranslation.lib
            MLIRArmSVETransforms.lib
            MLIRAsmParser.lib
            MLIRAsyncDialect.lib
            MLIRAsyncToLLVM.lib
            MLIRBufferizationDialect.lib
            MLIRBufferizationTransformOps.lib
            MLIRBufferizationTransforms.lib
            MLIRBytecodeReader.lib
            MLIRBytecodeWriter.lib
            MLIRCallInterfaces.lib
            MLIRCAPIIR.lib
            MLIRCastInterfaces.lib
            MLIRComplexDialect.lib
            MLIRControlFlowDialect.lib
            MLIRControlFlowInterfaces.lib
            MLIRControlFlowToLLVM.lib
            MLIRCopyOpInterface.lib
            MLIRDataLayoutInterfaces.lib
            MLIRDestinationStyleOpInterface.lib
            MLIRDialect.lib
            MLIRDialectUtils.lib
            MLIRDLTIDialect.lib
            MLIRExecutionEngineUtils.lib
            MLIRFuncDialect.lib
            MLIRFuncToLLVM.lib
            MLIRFuncTransforms.lib
            MLIRGPUOps.lib
            MLIRGPUToGPURuntimeTransforms.lib
            MLIRGPUToROCDLTransforms.lib
            MLIRGPUTransforms.lib
            MLIRInferIntRangeInterface.lib
            MLIRInferTypeOpInterface.lib
            MLIRIR.lib
            MLIRLinalgAnalysis.lib
            MLIRLinalgDialect.lib
            MLIRLinalgTransforms.lib
            MLIRLinalgUtils.lib
            MLIRLLVMCommonConversion.lib
            MLIRLLVMDialect.lib
            MLIRLLVMIRTransforms.lib
            MLIRLLVMToLLVMIRTranslation.lib
            MLIRLoopLikeInterface.lib
            MLIRMaskableOpInterface.lib
            MLIRMaskingOpInterface.lib
            MLIRMathDialect.lib
            MLIRMathToLLVM.lib
            MLIRMemRefDialect.lib
            MLIRMemRefToLLVM.lib
            MLIRMemRefTransforms.lib
            MLIRNVVMDialect.lib
            MLIRNVVMToLLVMIRTranslation.lib
            MLIROpenACCDialect.lib
            MLIROpenACCToLLVM.lib
            MLIROpenACCToLLVMIRTranslation.lib
            MLIROpenMPDialect.lib
            MLIROpenMPToLLVMIRTranslation.lib
            MLIRParallelCombiningOpInterface.lib
            MLIRParser.lib
            MLIRPass.lib
            MLIRPDLDialect.lib
            MLIRPDLInterpDialect.lib
            MLIRPDLToPDLInterp.lib
            MLIRPresburger.lib
            MLIRQuantDialect.lib
            MLIRQuantUtils.lib
            MLIRReconcileUnrealizedCasts.lib
            MLIRRewrite.lib
            MLIRROCDLDialect.lib
            MLIRROCDLToLLVMIRTranslation.lib
            MLIRRuntimeVerifiableOpInterface.lib
            MLIRSCFDialect.lib
            MLIRSCFToControlFlow.lib
            MLIRSCFTransforms.lib
            MLIRSCFUtils.lib
            MLIRShapedOpInterfaces.lib
            MLIRSideEffectInterfaces.lib
            MLIRSparseTensorDialect.lib
            MLIRSupport.lib
            MLIRSupportIndentedOstream.lib
            MLIRTableGen.lib
            MLIRTargetLLVMIRExport.lib
            MLIRTblgenLib.lib
            MLIRTensorDialect.lib
            MLIRTensorInferTypeOpInterfaceImpl.lib
            MLIRTensorTilingInterfaceImpl.lib
            MLIRTensorToLinalg.lib
            MLIRTensorTransforms.lib
            MLIRTensorUtils.lib
            MLIRTilingInterface.lib
            MLIRToLLVMIRTranslationRegistration.lib
            MLIRTosaDialect.lib
            MLIRTosaToArith.lib
            MLIRTosaToLinalg.lib
            MLIRTosaToSCF.lib
            MLIRTosaToTensor.lib
            MLIRTosaTransforms.lib
            MLIRTransformDialect.lib
            MLIRTransformDialectUtils.lib
            MLIRTransforms.lib
            MLIRTransformUtils.lib
            MLIRTranslateLib.lib
            MLIRVectorDialect.lib
            MLIRVectorInterfaces.lib
            MLIRVectorToLLVM.lib
            MLIRVectorToSCF.lib
            MLIRVectorTransforms.lib
            MLIRVectorUtils.lib
            MLIRViewLikeInterface.lib
            MLIRX86VectorDialect.lib
            MLIRX86VectorToLLVMIRTranslation.lib
            MLIRX86VectorTransforms.lib
            MLIRXModel.lib
            MLIRXModelTransforms.lib)

    foreach(__library ${__llvm_mlir_libraries})
      list(APPEND __install_libraries ${LLVM_EXTERNAL_LIB_DIR}/${__library})
      target_link_libraries(${LIBRARY_NAME}
          INTERFACE
              $<BUILD_INTERFACE:${LLVM_EXTERNAL_LIB_DIR}/${__library}>
              $<INSTALL_INTERFACE:\${_IMPORT_PREFIX}/lib/${PACKAGE_NAME}/llvm/${__library}>)
    endforeach()

    foreach(__library ${__rocmlir_libraries})
      target_link_libraries(${LIBRARY_NAME}
          INTERFACE
              $<BUILD_INTERFACE:${ROCMLIR_LIB_DIR}/${__library}>
              $<INSTALL_INTERFACE:\${_IMPORT_PREFIX}/lib/${__library}>)
    endforeach()

    install(TARGETS MLIRRockThin
            DESTINATION lib
            COMPONENT ${COMPONENT_NAME})

    install(FILES ${__install_libraries}
         DESTINATION lib/${PACKAGE_NAME}/llvm
         COMPONENT ${COMPONENT_NAME})
  else()

    set(output_archive_dummy_file ${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}.dummy.cpp)
    file(TOUCH ${output_archive_dummy_file})
    set(full_output_path ${ROCMLIR_LIB_DIR}/lib${LIBRARY_NAME}.a)
    set(mri_file ${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}.mri)

    # Step one: construct mri file.
    add_custom_command(OUTPUT ${LIBRARY_NAME}.mri
                       COMMAND echo "create ${full_output_path}" >> ${LIBRARY_NAME}.mri
                       COMMAND for archive in ${ROCMLIR_LIB_DIR}/*.a ${LLVM_EXTERNAL_LIB_DIR}/*.a\;
                       do echo "addlib $$archive" >> ${LIBRARY_NAME}.mri \; done
                       COMMAND echo "save" >> ${LIBRARY_NAME}.mri
                       COMMAND echo "end" >> ${LIBRARY_NAME}.mri)

    # Step two: use mri file to generate the fat library.
    llvm_add_library(${LIBRARY_NAME}
      PARTIAL_SOURCES_INTENDED
      STATIC ${output_archive_dummy_file})
    add_custom_command(TARGET ${LIBRARY_NAME}
                       POST_BUILD
                       COMMAND ${CMAKE_AR} -M < ${mri_file})
    add_dependencies(${LIBRARY_NAME} ${LIBRARY_NAME}.mri)
  endif()

  add_dependencies(${LIBRARY_NAME} MLIRRockThin)

  rocm_package_setup_component(${LIBRARY_NAME}
        LIBRARY_NAME lib${LIBRARY_NAME}
        PACKAGE_NAME ${PACKAGE_NAME})

  rocm_install(FILES LICENSE
    DESTINATION share/doc/${PACKAGE_NAME}
    COMPONENT ${COMPONENT_NAME})

  # Install Miir.h to ${CMAKE_INSTALL_PREFIX}/include/${PACKAGE_NAME}/
  # as part of component libroCompiler
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
    # so that when comsumer libraries (Rock) link in ${COMPONENT_NAME}, ${INCLUDE_DIR} is
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
