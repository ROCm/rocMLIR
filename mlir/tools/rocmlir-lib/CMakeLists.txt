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
  find_package(ROCM 0.8 REQUIRED PATHS /opt/rocm)
  include(ROCMInstallTargets)
  include(ROCMCreatePackage)

  set(ROCMCHECKS_WARN_TOOLCHAIN_VAR OFF)
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

  set(PACKAGE_NAME ${CMAKE_PROJECT_NAME})
  set(LIBRARY_NAME rockCompiler)
  set(COMPONENT_NAME ${LIBRARY_NAME})
  if (NOT MSVC)
    set(COMPONENT_NAME lib${COMPONENT_NAME})
  endif()

  function(combine_archives output_archive)
    set(output_archive_dummy_file ${CMAKE_CURRENT_BINARY_DIR}/${output_archive}.dummy.cpp)
    file(TOUCH ${output_archive_dummy_file})
    if (MSVC)
      llvm_add_library(${output_archive}
        PARTIAL_SOURCES_INTENDED
        STATIC ${output_archive_dummy_file})
      add_custom_command(
        TARGET ${output_archive}
        POST_BUILD
        COMMAND LIB.EXE /OUT:${output_archive}.lib
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
              MLIRTosaToRock.lib
              ${LLVM_EXTERNAL_LIB_DIR}/lldCommon.lib
              ${LLVM_EXTERNAL_LIB_DIR}/lldELF.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMAggressiveInstCombine.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMAMDGPUAsmParser.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMAMDGPUCodeGen.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMAMDGPUDesc.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMAMDGPUDisassembler.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMAMDGPUInfo.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMAMDGPUUtils.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMAnalysis.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMAsmParser.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMAsmPrinter.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMBinaryFormat.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMBitReader.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMBitstreamReader.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMBitWriter.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMCFGuard.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMCodeGen.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMCore.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMCoroutines.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMDebugInfoCodeView.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMDebugInfoDWARF.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMDebugInfoMSF.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMDebugInfoPDB.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMDemangle.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMExtensions.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMFrontendOpenMP.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMGlobalISel.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMInstCombine.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMInstrumentation.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMipo.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMIRReader.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMLinker.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMLTO.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMMC.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMMCDisassembler.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMMCParser.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMMIRParser.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMObjCARCOpts.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMObject.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMOption.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMPasses.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMProfileData.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMRemarks.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMScalarOpts.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMSelectionDAG.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMSupport.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMSymbolize.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMTableGen.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMTableGenGlobalISel.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMTarget.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMTextAPI.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMTransformUtils.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMVectorize.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMX86AsmParser.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMX86CodeGen.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMX86Desc.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMX86Disassembler.lib
              ${LLVM_EXTERNAL_LIB_DIR}/LLVMX86Info.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRAffineAnalysis.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRAffineDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRAffineToStandard.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRAffineTransforms.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRAffineUtils.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRAMDGPUDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRAMDGPUToROCDL.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRAMXDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRAMXToLLVMIRTranslation.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRAMXTransforms.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRAnalysis.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRArithmeticDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRArithmeticToLLVM.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRArithmeticTransforms.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRArithmeticUtils.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRArmNeonDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRArmNeonToLLVMIRTranslation.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRArmSVEDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRArmSVEToLLVMIRTranslation.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRArmSVETransforms.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRAsmParser.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRAsyncDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRAsyncToLLVM.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRBufferizationDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRBufferizationTransformOps.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRBufferizationTransforms.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRBytecodeReader.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRBytecodeWriter.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRCallInterfaces.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRCAPIIR.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRCastInterfaces.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRComplexDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRControlFlowDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRControlFlowInterfaces.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRControlFlowToLLVM.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRCopyOpInterface.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRDataLayoutInterfaces.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRDialectUtils.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRDLTIDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRExecutionEngineUtils.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRFuncDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRFuncToLLVM.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRFuncTransforms.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRGPUOps.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRGPUToGPURuntimeTransforms.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRGPUToROCDLTransforms.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRGPUTransforms.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRInferIntRangeInterface.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRInferTypeOpInterface.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRIR.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRLinalgAnalysis.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRLinalgDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRLinalgTransforms.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRLinalgUtils.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRLLVMCommonConversion.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRLLVMDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRLLVMIRTransforms.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRLLVMToLLVMIRTranslation.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRLoopLikeInterface.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRMathDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRMathToLLVM.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRMemRefDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRMemRefToLLVM.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRMemRefTransforms.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRNVVMDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRNVVMToLLVMIRTranslation.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIROpenACCDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIROpenACCToLLVM.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIROpenACCToLLVMIRTranslation.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIROpenMPDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIROpenMPToLLVMIRTranslation.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRParallelCombiningOpInterface.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRParser.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRPass.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRPDLDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRPDLInterpDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRPDLToPDLInterp.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRPresburger.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRQuantDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRQuantUtils.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRReconcileUnrealizedCasts.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRRewrite.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRROCDLDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRROCDLToLLVMIRTranslation.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRSCFDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRSCFToControlFlow.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRSCFTransforms.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRSCFUtils.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRSideEffectInterfaces.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRSparseTensorDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRSupport.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRSupportIndentedOstream.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTableGen.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTargetLLVMIRExport.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTblgenLib.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTensorDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTensorInferTypeOpInterfaceImpl.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTensorTilingInterfaceImpl.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTensorToLinalg.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTensorTransforms.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTensorUtils.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTilingInterface.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRToLLVMIRTranslationRegistration.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTosaDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTosaToArith.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTosaToLinalg.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTosaToSCF.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTosaToTensor.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTosaTransforms.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTransformDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTransforms.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTransformUtils.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRTranslateLib.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRVectorDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRVectorInterfaces.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRVectorToLLVM.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRVectorToSCF.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRVectorTransforms.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRVectorUtils.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRViewLikeInterface.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRX86VectorDialect.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRX86VectorToLLVMIRTranslation.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRX86VectorTransforms.lib
              ${LLVM_EXTERNAL_LIB_DIR}/MLIRXModel.lib
        WORKING_DIRECTORY ${ROCMLIR_LIB_DIR})
    else()
      set(full_output_path ${ROCMLIR_LIB_DIR}/lib${output_archive}.a)
      set(mri_file ${CMAKE_CURRENT_BINARY_DIR}/${output_archive}.mri)

      # Step one: construct mri file.
      add_custom_command(OUTPUT ${output_archive}.mri
                         COMMAND echo "create ${full_output_path}" >> ${output_archive}.mri
                         COMMAND for archive in ${ROCMLIR_LIB_DIR}/*.a ${LLVM_EXTERNAL_LIB_DIR}/*.a\;
                         do echo "addlib $$archive" >> ${output_archive}.mri \; done
                         COMMAND echo "save" >> ${output_archive}.mri
                         COMMAND echo "end" >> ${output_archive}.mri)

      # Step two: use mri file to generate the fat library.
      llvm_add_library(${output_archive}
        PARTIAL_SOURCES_INTENDED
        STATIC ${output_archive_dummy_file})
      add_custom_command(TARGET ${output_archive}
                         POST_BUILD
                         COMMAND ${CMAKE_AR} -M < ${mri_file})
      add_dependencies(${output_archive} ${output_archive}.mri)
    endif()
    add_dependencies(${output_archive} MLIRRockThin)
  endfunction()

  combine_archives(${LIBRARY_NAME})

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
      "${CMAKE_CURRENT_BINARY_DIR}/${pkg_version_file}"
      VERSION ${CMAKE_PROJECT_VERSION}
      COMPATIBILITY AnyNewerVersion
    )

    configure_package_config_file(cmake/${pkg_config_file}.in
      "${CMAKE_CURRENT_BINARY_DIR}/${pkg_config_file}"
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
      "${CMAKE_CURRENT_BINARY_DIR}/${pkg_config_file}"
      "${CMAKE_CURRENT_BINARY_DIR}/${pkg_version_file}"
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
