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

  string(TOLOWER ${CMAKE_PROJECT_NAME} CMAKE_PROJECT_NAME_LOWER)
  set(PACKAGE_NAME ${CMAKE_PROJECT_NAME_LOWER})

  set(LIBRARY_NAME rockCompiler)

  file(GLOB __llvm_libraries LIST_DIRECTORIES false
          CONFIGURE_DEPENDS ${LLVM_EXTERNAL_LIB_DIR}/*${CMAKE_STATIC_LIBRARY_SUFFIX})

  file(GLOB __rocmlir_libraries LIST_DIRECTORIES false
          CONFIGURE_DEPENDS ${ROCMLIR_LIB_DIR}/*${CMAKE_STATIC_LIBRARY_SUFFIX})

  # Remove the rockCompiler library from the list to not include it recursively.
  list(REMOVE_ITEM __rocmlir_libraries
          ${ROCMLIR_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${LIBRARY_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})

  # Static libraries on Windows have a 4GB size limit. The Debug version of the FAT library
  # exceeds that limit significantly. Therefore, the change is to expose CMake's INTERFACE
  # library instead as an interface to clients. Underneath, on Linux, the single monolithic
  # library is delivered (as it was before). However, we deliver a series of smaller static
  # libraries on Windows due to the size limit mentioned.
  # The client interface has not changed and is OS agnostic. Both MIOpen and MIGraphX do not
  # need to modify their CMake code to find the rocMLIR package and build with it.
  add_library(${LIBRARY_NAME} INTERFACE)

  if (WIN32)
    add_dependencies(${LIBRARY_NAME} MLIRRockThin)

    foreach(__library ${__llvm_libraries})
      file(RELATIVE_PATH __library_file ${LLVM_EXTERNAL_LIB_DIR} ${__library})
      target_link_libraries(${LIBRARY_NAME}
          INTERFACE
              $<BUILD_INTERFACE:${__library}>
              $<INSTALL_INTERFACE:\${_IMPORT_PREFIX}/lib/llvm/${__library_file}>)
    endforeach()

    foreach(__library ${__rocmlir_libraries})
      file(RELATIVE_PATH __library_file ${ROCMLIR_LIB_DIR} ${__library})
      target_link_libraries(${LIBRARY_NAME}
          INTERFACE
              $<BUILD_INTERFACE:${__library}>
              $<INSTALL_INTERFACE:\${_IMPORT_PREFIX}/lib/${__library_file}>)
    endforeach()

    rocm_install(FILES ${__llvm_libraries}
            DESTINATION lib/llvm)

    rocm_install(FILES ${ROCMLIR_LIB_DIR}/MLIRRockThin.lib
            DESTINATION lib)

  else()

    set(__library_name ${CMAKE_STATIC_LIBRARY_PREFIX}${LIBRARY_NAME})
    set(full_output_path ${ROCMLIR_LIB_DIR}/${__library_name}${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(mri_file ${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}.mri)

    # Step one: construct mri file.
    add_custom_command(OUTPUT ${mri_file}
                       COMMAND echo "create ${full_output_path}" > ${mri_file}
                       COMMAND for archive in ${__rocmlir_libraries} ${__llvm_libraries} \; do echo "addlib $$archive" >> ${mri_file} \; done
                       COMMAND echo "save" >> ${mri_file}
                       COMMAND echo "end" >> ${mri_file}
                       DEPENDS MLIRRockThin)

    # Step two: use mri file to generate the fat library.
    add_custom_command(OUTPUT ${full_output_path}
                       COMMAND ${CMAKE_AR} -M < ${mri_file}
                       DEPENDS ${mri_file})

    add_custom_target(__generate_fat_library DEPENDS ${full_output_path})
    add_dependencies(${LIBRARY_NAME} __generate_fat_library)

    target_link_libraries(${LIBRARY_NAME}
        INTERFACE
            $<BUILD_INTERFACE:${full_output_path}>
            $<INSTALL_INTERFACE:\${_IMPORT_PREFIX}/lib/${__library_name}${CMAKE_STATIC_LIBRARY_SUFFIX}>)

    rocm_install(FILES ${full_output_path}
            DESTINATION lib)

    # Backward compatibility to provide `librockCompiler` build target.
    # INTERFACE libraries are not exposed as build targets.
    add_custom_target(${__library_name} ALL DEPENDS ${LIBRARY_NAME})

    unset(mri_file)
    unset(full_output_path)
    unset(__library_name)

  endif()

  unset(__llvm_libraries)
  unset(__rocmlir_libraries)

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
endif()
