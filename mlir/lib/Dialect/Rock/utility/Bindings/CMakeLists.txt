if(NOT WIN32)
    find_package(Python3 COMPONENTS Interpreter Development QUIET)
    if(Python3_FOUND)
        include(${LLVM_EXTERNAL_SRC_DIR}/mlir/cmake/modules/MLIRDetectPythonEnv.cmake)
        mlir_detect_pybind11_install()
        find_package(pybind11 CONFIG QUIET)
    endif()

    if(Python3_FOUND AND pybind11_FOUND)
        pybind11_add_module(amd_arch_db AmdArchDbBindings.cpp)
        target_link_libraries(amd_arch_db PUBLIC MLIRRockUtility)
        set_target_properties(amd_arch_db PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${ROCMLIR_BIN_DIR}")
        message(STATUS "amd_arch_db Python binding will be built in ${ROCMLIR_BIN_DIR}")
        message(STATUS "To use it outside the build bin directory, set: "
                       "export PYTHONPATH=${ROCMLIR_BIN_DIR}:\$PYTHONPATH")
    else()
        message(STATUS "Python3 Development or pybind11 not found; "
                       "skipping amd_arch_db Python binding")
    endif()
endif()
