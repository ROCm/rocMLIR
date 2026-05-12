if(NOT WIN32)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    # Detects a pybind11 package installed in the current python environment
    # and sets variables to allow it to be found. This allows pybind11 to be
    # installed via pip, which typically yields a much more recent version than
    # the OS install, which will be available otherwise.
    if(pybind11_DIR)
        message(STATUS "Using explicit pybind11 cmake directory: ${pybind11_DIR} (-Dpybind11_DIR to change)")
    else()
        message(STATUS "Checking for pybind11 in python path...")
        execute_process(
            COMMAND "${Python3_EXECUTABLE}"
            -c "import pybind11;print(pybind11.get_cmake_dir(), end='')"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE STATUS
            OUTPUT_VARIABLE PACKAGE_DIR
            ERROR_QUIET)
        if(NOT STATUS EQUAL "0")
            message(STATUS "not found (install via 'pip install pybind11' or set pybind11_DIR)")
        endif()
        message(STATUS "found (${PACKAGE_DIR})")
        set(pybind11_DIR "${PACKAGE_DIR}")
    endif()

    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(amd_arch_db AmdArchDbBindings.cpp)
    target_link_libraries(amd_arch_db PUBLIC MLIRRockUtility)
    if(ROCMLIR_ENABLE_NATIVE_ARCH)
        target_compile_definitions(amd_arch_db PRIVATE ROCMLIR_ENABLE_NATIVE_ARCH=1)
    endif()
    set_target_properties(amd_arch_db PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${ROCMLIR_BIN_DIR}")
    message(VERBOSE "amd_arch_db Python binding will be built in ${ROCMLIR_BIN_DIR}")
    message(VERBOSE "To use it outside the build bin directory, set: "
                    "export PYTHONPATH=${ROCMLIR_BIN_DIR}:\$PYTHONPATH")
endif()
