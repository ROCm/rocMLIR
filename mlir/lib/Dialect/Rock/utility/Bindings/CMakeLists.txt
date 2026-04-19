if(NOT WIN32)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    include(${LLVM_EXTERNAL_SRC_DIR}/mlir/cmake/modules/MLIRDetectPythonEnv.cmake)
    mlir_detect_pybind11_install()

    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(amd_arch_db AmdArchDbBindings.cpp)
    target_link_libraries(amd_arch_db PUBLIC MLIRRockUtility)
    set_target_properties(amd_arch_db PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${ROCMLIR_BIN_DIR}")

    # Make amd_arch_db importable from any Python script by adding the
    # build bin dir to Python's module search path via a .pth file.
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
                "import site; print(site.getusersitepackages())"
        OUTPUT_VARIABLE _PYTHON_USER_SITE
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    file(MAKE_DIRECTORY "${_PYTHON_USER_SITE}")
    file(WRITE "${_PYTHON_USER_SITE}/rocmlir-amd-arch-db.pth"
         "${ROCMLIR_BIN_DIR}\n")
endif()
