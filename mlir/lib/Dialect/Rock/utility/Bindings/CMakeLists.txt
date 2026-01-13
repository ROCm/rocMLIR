# Python bindings for AMD arch database
# TODO: Re-enable once Python bindings are needed with Triton backend
if(FALSE AND NOT WIN32)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    # Use MLIR's Python detection from the found MLIR installation
    include(MLIRDetectPythonEnv OPTIONAL)
    if(COMMAND mlir_detect_pybind11_install)
    mlir_detect_pybind11_install()
    endif()

    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(amd_arch_db AmdArchDbBindings.cpp)
    target_link_libraries(amd_arch_db PUBLIC MLIRRockUtility)
endif()
