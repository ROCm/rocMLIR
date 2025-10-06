if(NOT WIN32)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    include(${LLVM_EXTERNAL_SRC_DIR}/mlir/cmake/modules/MLIRDetectPythonEnv.cmake)
    mlir_detect_pybind11_install()

    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(amd_arch_db AmdArchDbBindings.cpp)
    target_link_libraries(amd_arch_db PUBLIC MLIRRockUtility)
endif()
