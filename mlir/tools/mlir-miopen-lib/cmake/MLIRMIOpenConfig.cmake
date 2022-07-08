# Auto compute the include dir for the header file
get_filename_component(SELF_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
include("${SELF_DIR}/MLIRMIOpenTargets.cmake")
get_filename_component(MLIRMIOpen_INCLUDE_DIR "${SELF_DIR}/../../../include/@PACKAGE_DIR@" ABSOLUTE)
