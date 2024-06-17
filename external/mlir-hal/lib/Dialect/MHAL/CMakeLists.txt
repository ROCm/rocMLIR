add_subdirectory(IR)
add_subdirectory(Utility)

if(MHAL_ENABLE_TRANSFORMS)
add_subdirectory(Transforms)
add_subdirectory(Pipelines)
endif()
