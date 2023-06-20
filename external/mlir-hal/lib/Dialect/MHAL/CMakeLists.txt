add_subdirectory(IR)

if(MHAL_ENABLE_TRANSFORMS)
add_subdirectory(Transforms)
add_subdirectory(Pipelines)
endif()
